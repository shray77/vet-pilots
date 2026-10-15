# Пилот 1: IBM Granite TSPulse — детекция аномалий на суточном ряду вспышек (vet-heatmap)
# Модель: ibm-granite/granite-timeseries-tspulse-r1 (1M параметров), revision=main (AD-вариант)
# API по официальному ноутбуку granite-tsfm/notebooks/hfdemo/tspulse_anomaly_detection.ipynb
import os, json, collections
os.environ.setdefault("HF_HOME", "/home/z/my-project/.hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

torch.set_num_threads(4)
OUT = "/home/z/my-project/pilot_out/tspulse"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Данные: вспышки vet-heatmap -> суточный ряд ----------
raw = json.load(open("/home/z/my-project/repos/vet-heatmap/public/data/outbreaks.json"))
obs = raw["outbreaks"]
df_o = pd.DataFrame(obs)
df_o["date"] = pd.to_datetime(df_o["date"], errors="coerce")
df_o = df_o.dropna(subset=["date"])
print(f"вспышек всего: {len(df_o)}, горизонт {df_o.date.min().date()} .. {df_o.date.max().date()}")

# окно: последние 6 лет (требование TSPulse AD >=1536-2048 точек; ранние годы исторического
# слоя FSVPS кураторски неровные, 6 лет дают устойчивую годовую сезонность)
END = df_o["date"].max().normalize()
START = END - pd.DateOffset(years=6)
win = df_o[(df_o["date"] >= START) & (df_o["date"] <= END)]
daily = win.groupby(win["date"].dt.normalize()).size()
full_idx = pd.date_range(START, END, freq="D")
daily = daily.reindex(full_idx, fill_value=0).rename("count")
print(f"ряд: {len(daily)} точек, {START.date()} .. {END.date()}, среднее {daily.mean():.2f}/день, макс {daily.max()}")

ts_df = daily.reset_index()
ts_df.columns = ["timestamp", "x"]
ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
ts_df.to_csv(f"{OUT}/vet_outbreaks_daily.csv", index=False)

# ---------- 2. Модель + пайплайн (параметры официального ноутбука) ----------
print("загрузка TSPulse (revision=main, AD-вариант allhead-512-p8)...")
model = TSPulseForReconstruction.from_pretrained(
    "ibm-granite/granite-timeseries-tspulse-r1", revision="main"
)
pipeline = TimeSeriesAnomalyDetectionPipeline(
    model,
    timestamp_column="timestamp",
    target_columns=["x"],
    prediction_mode=[
        AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
        AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
    ],
    aggregation_length=64,
    aggr_function="max",
    smoothing_length=8,
    least_significant_scale=0.01,
    least_significant_score=0.1,
)
print("инференс (CPU, 1M параметров)...")
result = pipeline(ts_df, batch_size=256, predictive_score_smoothing=False)
result = result.reset_index()
result["timestamp"] = pd.to_datetime(result["timestamp"])
print(f"оценки готовы: {len(result)} строк, score [{result.anomaly_score.min():.3f} .. {result.anomaly_score.max():.3f}]")

result[["timestamp", "x", "anomaly_score"]].to_csv(f"{OUT}/anomaly_scores.csv", index=False)

# ---------- 3. Анализ: топ-аномалии + аномалия недели ----------
THR = 0.6  # порог из официального ноутбука (шкала 0..1)
meta = df_o.set_index("date")

def breakdown(day):
    if day not in meta.index:
        return {"outbreaks": 0, "diseases": {}, "top_regions": []}
    g = meta.loc[[day]]
    dis = collections.Counter(g["disease"])
    reg = collections.Counter(g["region"])
    return {
        "outbreaks": int(len(g)),
        "diseases": {k: int(v) for k, v in dis.most_common(4)},
        "top_regions": [f"{r} ({c})" for r, c in reg.most_common(3)],
        "cases": int(pd.to_numeric(g.get("cases"), errors="coerce").fillna(0).sum()),
    }

top = result.nlargest(10, "anomaly_score")
top_list = []
for _, r in top.iterrows():
    d = r["timestamp"]
    top_list.append({
        "date": str(d.date()),
        "score": round(float(r["anomaly_score"]), 3),
        "count": int(r["x"]),
        **breakdown(d),
    })

# аномалия недели: последние 7 дней против всей истории оценок
last7 = result[result["timestamp"] > END - pd.Timedelta(days=7)]
hist = result["anomaly_score"].values
week_scores = last7["anomaly_score"].values
week_mean = float(week_scores.mean()) if len(week_scores) else None
week_max = float(week_scores.max()) if len(week_scores) else None
pct_mean = float((hist < week_mean).mean() * 100) if week_mean is not None else None
pct_max = float((hist < week_max).mean() * 100) if week_max is not None else None
verdict = ("аномалия" if pct_max is not None and pct_max >= 95 else
           "повышенный фон" if pct_max is not None and pct_max >= 80 else "норма")

week = {
    "window": [str((END - pd.Timedelta(days=6)).date()), str(END.date())],
    "days_in_window": int(len(last7)),
    "mean_score": round(week_mean, 3) if week_mean is not None else None,
    "max_score": round(week_max, 3) if week_max is not None else None,
    "percentile_vs_history_mean": round(pct_mean, 1) if pct_mean is not None else None,
    "percentile_vs_history_max": round(pct_max, 1) if pct_max is not None else None,
    "threshold": THR,
    "days_over_threshold": int((week_scores > THR).sum()) if len(week_scores) else 0,
    "verdict": verdict,
    "week_breakdown": [breakdown(d) for d in last7["timestamp"]][-3:],
}

summary = {
    "pilot": "tspulse-anomaly-week",
    "model": "ibm-granite/granite-timeseries-tspulse-r1 @ main (1M params, CPU)",
    "series": {"source": "vet-heatmap outbreaks.json (WAHIS+FSVPS)",
               "points": int(len(result)), "from": str(START.date()), "to": str(END.date()),
               "mean_daily": round(float(daily.mean()), 2)},
    "anomaly_of_week": week,
    "top10_all_time": top_list,
    "pipeline": {"prediction_mode": ["time", "fft"], "aggregation_length": 64,
                 "aggr_function": "max", "smoothing_length": 8, "threshold": THR},
}
json.dump(summary, open(f"{OUT}/pilot_summary.json", "w"), ensure_ascii=False, indent=2)

# ---------- 4. График ----------
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
ax2 = ax.twinx()
ax.plot(result["timestamp"], result["x"], color="gray", lw=0.7, label="вспышек за сутки")
ax2.plot(result["timestamp"], result["anomaly_score"], color="#e8890c", lw=0.9, label="оценка аномальности (TSPulse)")
ax2.axhline(THR, color="red", ls="--", lw=0.8, label="порог 0.6")
for t in top_list[:3]:
    ax2.annotate(str(t["date"]), (pd.Timestamp(t["date"]), t["score"]),
                 xytext=(0, 6), textcoords="offset points", fontsize=8, color="darkred")
ax.set_title("Пилот TSPulse: аномальность суточного числа вспышек, окно 6 лет", fontsize=13)
ax.set_ylabel("вспышек/сутки"); ax2.set_ylabel("anomaly score (0-1)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncols=3, frameon=False)
for s in ["top"]: ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
fig.savefig(f"{OUT}/anomaly_chart.png", dpi=150)
print("АРТЕФАКТЫ:", sorted(os.listdir(OUT)))
print("ВЕРДИКТ НЕДЕЛИ:", json.dumps(week, ensure_ascii=False)[:600])
