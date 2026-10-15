# TSPulse weekly — «аномалия недели» на суточном ряду вспышек (прод-вариант пилота Task 71)
# Запуск: python3 tspulse/run_weekly.py [out_dir]
# Вход:  outbreaks.json из репо vet-heatmap (raw.githubusercontent, свежий бот-срез)
# Выход: tspulse/latest.json (человекочитаемая сводка для ВетХаба и дайджеста) + tspulse/chart.png
import os, sys, json, urllib.request, collections

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
OUT = sys.argv[1] if len(sys.argv) > 1 else "tspulse"
os.makedirs(OUT, exist_ok=True)
SRC_URL = "https://raw.githubusercontent.com/shray77/vet-heatmap/main/public/data/outbreaks.json"
THR = 0.6

# ---------- данные ----------
req = urllib.request.Request(SRC_URL, headers={"User-Agent": "vet-pilots/1.0"})
raw = json.loads(urllib.request.urlopen(req, timeout=60).read())
obs = raw["outbreaks"]
df_o = pd.DataFrame(obs)
df_o["date"] = pd.to_datetime(df_o["date"], errors="coerce")
df_o = df_o.dropna(subset=["date"])

END = df_o["date"].max().normalize()
START = END - pd.DateOffset(years=6)
win = df_o[(df_o["date"] >= START) & (df_o["date"] <= END)]
daily = win.groupby(win["date"].dt.normalize()).size().reindex(
    pd.date_range(START, END, freq="D"), fill_value=0).rename("count")
ts_df = daily.reset_index()
ts_df.columns = ["timestamp", "x"]
ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
print(f"ряд {len(ts_df)} точек {START.date()}..{END.date()}, среднее {daily.mean():.2f}/день")

# ---------- модель (та же, что в пилоте: revision=main = AD-вариант) ----------
model = TSPulseForReconstruction.from_pretrained(
    "ibm-granite/granite-timeseries-tspulse-r1", revision="main")
pipeline = TimeSeriesAnomalyDetectionPipeline(
    model, timestamp_column="timestamp", target_columns=["x"],
    prediction_mode=[AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                     AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value],
    aggregation_length=64, aggr_function="max", smoothing_length=8,
    least_significant_scale=0.01, least_significant_score=0.1)
result = pipeline(ts_df, batch_size=256, predictive_score_smoothing=False).reset_index()
result["timestamp"] = pd.to_datetime(result["timestamp"])
print(f"score [{result.anomaly_score.min():.3f}..{result.anomaly_score.max():.3f}]")

# ---------- сводка недели ----------
meta = df_o.set_index("date")
def tally(lo, hi):
    g = df_o[(df_o["date"] > lo) & (df_o["date"] <= hi)]
    return {k: int(v) for k, v in collections.Counter(g["disease"]).most_common(5)}

last7 = result[result["timestamp"] > END - pd.Timedelta(days=7)]
hist = result["anomaly_score"].values
w = last7["anomaly_score"].values
mean_s, max_s = float(w.mean()), float(w.max())
pct_mean = float((hist < mean_s).mean() * 100)
pct_max = float((hist < max_s).mean() * 100)
verdict = "аномалия" if pct_max >= 95 else "повышенный фон" if pct_max >= 80 else "норма"
notes = {
    "норма": "Аномальных всплесков суточного числа вспышек за неделю не выявлено.",
    "повышенный фон": "Умеренный подъём активности вспышек относительно обычного уровня.",
    "аномалия": "Зафиксирован аномальный всплеск суточных вспышек — см. карту и топ-регионы.",
}
recent90 = result[result["timestamp"] > END - pd.Timedelta(days=90)]
def top_rows(dfr, n):
    out = []
    for _, r in dfr.nlargest(n, "anomaly_score").iterrows():
        d = r["timestamp"]
        dis = {}
        if d in meta.index:
            dis = {k: int(v) for k, v in collections.Counter(meta.loc[[d]]["disease"]).most_common(3)}
        out.append({"date": str(d.date()), "score": round(float(r["anomaly_score"]), 3),
                    "count": int(r["x"]), "diseases": dis})
    return out

latest = {
    "service": "vet-pilots/tspulse",
    "generated_at": pd.Timestamp.utcnow().isoformat(),
    "model": "ibm-granite/granite-timeseries-tspulse-r1 @ main (1M params, CPU)",
    "week": {
        "window": [str((END - pd.Timedelta(days=6)).date()), str(END.date())],
        "verdict": verdict, "note": notes[verdict],
        "mean_score": round(mean_s, 3), "max_score": round(max_s, 3),
        "percentile_mean": round(pct_mean, 1), "percentile_max": round(pct_max, 1),
        "threshold": THR, "days_over_threshold": int((w > THR).sum()),
    },
    "week_diseases": tally(END - pd.Timedelta(days=7), END),
    "recent_anomalies_90d": top_rows(recent90, 3),
    "all_time_top": top_rows(result, 1),
    "series": {"points": int(len(result)), "from": str(START.date()), "to": str(END.date()),
               "mean_daily": round(float(daily.mean()), 2), "max_daily": int(daily.max())},
}
json.dump(latest, open(f"{OUT}/latest.json", "w"), ensure_ascii=False, indent=2)

# ---------- график ----------
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
ax2 = ax.twinx()
ax.plot(result["timestamp"], result["x"], color="gray", lw=0.7, label="вспышек за сутки")
ax2.plot(result["timestamp"], result["anomaly_score"], color="#e8890c", lw=0.9,
         label="оценка аномальности (TSPulse)")
ax2.axhline(THR, color="red", ls="--", lw=0.8, label="порог 0.6")
lastv = latest["week"]
ax2.annotate(f'неделя: {lastv["verdict"]} (max {lastv["max_score"]})',
             (last7["timestamp"].iloc[0], max_s), xytext=(0, 8),
             textcoords="offset points", fontsize=9, color="darkred")
ax.set_title("TSPulse: аномальность суточного числа вспышек (окно 6 лет)", fontsize=13)
ax.set_ylabel("вспышек/сутки"); ax2.set_ylabel("anomaly score (0-1)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncols=3, frameon=False)
for s_ in ["top"]: ax.spines[s_].set_visible(False); ax2.spines[s_].set_visible(False)
fig.savefig(f"{OUT}/chart.png", dpi=150)
print("VERDICT:", verdict, "|", notes[verdict])
print("OK:", sorted(os.listdir(OUT)))
