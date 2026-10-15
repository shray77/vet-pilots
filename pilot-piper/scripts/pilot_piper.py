# Пилот 2: Piper TTS — русская аудио-версия недельного дайджеста vet-digest-llm (2026-W41)
# Движок: rhasspy/piper 2023.11.14-2 (CPU, onnxruntime), голос ru_RU-dmitri-medium (22.05 кГц)
import os, json, re, subprocess, wave, contextlib

BASE = "/home/z/my-project"
DIGEST = f"{BASE}/repos/vet-digest-llm/data/llm_digest.json"
PIPER_DIR = f"{BASE}/pilot_data/piper/piper"          # бинарник + espeak-ng-data рядом
VOICE = f"{BASE}/pilot_data/piper/ru_RU-dmitri-medium.onnx"
OUT = f"{BASE}/pilot_out/piper"
os.makedirs(OUT, exist_ok=True)

d = json.load(open(DIGEST))

# ---------- 1. Нормализация текста под русский TTS ----------
def tts_text(dig):
    # флайвеи: аббревиатуры из vetflu -> произносимые полные названия;
    # первое предложение пересобираем целиком (грамматика для TTS)
    s = dig["summary"]
    s = re.sub(
        r"^Главные события недели:.*?R-индекс\s*(\d+)[–-](\d+)\)\.",
        r"Главные события недели: высокие окна риска птичьего гриппа отмечены "
        r"в Восточно-Атлантическом, Черноморском, Центрально-Азиатском и "
        r"Восточно-Азиатском флайвеях, Р-индекс от \1 до \2.",
        s, count=1)
    s = s.replace("R-индекс", "Р-индекс")
    b = dig["bullets"][0] if dig.get("bullets") else ""
    b = re.sub(r"\s*-\s*(Эпизоотология|Фармаконадзор|Наука):", r"\1: ", b.replace(" - ", ". "))
    p = dig["practical"]
    disc = re.sub(r"\s*\(Qwen3-1\.7B\)", "", dig["disclaimer"])
    disc = disc.replace("малой LLM", "малой языковой моделью")
    parts = [
        f"Ветеринарный дайджест, неделя {dig['week_key'].split('-W')[1]} {dig['week_key'].split('-')[0]} года, "
        "с девятого по пятнадцатое октября.",
        s, b, p, disc,
    ]
    return " ".join(x.strip() for x in parts if x.strip())

text = tts_text(d)
open(f"{OUT}/digest_w41.txt", "w").write(text)
print(f"текст: {len(text)} символов")
print(text[:300], "...")

# ---------- 2. Синтез ----------
wav_path = f"{OUT}/digest_w41.wav"
env = dict(os.environ, LD_LIBRARY_PATH=PIPER_DIR)
proc = subprocess.run(
    [f"{PIPER_DIR}/piper", "--model", VOICE, "--output_file", wav_path,
     "--sentence_silence", "0.25"],
    input=text.encode("utf-8"), env=env, capture_output=True)
if proc.returncode != 0:
    print("PIPER STDERR:", proc.stderr.decode()[:1000]); raise SystemExit(1)

with contextlib.closing(wave.open(wav_path)) as w:
    frames, rate = w.getnframes(), w.getframerate()
    dur = frames / rate
print(f"WAV: {dur:.1f} c, {rate} Гц, {os.path.getsize(wav_path)/1e6:.1f} МБ")

# ---------- 3. Сжатие в mp3 ----------
mp3_path = f"{OUT}/digest_w41.mp3"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                "-codec:a", "libmp3lame", "-q:a", "5", "-ar", "22050", "-ac", "1", mp3_path],
               check=True)
pb = subprocess.run(["ffprobe", "-v", "quiet", "-show_format", "-of", "json", mp3_path],
                    capture_output=True, check=True)
mp3_dur = float(json.loads(pb.stdout)["format"]["duration"])
print(f"MP3: {mp3_dur:.1f} c, {os.path.getsize(mp3_path)/1e3:.0f} кБ")
print(f"темп: {len(text)/mp3_dur:.1f} симв/с (норма Piper medium ~13-16)")

json.dump({
    "pilot": "piper-tts-ru",
    "engine": "rhasspy/piper 2023.11.14-2 (CPU onnxruntime)",
    "voice": "ru_RU-dmitri-medium (22.05 kHz, Apache/EUPL, rhasspy/piper-voices)",
    "source": "vet-digest-llm data/llm_digest.json, week 2026-W41",
    "chars": len(text), "wav_sec": round(dur, 1), "mp3_sec": round(mp3_dur, 1),
    "mp3_kb": round(os.path.getsize(mp3_path) / 1e3),
    "chars_per_sec": round(len(text) / mp3_dur, 1),
    "normalization": ["флайвеи раскрыты", "R-индекс -> Р-индекс, 100–85 -> от 100 до 85",
                      "LLM -> языковая модель", "даты развёрнуты"],
}, open(f"{OUT}/tts_stats.json", "w"), ensure_ascii=False, indent=2)
print("АРТЕФАКТЫ:", sorted(os.listdir(OUT)))
