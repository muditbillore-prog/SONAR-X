
import os
import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import periodogram
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

AUDIO_FILES = {
    "Whale": "whale.wav",
    "Dolphin": "dolphin.wav",
    "Ship": "ship.wav",
}

WINDOW_SECONDS = 0.10
CONTAMINATION = 0.05


def extract_features(audio, sample_rate):
    window_size = int(WINDOW_SECONDS * sample_rate)
    hop_size = window_size // 2
    rows = []

    if window_size < 16:
        raise ValueError("Audio sample rate is too low.")

    for start in range(0, len(audio) - window_size + 1, hop_size):
        segment = audio[start:start + window_size]

        rms = np.sqrt(np.mean(segment ** 2))
        std = np.std(segment)
        mad = np.mean(np.abs(np.diff(segment)))

        frequencies, power = periodogram(segment, fs=sample_rate)
        total_power = np.sum(power)

        if total_power > 0:
            centroid = np.sum(frequencies * power) / total_power
        else:
            centroid = 0.0

        peak_power = np.max(power) if len(power) else 0.0

        rows.append({
            "start_sec": start / sample_rate,
            "rms": rms,
            "std": std,
            "mad": mad,
            "centroid": centroid,
            "peak_power": peak_power,
        })

    return pd.DataFrame(rows)


all_features = []
recording_info = {}

for label, filename in AUDIO_FILES.items():
    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        print("Keep whale.wav, dolphin.wav and ship.wav in C:\\SONAR-X")
        raise SystemExit(1)

    audio, sample_rate = sf.read(filename)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        print(f"Empty audio file: {filename}")
        raise SystemExit(1)

    features = extract_features(audio, sample_rate)

    if features.empty:
        print(f"Recording too short: {filename}")
        raise SystemExit(1)

    features["recording"] = label
    all_features.append(features)

    recording_info[label] = {
        "filename": filename,
        "sample_rate": sample_rate,
        "duration_sec": len(audio) / sample_rate,
        "windows": len(features),
    }

data = pd.concat(all_features, ignore_index=True)

feature_columns = [
    "rms", "std", "mad", "centroid", "peak_power"
]

X = data[feature_columns].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(
    n_estimators=150,
    contamination=CONTAMINATION,
    random_state=42
)

data["anomaly"] = model.fit_predict(X_scaled) == -1

summary = []

for label, info in recording_info.items():
    subset = data[data["recording"] == label]
    flagged = int(subset["anomaly"].sum())
    total = len(subset)

    summary.append({
        "recording": label,
        "duration_sec": round(info["duration_sec"], 2),
        "sample_rate": info["sample_rate"],
        "windows": total,
        "flagged_windows": flagged,
        "flagged_percent": round(100 * flagged / total, 2),
    })

results = pd.DataFrame(summary)
results.to_csv("recording_comparison_results.csv", index=False)
data.to_csv("recording_comparison_windows.csv", index=False)

print("\nSONAR-X — Recording Comparison")
print("=" * 65)
print(results.to_string(index=False))
print("\nSaved:")
print("- recording_comparison_results.csv")
print("- recording_comparison_windows.csv")
print("\nImportant: These are exploratory anomaly flags, not verified")
print("anomalies or proof that a sound belongs to a particular animal.")
