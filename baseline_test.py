
import os
import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import periodogram
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

AUDIO_FILES = {
    "Background": "background.wav",
    "Whale": "whale.wav",
    "Dolphin": "dolphin.wav",
    "Ship": "ship.wav",
}

WINDOW_SECONDS = 0.10
FEATURES = ["rms", "std", "mad", "centroid", "peak_power"]


def extract_features(audio, sample_rate):
    window_size = int(WINDOW_SECONDS * sample_rate)
    hop_size = window_size // 2
    rows = []

    for start in range(0, len(audio) - window_size + 1, hop_size):
        segment = audio[start:start + window_size]

        rms = np.sqrt(np.mean(segment ** 2))
        std = np.std(segment)
        mad = np.mean(np.abs(np.diff(segment)))

        frequencies, power = periodogram(segment, fs=sample_rate)
        total_power = np.sum(power)

        centroid = (
            np.sum(frequencies * power) / total_power
            if total_power > 0 else 0.0
        )
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


def load_recording(label, filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing {filename}. Keep all WAV files in C:\\SONAR-X"
        )

    audio, sample_rate = sf.read(filename)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) < int(WINDOW_SECONDS * sample_rate):
        raise ValueError(f"{filename} is too short.")

    features = extract_features(audio, sample_rate)
    features["recording"] = label
    return features


def main():
    recordings = {}

    for label, filename in AUDIO_FILES.items():
        recordings[label] = load_recording(label, filename)

    # Fit the scaler and detector ONLY on the background recording.
    background = recordings["Background"]
    scaler = StandardScaler()
    X_background = scaler.fit_transform(background[FEATURES])

    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_background)

    results = []
    all_windows = []

    for label, frame in recordings.items():
        X = scaler.transform(frame[FEATURES])
        predictions = model.predict(X)
        scores = model.decision_function(X)

        output = frame.copy()
        output["anomaly"] = predictions == -1
        output["anomaly_score"] = scores
        all_windows.append(output)

        total = len(output)
        flagged = int(output["anomaly"].sum())

        results.append({
            "recording": label,
            "windows": total,
            "flagged_windows": flagged,
            "flagged_percent": round(100 * flagged / total, 2),
            "mean_anomaly_score": round(float(np.mean(scores)), 4),
        })

    summary = pd.DataFrame(results)
    summary.to_csv("baseline_test_results.csv", index=False)

    window_data = pd.concat(all_windows, ignore_index=True)
    window_data.to_csv("baseline_test_windows.csv", index=False)

    print("\nSONAR-X — Baseline Experiment")
    print("=" * 75)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print("- baseline_test_results.csv")
    print("- baseline_test_windows.csv")
    print("\nNote: This is a preliminary experiment using one short")
    print("background recording, not a validated accuracy test.")


if __name__ == "__main__":
    main()
