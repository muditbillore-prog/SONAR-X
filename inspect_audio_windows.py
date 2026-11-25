
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FILES = {
    "Whale": "whale.wav",
    "Ship": "ship.wav",
    "Dolphin": "dolphin.wav",
}

WINDOW_SIZE = 2048
HOP_SIZE = 1024
CONTAMINATION = 0.10


def extract_windows(audio, sr, label):
    rows = []
    features = []

    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

    for start in range(0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE):
        segment = audio[start:start + WINDOW_SIZE]
        rms = np.sqrt(np.mean(segment ** 2))
        std = np.std(segment)
        mad = np.mean(np.abs(np.diff(segment)))

        frequencies, power = welch(
            segment, fs=sr, nperseg=WINDOW_SIZE
        )
        total_power = np.sum(power)
        centroid = (
            np.sum(frequencies * power) / total_power
            if total_power > 0 else 0.0
        )
        peak_power = np.max(power) if total_power > 0 else 0.0

        features.append([rms, std, mad, centroid, peak_power])
        rows.append({
            "Recording": label,
            "Start (s)": start / sr,
            "End (s)": (start + WINDOW_SIZE) / sr,
            "RMS": rms,
            "Standard Deviation": std,
            "Mean Absolute Difference": mad,
            "Spectral Centroid": centroid,
            "Peak Spectral Power": peak_power,
        })

    return rows, np.asarray(features)


def main():
    folder = Path(__file__).resolve().parent
    all_rows = []
    all_features = []
    lengths = {}

    for label, filename in FILES.items():
        path = folder / filename

        if not path.exists():
            print(f"Missing file: {filename}")
            print("Put whale.wav, ship.wav and dolphin.wav in C:\\SONAR-X")
            return

        audio, sr = sf.read(path, dtype="float32")

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if len(audio) == 0 or not np.all(np.isfinite(audio)):
            print(f"Invalid audio data: {filename}")
            return

        rows, features = extract_windows(audio, sr, label)
        all_rows.extend(rows)
        all_features.extend(features)
        lengths[label] = len(rows)

    X = np.asarray(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=150,
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    offset = 0
    for label in FILES:
        count = lengths[label]
        for i in range(offset, offset + count):
            all_rows[i]["Anomaly Score"] = float(scores[i])
            all_rows[i]["Flagged"] = predictions[i] == -1
        offset += count

    results = pd.DataFrame(all_rows)
    output = folder / "audio_window_inspection.csv"
    results.to_csv(output, index=False)

    print("\nAUDIO WINDOW INSPECTION")
    print("=" * 65)

    for label in FILES:
        subset = results[results["Recording"] == label]
        flagged = subset[subset["Flagged"]]

        print(f"\n{label}: {len(flagged)} of {len(subset)} windows flagged")

        if flagged.empty:
            print("No windows flagged.")
            continue

        print(
            flagged.nsmallest(10, "Anomaly Score")[
                ["Start (s)", "End (s)", "Anomaly Score", "RMS",
                 "Spectral Centroid"]
            ].round(4).to_string(index=False)
        )

    print(f"\nFull results saved to: {output}")
    print("Lower scores indicate more unusual windows in this combined set.")
    print("These flags are exploratory, not verified detections.")


if __name__ == "__main__":
    main()
