
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

AUDIO_FILES = {
    "Whale": "whale.wav",
    "Ship": "ship.wav",
    "Dolphin": "dolphin.wav",
}

WINDOW_SIZE = 2048
HOP_SIZE = 1024
CONTAMINATION = 0.10


def extract_features(audio, sample_rate):
    results = []

    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

    for start in range(0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE):
        segment = audio[start:start + WINDOW_SIZE]

        rms = np.sqrt(np.mean(segment ** 2))
        std = np.std(segment)
        mean_difference = np.mean(np.abs(np.diff(segment)))

        frequencies, power = welch(
            segment, fs=sample_rate, nperseg=WINDOW_SIZE
        )
        total_power = np.sum(power)

        if total_power > 0:
            centroid = np.sum(frequencies * power) / total_power
            peak_power = np.max(power)
        else:
            centroid = 0.0
            peak_power = 0.0

        results.append([
            rms, std, mean_difference, centroid, peak_power
        ])

    return np.asarray(results)


def main():
    folder = Path(__file__).resolve().parent
    recordings = {}

    for label, filename in AUDIO_FILES.items():
        path = folder / filename

        if not path.exists():
            print(f"Missing file: {filename}")
            print("Place all three WAV files in the SONAR-X folder.")
            return

        audio, sample_rate = sf.read(path, dtype="float32")

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if len(audio) == 0 or not np.all(np.isfinite(audio)):
            print(f"Invalid audio data: {filename}")
            return

        features = extract_features(audio, sample_rate)

        recordings[label] = {
            "filename": filename,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "features": features,
        }

    all_features = np.vstack([
        item["features"] for item in recordings.values()
    ])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)

    model = IsolationForest(
        n_estimators=150,
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(scaled_features)

    # Lower decision scores indicate more unusual windows.
    predictions = model.predict(scaled_features)
    scores = model.decision_function(scaled_features)

    rows = []
    offset = 0

    for label, item in recordings.items():
        count = len(item["features"])
        file_predictions = predictions[offset:offset + count]
        file_scores = scores[offset:offset + count]
        flagged = int(np.sum(file_predictions == -1))

        rows.append({
            "Recording": label,
            "Duration (s)": round(item["duration"], 2),
            "Sample Rate (Hz)": item["sample_rate"],
            "Windows": count,
            "Flagged Windows": flagged,
            "Flagged (%)": round(100 * flagged / count, 2),
            "Mean Anomaly Score": round(float(np.mean(file_scores)), 4),
        })

        offset += count

    results = pd.DataFrame(rows)
    print("\nSONAR-X REAL AUDIO COMPARISON")
    print("=" * 75)
    print(results.to_string(index=False))
    print("\nLower mean anomaly scores indicate more unusual windows")
    print("relative to this combined set of recordings.")

    output_path = folder / "real_audio_test_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print("\nNote: This is exploratory testing, not an accuracy evaluation.")


if __name__ == "__main__":
    main()
