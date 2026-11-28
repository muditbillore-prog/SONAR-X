
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FILE = "background.wav"
WINDOW_SECONDS = 0.1
HOP_FRACTION = 0.5
CONTAMINATION = 0.05


def extract_features(audio, sample_rate):
    window_size = int(sample_rate * WINDOW_SECONDS)
    hop_size = int(window_size * HOP_FRACTION)
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

        rows.append([rms, std, mad, centroid, np.max(power)])

    return np.asarray(rows)


def main():
    audio, sample_rate = sf.read(FILE)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    features = extract_features(audio, sample_rate)

    # Keep chronological order: first 70% for training,
    # last 30% for testing.
    split = int(len(features) * 0.70)
    train = features[:split]
    test = features[split:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    model = IsolationForest(
        n_estimators=150,
        contamination=CONTAMINATION,
        random_state=42
    )
    model.fit(train_scaled)

    train_flags = model.predict(train_scaled) == -1
    test_flags = model.predict(test_scaled) == -1

    print("\nSONAR-X — Background Holdout Test")
    print("=" * 45)
    print(f"Total windows: {len(features)}")
    print(f"Training windows (first 70%): {len(train)}")
    print(f"Test windows (last 30%): {len(test)}")
    print(f"Training flagged: {train_flags.sum()} "
          f"({train_flags.mean() * 100:.2f}%)")
    print(f"Test flagged: {test_flags.sum()} "
          f"({test_flags.mean() * 100:.2f}%)")

    test_scores = model.decision_function(test_scaled)
    print(f"Test anomaly score min: {test_scores.min():.5f}")
    print(f"Test anomaly score max: {test_scores.max():.5f}")
    print("\nNote: These are anomaly flags, not verified real-world events.")

    results = pd.DataFrame({
        "window_index": np.arange(split, len(features)),
        "start_sec": np.arange(split, len(features))
                     * WINDOW_SECONDS * HOP_FRACTION,
        "anomaly_score": test_scores,
        "is_anomaly": test_flags
    })
    results.to_csv("holdout_test_results.csv", index=False)
    print("Saved: holdout_test_results.csv")


if __name__ == "__main__":
    main()
