
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

        rows.append({
            "rms": rms,
            "std": std,
            "mad": mad,
            "centroid": centroid,
            "peak_power": np.max(power),
        })

    return pd.DataFrame(rows)


def evaluate(name, data, train_end, feature_names):
    train = data.iloc[:train_end][feature_names].to_numpy()
    test = data.iloc[train_end:][feature_names].to_numpy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    model = IsolationForest(
        n_estimators=150,
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(train_scaled)

    train_flags = model.predict(train_scaled) == -1
    test_flags = model.predict(test_scaled) == -1
    scores = model.decision_function(test_scaled)

    return {
        "experiment": name,
        "features": ", ".join(feature_names),
        "train_flag_pct": round(train_flags.mean() * 100, 2),
        "test_flag_pct": round(test_flags.mean() * 100, 2),
        "test_flag_count": int(test_flags.sum()),
        "test_windows": len(test_flags),
        "test_score_min": round(float(scores.min()), 5),
        "test_score_median": round(float(np.median(scores)), 5),
    }


def main():
    audio, sample_rate = sf.read(FILE)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    data = extract_features(audio, sample_rate)
    train_end = int(len(data) * 0.70)

    experiments = {
        "All five features": [
            "rms", "std", "mad", "centroid", "peak_power"
        ],
        "Remove std": [
            "rms", "mad", "centroid", "peak_power"
        ],
        "Remove rms and std": [
            "mad", "centroid", "peak_power"
        ],
    }

    results = []

    for name, features in experiments.items():
        results.append(evaluate(name, data, train_end, features))

    output = pd.DataFrame(results)
    output.to_csv("feature_ablation_results.csv", index=False)

    print("\nSONAR-X — Feature Ablation Experiment")
    print("=" * 70)
    print(f"Total windows: {len(data)}")
    print(f"Training windows: {train_end}")
    print(f"Test windows: {len(data) - train_end}")
    print("\nResults:")
    print(output.to_string(index=False))
    print("\nSaved: feature_ablation_results.csv")
    print(
        "\nReminder: this is a comparison of flag behavior, "
        "not a measure of detection accuracy."
    )


if __name__ == "__main__":
    main()
