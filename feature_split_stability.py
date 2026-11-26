
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FILE = "background.wav"
WINDOW_SECONDS = 0.1
HOP_FRACTION = 0.5

CONFIGS = {
    "All five": ["rms", "std", "mad", "centroid", "peak_power"],
    "Remove std": ["rms", "mad", "centroid", "peak_power"],
    "Remove RMS and std": ["mad", "centroid", "peak_power"],
}

SPLITS = [0.60, 0.65, 0.70, 0.75, 0.80]


def extract_features(audio, sr):
    window = int(sr * WINDOW_SECONDS)
    hop = int(window * HOP_FRACTION)
    rows = []

    for start in range(0, len(audio) - window + 1, hop):
        x = audio[start:start + window]
        rms = np.sqrt(np.mean(x ** 2))
        std = np.std(x)
        mad = np.mean(np.abs(np.diff(x)))

        freq, power = periodogram(x, fs=sr)
        total = power.sum()
        centroid = np.sum(freq * power) / total if total > 0 else 0.0

        rows.append({
            "rms": rms,
            "std": std,
            "mad": mad,
            "centroid": centroid,
            "peak_power": np.max(power),
        })

    return pd.DataFrame(rows)


def main():
    audio, sr = sf.read(FILE)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    data = extract_features(audio, sr)
    results = []

    for split_ratio in SPLITS:
        cut = int(len(data) * split_ratio)

        for name, columns in CONFIGS.items():
            train = data.iloc[:cut][columns].to_numpy()
            test = data.iloc[cut:][columns].to_numpy()

            scaler = StandardScaler()
            train = scaler.fit_transform(train)
            test = scaler.transform(test)

            model = IsolationForest(
                n_estimators=150,
                contamination=0.05,
                random_state=42,
            )
            model.fit(train)

            flags = model.predict(test) == -1

            results.append({
                "split": split_ratio,
                "configuration": name,
                "train_windows": cut,
                "test_windows": len(flags),
                "test_flags": int(flags.sum()),
                "test_flag_pct": round(flags.mean() * 100, 2),
            })

    result = pd.DataFrame(results)
    result.to_csv("feature_split_stability_results.csv", index=False)

    print("\nSONAR-X — Feature Split Stability")
    print("=" * 65)
    print(result.to_string(index=False))

    print("\nAverage test flag rate by configuration:")
    print(
        result.groupby("configuration")["test_flag_pct"]
        .agg(["mean", "std", "min", "max"])
        .round(2)
        .to_string()
    )

    print("\nSaved: feature_split_stability_results.csv")
    print("Note: stability is not the same as detection accuracy.")


if __name__ == "__main__":
    main()
