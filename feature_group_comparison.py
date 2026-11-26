
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.signal import periodogram

FILES = {
    "Background": "background.wav",
    "Whale": "whale.wav",
    "Dolphin": "dolphin.wav",
    "Ship": "ship.wav",
}

WINDOW_SECONDS = 0.1
HOP_FRACTION = 0.5


def extract_features(audio, sr):
    window = int(sr * WINDOW_SECONDS)
    hop = int(window * HOP_FRACTION)
    rows = []

    for start in range(0, len(audio) - window + 1, hop):
        x = audio[start:start + window]

        rms = np.sqrt(np.mean(x ** 2))
        std = np.std(x)
        mad = np.mean(np.abs(np.diff(x)))

        frequencies, power = periodogram(x, fs=sr)
        total_power = power.sum()
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


def main():
    summaries = []

    for label, filename in FILES.items():
        audio, sr = sf.read(filename)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        features = extract_features(audio, sr)

        for feature in ["rms", "std", "mad", "centroid", "peak_power"]:
            summaries.append({
                "recording": label,
                "sample_rate": sr,
                "duration_sec": round(len(audio) / sr, 2),
                "feature": feature,
                "median": features[feature].median(),
                "q25": features[feature].quantile(0.25),
                "q75": features[feature].quantile(0.75),
            })

    result = pd.DataFrame(summaries)
    result.to_csv("feature_group_comparison_results.csv", index=False)

    print("\nSONAR-X — Recording Feature Comparison")
    print("=" * 65)

    for feature in ["rms", "std", "mad", "centroid", "peak_power"]:
        print(f"\n--- {feature} ---")
        subset = result[result["feature"] == feature]
        print(
            subset[
                ["recording", "median", "q25", "q75"]
            ].round(8).to_string(index=False)
        )

    print("\nSaved: feature_group_comparison_results.csv")
    print(
        "\nReminder: these are recording-level feature comparisons, "
        "not verified anomaly or species classifications."
    )


if __name__ == "__main__":
    main()
