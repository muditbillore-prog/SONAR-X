
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram

FILES = {
    "Background": "background.wav",
    "Whale": "whale.wav",
    "Dolphin": "dolphin.wav",
    "Ship": "ship.wav",
}

WINDOW_SECONDS = 0.1
HOP_FRACTION = 0.5

FEATURES = ["mad", "centroid", "peak_power"]


def extract_features(audio, sr):
    window = int(sr * WINDOW_SECONDS)
    hop = int(window * HOP_FRACTION)
    rows = []

    for start in range(0, len(audio) - window + 1, hop):
        x = audio[start:start + window]

        mad = np.mean(np.abs(np.diff(x)))
        frequencies, power = periodogram(x, fs=sr)
        total_power = power.sum()

        centroid = (
            np.sum(frequencies * power) / total_power
            if total_power > 0 else 0.0
        )

        rows.append({
            "mad": mad,
            "centroid": centroid,
            "peak_power": np.max(power),
        })

    return pd.DataFrame(rows)


def main():
    all_data = []

    for label, filename in FILES.items():
        audio, sr = sf.read(filename)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        features = extract_features(audio, sr)
        features["recording"] = label
        all_data.append(features)

    data = pd.concat(all_data, ignore_index=True)
    data.to_csv("feature_distribution_data.csv", index=False)

    for feature in FEATURES:
        plt.figure(figsize=(9, 5))

        groups = [
            data.loc[data["recording"] == label, feature].to_numpy()
            for label in FILES
        ]

        plt.boxplot(groups, tick_labels=list(FILES.keys()), showfliers=False)
        plt.title(f"{feature} distribution across recordings")
        plt.xlabel("Recording")
        plt.ylabel(feature)
        plt.tight_layout()

        output = f"{feature}_distribution.png"
        plt.savefig(output, dpi=150)
        plt.close()
        print(f"Saved: {output}")

    print("Saved: feature_distribution_data.csv")
    print("Reminder: distributions show differences, not verified classification accuracy.")


if __name__ == "__main__":
    main()
