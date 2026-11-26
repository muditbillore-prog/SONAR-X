
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from pathlib import Path

AUDIO_FILES = [
    "background.wav",
    "whale.wav",
    "dolphin.wav",
    "ship.wav",
]

WINDOW_SECONDS = 0.1
OVERLAP = 0.5


def extract_features(file_path):
    audio, sample_rate = sf.read(file_path)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    window_size = int(sample_rate * WINDOW_SECONDS)
    hop_size = int(window_size * (1 - OVERLAP))

    if len(audio) < window_size:
        raise ValueError("Audio is shorter than one analysis window.")

    features = []

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

        peak_power = np.max(power)

        features.append({
            "rms": rms,
            "std": std,
            "mad": mad,
            "centroid": centroid,
            "peak_power": peak_power,
        })

    return pd.DataFrame(features)


def main():
    summaries = []

    for filename in AUDIO_FILES:
        path = Path(filename)

        if not path.exists():
            print(f"\nMissing file: {filename}")
            continue

        data = extract_features(path)

        row = {
            "audio": filename,
            "windows": len(data),
        }

        for feature in ["rms", "std", "mad", "centroid", "peak_power"]:
            row[f"{feature}_median"] = data[feature].median()
            row[f"{feature}_q25"] = data[feature].quantile(0.25)
            row[f"{feature}_q75"] = data[feature].quantile(0.75)

        summaries.append(row)

    if not summaries:
        print("No audio files found. Check the filenames and folder.")
        return

    result = pd.DataFrame(summaries)
    result.to_csv("feature_comparison_summary.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\nSONAR-X — Feature Comparison")
    print("=" * 80)
    print(result.round(8).to_string(index=False))
    print("\nSaved: feature_comparison_summary.csv")


if __name__ == "__main__":
    main()
