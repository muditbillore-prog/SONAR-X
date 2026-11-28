
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent

AUDIO_FILES = {
    "Whale": BASE_DIR / "whale.wav",
    "Ship": BASE_DIR / "ship.wav",
    "Dolphin": BASE_DIR / "dolphin.wav",
}

WINDOW_SECONDS = [0.05, 0.085, 0.10, 0.20]
OVERLAP = 0.5


def extract_features(audio, sample_rate, window_seconds):
    window_size = max(32, int(sample_rate * window_seconds))
    hop_size = max(1, int(window_size * (1 - OVERLAP)))
    features = []
    times = []

    for start in range(0, len(audio) - window_size + 1, hop_size):
        segment = audio[start:start + window_size]

        rms = np.sqrt(np.mean(segment ** 2))
        std = np.std(segment)
        mad = np.mean(np.abs(np.diff(segment)))

        spectrum = np.abs(np.fft.rfft(segment))
        frequencies = np.fft.rfftfreq(
            len(segment), d=1 / sample_rate
        )
        power = spectrum ** 2

        centroid = (
            np.sum(frequencies * spectrum) / np.sum(spectrum)
            if np.sum(spectrum) > 0 else 0.0
        )
        peak_power = np.max(power) if len(power) else 0.0

        features.append([rms, std, mad, centroid, peak_power])
        times.append(start / sample_rate)

    return np.asarray(features), np.asarray(times)


audio_data = {}

for name, path in AUDIO_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(path)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio_data[name] = (audio.astype(float), sr)

results = []

print("\nSONAR-X WINDOW SIZE STABILITY TEST")
print("=" * 75)

for window_seconds in WINDOW_SECONDS:
    all_features = []
    recording_features = {}

    for name, (audio, sr) in audio_data.items():
        features, times = extract_features(
            audio, sr, window_seconds
        )
        recording_features[name] = (features, times)
        all_features.append(features)

    combined = np.vstack(all_features)
    scaled = StandardScaler().fit_transform(combined)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.10,
        random_state=42
    )
    model.fit(scaled)

    print(f"\nWindow: {window_seconds:.3f} sec")

    offset = 0
    for name, (features, times) in recording_features.items():
        count = len(features)
        local_scaled = scaled[offset:offset + count]
        offset += count

        predictions = model.predict(local_scaled)
        flagged = int(np.sum(predictions == -1))
        total = len(predictions)
        percentage = 100 * flagged / total if total else 0

        results.append({
            "Recording": name,
            "Window_seconds": window_seconds,
            "Total_windows": total,
            "Flagged_windows": flagged,
            "Flagged_percent": round(percentage, 2),
        })

        print(
            f"  {name:8s}: {flagged:4d}/{total:<4d} "
            f"windows flagged ({percentage:.2f}%)"
        )

output = BASE_DIR / "window_stability_results.csv"
pd.DataFrame(results).to_csv(output, index=False)

print("\nResults saved to:", output)
print("\nNote: These are exploratory comparisons, not accuracy scores.")
