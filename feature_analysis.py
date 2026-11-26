
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAMPLE_RATE = 8000
WINDOW_SIZE = 2048
RNG = np.random.default_rng(42)

FEATURE_NAMES = [
    "RMS Energy",
    "Standard Deviation",
    "Mean Absolute Difference",
    "Spectral Centroid",
    "Peak Spectral Power",
]


def make_normal_audio():
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    signal = RNG.normal(0, 0.12, WINDOW_SIZE)
    frequency = RNG.uniform(100, 900)
    signal += RNG.uniform(0.05, 0.25) * np.sin(
        2 * np.pi * frequency * t
    )
    return signal


def make_anomalous_audio():
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    signal = RNG.normal(0, 0.12, WINDOW_SIZE)

    if RNG.random() < 0.5:
        frequency = RNG.uniform(1500, 3500)
        signal += RNG.uniform(0.5, 1.0) * np.sin(
            2 * np.pi * frequency * t
        )
    else:
        signal[RNG.integers(100, WINDOW_SIZE - 100)] += RNG.uniform(
            2.0, 4.0
        )

    return signal


def extract_features(signal):
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(
        len(signal), d=1 / SAMPLE_RATE
    )
    power = spectrum ** 2

    centroid = np.sum(frequencies * spectrum) / (
        np.sum(spectrum) + 1e-10
    )

    return [
        np.sqrt(np.mean(signal ** 2)),
        np.std(signal),
        np.mean(np.abs(np.diff(signal))),
        centroid,
        np.max(power),
    ]


rows = []

for _ in range(1000):
    for label, generator in [
        ("Normal", make_normal_audio),
        ("Anomaly", make_anomalous_audio),
    ]:
        values = extract_features(generator())
        row = dict(zip(FEATURE_NAMES, values))
        row["Class"] = label
        rows.append(row)

df = pd.DataFrame(rows)

print("\n========== SONAR-X FEATURE ANALYSIS ==========")
print("Samples: 1,000 normal + 1,000 simulated anomalies")

print("\nFeature averages by class:")
print(
    df.groupby("Class")[FEATURE_NAMES]
    .mean()
    .T
    .round(4)
    .to_string()
)

print("\nFeature spread by class (standard deviation):")
print(
    df.groupby("Class")[FEATURE_NAMES]
    .std()
    .T
    .round(4)
    .to_string()
)

print("\nGenerating feature distribution plots...")

for feature in FEATURE_NAMES:
    plt.figure(figsize=(8, 4))

    plt.hist(
        df.loc[df["Class"] == "Normal", feature],
        bins=35,
        alpha=0.6,
        label="Normal",
    )

    plt.hist(
        df.loc[df["Class"] == "Anomaly", feature],
        bins=35,
        alpha=0.6,
        label="Simulated anomaly",
    )

    plt.title(f"SONAR-X: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of samples")
    plt.legend()
    plt.tight_layout()

    filename = (
        feature.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    plt.savefig(f"feature_{filename}.png", dpi=150)
    plt.close()

print("\nSaved 5 feature plots in your project folder.")
print("Open the PNG files in VS Code to inspect them.")
print("\nNote: These findings apply only to our simulated audio.")
