
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


np.random.seed(42)

SAMPLE_RATE = 8000
N_NORMAL = 600
N_ANOMALY = 200
WINDOW_SIZE = 2048

FEATURE_NAMES = [
    "RMS Energy",
    "Standard Deviation",
    "Mean Absolute Difference",
    "Spectral Centroid",
    "Peak Spectral Power",
]


def extract_features(signal):
    signal = np.asarray(signal, dtype=float)

    rms = np.sqrt(np.mean(signal ** 2))
    std = np.std(signal)
    mean_diff = np.mean(np.abs(np.diff(signal)))

    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(
        len(signal), d=1 / SAMPLE_RATE
    )

    centroid = (
        np.sum(frequencies * spectrum) / (np.sum(spectrum) + 1e-10)
    )
    peak_power = np.max(spectrum ** 2)

    return [rms, std, mean_diff, centroid, peak_power]


def generate_normal():
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    noise = np.random.normal(0, 0.16, WINDOW_SIZE)
    background = 0.025 * np.sin(2 * np.pi * 180 * t)
    return noise + background


def generate_anomaly():
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    signal = np.random.normal(0, 0.16, WINDOW_SIZE)

    if np.random.rand() < 0.5:
        # Unusual high-frequency tone
        signal += 0.8 * np.sin(2 * np.pi * 2400 * t)
    else:
        # Short, strong impulse
        index = np.random.randint(100, WINDOW_SIZE - 100)
        signal[index:index + 20] += 3.0

    return signal


print("Generating synthetic audio examples...")

X_normal = np.array([
    extract_features(generate_normal())
    for _ in range(N_NORMAL)
])

X_anomaly = np.array([
    extract_features(generate_anomaly())
    for _ in range(N_ANOMALY)
])

X = np.vstack([X_normal, X_anomaly])
y = np.concatenate([
    np.zeros(N_NORMAL, dtype=int),
    np.ones(N_ANOMALY, dtype=int)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

results = []

print("\nTesting each feature separately...\n")

for i, name in enumerate(FEATURE_NAMES):
    # Train the scaler only on normal training examples
    normal_train = X_train[y_train == 0, i].reshape(-1, 1)
    X_test_feature = X_test[:, i].reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(normal_train)

    normal_scaled = scaler.transform(normal_train)
    test_scaled = scaler.transform(X_test_feature)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.03,
        random_state=42
    )

    model.fit(normal_scaled)

    predictions = model.predict(test_scaled)
    predictions = (predictions == -1).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Feature": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "F1 Score", ascending=False
)

print(results_df.to_string(
    index=False,
    formatters={
        "Precision": "{:.3f}".format,
        "Recall": "{:.3f}".format,
        "F1 Score": "{:.3f}".format
    }
))

results_df.to_csv("feature_importance_results.csv", index=False)

plt.figure(figsize=(10, 5))
plt.bar(results_df["Feature"], results_df["F1 Score"])
plt.ylabel("F1 Score")
plt.title("Individual Audio Feature Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("feature_importance_chart.png", dpi=150)
plt.show()

print("\nSaved:")
print("- feature_importance_results.csv")
print("- feature_importance_chart.png")
print("\nNote: Results use synthetic audio, not real ocean recordings.")
