
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Reproduce the synthetic experiment with fixed random seeds
rng = np.random.default_rng(42)

n_normal = 600
n_anomaly = 200

def make_signal(kind, rng):
    length = 2048
    t = np.arange(length) / 8000

    if kind == "normal":
        return rng.normal(0, 0.15, length)

    signal = rng.normal(0, 0.15, length)

    if rng.random() < 0.5:
        signal += 0.9 * np.sin(2 * np.pi * 2400 * t)
    else:
        signal[length // 2:length // 2 + 20] += 2.0

    return signal

def extract_features(signal):
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1 / 8000)

    total_power = spectrum.sum() + 1e-12

    return [
        np.sqrt(np.mean(signal ** 2)),
        np.std(signal),
        np.mean(np.abs(np.diff(signal))),
        np.sum(frequencies * spectrum) / total_power,
        np.max(spectrum),
    ]

normal = np.array([
    extract_features(make_signal("normal", rng))
    for _ in range(n_normal)
])

anomalies = np.array([
    extract_features(make_signal("anomaly", rng))
    for _ in range(n_anomaly)
])

X = np.vstack([normal, anomalies])
y = np.r_[np.zeros(n_normal), np.ones(n_anomaly)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Train only on normal training examples
normal_train = X_train[y_train == 0]

feature_sets = {
    "All 5 features": [0, 1, 2, 3, 4],
    "Energy + spectral": [0, 3, 4],
    "Spectral only": [3, 4],
    "Energy only": [0, 1],
}

results = []

for name, columns in feature_sets.items():
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_train[:, columns])
    test_scaled = scaler.transform(X_test[:, columns])

    for contamination in [0.02, 0.05, 0.10]:
        model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42
        )
        model.fit(normal_scaled)

        predictions = (model.predict(test_scaled) == -1).astype(int)

        results.append({
            "Feature Set": name,
            "Contamination": contamination,
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
        })

results_df = pd.DataFrame(results)
results_df.to_csv("feature_comparison_results.csv", index=False)

print("\nSONAR-X FEATURE COMPARISON")
print(results_df.round(3).to_string(index=False))
print("\nSaved: feature_comparison_results.csv")
print("Note: Synthetic test only; not real-world performance.")
