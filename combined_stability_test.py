
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


SAMPLE_RATE = 8000
WINDOW_SIZE = 2048
N_NORMAL = 600
N_ANOMALY = 200

CONTAMINATIONS = [0.02, 0.03, 0.04, 0.05, 0.10]
SEEDS = [0, 1, 2, 3, 4]


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
        np.sum(frequencies * spectrum)
        / (np.sum(spectrum) + 1e-10)
    )
    peak_power = np.max(spectrum ** 2)

    return [rms, std, mean_diff, centroid, peak_power]


def generate_normal(rng):
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    noise = rng.normal(0, 0.16, WINDOW_SIZE)
    background = 0.025 * np.sin(2 * np.pi * 180 * t)
    return noise + background


def generate_anomaly(rng):
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    signal = rng.normal(0, 0.16, WINDOW_SIZE)

    if rng.random() < 0.5:
        signal += 0.8 * np.sin(2 * np.pi * 2400 * t)
    else:
        index = rng.integers(100, WINDOW_SIZE - 100)
        signal[index:index + 20] += 3.0

    return signal


def create_dataset(seed):
    rng = np.random.default_rng(seed)

    normal = np.array([
        extract_features(generate_normal(rng))
        for _ in range(N_NORMAL)
    ])

    anomaly = np.array([
        extract_features(generate_anomaly(rng))
        for _ in range(N_ANOMALY)
    ])

    X = np.vstack([normal, anomaly])
    y = np.concatenate([
        np.zeros(N_NORMAL, dtype=int),
        np.ones(N_ANOMALY, dtype=int)
    ])

    return X, y


all_results = []

print("SONAR-X: Combined Model Stability Test")
print("=" * 65)

for seed in SEEDS:
    X, y = create_dataset(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=seed + 100,
        stratify=y
    )

    # Fit preprocessing only on normal training samples
    normal_train = X_train[y_train == 0]

    scaler = StandardScaler()
    scaler.fit(normal_train)

    normal_scaled = scaler.transform(normal_train)
    test_scaled = scaler.transform(X_test)

    for contamination in CONTAMINATIONS:
        model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=seed
        )

        model.fit(normal_scaled)

        predictions = (
            model.predict(test_scaled) == -1
        ).astype(int)

        all_results.append({
            "Seed": seed,
            "Contamination": contamination,
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

results_df = pd.DataFrame(all_results)

print("\nResults for each run:")
print(
    results_df.to_string(
        index=False,
        formatters={
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
            "F1 Score": "{:.3f}".format
        }
    )
)

summary = results_df.groupby("Contamination").agg(
    Mean_Precision=("Precision", "mean"),
    Mean_Recall=("Recall", "mean"),
    Mean_F1=("F1 Score", "mean"),
    F1_Std=("F1 Score", "std")
).reset_index()

summary = summary.sort_values("Mean_F1", ascending=False)

print("\nAverage performance across five seeds:")
print(
    summary.to_string(
        index=False,
        formatters={
            "Mean_Precision": "{:.3f}".format,
            "Mean_Recall": "{:.3f}".format,
            "Mean_F1": "{:.3f}".format,
            "F1_Std": "{:.3f}".format
        }
    )
)

results_df.to_csv("combined_stability_runs.csv", index=False)
summary.to_csv("combined_stability_summary.csv", index=False)

print("\nSaved:")
print("- combined_stability_runs.csv")
print("- combined_stability_summary.csv")
print("\nReminder: This evaluates synthetic signals, not real underwater recordings.")
