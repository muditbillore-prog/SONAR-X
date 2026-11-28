
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 8000
WINDOW_SIZE = 2048


def make_normal_audio(rng):
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    signal = rng.normal(0, 0.12, WINDOW_SIZE)
    frequency = rng.uniform(100, 900)
    signal += rng.uniform(0.05, 0.25) * np.sin(
        2 * np.pi * frequency * t
    )
    return signal


def make_anomalous_audio(rng):
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    signal = rng.normal(0, 0.12, WINDOW_SIZE)

    if rng.random() < 0.5:
        frequency = rng.uniform(1500, 3500)
        signal += rng.uniform(0.5, 1.0) * np.sin(
            2 * np.pi * frequency * t
        )
    else:
        signal[rng.integers(100, WINDOW_SIZE - 100)] += rng.uniform(
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


print("\n========== SONAR-X STABILITY TEST ==========")
print("Testing contamination settings on 5 different datasets.")
print("All results are from simulated audio.\n")

all_results = {rate: [] for rate in [0.03, 0.04, 0.05]}

for seed in range(5):
    rng = np.random.default_rng(seed)

    features = []
    labels = []

    for _ in range(600):
        features.append(extract_features(make_normal_audio(rng)))
        labels.append(0)

    for _ in range(200):
        features.append(extract_features(make_anomalous_audio(rng)))
        labels.append(1)

    X = np.asarray(features)
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(
        X_train[y_train == 0]
    )
    X_test_scaled = scaler.transform(X_test)

    for rate in all_results:
        model = IsolationForest(
            n_estimators=150,
            contamination=rate,
            random_state=seed,
        )
        model.fit(X_train_normal)

        predictions = (
            model.predict(X_test_scaled) == -1
        ).astype(int)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        all_results[rate].append((precision, recall, f1))

    print(f"Dataset {seed + 1}/5 completed.")

print("\n========== AVERAGE RESULTS ==========")

for rate, results in all_results.items():
    averages = np.mean(results, axis=0)
    print(
        f"Contamination={rate:.2f} | "
        f"Precision={averages[0]:.3f} | "
        f"Recall={averages[1]:.3f} | "
        f"F1={averages[2]:.3f}"
    )

print("\nHigher F1 indicates a better precision-recall balance.")
print("These results do not establish real-world ocean accuracy.")
