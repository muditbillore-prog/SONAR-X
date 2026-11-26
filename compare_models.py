
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

data = pd.read_csv("clip_acoustic_stats.csv")

classes = ["background", "cargo", "tug"]
data = data[data["label"].isin(classes)].copy()

features = [
    "rms_dbfs",
    "peak_dbfs",
    "centroid_hz",
    "lowband_frac",
    "band_db_20_200",
    "band_db_200_1k",
    "band_db_1k_8k",
    "frame_rms_p50_dbfs",
    "frame_rms_p95_dbfs",
    "frame_rms_max_dbfs",
    "spectral_flux",
]

data[features] = data[features].replace(
    [float("inf"), -float("inf")], float("nan")
)

train = data[data["split"] == "train"]
val = data[data["split"] == "val"]

X_train, y_train = train[features], train["label"]
X_val, y_val = val[features], val["label"]

settings = [
    {"n_estimators": 100, "min_samples_leaf": 1},
    {"n_estimators": 200, "min_samples_leaf": 1},
    {"n_estimators": 200, "min_samples_leaf": 2},
    {"n_estimators": 300, "min_samples_leaf": 3},
]

for config in settings:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            **config,
            random_state=42,
            class_weight="balanced",
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    print("\n" + "=" * 55)
    print("Settings:", config)
    print("Validation accuracy:",
          round(accuracy_score(y_val, predictions) * 100, 2), "%")

    print("\nClassification report:")
    print(classification_report(
        y_val,
        predictions,
        labels=classes,
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_val, predictions, labels=classes),
        index=classes,
        columns=classes,
    ))
