import pandas as pd
import joblib
import numpy as np

# Load training data, extracted audio features, and trained model
data = pd.read_csv("clip_acoustic_stats.csv")
audio = pd.read_csv("audio_features.csv")
saved = joblib.load("sonar_classifier.joblib")

features = saved["features"]

# Use only the training split for comparison
train = data[data["split"] == "train"]

print("\nSONAR-X FEATURE COMPATIBILITY CHECK")
print("=" * 50)

print("\nComparing whale.wav features with training data:")

for feature in features:
    train_values = pd.to_numeric(
        train[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    value = float(audio.iloc[0][feature])

    if train_values.empty:
        print(f"\n{feature}: No valid training values")
        continue

    low = train_values.quantile(0.05)
    high = train_values.quantile(0.95)

    status = "WITHIN RANGE" if low <= value <= high else "OUTSIDE RANGE"

    print(f"\nFeature: {feature}")
    print(f"  Whale value: {value:.6g}")
    print(f"  Training 5th–95th percentile: {low:.6g} to {high:.6g}")
    print(f"  Status: {status}")

# Experimental prediction only
model = saved["model"]
prediction = model.predict(audio[features])[0]

print("\n" + "=" * 50)
print("Experimental prediction:", prediction)
print("IMPORTANT: This prediction is not validated.")
print("Feature compatibility must be checked before relying on it.")
