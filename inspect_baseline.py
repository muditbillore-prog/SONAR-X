
import pandas as pd

# Read the window-level results from our previous experiment.
df = pd.read_csv("baseline_test_windows.csv")

background = df[df["recording"] == "Background"].copy()

print("\nSONAR-X — Background Baseline Inspection")
print("=" * 55)

print("Total windows:", len(background))
print("Flagged windows:", int(background["anomaly"].sum()))
print("Flagged percentage:",
      round(background["anomaly"].mean() * 100, 2), "%")

print("\nFeature ranges:")
features = ["rms", "std", "mad", "centroid", "peak_power"]
print(background[features].describe().round(4).to_string())

print("\nFirst 10 flagged windows:")
flagged = background[background["anomaly"]]

if flagged.empty:
    print("No windows flagged.")
else:
    print(
        flagged[
            ["start_sec", *features, "anomaly_score"]
        ].head(10).round(4).to_string(index=False)
    )

background.to_csv("background_inspection.csv", index=False)
print("\nSaved: background_inspection.csv")

print("\nPeak power precise values:")
print(
    background["peak_power"]
    .describe()
    .to_string()
)

print(
    "Non-zero peak power values:",
    (background["peak_power"] > 0).sum()
)
