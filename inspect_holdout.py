
import pandas as pd

df = pd.read_csv("holdout_test_results.csv")

print("\nSONAR-X — Holdout Flag Inspection")
print("=" * 45)

flagged = df[df["is_anomaly"] == True].copy()

print(f"Total test windows: {len(df)}")
print(f"Flagged windows: {len(flagged)}")

if flagged.empty:
    print("No flagged windows found.")
else:
    print("\nFlagged windows:")
    print(
        flagged[["start_sec", "anomaly_score"]]
        .round(4)
        .to_string(index=False)
    )

    print("\nTime span of flagged windows:")
    print(f"First: {flagged['start_sec'].min():.2f} sec")
    print(f"Last:  {flagged['start_sec'].max():.2f} sec")

    flagged.to_csv("holdout_flagged_windows.csv", index=False)
    print("\nSaved: holdout_flagged_windows.csv")
