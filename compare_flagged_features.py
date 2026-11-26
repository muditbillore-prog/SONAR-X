
import pandas as pd
import numpy as np

baseline = pd.read_csv("baseline_test_windows.csv")
holdout = pd.read_csv("holdout_test_results.csv")

# Keep only the background recording
background = baseline[
    baseline["recording"].astype(str).str.lower() == "background"
].copy()

features = ["rms", "std", "mad", "centroid", "peak_power"]

if background.empty:
    print("No Background rows found.")
else:
    flagged_times = holdout.loc[
        holdout["is_anomaly"].astype(bool), "start_sec"
    ].to_numpy()

    # Match only identical time windows, within floating-point tolerance
    background["is_flagged"] = background["start_sec"].apply(
        lambda t: np.any(np.isclose(flagged_times, t, atol=1e-6))
    )

    print("\nSONAR-X — Correct Background Comparison")
    print("=" * 50)
    print("Background windows:", len(background))
    print("Matched flagged windows:", int(background["is_flagged"].sum()))

    for name, group in [
        ("Flagged", background[background["is_flagged"]]),
        ("Non-flagged", background[~background["is_flagged"]]),
    ]:
        print(f"\n{name} windows: {len(group)}")
        if not group.empty:
            print(group[features].median().to_string())

    print(
        "\nNote: Matching by time is valid only when the files "
        "use the same background recording and window settings."
    )
