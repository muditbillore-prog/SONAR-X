
import pandas as pd

df = pd.read_csv("audio_window_inspection.csv")

print("\nSONAR-X REAL AUDIO SUMMARY")
print("=" * 65)

for recording, group in df.groupby("Recording"):
    flagged = group[group["Flagged"] == True]

    print(f"\nRecording: {recording}")
    print(f"Total windows: {len(group)}")
    print(f"Flagged windows: {len(flagged)}")
    print(f"Flagged percentage: {len(flagged) / len(group) * 100:.2f}%")

    if not flagged.empty:
        print(
            f"Flagged RMS range: "
            f"{flagged['RMS'].min():.4f} - "
            f"{flagged['RMS'].max():.4f}"
        )
        print(
            f"Flagged spectral centroid range: "
            f"{flagged['Spectral Centroid'].min():.1f} - "
            f"{flagged['Spectral Centroid'].max():.1f} Hz"
        )

print("\nNote: These are exploratory detections, not verified anomalies.")
