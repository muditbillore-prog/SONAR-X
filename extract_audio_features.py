import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import periodogram

def extract_features(file_path):
    audio, sample_rate = sf.read(file_path)

    # Convert stereo audio to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(float)

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # Split audio into short frames
    frame_size = int(sample_rate * 3)
    frames = [
        audio[i:i + frame_size]
        for i in range(0, len(audio), frame_size)
        if len(audio[i:i + frame_size]) > 0
    ]

    frame_rms = []
    centroids = []
    lowband_fractions = []
    band_20_200 = []
    band_200_1k = []
    band_1k_8k = []
    spectral_flux = []

    previous_spectrum = None

    for frame in frames:
        rms = np.sqrt(np.mean(frame ** 2))
        frame_rms.append(rms)

        frequencies, power = periodogram(frame, fs=sample_rate)
        total_power = power.sum() + 1e-12

        centroid = np.sum(frequencies * power) / total_power
        centroids.append(centroid)

        lowband_fractions.append(
            power[frequencies < 200].sum() / total_power
        )

        band_20_200.append(
            10 * np.log10(power[
                (frequencies >= 20) & (frequencies < 200)
            ].sum() + 1e-12)
        )
        band_200_1k.append(
            10 * np.log10(power[
                (frequencies >= 200) & (frequencies < 1000)
            ].sum() + 1e-12)
        )
        band_1k_8k.append(
            10 * np.log10(power[
                (frequencies >= 1000) & (frequencies < 8000)
            ].sum() + 1e-12)
        )

        spectrum = np.sqrt(power)
        
        if previous_spectrum is None or len(spectrum) != len(previous_spectrum):
            spectral_flux.append(0.0)
        else:
            spectral_flux.append(
            np.mean((spectrum - previous_spectrum) ** 2)
    )

        previous_spectrum = spectrum

    # Recording-level features
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    peak_dbfs = 20 * np.log10(np.max(np.abs(audio)) + 1e-12)

    return {
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "centroid_hz": np.mean(centroids),
        "lowband_frac": np.mean(lowband_fractions),
        "band_db_20_200": np.mean(band_20_200),
        "band_db_200_1k": np.mean(band_200_1k),
        "band_db_1k_8k": np.mean(band_1k_8k),
        "frame_rms_p50_dbfs": 20 * np.log10(
            np.percentile(frame_rms, 50) + 1e-12
        ),
        "frame_rms_p95_dbfs": 20 * np.log10(
            np.percentile(frame_rms, 95) + 1e-12
        ),
        "frame_rms_max_dbfs": 20 * np.log10(
            np.max(frame_rms) + 1e-12
        ),
        "spectral_flux": np.mean(spectral_flux),
    }


if __name__ == "__main__":
    file_path = input("Enter WAV filename (e.g. whale.wav): ").strip()

    try:
        features = extract_features(file_path)
        result = pd.DataFrame([features])
        print("\nExtracted acoustic features:")
        print(result.T.to_string(header=False))
        result.to_csv("audio_features.csv", index=False)
        print("\nSaved features to audio_features.csv")
    except Exception as error:
        print(f"\nCould not process audio: {error}")
