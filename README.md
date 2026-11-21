# SONAR-X — Underwater Acoustic Monitoring System

## Project Overview

SONAR-X is a Python-based research prototype for exploring underwater acoustic signals. It provides an interactive dashboard for audio inspection, signal visualization, and exploratory anomaly detection.

The application supports synthetic acoustic simulation and WAV audio uploads. It displays waveform and spectrogram visualizations, calculates audio-window features, highlights potentially unusual windows, and allows analysis results to be exported as CSV.

## Key Features

- Synthetic underwater-style acoustic signal generation
- WAV audio upload and inspection
- Waveform visualization
- Frequency-time spectrogram
- Audio-window feature extraction
- Exploratory anomaly detection using Isolation Forest
- Anomaly review table and window-level results
- CSV export of analysis results
- Separate experimental vessel-classification and model-evaluation scripts

## Technology Stack

- Python 3.12
- Streamlit
- NumPy and Pandas
- SciPy
- SoundFile
- Plotly
- scikit-learn

## Project Structure

- `app.py` — Main interactive dashboard
- `app.pybackup.py` — Backup of the dashboard
- `background.wav`, `ship.wav`, `dolphin.wav`, `whale.wav` — Audio samples
- `sonar_classifier.joblib` — Saved experimental classifier
- `*.csv` — Extracted features, test results, and analysis outputs
- `*.py` — Feature extraction, model evaluation, comparison, and testing scripts
- `*.png` — Generated analysis plots
- `.venv/` — Local Python virtual environment

## Setup and Execution

### 1. Install Python

Install Python 3.12 and ensure the Python launcher is available.

### 2. Open the project folder

Open a terminal and run:

```powershell
cd D:\SONAR-X
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Start the dashboard

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

If the existing virtual environment is unavailable on another computer, create a new Python 3.12 environment and install the required dependencies before launching the application.

## How to Use

1. Start the Streamlit application.
2. Select synthetic simulation or WAV upload.
3. Configure the available simulation settings, or upload a supported WAV file.
4. Inspect the waveform, spectrogram, and calculated audio features.
5. Review windows flagged as potentially unusual.
6. Export the analysis results as a CSV file if required.

## Methodology

The dashboard divides audio into overlapping windows and calculates statistical and spectral features, including RMS energy, standard deviation, mean absolute difference, spectral centroid, and peak spectral power.

The extracted features are scaled and passed to an Isolation Forest model. The model assigns anomaly scores and flags windows for review.

Separate experimental scripts explore feature extraction, model comparison, stability, and vessel classification using available audio samples and dataset metadata.

## Limitations

- The dashboard's anomaly detector is exploratory and is not independently validated for reliable real-world underwater anomaly detection.
- Its Isolation Forest is fitted to the recording being analyzed, so flagged windows should not be interpreted as confirmed events.
- A flagged window does not establish the identity, location, or cause of an underwater sound.
- The saved vessel classifier is a separate experimental artifact; its presence does not mean it is integrated into the dashboard.
- Experimental classification results are limited by the available data, class distribution, and evaluation-set size.
- The illustrative radar graphic does not represent actual underwater positioning or tracking.

## Future Scope

- Evaluate the anomaly detector using labelled and independent underwater recordings.
- Improve generalization across different recording devices and acoustic environments.
- Expand and validate vessel classification using appropriately labelled data.
- Add reproducible evaluation metrics and model-version tracking.
- Explore real-time processing only after suitable testing and validation.

## Disclaimer

SONAR-X is an academic research prototype intended for learning, experimentation, and acoustic data exploration. Its outputs should not be treated as verified operational or safety-critical detections.
