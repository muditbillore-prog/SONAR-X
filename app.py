
import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="SONAR-X | Acoustic Intelligence",
    page_icon="🌊",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #07111f;
    color: #e6edf7;
}
[data-testid="stSidebar"] {
    background: #0b1727;
}
[data-testid="stMetric"] {
    background: #101e30;
    border: 1px solid #243950;
    padding: 14px;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #e6edf7;
}
[data-testid="stMetricLabel"] {
    color: #8ca6c2;
}
</style>
""", unsafe_allow_html=True)


def generate_audio(duration, sample_rate, noise_level, add_event):
    rng = np.random.default_rng(42)
    t = np.arange(int(duration * sample_rate)) / sample_rate

    audio = noise_level * rng.normal(size=len(t))
    audio += 0.08 * np.sin(2 * np.pi * 120 * t)
    audio += 0.04 * np.sin(2 * np.pi * 240 * t)

    if add_event:
        start = int(0.60 * len(t))
        end = min(start + int(0.12 * sample_rate), len(t))
        event_t = np.arange(end - start) / sample_rate
        envelope = np.hanning(len(event_t))
        audio[start:end] += (
            0.85 * envelope
            * np.sin(2 * np.pi * 850 * event_t)
        )

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = 0.95 * audio / peak

    return t, audio.astype(np.float32)


def extract_features(audio, sample_rate, window_size=2048):
    hop = window_size // 2
    rows = []
    starts = []

    for start in range(0, len(audio) - window_size + 1, hop):
        chunk = audio[start:start + window_size]
        frequencies, power = signal.periodogram(
            chunk, fs=sample_rate
        )

        total_power = np.sum(power) + 1e-12
        centroid = np.sum(frequencies * power) / total_power

        rows.append([
            np.sqrt(np.mean(chunk ** 2)),
            np.std(chunk),
            np.mean(np.abs(np.diff(chunk))),
            centroid,
            np.max(power),
        ])
        starts.append(start / sample_rate)

    names = [
        "RMS Energy",
        "Standard Deviation",
        "Mean Absolute Difference",
        "Spectral Centroid",
        "Peak Spectral Power",
    ]

    return pd.DataFrame(rows, columns=names), np.array(starts)


def style_plot(fig, height=300):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=35, b=15),
        paper_bgcolor="#101e30",
        plot_bgcolor="#101e30",
        font=dict(color="#d9e7f5"),
        xaxis=dict(gridcolor="#243950"),
        yaxis=dict(gridcolor="#243950"),
        legend=dict(font=dict(color="#d9e7f5")),
    )
    return fig


st.markdown(
    "<h1>🌊 SONAR-X</h1>"
    "<p style='color:#8ca6c2'>"
    "UNDERWATER ACOUSTIC MONITORING SYSTEM · RESEARCH PROTOTYPE"
    "</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## SYSTEM CONTROL")

    mode = st.radio(
        "Audio source",
        ["Synthetic Simulation", "Upload WAV"],
        key="audio_source",
    )

    if mode == "Synthetic Simulation":
        duration = st.slider(
            "Simulation duration (seconds)",
            min_value=2,
            max_value=10,
            value=4,
            key="simulation_duration",
        )
        noise = st.slider(
            "Background noise",
            min_value=0.02,
            max_value=0.30,
            value=0.12,
            key="background_noise",
        )
        add_event = st.checkbox(
            "Insert unusual acoustic event",
            value=True,
            key="insert_event",
        )
        uploaded = None
    else:
        uploaded = st.file_uploader(
            "Upload an audio file",
            type=["wav"],
            key="wav_upload",
        )

    st.caption(
        "Anomaly flags are experimental and do not identify "
        "specific animals, objects, or real-world threats."
    )

if mode == "Upload WAV":
    if uploaded is None:
        st.info("Choose a WAV recording in the sidebar to begin.")
        st.stop()

    try:
        audio, sample_rate = sf.read(uploaded, dtype="float32")
    except Exception as exc:
        st.error(f"Could not read this WAV file: {exc}")
        st.stop()

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) < 2048:
        st.error("The recording is too short. Use at least 2048 samples.")
        st.stop()

    if not np.all(np.isfinite(audio)):
        st.error("The audio contains invalid values.")
        st.stop()

    audio = audio.astype(np.float32)
    t = np.arange(len(audio)) / sample_rate
    source_name = uploaded.name

else:
    sample_rate = 8000
    t, audio = generate_audio(
        duration, sample_rate, noise, add_event
    )
    source_name = "Synthetic simulation"

duration_actual = len(audio) / sample_rate

features, window_starts = extract_features(audio, sample_rate)

if len(features) < 3:
    st.error("Not enough audio windows for analysis.")
    st.stop()

if np.allclose(audio, audio[0]):
    st.warning("This recording is nearly constant; results may not be useful.")

# Standardize feature scales before fitting the experimental detector.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

model = IsolationForest(
    n_estimators=150,
    contamination=0.10,
    random_state=42,
)
model.fit(scaled_features)

predictions = model.predict(scaled_features)
scores = -model.decision_function(scaled_features)
anomalies = predictions == -1

m1, m2, m3, m4 = st.columns(4)
m1.metric("SYSTEM STATUS", "ONLINE")
m2.metric("SAMPLE RATE", f"{sample_rate:,} Hz")
m3.metric("AUDIO DURATION", f"{duration_actual:.2f} s")
m4.metric("FLAGGED WINDOWS", int(anomalies.sum()))

st.caption(f"Audio source: **{source_name}**")

st.markdown("---")
left, right = st.columns([1, 1.4])

with left:
    st.subheader("◉ Acoustic Monitor")

    fig_radar = go.Figure()
    theta = np.linspace(0, 360, 361)

    for radius in [0.25, 0.5, 0.75, 1.0]:
        fig_radar.add_trace(go.Scatterpolar(
            r=np.full(len(theta), radius),
            theta=theta,
            mode="lines",
            line=dict(color="#24495a", width=1),
            hoverinfo="skip",
        ))

    angle = (time.time() * 35) % 360
    fig_radar.add_trace(go.Scatterpolar(
        r=[0, 1],
        theta=[angle, angle],
        mode="lines",
        line=dict(color="#4de3d1", width=3),
        hoverinfo="skip",
    ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="#101e30",
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(color="#6d91a9", gridcolor="#243950"),
        ),
        height=330,
        margin=dict(l=20, r=20, t=15, b=15),
        paper_bgcolor="#101e30",
        font=dict(color="#d9e7f5"),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("Illustrative radar display; it does not show real positions.")

with right:
    st.subheader("〰 Acoustic Waveform")

    stride = max(1, len(audio) // 5000)
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=t[::stride],
        y=audio[::stride],
        mode="lines",
        name="Amplitude",
        line=dict(color="#4de3d1", width=1),
    ))
    fig_wave.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=False,
    )
    st.plotly_chart(style_plot(fig_wave), use_container_width=True)

st.subheader("▥ Frequency-Time Spectrogram")

frequencies, times, spectrum = signal.spectrogram(
    audio,
    fs=sample_rate,
    nperseg=min(512, len(audio)),
)

fig_spec = go.Figure(go.Heatmap(
    x=times,
    y=frequencies,
    z=10 * np.log10(spectrum + 1e-12),
    colorscale=[
        [0, "#07111f"],
        [0.45, "#125d73"],
        [0.75, "#24c7b5"],
        [1, "#e4f8c7"],
    ],
    colorbar=dict(title="Power (dB)"),
))
fig_spec.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Frequency (Hz)",
)
st.plotly_chart(style_plot(fig_spec, 350), use_container_width=True)

st.subheader("🧠 Machine Learning Anomaly Analysis")
st.caption(
    "Isolation Forest flags windows that differ from the fitted "
    "feature distribution. This exploratory detector has not been "
    "validated against labelled underwater recordings."
)

results = features.copy()
results.insert(0, "Window start (s)", np.round(window_starts, 3))
results["Anomaly score"] = np.round(scores, 5)
results["Status"] = np.where(anomalies, "REVIEW", "NORMAL")

if anomalies.any():
    st.dataframe(
        results.loc[anomalies].reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.info("No windows flagged by this model.")

with st.expander("View all analyzed audio windows"):
    st.dataframe(results, use_container_width=True)

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Export complete analysis (CSV)",
    data=csv_bytes,
    file_name="sonar_x_analysis.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "SONAR-X Research Prototype | Audio signal analysis and "
    "experimental anomaly detection | Not for operational targeting"
)
