
import librosa
import soundfile as sf

input_file = "Quiet-AWI-Van-Opzeeland-02-quiet-ocean-clip.mp3"
output_file = "background.wav"

audio, sample_rate = librosa.load(
    input_file,
    sr=None,
    mono=True
)

sf.write(output_file, audio, sample_rate, subtype="PCM_16")

print("Conversion successful!")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
print(f"Saved as: {output_file}")
