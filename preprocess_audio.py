import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Configuration
# -----------------------------
TARGET_SR = 16000
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/waveforms", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/features", exist_ok=True)

# -----------------------------
# Load Dataset (LibriSpeech)
# -----------------------------
dataset = torchaudio.datasets.LIBRISPEECH(
    root="./data",
    url="test-clean",
    download=True
)

# Select two samples
wave1, sr1, *_ = dataset[0]
wave2, sr2, *_ = dataset[1]

# -----------------------------
# Preprocessing Functions
# -----------------------------
def preprocess_audio(waveform, sr):
    waveform = waveform.numpy().squeeze()

    # Resample
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

    # Normalize
    waveform = waveform / np.max(np.abs(waveform))

    # Trim silence
    waveform, _ = librosa.effects.trim(waveform, top_db=20)

    return waveform

# -----------------------------
# Apply Preprocessing
# -----------------------------
proc1 = preprocess_audio(wave1, sr1)
proc2 = preprocess_audio(wave2, sr2)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Processed Audio 1")
plt.plot(proc1)

plt.subplot(1, 2, 2)
plt.title("Processed Audio 2")
plt.plot(proc2)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/waveforms/processed_waveforms.png")
plt.close()

# -----------------------------
# Feature Extraction (MFCCs)
# -----------------------------
mfcc1 = librosa.feature.mfcc(y=proc1, sr=TARGET_SR, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=proc2, sr=TARGET_SR, n_mfcc=13)

# Save MFCC plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("MFCC Audio 1")
plt.imshow(mfcc1, aspect='auto', origin='lower')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("MFCC Audio 2")
plt.imshow(mfcc2, aspect='auto', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/features/mfcc_features.png")
plt.close()

# -----------------------------
# Combine Features
# -----------------------------
features = np.stack([mfcc1, mfcc2])
np.save(f"{OUTPUT_DIR}/features/combined_features.npy", features)

print("Preprocessing complete!")
print("Feature tensor shape:", features.shape)
