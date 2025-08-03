import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_audio(file_path):
    print(f"Loading audio file: {file_path}")
    
    y, sr = librosa.load(file_path, duration=3.0)
    
    print(f"Sample rate: {sr} Hz")
    print(f"Audio length: {len(y)} samples ({len(y)/sr:.2f} seconds)")
    print(f"Audio range: {y.min():.3f} to {y.max():.3f}")
    
    # plot wave
    plt.figure(figsize=(12, 4))
    plt.plot(y)
    plt.title("Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()
    
    return y, sr

if __name__ == "__main__":
    audio_file = "ahem_x.wav"
    y, sr = analyze_audio(audio_file)
