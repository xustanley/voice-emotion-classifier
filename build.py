import pandas as pd
import numpy as np
from parse import process_ravdess_dataset
from features import extract_features
import os
from tqdm import tqdm

def build_emotion_dataset(data_folder, max_files_per_emotion=40):
    """Build dataset from RAVDESS audio files"""
    
    audio_files, emotions = process_ravdess_dataset(data_folder)
    
    emotion_counts = {}
    filtered_files = []
    filtered_emotions = []
    
    for file, emotion in zip(audio_files, emotions):
        if emotion_counts.get(emotion, 0) < max_files_per_emotion:
            filtered_files.append(file)
            filtered_emotions.append(emotion)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"Processing {len(filtered_files)} files (max {max_files_per_emotion} per emotion)")
    print("Files per emotion:", emotion_counts)
    
    all_features = []
    all_labels = []
    
    print("Extracting features...")
    for i, (audio_file, emotion) in enumerate(zip(filtered_files, filtered_emotions)):
        try:
            print(f"Processing {i+1}/{len(filtered_files)}: {os.path.basename(audio_file)}")
            
            features = extract_features(audio_file)
            
            feature_vector = {}
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    for j, v in enumerate(value):
                        feature_vector[f"{key}_{j}"] = v
                else:
                    feature_vector[key] = value
            
            all_features.append(feature_vector)
            all_labels.append(emotion)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    df = pd.DataFrame(all_features)
    df['emotion'] = all_labels
    
    print(f"\nDataset created!")
    print(f"Shape: {df.shape}")
    print(f"Emotions: {df['emotion'].value_counts()}")
    
    df.to_csv('ravdess_emotion_dataset.csv', index=False)
    print("Saved to ravdess_emotion_dataset.csv")
    
    return df

if __name__ == "__main__":
    dataset = build_emotion_dataset("Audio_Speech_Actors_01-24", max_files_per_emotion=40)