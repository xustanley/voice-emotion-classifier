import os
import pandas as pd
from features import extract_features

def parse_filename(filename):
    """Parse RAVDESS filename to get emotion label"""
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = int(parts[2])
        
        emotion_map = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }
        
        return emotion_map.get(emotion_code, 'unknown')
    return 'unknown'

def process_ravdess_dataset(data_folder):
    """Process all RAVDESS audio files"""
    audio_files = []
    emotions = []
    
    print("Scanning audio files...")
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                emotion = parse_filename(file)
                
                if emotion != 'unknown':
                    audio_files.append(filepath)
                    emotions.append(emotion)
    
    print(f"Found {len(audio_files)} valid audio files")
    print(f"Emotions: {set(emotions)}")
    
    return audio_files, emotions