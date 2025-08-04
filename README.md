# Voice Emotion Classifier
Detects emotions from speech audio (recordings) using acoustic feature extraction and Random Forest classification.

- predicts the speaker's emotion with **70% accuracy** across **8 different emotion categories**:
  - Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised
- web interface for easy testing and demonstration
- professional dataset training (RAVDESS):
  - Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.
- audio feature extraction including MFCC, pitch, energy, and spectral features

## Tech Stack
- Python 3.13 - main
- Streamlit - web interface
- RAVDESS - emotional speech data

- scikit-learn - Random Forest classifier & model evaluation
- pandas - data manipulation & csv handling
- numpy - numerical computations

- librosa - feature extraction

- matplotlib/seaborn - Plotting & confusion matrices
- joblib - Model serialization

## Installation
Clone this repo. Create a venv.  
Install dependencies: `pip install -r requirements.txt`  
For web interface: `streamlit run app.py`  
