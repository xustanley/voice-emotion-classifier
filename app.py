import streamlit as st
import joblib
import numpy as np
import tempfile
import os
from features import extract_features

def load_model():
    """Load the trained model"""
    model_data = joblib.load('emotion_classifier_model.pkl')
    return model_data

def predict_emotion(audio_file):
    """Predict emotion from audio file"""
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    emotions = model_data['emotions']
    
    features = extract_features(audio_file)
    
    feature_list = []
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            for v in value:
                feature_list.append(v)
        else:
            feature_list.append(value)
    
    while len(feature_list) < len(feature_names):
        feature_list.append(0)
    
    X = np.array(feature_list[:len(feature_names)]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    return prediction, probabilities, emotions

st.title("Voice Emotion Classifier")
st.write("Upload an audio file and I'll tell you what emotion it sounds like!")

uploaded_file = st.file_uploader("Choose an audio file (WAV or MP3):", type=['wav', 'mp3'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    
    if st.button("Analyze Emotion"):
        with st.spinner("Thinking..."):
            try:
                prediction, probabilities, emotions = predict_emotion(temp_path)
                
                st.success(f"I think this sounds: **{prediction.upper()}**")
                
                confidence = max(probabilities) * 100
                st.write(f"Confidence: {confidence:.1f}%")
                
                st.write("Here's how sure I am about each emotion:")
                for emotion, prob in zip(emotions, probabilities):
                    percentage = prob * 100
                    st.write(f"• {emotion}: {percentage:.1f}%")
                
            except Exception as e:
                st.error("Sorry, I couldn't analyze that audio file. Try a different one!")
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)

st.write("This model was trained on professional actors as provided by:")
st.write("Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): " \
"A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391")
st.write("Model Accuracy: 70.3%")