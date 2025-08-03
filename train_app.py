import joblib
import numpy as np
from features import extract_features

def predict_emotion(audio_file):
    """Predict emotion from audio file using trained model"""
    
    model_data = joblib.load('emotion_classifier_model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    emotions = model_data['emotions']
    
    print(f"Analyzing: {audio_file}")
    
    features = extract_features(audio_file)
    
    feature_vector = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            for j, v in enumerate(value):
                feature_vector[f"{key}_{j}"] = v
        else:
            feature_vector[key] = value
    
    X = np.array([feature_vector.get(name, 0) for name in feature_names]).reshape(1, -1)
    
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    print(f"\nPredicted emotion: {prediction}")
    print("\nConfidence scores:")
    for emotion, prob in zip(emotions, probabilities):
        print(f"  {emotion}: {prob:.3f}")
    
    return prediction, probabilities

if __name__ == "__main__":
    predict_emotion("happy_me.wav")