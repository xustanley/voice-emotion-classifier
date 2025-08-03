import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_emotion_classifier():
    """Train classifier on RAVDESS dataset"""
    
    print("Loading dataset...")
    df = pd.read_csv('ravdess_emotion_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Emotions: {df['emotion'].value_counts()}")
    
    X = df.drop('emotion', axis=1)
    y = df['emotion']
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X)}")
    
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.3f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Emotion Classification Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.show()
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'emotions': model.classes_.tolist()
    }
    
    joblib.dump(model_data, 'emotion_classifier_model.pkl')
    print("\nModel saved as 'emotion_classifier_model.pkl'")
    
    return model, scaler, accuracy

if __name__ == "__main__":
    model, scaler, accuracy = train_emotion_classifier()
    print(f"\n🎉 Your emotion classifier is trained! Accuracy: {accuracy:.1%}")