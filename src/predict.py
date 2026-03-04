# =========================================================
# PREDICTION
# =========================================================

import joblib
import numpy as np

from tensorflow.keras.models import load_model

from .feature_extraction import extract_mfcc


def predict_emotion(file_path):
    
    model = load_model("../models/emotion_model.h5")
    
    scaler = joblib.load("../models/scaler.pkl")
    
    encoder = joblib.load("../models/label_encoder.pkl")
    
    
    feature = extract_mfcc(file_path)
    
    feature = scaler.transform([feature])
    
    
    prediction = model.predict(feature)
    
    predicted_class = np.argmax(prediction)
    
    
    emotion = encoder.inverse_transform([predicted_class])
    
    
    return emotion[0]