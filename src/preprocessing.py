# =========================================================
# PREPROCESSING
# =========================================================

import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from .config import TEST_SIZE, RANDOM_SEED


def preprocess_data(X, y):
    
    encoder = LabelEncoder()
    
    y_encoded = encoder.fit_transform(y)
    
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    
    
    joblib.dump(encoder, "../models/label_encoder.pkl")
    
    joblib.dump(scaler, "../models/scaler.pkl")
    
    
    return X_train, X_test, y_train, y_test