# =========================================================
# FEATURE EXTRACTION
# =========================================================

import librosa
import numpy as np
from .config import SAMPLE_RATE, N_MFCC


def extract_mfcc(file_path):
    
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )
    
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )
    
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    return mfcc_mean


def extract_features(df):
    
    features = []
    labels = []
    
    for index, row in df.iterrows():
        
        file_path = row["path"]
        emotion = row["emotion"]
        
        mfcc = extract_mfcc(file_path)
        
        features.append(mfcc)
        labels.append(emotion)
    
    
    return np.array(features), np.array(labels)