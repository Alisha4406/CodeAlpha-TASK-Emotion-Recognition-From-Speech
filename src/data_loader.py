# =========================================================
# DATA LOADER
# =========================================================

import os
import pandas as pd
from .config import EMOTIONS


def load_RAVDNESS_data(dataset_path):
    """
    Load RAVDESS dataset paths and labels
    
    Returns:
        DataFrame with path and emotion
    """
    
    paths = []
    emotions = []
    
    for actor in os.listdir(dataset_path):
        
        actor_path = os.path.join(dataset_path, actor)
        
        if os.path.isdir(actor_path):
            
            for file in os.listdir(actor_path):
                
                if file.endswith(".wav"):
                    
                    file_path = os.path.join(actor_path, file)
                    
                    emotion_code = file.split("-")[2]
                    
                    emotion = EMOTIONS[emotion_code]
                    
                    paths.append(file_path)
                    emotions.append(emotion)
    
    
    df = pd.DataFrame({
        "path": paths,
        "emotion": emotions
    })
    
    
    print("Total audio files:", len(df))
    
    return df