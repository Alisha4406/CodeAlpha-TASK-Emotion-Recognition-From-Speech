# ====================================================
# STREAMLIT APP: BEAUTIFUL SPEECH EMOTION RECOGNITION
# ====================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from tensorflow.keras.models import load_model
from src.feature_extraction import extract_mfcc

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🎤 Speech Emotion Recognition",
    page_icon="🎧",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #e4e7eb);
}

/* Main title */
.main-title {
    font-size: 50px;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
}

/* Prediction Box */
.prediction-box {
    background-color: #3498db;
    color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin: 20px auto;
    width: 60%;
    transition: transform 0.2s;
}

.prediction-box:hover {
    transform: scale(1.05);
}

.prediction-box span {
    font-size: 50px;
    margin-left: 10px;
}

/* Sidebar Background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50, #4ca1af);
    color: white;
}

/* File uploader text BLACK */
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stFileUploader div,
section[data-testid="stSidebar"] .stFileUploader small {
    color: black !important;
}

/* Browse files button styling */
section[data-testid="stSidebar"] .stFileUploader button {
    background-color: black !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 6px 14px !important;
    font-weight: bold !important;
}

/* Hover effect */
section[data-testid="stSidebar"] .stFileUploader button:hover {
    background-color: #333 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# APP TITLE
# -------------------------------
st.markdown("<h1 class='main-title'>🎤 Speech Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:20px;color:#34495e'>Upload your voice and see the emotion instantly!</p>", unsafe_allow_html=True)

# -------------------------------
# IMAGE
# -------------------------------
st.image(
    "C:\Emotion Recognition Speech\images.jpg",
    width=350,
    caption="Feel the emotion in voice!"
)

# -------------------------------
# SIDEBAR: UPLOAD AUDIO
# -------------------------------
st.sidebar.header("Upload Audio File")
uploaded_file = st.sidebar.file_uploader("Choose a WAV file", type=["wav"])

# -------------------------------
# EMOJI MAP
# -------------------------------
emoji_map = {
    "neutral": "😐","calm": "😌","happy": "😄",
    "sad": "😢","angry": "😡","fearful": "😱",
    "disgust": "🤢","surprised": "😲"
}

# -------------------------------
# MODEL PATHS
# -------------------------------
BASE_PATH = os.getcwd()
MODEL_PATH = os.path.join(BASE_PATH, "models", "emotion_model.h5")
SCALER_PATH = os.path.join(BASE_PATH, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_PATH, "models", "label_encoder.pkl")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_models():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

try:
    model, scaler, encoder = load_models()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -------------------------------
# MAIN INTERACTION
# -------------------------------
if uploaded_file is not None:
    
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Audio uploaded successfully!")
    st.audio(file_path)

    feature = extract_mfcc(file_path)
    feature_scaled = scaler.transform([feature])
    pred_prob = model.predict(feature_scaled)[0]
    pred_index = np.argmax(pred_prob)
    predicted_emotion = encoder.inverse_transform([pred_index])[0]
    emoji = emoji_map.get(predicted_emotion, "")

    st.markdown(
        f"""
        <div class='prediction-box'>
            Prediction: {predicted_emotion.upper()} <span>{emoji}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    audio, sr = librosa.load(file_path, sr=22050)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Waveform 🎵")
        fig, ax = plt.subplots(figsize=(10,3))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Spectrogram 🌈")
        X = librosa.stft(audio)
        Xdb = librosa.amplitude_to_db(abs(X))
        fig, ax = plt.subplots(figsize=(10,3))
        img = librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

    st.subheader("MFCC Heatmap 🔥")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots(figsize=(12,4))
    sns.heatmap(mfcc, cmap="inferno", ax=ax)
    st.pyplot(fig)

else:
    st.info("📌 Please upload a WAV file to start the prediction!")