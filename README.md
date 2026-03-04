🎙️ Emotion Recognition from Speech

## Streamlit App Preview

![Streamlit App](images/streamlit_app.png)

A Deep Learning based Speech Emotion Recognition (SER) system that detects human emotions from voice recordings using MFCC feature extraction and a Neural Network model.

📌 Project Overview

This project analyzes speech audio files and predicts emotions such as:

😄 Happy

😢 Sad

😠 Angry

😐 Neutral

😲 Surprise

😨 Fear

🤢 Disgust

😌 Calm

The model is trained on the RAVDESS dataset.

📂 Dataset

Dataset Used: RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song

🔗 Download here: [RAVDESS Dataset](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip)

🧠 Technologies Used

Python

Librosa

NumPy

TensorFlow / Keras

Scikit-learn

Streamlit

Matplotlib / Seaborn

🏗️ Project Structure
Emotion-Recognition-Speech/
│
├── data/
│   └── raw/
│       └── RAVDESS/
│           ├── Actor_01/
│           ├── Actor_02/
│           └── ...
|
│       └── processed/
│           ├── features.npy
│           ├── labels.npy
│
├── notebooks/
│   ├── 01_EDA_RAVDESS.ipynb
│   ├── 02_Feature_Extraction.ipynb
│   ├── 03_Model_Training_CNN.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
|   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   └── emotion_model.h5
│   └── history.pkl
│   └── label_encoder.pkl
│   └── scaler.pkl
|
│
├── .gitignore
├── app.py
├── requirements.txt
└── README.md

⚙️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/Alisha4406/CodeAlpha-TASK-Emotion-Recognition-From-Speech.git

2️⃣ Create virtual environment
python -m venv emotion_env
emotion_env\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app.py

📊 Model Performance

Feature Extraction: MFCC

Model: Deep Neural Network

Evaluation Metrics: Accuracy, Confusion Matrix

ACCURACY: 93%

🚀 Live Prediction

Upload a .wav file in the Streamlit app and get emotion prediction instantly.

# CodeAlpha-TASK-Emotion-Recognition-From-Speech
🎙️ Deep Learning based Speech Emotion Recognition system using RAVDESS dataset with MFCC feature extraction and a CNN model, deployed using Streamlit.

