import os
import sys
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.inference import (
    load_artifacts, 
    load_selected_model, 
    load_audio,
    run_inference
)
from src.config import (
    SAVED_MODELS_DIR, RESULTS_DIR, SCALER_PATH, LABEL_ENCODER_PATH, SEX_ENCODER_PATH, SAMPLE_RATE,
    APP_WAVLM_RUN, ENSEMBLE_WEIGHTS_PATH
)

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

st.title("Multi-Model Speech Emotion Recognition")
st.write("Record live audio through your browser to classify vocal emotion across trained architectures.")

# 1. Discover available models on disk FIRST (a model is available only if all its files exist)
MODEL_REGISTRY = {
    'Random Forest': ['best_rf_model.joblib'],
    'SVM': ['best_svm_model.joblib'],
    '1D CNN': ['cnn_1d_model.pth'],
    'Stacked CNN': ['cnn_stacked_model.pth'],
    'Ensemble CNN': ['ensemble_gender_model.pth', 'ensemble_male_model.pth', 'ensemble_female_model.pth'], 
    'Vision Transformer': ['VIT_model.pth'],
    'WavLM': [f'{APP_WAVLM_RUN}_model.pth', f'{APP_WAVLM_RUN}_config/config.json'],
    'Best Ensemble': [ENSEMBLE_WEIGHTS_PATH.name]
}

def ensemble_member_files():
    """Every file the Best Ensemble's members need (members are listed in its weights file)."""
    if not ENSEMBLE_WEIGHTS_PATH.exists():
        return []
    runs = json.loads(ENSEMBLE_WEIGHTS_PATH.read_text())
    return [f for run in runs for f in (['VIT_model.pth'] if run == 'vit' else [f'{run}_model.pth', f'{run}_config/config.json'])]

MODEL_REGISTRY['Best Ensemble'] += ensemble_member_files()

# results/<name>.json written by each training script; used to list the best model first
RESULT_FILES = {
    'Random Forest': 'random_forest', 'SVM': 'svm', '1D CNN': 'cnn1d', 'Stacked CNN': 'cnnstacked',
    'Ensemble CNN': 'cnnensemble', 'Vision Transformer': 'vit', 'WavLM': APP_WAVLM_RUN, 'Best Ensemble': 'ensemble_best'
}

def test_accuracy(name):
    """Test accuracy from results/<name>.json, or None when the model has no saved evaluation."""
    path = RESULTS_DIR / f'{RESULT_FILES[name]}.json'
    return json.loads(path.read_text())['accuracy'] if path.exists() else None

available_models = [
    name for name, files in MODEL_REGISTRY.items()
    if all(os.path.exists(SAVED_MODELS_DIR / f) for f in files)
]
# Highest test accuracy first, so the app opens on the best model (unevaluated models go last)
model_accuracy = {name: test_accuracy(name) for name in available_models}
available_models.sort(key=lambda name: -(model_accuracy[name] or 0))

# Sidebar Configuration
st.sidebar.header("Model Settings")

# Stop execution immediately if NO models exist, preventing the load crash
if not available_models:
    st.sidebar.error("No trained models found in `saved_models/`. Please run a training script first.")
    st.stop()

# Stop if foundational artifacts are missing
if not all(os.path.exists(p) for p in [SCALER_PATH, LABEL_ENCODER_PATH, SEX_ENCODER_PATH]):
    st.sidebar.error("Missing `scaler.joblib`, `label_encoder.joblib` or `sex_encoder.joblib`. Please run `src/extraction_1d.py` to generate them.")
    st.stop()

# 2. Load artifacts ONLY after confirming files exist
scaler, encoder, sex_encoder = load_artifacts()
num_emotions = len(encoder.classes_)
# Feature length comes from the fitted scaler instead of being hard-coded
input_dim = scaler.n_features_in_

selected_model_name = st.sidebar.selectbox(
    "Select Active Inference Engine",
    available_models
)

st.sidebar.markdown("---")
st.sidebar.subheader("Active Model Info")
st.sidebar.info(f"Currently loaded: **{selected_model_name}**")
if model_accuracy[selected_model_name] is not None:
    st.sidebar.caption(f"Test accuracy on unseen speakers: {model_accuracy[selected_model_name]:.1%}")

# Load only the chosen model into GPU memory
active_model = load_selected_model(selected_model_name, num_emotions, input_dim)

# Main Interface Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Audio Input")
    st.write("Click the microphone to start recording. Click again to stop.")
    
    # Native browser-based audio recorder
    audio_data = st.audio_input("Record your voice")

# Load the recording once; both inference and the waveform plot reuse the same array
y = load_audio(audio_data) if audio_data is not None else None
if y is not None and len(y) < SAMPLE_RATE:
    st.warning("Recording is too short. Please record at least one second of speech.")
    y = None

with col2:
    st.subheader("2. Classification Results")
    if y is not None:
        with st.spinner(f"Running inference through {selected_model_name}..."):
            # Execute processing & inference
            pred_emotion, probabilities, gender_route = run_inference(
                y, 
                selected_model_name, 
                active_model, 
                scaler, 
                encoder,
                sex_encoder
            )
            
            # Display primary metric
            st.metric(label="Predicted Emotion", value=pred_emotion.upper())
            
            if gender_route:
                st.caption(f"🔀 **Hierarchical Router Decision:** Classified voice profile as **{gender_route}**.")

# Waveform & Probability Visualizations
if y is not None:
    st.markdown("---")
    st.subheader("3. Audio & Model Confidence Visualizations")
    
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        st.write("**Raw Audio Waveform**")
        sr = SAMPLE_RATE
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(np.linspace(0, len(y) / sr, num=len(y)), y, color='#1f77b4', alpha=0.8)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    with viz_col2:
        st.write("**Probability Distribution Across Emotions**")
        
        fig, ax = plt.subplots(figsize=(8, 3))
        y_pos = np.arange(len(encoder.classes_))
        
        bars = ax.barh(y_pos, probabilities * 100, align='center', color='#2ca02c', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(encoder.classes_)
        ax.invert_yaxis()
        ax.set_xlabel('Confidence (%)')
        ax.set_xlim(0, 100)
        ax.set_title("Softmax Output")
        
        # Add values next to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    va='center', ha='left', fontsize=8)
            
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)