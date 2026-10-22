import os
import sys
import json
import joblib
import numpy as np
import librosa
import torch
import streamlit as st

# Add project root to path to import models and the shared feature pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CNN1D
from models import CNNStacked
from models import EnsembleBaseCNN
from models import build_vit
from models import build_wavlm
from models.wavlm import wavlm_logits
from src.config import (
    SCALER_PATH, LABEL_ENCODER_PATH, SEX_ENCODER_PATH, SAVED_MODELS_DIR, SAMPLE_RATE,
    APP_WAVLM_RUN, ENSEMBLE_WEIGHTS_PATH
)
from src.features import split_windows, extract_features, spectrogram_image, vit_input, fixed_length_wave, trimmed_windows

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_artifacts():
    """Loads the scaler and the emotion/sex label encoders exactly once."""
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    sex_encoder = joblib.load(SEX_ENCODER_PATH)
    return scaler, encoder, sex_encoder

def load_torch_model(model, filename):
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / filename, map_location=DEVICE))
    return model.to(DEVICE).eval()

@st.cache_resource
def load_wavlm_run(run, num_emotions):
    """Loads one trained WavLM run (saved_models/<run>_model.pth), rebuilt offline from its saved config."""
    model = build_wavlm(num_emotions, pretrained=False, config_dir=SAVED_MODELS_DIR / f'{run}_config')
    return load_torch_model(model, f'{run}_model.pth')

@st.cache_resource
def load_selected_model(model_choice, num_emotions, input_dim):
    """Dynamically loads the requested model into VRAM."""
    if model_choice == 'Random Forest':
        return joblib.load(SAVED_MODELS_DIR / 'best_rf_model.joblib')

    elif model_choice == 'SVM':
        return joblib.load(SAVED_MODELS_DIR / 'best_svm_model.joblib')

    elif model_choice == '1D CNN':
        return load_torch_model(CNN1D(input_dim=input_dim, num_classes=num_emotions), 'cnn_1d_model.pth')

    elif model_choice == 'Stacked CNN':
        return load_torch_model(CNNStacked(input_dim=input_dim, num_classes=num_emotions), 'cnn_stacked_model.pth')

    elif model_choice == 'Ensemble CNN':
        return {
            'router': load_torch_model(EnsembleBaseCNN(input_dim=input_dim, num_classes=2), 'ensemble_gender_model.pth'),
            'male': load_torch_model(EnsembleBaseCNN(input_dim=input_dim, num_classes=num_emotions), 'ensemble_male_model.pth'),
            'female': load_torch_model(EnsembleBaseCNN(input_dim=input_dim, num_classes=num_emotions), 'ensemble_female_model.pth')
        }

    elif model_choice == 'Vision Transformer':
        # Built offline from the config; the fine-tuned weights replace everything anyway
        return load_torch_model(build_vit(num_emotions, pretrained=False), 'VIT_model.pth')
        
    elif model_choice == 'WavLM':
        # The best single WavLM run (src/config.py: APP_WAVLM_RUN)
        return load_wavlm_run(APP_WAVLM_RUN, num_emotions)

    elif model_choice == 'Best Ensemble':
        # Members and weights chosen on the validation split by scripts/evaluate_ensemble.py;
        # 'vit' is the Vision Transformer, every other member is a WavLM run name
        weights = json.loads(ENSEMBLE_WEIGHTS_PATH.read_text())
        return {
            run: (load_selected_model('Vision Transformer', num_emotions, input_dim) if run == 'vit'
                  else load_wavlm_run(run, num_emotions), weight)
            for run, weight in weights.items()
        }

def load_audio(audio_file):
    """Loads the full recording at the training sample rate (librosa supports BytesIO natively)."""
    data, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
    return data

def softmax_probs(model, inputs):
    with torch.no_grad():
        return torch.softmax(model(inputs), dim=1).cpu().numpy()

def run_inference(data, model_choice, model, scaler, encoder, sex_encoder):
    """Routes the audio to the correct preprocessing and inference pipeline.

    The recording is cut into the same 2.5 s windows used in training (after the 0.6 s offset),
    every window is classified, and the window probabilities are averaged.
    """
    windows = split_windows(data)

    # Ensemble Pipeline: weighted average of the member models' probabilities
    if model_choice == 'Best Ensemble':
        probs = sum(weight * run_inference(data, 'Vision Transformer' if run == 'vit' else 'WavLM',
                                           member, scaler, encoder, sex_encoder)[1]
                    for run, (member, weight) in model.items())
        return encoder.classes_[np.argmax(probs)], probs, None

    # Vision Transformer Pipeline
    if model_choice == 'Vision Transformer':
        images = torch.from_numpy(np.stack([spectrogram_image(w) for w in windows])).to(DEVICE)
        with torch.no_grad():
            window_probs = torch.softmax(model(vit_input(images)).logits, dim=1).cpu().numpy()
        probs = window_probs.mean(axis=0)
        return encoder.classes_[np.argmax(probs)], probs, None # Return None for router_info

    # WavLM Pipeline: raw waveform windows resampled to 16 kHz. Models trained on silence-trimmed
    # clips store their window length in the config (clip_seconds); older ones use the 2.5 s windows
    if model_choice == 'WavLM':
        clip_seconds = getattr(model.config, 'clip_seconds', None)
        wave_windows = trimmed_windows(data, seconds=clip_seconds) if clip_seconds else [fixed_length_wave(w) for w in windows]
        waves = torch.from_numpy(np.stack(wave_windows)).to(DEVICE)
        probs = softmax_probs(lambda x: wavlm_logits(model, x), waves).mean(axis=0)
        return encoder.classes_[np.argmax(probs)], probs, None
        
    # 1D Feature Pipeline (standard scaling with the training-split scaler)
    features_scaled = scaler.transform(np.array([extract_features(w) for w in windows]))

    # Scikit-Learn Pipeline
    if model_choice in ['Random Forest', 'SVM']:
        probs = model.predict_proba(features_scaled).mean(axis=0)
        return encoder.classes_[np.argmax(probs)], probs, None

    # PyTorch 1D CNN Pipeline: (Batch, Length, Channels)
    tensor_1d = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    if model_choice in ['1D CNN', 'Stacked CNN']:
        probs = softmax_probs(model, tensor_1d).mean(axis=0)
        return encoder.classes_[np.argmax(probs)], probs, None

    # Hierarchical Ensemble Pipeline
    if model_choice == 'Ensemble CNN':
        # Run Gender Router once over the whole recording
        gender_probs = softmax_probs(model['router'], tensor_1d).mean(axis=0)
        gender_string = sex_encoder.classes_[np.argmax(gender_probs)]

        # Route to Specific Model
        probs = softmax_probs(model[gender_string], tensor_1d).mean(axis=0)
        return encoder.classes_[np.argmax(probs)], probs, gender_string.capitalize()
