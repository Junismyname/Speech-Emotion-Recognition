import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import build_vit, build_wavlm
from src.config import (
    VIT_X_PATH, WAVE_TENSORS_PATH, SAVED_MODELS_DIR, RESULTS_DIR, EMOTIONS
)
from src.training import predict
from src.evaluation import evaluate
from train_vit import vit_forward
from train_wavlm import wavlm_forward, WAVLM_WEIGHTS_PATH, WAVLM_CONFIG_DIR

# Late fusion of the two strongest models: weighted average of WavLM (raw waveform) and ViT
# (spectrogram image) probabilities. Both tensor files follow ser_metadata.csv order, so rows align.
# The weight is chosen on the validation split; the test split is evaluated once with that weight.

def split_probs(model, X, y, split, forward, device, batch_size):
    probs = {}
    for name in ['val', 'test']:
        idx = torch.from_numpy(np.where(split == name)[0])
        loader = DataLoader(TensorDataset(X[idx], y[idx]), batch_size=batch_size)
        _, probs[name] = predict(model, loader, device, forward=forward)
    return probs

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    categories = np.array(sorted(EMOTIONS))
    waves = torch.load(WAVE_TENSORS_PATH, weights_only=False)
    split, y, sources = waves['split'], waves['y'], waves['source']

    # One large model on the GPU at a time
    wavlm = build_wavlm(len(categories), pretrained=False, config_dir=WAVLM_CONFIG_DIR)
    wavlm.load_state_dict(torch.load(WAVLM_WEIGHTS_PATH, map_location='cpu'))
    wavlm_probs = split_probs(wavlm.to(device), waves['X'], y, split, wavlm_forward, device, 32)
    del wavlm, waves
    torch.cuda.empty_cache()

    vit = build_vit(len(categories), pretrained=False)
    vit.load_state_dict(torch.load(SAVED_MODELS_DIR / 'VIT_model.pth', map_location='cpu'))
    vit_probs = split_probs(vit.to(device), torch.load(VIT_X_PATH), y, split, vit_forward, device, 64)
    del vit

    y_val, y_test = y[torch.from_numpy(split == 'val')].numpy(), y[torch.from_numpy(split == 'test')].numpy()

    # Saved for scripts/evaluate_ensemble.py
    np.savez(RESULTS_DIR / 'wavlm_probs.npz', val=wavlm_probs['val'], test=wavlm_probs['test'])
    np.savez(RESULTS_DIR / 'vit_probs.npz', val=vit_probs['val'], test=vit_probs['test'])

    # Pick the WavLM weight on validation
    weights = np.round(np.arange(0, 1.01, 0.05), 2)
    val_scores = [accuracy_score(y_val, (w * wavlm_probs['val'] + (1 - w) * vit_probs['val']).argmax(1)) for w in weights]
    best_w = float(weights[int(np.argmax(val_scores))])
    print("Val accuracy by WavLM weight: " + ", ".join(f"{w:.2f}={s:.4f}" for w, s in zip(weights, val_scores)))
    print(f"Best WavLM weight on val: {best_w} (val acc {max(val_scores):.4f} vs WavLM alone {val_scores[-1]:.4f})")

    fused = best_w * wavlm_probs['test'] + (1 - best_w) * vit_probs['test']
    evaluate('fusion', 'Fusion (WavLM + ViT)', y_test, fused.argmax(1), categories, sources[split == 'test'],
             extra={'wavlm_weight': best_w, 'val_accuracy_by_weight': dict(zip(map(str, weights), val_scores))})

    with open(SAVED_MODELS_DIR / 'fusion_weights.json', 'w') as f:
        json.dump({'wavlm': best_w, 'vit': round(1 - best_w, 2)}, f, indent=4)
    print(f"Saved fusion weights to {SAVED_MODELS_DIR / 'fusion_weights.json'}")
