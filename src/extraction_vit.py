import os
import sys
import warnings
import numpy as np
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Add the project root to the system path so Python can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import load_clip, spectrogram_image
from src.config import (
    METADATA_CSV, VIT_X_PATH, VIT_Y_PATH, VIT_SPLIT_PATH,
    VIT_SOURCE_PATH, EMOTIONS, to_project_path
)

# Suppress the FutureWarnings and RuntimeWarnings from librosa/numba
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def preprocess_vit_audio(path):
    """Loads audio and converts it into a 224x224 uint8 MelSpectrogram image tensor."""
    return spectrogram_image(load_clip(to_project_path(path)))

def generate_vit_features(metadata_csv_path):
    # torch is imported here, not at module level: on Windows every worker process re-imports
    # this module, and loading torch's CUDA DLLs in 32 workers exhausts the paging file (WinError 1455)
    import torch
    
    dataset = pd.read_csv(metadata_csv_path)

    print("Converting audio to 224x224 Spectrograms...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        X_vit = list(tqdm(executor.map(preprocess_vit_audio, dataset['filepath'], chunksize=16), total=len(dataset)))

    X_vit_tensor = torch.from_numpy(np.stack(X_vit))

    # Same fixed class order as the 1D label encoder, so the app can share one encoder
    encoder_vit = LabelEncoder().fit(EMOTIONS)
    Y_vit_tensor = torch.tensor(encoder_vit.transform(dataset['emotion']), dtype=torch.long)

    # Save the tensors to disk for fast loading during training
    torch.save(X_vit_tensor, VIT_X_PATH)
    torch.save(Y_vit_tensor, VIT_Y_PATH)

    # Save the split and source of every row so training uses the same speaker-level split
    np.save(VIT_SPLIT_PATH, dataset['split'].to_numpy())
    np.save(VIT_SOURCE_PATH, dataset['source'].to_numpy())

    print(f"\nViT feature extraction complete!")
    print(f"X Tensor Shape: {X_vit_tensor.shape} ({X_vit_tensor.dtype})")
    print(f"Saved to {VIT_X_PATH} and {VIT_Y_PATH}")

if __name__ == "__main__":
    generate_vit_features(METADATA_CSV)
