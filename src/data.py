import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from .config import FEATURES_CSV, SCALER_PATH, LABEL_ENCODER_PATH, SEX_ENCODER_PATH, META_COLUMNS

SPLITS = ['train', 'val', 'test']

def load_1d_splits(csv_path=FEATURES_CSV):
    """Loads features.csv and returns scaled train/val/test arrays using the saved shared artifacts."""
    print("Loading extracted features...")
    df = pd.read_csv(csv_path)

    # Scaler and encoders come from src/extraction_1d.py so every model (and the app) uses the same ones
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    sex_encoder = joblib.load(SEX_ENCODER_PATH)

    splits = {}
    for name in SPLITS:
        part = df[df['split'] == name]
        splits[name] = {
            'X': scaler.transform(part.drop(columns=META_COLUMNS).values),
            'y_emotion': encoder.transform(part['labels_emotion']),
            'y_sex': sex_encoder.transform(part['labels_sex']),
            'speaker': part['speaker'].to_numpy(),
            'source': part['source'].to_numpy()
        }
        print(f"{name}: {len(part)} rows")

    return splits, encoder, sex_encoder

def make_loader(X, y, batch_size=64, shuffle=False):
    """Wraps 1D features as (Batch, Length, Channels) tensors, the layout the app also uses."""
    X_tensor = torch.tensor(np.expand_dims(X, axis=-1), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)
