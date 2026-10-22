import os
import sys
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add the project root to the system path so Python can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import augmentations and the shared feature pipeline
from src import noise, stretch, shift, pitch
from src.features import load_clip, extract_features
from src.config import (
    METADATA_CSV, FEATURES_CSV, SAVED_MODELS_DIR, SCALER_PATH,
    LABEL_ENCODER_PATH, SEX_ENCODER_PATH, EMOTIONS, SEXES,
    SAMPLE_RATE, RANDOM_STATE, META_COLUMNS, FEATURE_SET, to_project_path
)

# Suppress the FutureWarnings and RuntimeWarnings from librosa/numba
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_features(path, augment, seed, feature_set=FEATURE_SET):
    """Loads audio and returns its feature rows (5 augmented rows for training files, 1 otherwise)."""
    data = load_clip(to_project_path(path))
    if not augment:
        return np.array([extract_features(data, SAMPLE_RATE, feature_set)])

    # Per-file seed so the random augmentations are reproducible across runs and workers
    rng = np.random.default_rng(seed)
    return np.array([
        extract_features(data, SAMPLE_RATE, feature_set),
        extract_features(noise(data, rng), SAMPLE_RATE, feature_set),
        extract_features(stretch(data), SAMPLE_RATE, feature_set),
        extract_features(shift(data, rng), SAMPLE_RATE, feature_set),
        extract_features(pitch(data, SAMPLE_RATE), SAMPLE_RATE, feature_set)
    ])

def process_file(args):
    """Helper function for ProcessPoolExecutor mapping."""
    index, row, feature_set = args
    # Augment only the training split; val/test must stay untouched real recordings
    features = get_features(row['filepath'], augment=row['split'] == 'train', seed=RANDOM_STATE + index,
                            feature_set=feature_set)

    meta = [[row['emotion'], row['sex'], row['speaker'], row['source'], row['split']]] * len(features)
    return features, meta

def generate_1d_features(metadata_csv_path, output_csv_path, feature_set=FEATURE_SET, max_workers=16):
    dataset = pd.read_csv(metadata_csv_path)
    file_data = [(index, row, feature_set) for index, row in dataset.to_dict('index').items()]

    X, meta = [], []

    # Processes instead of threads: librosa feature code is mostly bound by the GIL
    # max_workers is capped so extraction can share the machine with a GPU training job
    print(f"Extracting 1D features ({feature_set}) with ProcessPoolExecutor...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_file, file_data, chunksize=16), total=len(file_data)))

    for features, rows in results:
        X.extend(features)
        meta.extend(rows)

    ensemble_features = pd.DataFrame(X, columns=[f'f{i}' for i in range(len(X[0]))])
    ensemble_features[META_COLUMNS] = pd.DataFrame(meta, columns=META_COLUMNS)

    # Shuffle the entire dataset to prevent sequential bias during training
    ensemble_features = ensemble_features.sample(frac=1, random_state=0).reset_index(drop=True)

    ensemble_features.to_csv(output_csv_path, index=False)
    print(f"Extraction complete! Features shape: {ensemble_features.shape}")
    print(f"Saved to {output_csv_path}")
    return ensemble_features

def save_preprocessing_artifacts(features_df):
    """Fits the scaler on the training split only and saves it with the label encoders.

    Every 1D model and the live app share these exact artifacts.
    """
    train = features_df[features_df['split'] == 'train']
    scaler = StandardScaler().fit(train.drop(columns=META_COLUMNS).values)

    # Fit on the fixed class lists so the index order never depends on which rows are present
    encoder = LabelEncoder().fit(EMOTIONS)
    sex_encoder = LabelEncoder().fit(SEXES)

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, LABEL_ENCODER_PATH)
    joblib.dump(sex_encoder, SEX_ENCODER_PATH)
    print(f"Saved scaler and label encoders to {SAVED_MODELS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 1D features from data/ser_metadata.csv.")
    parser.add_argument('--feature-set', choices=['v1', 'v2'], default=FEATURE_SET)
    parser.add_argument('--output', default=str(FEATURES_CSV), help="Output CSV path.")
    parser.add_argument('--skip-artifacts', action='store_true',
                        help="Do not overwrite saved_models/ scaler and encoders (for feature experiments).")
    args = parser.parse_args()

    features_df = generate_1d_features(METADATA_CSV, args.output, args.feature_set)
    if not args.skip_artifacts:
        save_preprocessing_artifacts(features_df)
