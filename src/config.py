from pathlib import Path

# All paths are resolved from the project root so scripts can be run from any directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data'
DATASET_DIR = DATA_DIR / 'dataset'
METADATA_CSV = DATA_DIR / 'ser_metadata.csv'
FEATURES_CSV = DATA_DIR / 'features.csv'
VIT_X_PATH = DATA_DIR / 'X_vit_tensor.pt'
VIT_Y_PATH = DATA_DIR / 'Y_vit_tensor.pt'
VIT_SPLIT_PATH = DATA_DIR / 'split_vit.npy'
VIT_SOURCE_PATH = DATA_DIR / 'source_vit.npy'
# Raw 16 kHz waveforms (float16) with labels/split/source for the WavLM model
WAVE_TENSORS_PATH = DATA_DIR / 'wave_tensors.pt'

SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'
SCALER_PATH = SAVED_MODELS_DIR / 'scaler.joblib'
LABEL_ENCODER_PATH = SAVED_MODELS_DIR / 'label_encoder.joblib'
SEX_ENCODER_PATH = SAVED_MODELS_DIR / 'sex_encoder.joblib'

RESULTS_DIR = PROJECT_ROOT / 'results'
IMAGES_DIR = PROJECT_ROOT / 'images'

# 'calm' (RAVDESS only) is merged into 'neutral', leaving 7 classes
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SEXES = ['female', 'male']

# Audio window used for both training and live inference
SAMPLE_RATE = 22050
CLIP_OFFSET = 0.6
CLIP_DURATION = 2.5
# WavLM was pre-trained on 16 kHz audio
WAVE_SAMPLE_RATE = 16000

# Which trained WavLM run the app serves as "WavLM" (saved_models/<run>_model.pth + <run>_config/),
# and the validation-tuned ensemble written by scripts/evaluate_ensemble.py
APP_WAVLM_RUN = 'wavlm_large'
ENSEMBLE_WEIGHTS_PATH = SAVED_MODELS_DIR / 'ensemble_best_weights.json'

# 1D feature set used by extraction_1d.py and the app: 'v1' (162 dims) or 'v2' (550 dims), see src/features.py
FEATURE_SET = 'v2'

# Speaker-level split proportions and the seed used everywhere
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_STATE = 42

# Columns of features.csv that are not features
META_COLUMNS = ['labels_emotion', 'labels_sex', 'speaker', 'source', 'split']


def to_project_path(relative_path):
    """Turns a path stored in the metadata CSV back into an absolute path."""
    return PROJECT_ROOT / relative_path
