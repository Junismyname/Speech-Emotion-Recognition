import os
import sys
import shutil
import kagglehub
import numpy as np
import pandas as pd

# Add the project root to the system path so Python can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    PROJECT_ROOT, DATASET_DIR, METADATA_CSV,
    VAL_FRACTION, TEST_FRACTION, RANDOM_STATE
)

def download_datasets(target_base_dir):
    datasets = {
        "ravdess": "uwrfkaggler/ravdess-emotional-speech-audio",
        "tess": "ejlok1/toronto-emotional-speech-set-tess",
        "cremad": "ejlok1/cremad",
        "savee": "ejlok1/surrey-audiovisual-expressed-emotion-savee"
    }

    os.makedirs(target_base_dir, exist_ok=True)

    for name, handle in datasets.items():
        dataset_dir = os.path.join(target_base_dir, name)

        # Duplicate-safe check: verify if the folder exists and is not empty
        if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
            print(f"Dataset '{name}' already exists. Skipping download.")
        else:
            print(f"Downloading '{name}' dataset...")
             # Download using kagglehub (downloads to a local cache directory)
            cached_path = kagglehub.dataset_download(handle)
            shutil.copytree(cached_path, dataset_dir, dirs_exist_ok=True)
            print(f"Successfully saved '{name}'.")

def get_ravdess_emotion(filename):
    # RAVDESS format: 03-01-05-01-01-01-01.wav (3rd part is emotion)
    # 'calm' (02) is merged into 'neutral'
    emotion_dict = {'01':'neutral', '02':'neutral', '03':'happy', '04':'sad',
                    '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}
    parts = filename.split('-')
    return emotion_dict.get(parts[2]) if len(parts) > 2 else None

def get_tess_emotion(filename):
    # TESS format: OAF_back_angry.wav (last part is emotion, 'ps' = pleasant surprise)
    code = filename.lower().split('.')[0].split('_')[-1]
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    if code in emotions:
        return 'surprise' if code == 'ps' else code
    return None

def get_cremad_emotion(filename):
    # CREMA-D format: 1001_DFA_ANG_XX.wav (3rd part is emotion)
    emotion_dict = {'ANG':'angry', 'DIS':'disgust', 'FEA':'fear',
                    'HAP':'happy', 'NEU':'neutral', 'SAD':'sad'}
    parts = filename.split('_')
    return emotion_dict.get(parts[2]) if len(parts) > 2 else None

def get_savee_emotion(filename):
    # SAVEE format: DC_a01.wav (letter code before numbers)
    emotion_dict = {'a':'angry', 'd':'disgust', 'f':'fear', 'h':'happy',
                    'n':'neutral', 'sa':'sad', 'su':'surprise'}
    parts = filename.split('_')
    if len(parts) < 2:
        return None
    code = ''.join([c for c in parts[1] if not c.isdigit()]).split('.')[0]
    return emotion_dict.get(code)

def get_ravdess_sex(filename):
    # The 7th identifier (index 6) is the Actor ID. Odd = Male, Even = Female.
    parts = filename.split('-')
    if len(parts) > 6:
        actor_id = int(parts[6].split('.')[0])
        return 'male' if actor_id % 2 != 0 else 'female'
    return None

def get_speaker(dataset, file, root):
    """Returns a dataset-prefixed speaker ID so speakers never collide across datasets."""
    if dataset == 'ravdess':
        # 7th identifier is the Actor ID
        return f"ravdess_{file.split('-')[6].split('.')[0]}"
    if dataset == 'tess':
        # Folder name (e.g. OAF_angry) is used because one filename has a typo ('OA_')
        return f"tess_{os.path.basename(root).split('_')[0].upper()}"
    if dataset == 'cremad':
        return f"cremad_{file.split('_')[0]}"
    # SAVEE: speaker initials prefix (e.g. DC_a01.wav)
    return f"savee_{file.split('_')[0]}"

def get_cremad_demographics(cremad_dir):
    demo_path = os.path.join(cremad_dir, 'Metadata/VideoDemographics.csv')
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        return dict(zip(df['ActorID'].astype(str), df['Sex'].str.lower()))
    return {}

def create_unified_dataset(base_dir):
    data = []
    seen_files = set()
    cremad_dir = os.path.join(base_dir, 'cremad')
    cremad_sex_map = get_cremad_demographics(cremad_dir)

    for dataset in ['ravdess', 'tess', 'cremad', 'savee']:
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_path):
            continue

        for root, dirs, files in os.walk(dataset_path):
            # Sort for a deterministic walk order (and therefore deterministic duplicate removal)
            dirs.sort()
            for file in sorted(files):
                if not file.endswith('.wav'):
                    continue

                # The Kaggle archives for RAVDESS and TESS contain a nested second copy of every file
                if (dataset, file) in seen_files:
                    continue
                seen_files.add((dataset, file))

                file_path = os.path.join(root, file)
                emotion, sex = None, None

                if dataset == 'ravdess':
                    emotion = get_ravdess_emotion(file)
                    sex = get_ravdess_sex(file)
                elif dataset == 'tess':
                    emotion = get_tess_emotion(file)
                    sex = 'female'
                elif dataset == 'cremad':
                    emotion = get_cremad_emotion(file)
                    parts = file.split('_')
                    actor_id = parts[0] if len(parts) > 0 else ''
                    sex = cremad_sex_map.get(actor_id)
                elif dataset == 'savee':
                    emotion = get_savee_emotion(file)
                    sex = 'male'

                # Skip files whose emotion or sex cannot be determined
                if emotion and sex:
                    data.append({
                        # Stored relative to the project root with forward slashes so the CSV is portable
                        'filepath': os.path.relpath(file_path, PROJECT_ROOT).replace('\\', '/'),
                        'emotion': emotion,
                        'sex': sex,
                        'speaker': get_speaker(dataset, file, root),
                        'source': dataset
                    })
                else:
                    print(f"Skipping {file_path} (emotion={emotion}, sex={sex})")

    return pd.DataFrame(data)

def assign_splits(dataset, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=RANDOM_STATE):
    """Assigns train/val/test by speaker so no voice appears in more than one split.

    Speakers are split separately inside each (source, sex) group, which keeps every
    dataset and both sexes represented in all three splits.
    """
    rng = np.random.default_rng(seed)
    dataset = dataset.copy()
    dataset['split'] = 'train'

    for (source, sex), group in dataset.groupby(['source', 'sex']):
        speakers = sorted(group['speaker'].unique())
        rng.shuffle(speakers)
        n = len(speakers)

        if n >= 3:
            n_test = max(1, round(n * test_fraction))
            n_val = max(1, round(n * val_fraction))
            test_speakers = speakers[:n_test]
            val_speakers = speakers[n_test:n_test + n_val]
            dataset.loc[dataset['speaker'].isin(test_speakers), 'split'] = 'test'
            dataset.loc[dataset['speaker'].isin(val_speakers), 'split'] = 'val'
        else:
            # Too few speakers (TESS has 2): the last speaker is held out and its files are
            # shared between val and test, so train and test still never share a voice
            held_out = group[group['speaker'] == speakers[-1]].index.to_numpy()
            held_out = rng.permutation(held_out)
            dataset.loc[held_out[:len(held_out) // 2], 'split'] = 'val'
            dataset.loc[held_out[len(held_out) // 2:], 'split'] = 'test'

    return dataset

if __name__ == "__main__":
    download_datasets(DATASET_DIR)
    dataset = create_unified_dataset(DATASET_DIR)
    dataset = assign_splits(dataset)

    dataset.to_csv(METADATA_CSV, index=False)
    print(f"\nUnified dataset created with {len(dataset)} files and saved to {METADATA_CSV}.")
    print(pd.crosstab(dataset['source'], dataset['split']))
