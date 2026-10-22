import os
import sys
import argparse
import warnings
import numpy as np
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Add the project root to the system path so Python can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import load_clip, fixed_length_wave, trimmed_wave
from src.config import METADATA_CSV, WAVE_TENSORS_PATH, WAVE_SAMPLE_RATE, EMOTIONS, DATA_DIR, to_project_path

# Suppress the FutureWarnings and RuntimeWarnings from librosa/numba
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def preprocess_wave(args):
    """Loads one recording as a fixed-length 16 kHz waveform.

    seconds=None keeps the original window (2.5 s from 0.6 s); otherwise silence is trimmed
    and up to `seconds` of speech are kept.
    """
    path, seconds = args
    if seconds is None:
        data = load_clip(to_project_path(path), sr=WAVE_SAMPLE_RATE)
        return fixed_length_wave(data, orig_sr=WAVE_SAMPLE_RATE).astype(np.float16)
    data = load_clip(to_project_path(path), offset=0.0, duration=None, sr=WAVE_SAMPLE_RATE)
    return trimmed_wave(data, orig_sr=WAVE_SAMPLE_RATE, seconds=seconds).astype(np.float16)

def generate_wave_tensors(metadata_csv_path, output_path, seconds=None, max_workers=16):
    # torch is imported here, not at module level, so the worker processes never load it (see extraction_vit.py)
    import torch

    dataset = pd.read_csv(metadata_csv_path)

    print(f"Loading 16 kHz waveforms ({'2.5 s from 0.6 s' if seconds is None else f'silence-trimmed, {seconds} s'})...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        waves = list(tqdm(executor.map(preprocess_wave, [(p, seconds) for p in dataset['filepath']], chunksize=32),
                          total=len(dataset)))

    # float16 halves the file size; training casts batches back to float32
    X_wave = torch.from_numpy(np.stack(waves))
    encoder = LabelEncoder().fit(EMOTIONS)

    torch.save({
        'X': X_wave,
        'y': torch.tensor(encoder.transform(dataset['emotion']), dtype=torch.long),
        'split': dataset['split'].to_numpy(),
        'source': dataset['source'].to_numpy(),
        'clip_seconds': seconds
    }, output_path)

    print(f"\nWaveform extraction complete! X shape: {tuple(X_wave.shape)} ({X_wave.dtype})")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract fixed-length 16 kHz waveforms for WavLM.")
    parser.add_argument('--seconds', type=float, default=None,
                        help="Trim silence and keep this many seconds (default: the 2.5 s training window).")
    parser.add_argument('--output', default=None, help="Output path (default: data/wave_tensors.pt, "
                                                       "or data/wave_tensors_<seconds>s.pt with --seconds).")
    args = parser.parse_args()

    output = args.output or (WAVE_TENSORS_PATH if args.seconds is None else DATA_DIR / f'wave_tensors_{args.seconds:g}s.pt')
    generate_wave_tensors(METADATA_CSV, output, args.seconds)
