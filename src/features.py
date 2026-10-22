import numpy as np
import librosa
from PIL import Image

from .config import SAMPLE_RATE, CLIP_OFFSET, CLIP_DURATION, FEATURE_SET, WAVE_SAMPLE_RATE

# Shared by training (src/extraction_*.py) and live inference (app/inference.py)
# so both sides always compute identical features.

VIT_IMAGE_SIZE = 224
# google/vit-base-patch16-224 was pre-trained on images normalized with mean 0.5 / std 0.5
VIT_MEAN = 0.5
VIT_STD = 0.5

def load_clip(path, offset=CLIP_OFFSET, duration=CLIP_DURATION, sr=SAMPLE_RATE):
    """Loads the fixed training window (2.5 s starting at 0.6 s) at the shared sample rate."""
    data, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return data

def trimmed_wave(data, orig_sr=SAMPLE_RATE, target_sr=WAVE_SAMPLE_RATE, seconds=4.0, top_db=30):
    """Trims leading/trailing silence, then keeps up to `seconds` of speech at 16 kHz (zero-padded).

    Adapts to each recording's own silence instead of a fixed 0.6 s offset, so long clips
    (RAVDESS, SAVEE: 3-7 s) are no longer cut mid-sentence.
    """
    trimmed, _ = librosa.effects.trim(data, top_db=top_db)
    if len(trimmed) < int(0.2 * orig_sr):
        trimmed = data  # Nothing above the threshold: keep the raw clip rather than an empty one
    return fixed_length_wave(trimmed, orig_sr=orig_sr, target_sr=target_sr, duration=seconds)

def trimmed_windows(data, orig_sr=SAMPLE_RATE, seconds=4.0, top_db=30, min_duration=1.0):
    """Live inference counterpart of trimmed_wave: trims silence, then cuts long recordings into
    `seconds`-long 16 kHz windows (a remainder of at least `min_duration` is kept)."""
    trimmed, _ = librosa.effects.trim(data, top_db=top_db)
    if len(trimmed) < int(0.2 * orig_sr):
        trimmed = data
    window = int(seconds * orig_sr)
    chunks = [trimmed[i:i + window] for i in range(0, len(trimmed), window)]
    chunks = [c for c in chunks if len(c) >= int(min_duration * orig_sr)] or [trimmed]
    return [fixed_length_wave(c, orig_sr=orig_sr, duration=seconds) for c in chunks]

def fixed_length_wave(data, orig_sr=SAMPLE_RATE, target_sr=WAVE_SAMPLE_RATE, duration=CLIP_DURATION):
    """Resamples a window to 16 kHz (if needed) and zero-pads/truncates it to exactly 2.5 s for WavLM."""
    if orig_sr != target_sr:
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    length = int(duration * target_sr)
    return np.pad(data, (0, max(0, length - len(data))))[:length].astype(np.float32)

def split_windows(data, offset=CLIP_OFFSET, duration=CLIP_DURATION, min_duration=1.0):
    """Cuts a variable-length recording into training-sized windows for live inference."""
    start = int(offset * SAMPLE_RATE)
    window = int(duration * SAMPLE_RATE)
    # Short recordings fall back to the whole clip instead of an empty window
    if len(data) - start < int(min_duration * SAMPLE_RATE):
        return [data]

    windows = []
    for begin in range(start, len(data), window):
        chunk = data[begin:begin + window]
        if len(chunk) >= int(min_duration * SAMPLE_RATE):
            windows.append(chunk)
    return windows

def extract_features(data, sample_rate=SAMPLE_RATE, feature_set=FEATURE_SET):
    """Extracts the configured 1D feature set (src/config.py: FEATURE_SET) for training and the app."""
    if feature_set == 'v2':
        return extract_features_v2(data, sample_rate)
    return extract_features_v1(data, sample_rate)

def extract_features_v2(data, sample_rate=SAMPLE_RATE):
    """Richer descriptor set (550 dims): mean and std over time of frame-level features, plus pitch.

    Compared to v1: log-scaled mel instead of raw power, 40 MFCCs with deltas, spectral shape
    descriptors, std pooling (how the voice varies, not only its average) and pitch statistics.
    """
    # One magnitude STFT (n_fft=2048, hop=512) shared by every spectral feature
    stft = np.abs(librosa.stft(data))
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(S=stft ** 2, sr=sample_rate))
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=40)

    frames = np.vstack([
        mfcc,                                                       # 40 timbre coefficients
        librosa.feature.delta(mfcc, mode='nearest'),                # 40 MFCC velocity
        librosa.feature.delta(mfcc, order=2, mode='nearest'),       # 40 MFCC acceleration
        mel_db,                                                     # 128 log-mel bands
        librosa.feature.chroma_stft(S=stft, sr=sample_rate),        # 12 pitch classes
        librosa.feature.spectral_contrast(S=stft, sr=sample_rate),  # 7 peak/valley contrast bands
        librosa.feature.spectral_centroid(S=stft, sr=sample_rate),  # brightness
        librosa.feature.spectral_bandwidth(S=stft, sr=sample_rate),
        librosa.feature.spectral_rolloff(S=stft, sr=sample_rate),
        librosa.feature.spectral_flatness(S=stft),                  # noisiness
        librosa.feature.zero_crossing_rate(y=data),
        librosa.feature.rms(S=stft)                                 # loudness
    ])

    # Fundamental frequency (pitch) per frame; percentiles are robust to unvoiced frames
    f0 = librosa.yin(data, fmin=65, fmax=600, sr=sample_rate)
    pitch = np.hstack((np.percentile(f0, [10, 50, 90]), np.std(f0)))

    return np.hstack((frames.mean(axis=1), frames.std(axis=1), pitch))

def extract_features_v1(data, sample_rate=SAMPLE_RATE):
    """Extracts ZCR, Chroma, MFCC, RMS, and MelSpectrogram and horizontally stacks them (162 dims)."""
    result = np.array([])

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def spectrogram_image(data, sample_rate=SAMPLE_RATE):
    """Converts audio into a 224x224 grayscale MelSpectrogram stored as uint8 (1, H, W).

    Stored as a single uint8 channel to keep the ViT tensor file ~12x smaller than
    3-channel float32. vit_input() expands it to the 3-channel input the ViT expects.
    """
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize negative decibel values to 0-255 before uint8 casting
    span = mel_spec_db.max() - mel_spec_db.min()
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (span if span > 0 else 1.0) * 255.0

    image = Image.fromarray(mel_spec_db.astype(np.uint8)).resize(
        (VIT_IMAGE_SIZE, VIT_IMAGE_SIZE), Image.BILINEAR
    )
    return np.array(image)[np.newaxis, :, :]

def vit_input(images_uint8):
    """Turns a (B, 1, H, W) uint8 torch batch into the normalized (B, 3, H, W) float batch the ViT expects."""
    images = images_uint8.float() / 255.0
    images = (images - VIT_MEAN) / VIT_STD
    return images.expand(-1, 3, -1, -1)
