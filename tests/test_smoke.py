import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

# Add the project root to the system path so Python can find the 'src' and 'models' packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import ingestion
from src.config import EMOTIONS, SAMPLE_RATE, METADATA_CSV
from src.features import extract_features, split_windows, spectrogram_image, vit_input
from models import CNN1D, CNNStacked, EnsembleBaseCNN, build_vit

FEATURE_DIM = 162

# Label parsers

def test_ravdess_parsers():
    assert ingestion.get_ravdess_emotion('03-01-05-01-01-01-01.wav') == 'angry'
    # 'calm' is merged into 'neutral'
    assert ingestion.get_ravdess_emotion('03-01-02-01-01-01-01.wav') == 'neutral'
    assert ingestion.get_ravdess_sex('03-01-05-01-01-01-01.wav') == 'male'
    assert ingestion.get_ravdess_sex('03-01-05-01-01-01-02.wav') == 'female'

def test_tess_parser():
    assert ingestion.get_tess_emotion('OAF_back_angry.wav') == 'angry'
    assert ingestion.get_tess_emotion('YAF_chair_ps.wav') == 'surprise'
    assert ingestion.get_tess_emotion('OA_bite_neutral.wav') == 'neutral'

def test_cremad_and_savee_parsers():
    assert ingestion.get_cremad_emotion('1001_DFA_ANG_XX.wav') == 'angry'
    assert ingestion.get_savee_emotion('DC_sa01.wav') == 'sad'
    assert ingestion.get_savee_emotion('DC_su15.wav') == 'surprise'
    assert ingestion.get_savee_emotion('nounderscore.wav') is None

def test_speaker_ids():
    assert ingestion.get_speaker('ravdess', '03-01-05-01-01-01-07.wav', 'x/Actor_07') == 'ravdess_07'
    assert ingestion.get_speaker('tess', 'OA_bite_neutral.wav', 'x/OAF_neutral') == 'tess_OAF'
    assert ingestion.get_speaker('cremad', '1001_DFA_ANG_XX.wav', 'x') == 'cremad_1001'
    assert ingestion.get_speaker('savee', 'DC_a01.wav', 'x') == 'savee_DC'

# Metadata (only when data/ser_metadata.csv has been generated)

@pytest.mark.skipif(not METADATA_CSV.exists(), reason="run src/ingestion.py first")
def test_metadata_split_is_speaker_disjoint():
    meta = pd.read_csv(METADATA_CSV)
    assert not meta.duplicated(subset=['source', 'filepath']).any()
    assert set(meta['emotion']) == set(EMOTIONS)
    train_speakers = set(meta.loc[meta['split'] == 'train', 'speaker'])
    held_out_speakers = set(meta.loc[meta['split'] != 'train', 'speaker'])
    assert train_speakers.isdisjoint(held_out_speakers)

# Features

@pytest.mark.parametrize('feature_set, dim', [('v1', 162), ('v2', 550)])
def test_feature_vector_length(feature_set, dim):
    audio = np.random.default_rng(0).normal(size=int(2.5 * SAMPLE_RATE)).astype(np.float32)
    features = extract_features(audio, feature_set=feature_set)
    assert features.shape == (dim,) and np.isfinite(features).all()

def test_split_windows():
    # 10 s recording -> 9.4 s after the 0.6 s offset -> three 2.5 s windows plus a 1.9 s remainder (kept, >= 1 s)
    assert len(split_windows(np.zeros(10 * SAMPLE_RATE))) == 4
    # Very short recording falls back to the whole clip
    assert len(split_windows(np.zeros(SAMPLE_RATE))) == 1

def test_vit_image_pipeline():
    audio = np.random.default_rng(0).normal(size=int(2.5 * SAMPLE_RATE)).astype(np.float32)
    image = spectrogram_image(audio)
    assert image.shape == (1, 224, 224) and image.dtype == np.uint8
    batch = vit_input(torch.from_numpy(image).unsqueeze(0))
    assert batch.shape == (1, 3, 224, 224)
    assert batch.min() >= -1.0 and batch.max() <= 1.0

# Model output shapes for both (B, L, 1) and (B, 1, L) inputs

@pytest.mark.parametrize('model_cls', [CNN1D, CNNStacked, EnsembleBaseCNN])
def test_cnn_output_shapes(model_cls):
    model = model_cls(input_dim=FEATURE_DIM, num_classes=len(EMOTIONS)).eval()
    with torch.no_grad():
        assert model(torch.zeros(4, FEATURE_DIM, 1)).shape == (4, len(EMOTIONS))
        assert model(torch.zeros(4, 1, FEATURE_DIM)).shape == (4, len(EMOTIONS))

def test_vit_builds_offline():
    model = build_vit(len(EMOTIONS), pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).logits.shape == (1, len(EMOTIONS))

# Training-time augmentations

def test_spec_augment_masks_without_changing_shape():
    from src.gpu_augmentations import spec_augment
    images = torch.full((4, 1, 224, 224), 200, dtype=torch.uint8)
    out = spec_augment(images)
    assert out.shape == images.shape and out.dtype == torch.uint8
    assert (out == 0).any() and (out == 200).any()

def test_waveform_augment_keeps_shape():
    from src.gpu_augmentations import waveform_augment
    waves = torch.randn(4, 40000)
    assert waveform_augment(waves).shape == waves.shape

def test_fixed_length_wave():
    from src.features import fixed_length_wave
    # 1 s at 22,050 Hz -> resampled to 16 kHz and padded to 2.5 s
    assert fixed_length_wave(np.zeros(SAMPLE_RATE, dtype=np.float32)).shape == (40000,)

def test_wavlm_head_parameter_names():
    # train_wavlm.py gives these prefixes a higher learning rate; make sure they exist
    from transformers import WavLMConfig, WavLMForSequenceClassification
    model = WavLMForSequenceClassification(WavLMConfig(num_labels=len(EMOTIONS), use_weighted_layer_sum=True,
                                                       num_hidden_layers=2))
    names = [n for n, _ in model.named_parameters()]
    for prefix in ('projector', 'classifier', 'layer_weights'):
        assert any(n.startswith(prefix) for n in names)
