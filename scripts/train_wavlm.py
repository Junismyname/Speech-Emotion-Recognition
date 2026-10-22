import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import build_wavlm
from models.wavlm import PRETRAINED_WAVLM, wavlm_logits
from src.config import WAVE_TENSORS_PATH, SAVED_MODELS_DIR, RESULTS_DIR, EMOTIONS, RANDOM_STATE, PROJECT_ROOT
from src.gpu_augmentations import waveform_augment
from src.training import train_model, predict
from src.evaluation import evaluate, plot_history

WAVLM_WEIGHTS_PATH = SAVED_MODELS_DIR / 'wavlm_model.pth'
# Config is saved next to the weights so the app can rebuild the model without downloading
WAVLM_CONFIG_DIR = SAVED_MODELS_DIR / 'wavlm_config'

def output_paths(name):
    """Weights and config locations for a run name ('wavlm' keeps the original file names)."""
    return SAVED_MODELS_DIR / f'{name}_model.pth', SAVED_MODELS_DIR / f'{name}_config'

def make_forward(bf16=False):
    def forward(model, inputs):
        # Waveforms are stored as float16 and cast to float32. bfloat16 autocast is about 2x faster but made
        # fine-tuning collapse (~40% accuracy) in most seeds and every WavLM-Large run, so it is opt-in
        with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = wavlm_logits(model, inputs.float())
        return logits.float()
    return forward

# Default forward (float32), also used by scripts/evaluate_fusion.py
wavlm_forward = make_forward(bf16=False)

def wavlm_augment(inputs):
    return waveform_augment(inputs.float())

def load_wave_data(batch_size=16, waves_path=WAVE_TENSORS_PATH, balance_sources=False):
    print(f"Loading pre-computed waveforms from {waves_path}...")
    data = torch.load(waves_path, weights_only=False)

    loaders = {}
    for name in ['train', 'val', 'test']:
        idx = np.where(data['split'] == name)[0]
        dataset = TensorDataset(data['X'][torch.from_numpy(idx)], data['y'][torch.from_numpy(idx)])
        if name == 'train' and balance_sources:
            # Oversample small corpora (SAVEE is 3% of training clips): weight 1/sqrt(corpus size),
            # a middle ground between natural frequency and equal weight per corpus
            sources, counts = np.unique(data['source'][idx], return_counts=True)
            per_source = dict(zip(sources, 1 / np.sqrt(counts)))
            weights = torch.tensor([per_source[s] for s in data['source'][idx]], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(idx), replacement=True)
            loaders[name] = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=(name == 'train'))
        print(f"{name}: {len(idx)} clips")

    test_mask = data['split'] == 'test'
    return loaders, data['y'][torch.from_numpy(test_mask)].numpy(), data['source'][test_mask], data.get('clip_seconds')

def train_wavlm(loaders, y_test, test_sources, categories, epochs=12, checkpoint=PRETRAINED_WAVLM,
                name='wavlm', display_name='WavLM', lr=3e-5, head_lr=1e-3, bf16=False, clip_seconds=None, seed=RANDOM_STATE,
                warmup_epochs=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(seed)
    forward = make_forward(bf16)

    print(f"Downloading {checkpoint} weights...")
    model = build_wavlm(len(categories), pretrained=True, checkpoint=checkpoint).to(device)
    # The convolutional waveform encoder stays frozen; only the transformer and the head are fine-tuned
    model.freeze_feature_encoder()
    # Window the app must use for this model (None = the original 2.5 s from 0.6 s)
    model.config.clip_seconds = clip_seconds

    # New head (projector, classifier, layer weights) learns faster than the pre-trained transformer
    head_names = ('projector', 'classifier', 'layer_weights')
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(head_names)]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head_names)]
    optimizer = optim.AdamW([{'params': body, 'lr': lr}, {'params': head, 'lr': head_lr}], weight_decay=0.01)
    # Per-batch warmup then cosine decay, plus gradient clipping. Without warmup, the random new head's
    # early gradients sometimes wrecked the pre-trained transformer in epoch 1 (runs stuck near 40%)
    steps_per_epoch = len(loaders['train'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(warmup_epochs * steps_per_epoch), epochs * steps_per_epoch)

    history = train_model(model, loaders['train'], loaders['val'], optimizer, device, epochs=epochs,
                          model_name=display_name, patience=4, batch_scheduler=scheduler, max_grad_norm=1.0,
                          forward=forward, augment=wavlm_augment,
                          criterion=nn.CrossEntropyLoss(label_smoothing=0.1))
    plot_history(name, display_name, history)

    print("\n--- Evaluation ---")
    _, val_probs = predict(model, loaders['val'], device, forward=forward)
    pred_indices, test_probs = predict(model, loaders['test'], device, forward=forward)
    evaluate(name, display_name, y_test, pred_indices, categories, test_sources)
    # Probabilities kept for ensembling experiments (weights are always chosen on val)
    np.savez(RESULTS_DIR / f'{name}_probs.npz', val=val_probs, test=test_probs)

    weights_path, config_dir = output_paths(name)
    torch.save(model.state_dict(), weights_path)
    model.config.save_pretrained(config_dir)
    print(f"{display_name} model saved to {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune WavLM on raw waveforms.")
    parser.add_argument('--epochs', type=int, default=12, help="Epochs (also the cosine LR schedule length).")
    parser.add_argument('--checkpoint', default=PRETRAINED_WAVLM, help="Hugging Face WavLM checkpoint.")
    parser.add_argument('--name', default='wavlm', help="Output name: saved_models/<name>_model.pth, results/<name>.json.")
    parser.add_argument('--display-name', default='WavLM')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate of the pre-trained transformer.")
    parser.add_argument('--head-lr', type=float, default=1e-3, help="Learning rate of the new classification head.")
    parser.add_argument('--waves', default=str(WAVE_TENSORS_PATH), help="Waveform tensors from src/extraction_wave.py.")
    parser.add_argument('--balance-sources', action='store_true', help="Oversample small corpora (SAVEE, RAVDESS).")
    parser.add_argument('--bf16', action='store_true', help="bfloat16 autocast: ~2x faster but unstable for WavLM fine-tuning.")
    parser.add_argument('--seed', type=int, default=RANDOM_STATE)
    parser.add_argument('--warmup-epochs', type=float, default=1.0, help="Linear LR warmup length, in epochs.")
    args = parser.parse_args()

    loaders, y_test, test_sources, clip_seconds = load_wave_data(
        args.batch_size, PROJECT_ROOT / args.waves, args.balance_sources)
    # Same fixed class order as saved_models/label_encoder.joblib
    train_wavlm(loaders, y_test, test_sources, np.array(sorted(EMOTIONS)), epochs=args.epochs,
                checkpoint=args.checkpoint, name=args.name, display_name=args.display_name, lr=args.lr,
                head_lr=args.head_lr, bf16=args.bf16, clip_seconds=clip_seconds, seed=args.seed,
                warmup_epochs=args.warmup_epochs)
