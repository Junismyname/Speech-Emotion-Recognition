import torch

# Batch-level augmentations applied on the GPU during training only (see `augment` in src/training.py).
# Kept separate from src/augmentations.py so the multiprocessing extraction workers never import torch.

def spec_augment(images, n_freq_masks=2, max_freq=30, n_time_masks=2, max_time=40):
    """SpecAugment: blanks random frequency bands (rows) and time spans (columns) of each spectrogram.

    Works on the (B, 1, H, W) uint8 spectrogram batches; masked pixels are set to 0 (the quietest dB value).
    """
    batch, _, height, width = images.shape
    rows = torch.arange(height, device=images.device).view(1, height, 1)
    cols = torch.arange(width, device=images.device).view(1, 1, width)
    mask = torch.zeros(batch, height, width, dtype=torch.bool, device=images.device)

    for _ in range(n_freq_masks):
        size = torch.randint(0, max_freq + 1, (batch, 1, 1), device=images.device)
        start = (torch.rand(batch, 1, 1, device=images.device) * (height - size)).long()
        mask |= (rows >= start) & (rows < start + size)

    for _ in range(n_time_masks):
        size = torch.randint(0, max_time + 1, (batch, 1, 1), device=images.device)
        start = (torch.rand(batch, 1, 1, device=images.device) * (width - size)).long()
        mask |= (cols >= start) & (cols < start + size)

    return images.masked_fill(mask.unsqueeze(1), 0)

def waveform_augment(waves, sample_rate=16000, min_snr_db=15.0, max_snr_db=35.0, max_shift=0.25):
    """Random gain, background noise at a random SNR, and a random time shift for (B, L) waveforms."""
    batch = waves.shape[0]
    device = waves.device

    # Random gain between 0.8x and 1.2x
    waves = waves * (0.8 + 0.4 * torch.rand(batch, 1, device=device))

    # Gaussian noise scaled to a random signal-to-noise ratio
    snr_db = min_snr_db + (max_snr_db - min_snr_db) * torch.rand(batch, 1, device=device)
    signal_power = waves.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8)
    noise_power = signal_power / (10 ** (snr_db / 10))
    waves = waves + torch.randn_like(waves) * noise_power.sqrt()

    # Roll each clip by up to +/- max_shift seconds
    shifts = torch.randint(-int(max_shift * sample_rate), int(max_shift * sample_rate) + 1, (batch,))
    return torch.stack([torch.roll(w, int(s)) for w, s in zip(waves, shifts)])
