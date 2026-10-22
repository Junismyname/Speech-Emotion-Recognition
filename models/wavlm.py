from transformers import AutoFeatureExtractor, WavLMConfig, WavLMForSequenceClassification

PRETRAINED_WAVLM = 'microsoft/wavlm-base-plus'

def build_wavlm(num_classes, pretrained=True, config_dir=None, checkpoint=PRETRAINED_WAVLM):
    """Builds WavLM (Base+ by default, or e.g. microsoft/wavlm-large) with a mean-pooled classification head.

    use_weighted_layer_sum learns a weighted mix of all transformer layers instead of using only
    the last one; emotion cues live mostly in the middle layers (the SUPERB benchmark setup).
    pretrained=False rebuilds the architecture from a saved config (config_dir) without downloading,
    for loading fine-tuned weights in the app.
    """
    if pretrained:
        # layerdrop=0.0: with LayerDrop, a skipped layer records no hidden state in transformers 5.x,
        # so the weighted layer sum gets 12 states for 13 weights and crashes during training
        model = WavLMForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_classes, use_weighted_layer_sum=True, layerdrop=0.0
        )
        # Large checkpoints expect zero-mean/unit-variance input, Base+ expects raw audio. Stored in the
        # config (saved with the weights) so training and the app apply the same preprocessing.
        model.config.do_normalize = AutoFeatureExtractor.from_pretrained(checkpoint).do_normalize
        return model
    config = WavLMConfig.from_pretrained(config_dir)
    return WavLMForSequenceClassification(config)

def wavlm_logits(model, waves):
    """Forward pass on (B, L) float waveforms, normalizing each clip if the checkpoint expects it."""
    if getattr(model.config, 'do_normalize', False):
        waves = (waves - waves.mean(dim=1, keepdim=True)) / (waves.std(dim=1, keepdim=True) + 1e-7)
    return model(waves).logits
