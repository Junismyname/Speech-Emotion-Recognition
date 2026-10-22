from transformers import ViTConfig, ViTForImageClassification

PRETRAINED_VIT = 'google/vit-base-patch16-224'

def build_vit(num_classes, pretrained=True):
    """Builds the ViT with an emotion-sized classifier head.

    pretrained=True downloads the ImageNet weights for fine-tuning and replaces the
    1000-class head. pretrained=False builds the same architecture offline (ViTConfig
    defaults match vit-base-patch16-224); use it when a fine-tuned state_dict will be loaded.
    """
    if pretrained:
        return ViTForImageClassification.from_pretrained(
            PRETRAINED_VIT, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    return ViTForImageClassification(ViTConfig(num_labels=num_classes))
