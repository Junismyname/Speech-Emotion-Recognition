import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import build_vit
from src.config import (
    VIT_X_PATH, VIT_Y_PATH, VIT_SPLIT_PATH, VIT_SOURCE_PATH,
    SAVED_MODELS_DIR, EMOTIONS, RANDOM_STATE
)
from src.features import vit_input
from src.gpu_augmentations import spec_augment
from src.training import train_model, predict
from src.evaluation import evaluate, plot_history

def vit_forward(model, inputs):
    # Tensors are stored as 1-channel uint8; expand and normalize on the GPU, batch by batch
    return model(vit_input(inputs)).logits

def load_vit_data():
    print("Loading pre-computed ViT tensors...")
    X_vit_tensor = torch.load(VIT_X_PATH)
    Y_vit_tensor = torch.load(VIT_Y_PATH)
    split = np.load(VIT_SPLIT_PATH, allow_pickle=True)
    sources = np.load(VIT_SOURCE_PATH, allow_pickle=True)

    loaders = {}
    for name in ['train', 'val', 'test']:
        idx = torch.from_numpy(np.where(split == name)[0])
        # Batch size reduced to 32
        loaders[name] = DataLoader(TensorDataset(X_vit_tensor[idx], Y_vit_tensor[idx]),
                                   batch_size=32, shuffle=(name == 'train'))
        print(f"{name}: {len(idx)} images")

    test_mask = split == 'test'
    return loaders, Y_vit_tensor[torch.from_numpy(test_mask)].numpy(), sources[test_mask]

def train_vision_transformer(loaders, y_test, test_sources, categories):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(RANDOM_STATE)

    print("Downloading google/vit-base-patch16-224 weights...")
    # Replaces the default 1000-class ImageNet head (see models/vit.py)
    model_vit = build_vit(len(categories), pretrained=True).to(device)

    # ViTs require much smaller learning rates; weight decay, SpecAugment and label smoothing
    # fight the fast overfitting seen without them (train acc ~97% vs val ~62% by epoch 6)
    epochs = 15
    optimizer = optim.AdamW(model_vit.parameters(), lr=2e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = train_model(model_vit, loaders['train'], loaders['val'], optimizer, device, epochs=epochs,
                          model_name="Vision Transformer", scheduler=scheduler, patience=5,
                          forward=vit_forward, augment=spec_augment,
                          criterion=nn.CrossEntropyLoss(label_smoothing=0.1))
    plot_history('vit', 'Vision Transformer', history)

    print("\n--- Evaluation ---")
    pred_indices, _ = predict(model_vit, loaders['test'], device, forward=vit_forward)
    evaluate('vit', 'Vision Transformer', y_test, pred_indices, categories, test_sources)

    save_path = SAVED_MODELS_DIR / 'VIT_model.pth'
    torch.save(model_vit.state_dict(), save_path)
    print(f"Vision Transformer model saved to {save_path}")

if __name__ == "__main__":
    loaders, y_test, test_sources = load_vit_data()
    # Same fixed class order as saved_models/label_encoder.joblib
    train_vision_transformer(loaders, y_test, test_sources, np.array(sorted(EMOTIONS)))
