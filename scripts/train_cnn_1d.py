import os
import sys
import torch
import torch.optim as optim

# Add the project root to the system path so Python can find the 'models' and 'src' packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CNN1D
from src.config import SAVED_MODELS_DIR, RANDOM_STATE
from src.data import load_1d_splits, make_loader
from src.training import train_model, predict
from src.evaluation import evaluate, plot_history

def train_cnn(splits, categories):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(RANDOM_STATE)

    train_loader = make_loader(splits['train']['X'], splits['train']['y_emotion'], shuffle=True)
    val_loader = make_loader(splits['val']['X'], splits['val']['y_emotion'])
    test_loader = make_loader(splits['test']['X'], splits['test']['y_emotion'])

    model = CNN1D(input_dim=splits['train']['X'].shape[1], num_classes=len(categories)).to(device)

    optimizer = optim.Adam(model.parameters())
    # Steps on validation loss (see src/training.py)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, min_lr=0.0000001)

    history = train_model(model, train_loader, val_loader, optimizer, device, epochs=50,
                          model_name="1D CNN", scheduler=scheduler)
    plot_history('cnn1d', '1D CNN', history)

    print("\n--- Evaluation ---")
    pred_indices, _ = predict(model, test_loader, device)
    evaluate('cnn1d', '1D CNN', splits['test']['y_emotion'], pred_indices, categories, splits['test']['source'])

    save_path = SAVED_MODELS_DIR / 'cnn_1d_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    splits, encoder, _ = load_1d_splits()
    train_cnn(splits, encoder.classes_)
