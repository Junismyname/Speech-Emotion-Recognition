import os
import sys
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CNNStacked
from src.config import SAVED_MODELS_DIR, RANDOM_STATE
from src.data import load_1d_splits, make_loader
from src.training import train_model, predict
from src.evaluation import evaluate, plot_history

def train_cnn_stacked(splits, categories):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(RANDOM_STATE)

    train_loader = make_loader(splits['train']['X'], splits['train']['y_emotion'], shuffle=True)
    val_loader = make_loader(splits['val']['X'], splits['val']['y_emotion'])
    test_loader = make_loader(splits['test']['X'], splits['test']['y_emotion'])

    model = CNNStacked(input_dim=splits['train']['X'].shape[1], num_classes=len(categories)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Early stopping on validation accuracy; the best weights are restored afterwards
    history = train_model(model, train_loader, val_loader, optimizer, device, epochs=50,
                          model_name="Stacked CNN", scheduler=scheduler, patience=5)
    plot_history('cnnstacked', 'Stacked CNN', history)

    print("\n--- Evaluation ---")
    pred_indices, _ = predict(model, test_loader, device)
    evaluate('cnnstacked', 'Stacked CNN', splits['test']['y_emotion'], pred_indices, categories, splits['test']['source'])

    save_path = SAVED_MODELS_DIR / 'cnn_stacked_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    splits, encoder, _ = load_1d_splits()
    train_cnn_stacked(splits, encoder.classes_)
