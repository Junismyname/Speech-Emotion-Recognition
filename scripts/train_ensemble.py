import os
import sys
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import EnsembleBaseCNN
from src.config import SAVED_MODELS_DIR, RANDOM_STATE
from src.data import load_1d_splits, make_loader
from src.training import train_model, predict
from src.evaluation import evaluate, plot_history

def sex_subset(split, sex_idx):
    """Rows of one split belonging to a single sex (used for the male/female emotion models)."""
    mask = split['y_sex'] == sex_idx
    return split['X'][mask], split['y_emotion'][mask]

def evaluate_and_save(gender_model, emo_model_m, emo_model_f, splits, sex_encoder, categories, device):
    print("\n--- Running Hierarchical Inference ---")
    test = splits['test']
    test_loader = make_loader(test['X'], test['y_emotion'])

    MALE_IDX = sex_encoder.transform(['male'])[0]

    # Route every test clip through the predicted (not the true) sex, exactly like the app
    gender_preds, _ = predict(gender_model, test_loader, device)
    male_preds, _ = predict(emo_model_m, test_loader, device)
    female_preds, _ = predict(emo_model_f, test_loader, device)
    emotion_preds = np.where(gender_preds == MALE_IDX, male_preds, female_preds)

    gender_acc = accuracy_score(test['y_sex'], gender_preds)
    print(f"Gender Routing Accuracy: {gender_acc * 100:.2f}%")
    evaluate('cnnensemble', 'Ensemble CNN', test['y_emotion'], emotion_preds, categories, test['source'],
             extra={'gender_routing_accuracy': float(gender_acc)})

    torch.save(gender_model.state_dict(), SAVED_MODELS_DIR / 'ensemble_gender_model.pth')
    torch.save(emo_model_m.state_dict(), SAVED_MODELS_DIR / 'ensemble_male_model.pth')
    torch.save(emo_model_f.state_dict(), SAVED_MODELS_DIR / 'ensemble_female_model.pth')
    print("All ensemble models saved to saved_models/.")

if __name__ == "__main__":
    # Scaler comes from src/extraction_1d.py and is fit on the training split only
    splits, encoder_emo, encoder_sex = load_1d_splits()
    categories = encoder_emo.classes_
    input_dim = splits['train']['X'].shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(RANDOM_STATE)
    num_emotions = len(categories)
    male_idx = encoder_sex.transform(['male'])[0]
    female_idx = encoder_sex.transform(['female'])[0]

    gender_model = EnsembleBaseCNN(input_dim, num_classes=len(encoder_sex.classes_)).to(device)
    emo_model_m = EnsembleBaseCNN(input_dim, num_classes=num_emotions).to(device)
    emo_model_f = EnsembleBaseCNN(input_dim, num_classes=num_emotions).to(device)

    # Gender router: all training clips, sex labels
    history = train_model(
        gender_model,
        make_loader(splits['train']['X'], splits['train']['y_sex'], shuffle=True),
        make_loader(splits['val']['X'], splits['val']['y_sex']),
        optim.Adam(gender_model.parameters(), lr=1e-4), device, epochs=10, model_name="Gender Router"
    )
    plot_history('cnnensemble_gender', 'Gender Router', history)

    # Sex-specific emotion models: each trained and validated on its own subset
    for model, sex_idx, name, label in [(emo_model_m, male_idx, 'cnnensemble_male', 'Male Emotion Model'),
                                        (emo_model_f, female_idx, 'cnnensemble_female', 'Female Emotion Model')]:
        history = train_model(
            model,
            make_loader(*sex_subset(splits['train'], sex_idx), shuffle=True),
            make_loader(*sex_subset(splits['val'], sex_idx)),
            optim.Adam(model.parameters(), lr=1e-4), device, epochs=30, model_name=label
        )
        plot_history(name, label, history)

    evaluate_and_save(gender_model, emo_model_m, emo_model_f, splits, encoder_sex, categories, device)
