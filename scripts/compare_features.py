import os
import sys
import json
import argparse
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CNNStacked
from src.config import META_COLUMNS, EMOTIONS, RANDOM_STATE, RESULTS_DIR, PROJECT_ROOT
from src.data import make_loader
from src.training import train_model, predict

# Compares 1D feature sets on the VALIDATION split only, so choosing a feature set never looks at test.

def load_split(csv_path):
    df = pd.read_csv(csv_path)
    encoder = LabelEncoder().fit(EMOTIONS)
    train, val = df[df['split'] == 'train'], df[df['split'] == 'val']
    scaler = StandardScaler().fit(train.drop(columns=META_COLUMNS).values)
    return (scaler.transform(train.drop(columns=META_COLUMNS).values), encoder.transform(train['labels_emotion']),
            scaler.transform(val.drop(columns=META_COLUMNS).values), encoder.transform(val['labels_emotion']))

def score(y_true, y_pred):
    return {'val_accuracy': float(accuracy_score(y_true, y_pred)),
            'val_macro_f1': float(f1_score(y_true, y_pred, average='macro'))}

def compare(feature_csvs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for name, csv_path in feature_csvs.items():
        print(f"\n===== Feature set: {name} ({csv_path}) =====")
        x_train, y_train, x_val, y_val = load_split(csv_path)
        print(f"Feature dimension: {x_train.shape[1]}")
        results[name] = {'dimension': int(x_train.shape[1])}

        rf = RandomForestClassifier(n_estimators=300, n_jobs=8, random_state=RANDOM_STATE).fit(x_train, y_train)
        results[name]['random_forest'] = score(y_val, rf.predict(x_val))

        logreg = LogisticRegression(max_iter=2000, C=0.1).fit(x_train, y_train)
        results[name]['logistic_regression'] = score(y_val, logreg.predict(x_val))

        torch.manual_seed(RANDOM_STATE)
        cnn = CNNStacked(input_dim=x_train.shape[1], num_classes=len(EMOTIONS)).to(device)
        train_model(cnn, make_loader(x_train, y_train, shuffle=True), make_loader(x_val, y_val),
                    optim.Adam(cnn.parameters(), lr=1e-4), device, epochs=30, model_name=f"Stacked CNN ({name})",
                    patience=5)
        pred, _ = predict(cnn, make_loader(x_val, y_val), device)
        results[name]['stacked_cnn'] = score(y_val, pred)

        print(json.dumps(results[name], indent=2))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / 'feature_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {RESULTS_DIR / 'feature_comparison.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 1D feature sets on the validation split.")
    parser.add_argument('--sets', nargs='+', default=['v1=data/features.csv', 'v2=data/features_v2.csv'],
                        help="name=path pairs, paths relative to the project root.")
    args = parser.parse_args()
    compare({pair.split('=')[0]: PROJECT_ROOT / pair.split('=')[1] for pair in args.sets})
