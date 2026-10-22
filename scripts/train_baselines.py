import os
import sys
import json
import joblib
import argparse
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Add the project root to the system path so Python can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import SAVED_MODELS_DIR, RANDOM_STATE
from src.data import load_1d_splits
from src.evaluation import evaluate

RF_PARAMS_PATH = SAVED_MODELS_DIR / 'rf_params.json'
SVM_PARAMS_PATH = SAVED_MODELS_DIR / 'svm_params.json'

def grid_search(estimator, param_grid, train):
    """GridSearchCV with folds grouped by speaker, so no voice is in both a training and a scoring fold."""
    grid = GridSearchCV(estimator, param_grid, refit=False, verbose=3, cv=GroupKFold(n_splits=5), n_jobs=-1)
    grid.fit(train['X'], train['y_emotion'], groups=train['speaker'])
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_params_

def load_params(path, defaults):
    if os.path.exists(path):
        print("Loading previously optimized parameters...")
        with open(path, 'r') as f:
            return json.load(f)
    print("No saved parameters found. Using defaults...")
    return defaults

def train_random_forest(splits, categories, optimize=False):
    print("\n--- Training Random Forest ---")

    if optimize:
        print("Running GridSearchCV (this may take a while)...")
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        params = grid_search(RandomForestClassifier(random_state=RANDOM_STATE), param_grid_rf, splits['train'])
        with open(RF_PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=4)
    else:
        params = load_params(RF_PARAMS_PATH, {'n_estimators': 100})

    best_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    best_rf.fit(splits['train']['X'], splits['train']['y_emotion'])

    test = splits['test']
    y_pred = best_rf.predict(test['X'])
    evaluate('random_forest', 'Random Forest', test['y_emotion'], y_pred, categories, test['source'])

    # Compressed: a fully grown 300-tree forest is several hundred MB uncompressed
    joblib.dump(best_rf, SAVED_MODELS_DIR / 'best_rf_model.joblib', compress=3)
    print("Random Forest model saved.")

def train_svm(splits, categories, optimize=False):
    print("\n--- Training SVM ---")

    if optimize:
        print("Running GridSearchCV (this may take a while)...")
        # RBF only: poly/sigmoid were never competitive and tripled the search time on 550-dim features
        param_grid_svm = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.001, 0.0003],
            'kernel': ['rbf']
        }
        # probability=False during the search: Platt scaling runs an extra internal 5-fold CV per fit
        params = grid_search(SVC(), param_grid_svm, splits['train'])
        with open(SVM_PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=4)
    else:
        params = load_params(SVM_PARAMS_PATH, {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'})

    # probability=True only on the final model, which the app needs for predict_proba
    best_svm = SVC(probability=True, random_state=RANDOM_STATE, **params)
    best_svm.fit(splits['train']['X'], splits['train']['y_emotion'])

    test = splits['test']
    y_pred = best_svm.predict(test['X'])
    evaluate('svm', 'SVM', test['y_emotion'], y_pred, categories, test['source'])

    joblib.dump(best_svm, SAVED_MODELS_DIR / 'best_svm_model.joblib')
    print("SVM model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline ML models.")
    parser.add_argument('--optimize', action='store_true', help="Run GridSearchCV for hyperparameter tuning.")
    parser.add_argument('--model', choices=['rf', 'svm', 'all'], default='all', help="Which baseline to train.")
    args = parser.parse_args()

    splits, encoder, _ = load_1d_splits()
    categories = encoder.classes_

    if args.model in ('rf', 'all'):
        train_random_forest(splits, categories, optimize=args.optimize)
    if args.model in ('svm', 'all'):
        train_svm(splits, categories, optimize=args.optimize)
