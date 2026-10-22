import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import WAVE_TENSORS_PATH, SAVED_MODELS_DIR, RESULTS_DIR, EMOTIONS
from src.evaluation import evaluate

# Weighted average of several models' probabilities. Every member's val/test probabilities come from
# results/<name>_probs.npz (written by the training scripts). Members and weights are chosen on the
# VALIDATION split by greedy forward selection with replacement (Caruana et al., 2004); the test
# split is scored once with the chosen weights.

def load_probs(names):
    probs = {}
    for name in names:
        data = np.load(RESULTS_DIR / f'{name}_probs.npz')
        probs[name] = {'val': data['val'], 'test': data['test']}
    return probs

def greedy_weights(probs, y_val, rounds=20):
    """Repeatedly adds the member (repeats allowed) that most improves val accuracy of the running average."""
    chosen, total = [], None
    for _ in range(rounds):
        scores = {name: accuracy_score(y_val, (p['val'] if total is None else total + p['val']).argmax(1))
                  for name, p in probs.items()}
        best = max(scores, key=scores.get)
        chosen.append(best)
        total = probs[best]['val'] if total is None else total + probs[best]['val']
    counts = {name: chosen.count(name) for name in probs if name in chosen}
    return {name: count / len(chosen) for name, count in counts.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Val-tuned probability ensemble of saved models.")
    parser.add_argument('--members', nargs='+', required=True, help="Run names with results/<name>_probs.npz.")
    parser.add_argument('--name', default='ensemble_best', help="Output name: results/<name>.json, saved_models/<name>_weights.json.")
    parser.add_argument('--display-name', default='Best Ensemble')
    args = parser.parse_args()

    # Labels/sources in metadata order (identical for every tensor file)
    data = torch.load(WAVE_TENSORS_PATH, weights_only=False)
    split, y, sources = data['split'], data['y'].numpy(), data['source']
    y_val, y_test = y[split == 'val'], y[split == 'test']
    categories = np.array(sorted(EMOTIONS))

    probs = load_probs(args.members)
    for name, p in probs.items():
        print(f"{name:28s} val acc {accuracy_score(y_val, p['val'].argmax(1)):.4f}")

    weights = greedy_weights(probs, y_val)
    fused_val = sum(w * probs[n]['val'] for n, w in weights.items())
    print(f"\nChosen weights (on val): {weights} -> val acc {accuracy_score(y_val, fused_val.argmax(1)):.4f}")

    fused_test = sum(w * probs[n]['test'] for n, w in weights.items())
    evaluate(args.name, args.display_name, y_test, fused_test.argmax(1), categories, sources[split == 'test'],
             extra={'members': weights, 'val_accuracy': float(accuracy_score(y_val, fused_val.argmax(1)))})
    with open(SAVED_MODELS_DIR / f'{args.name}_weights.json', 'w') as f:
        json.dump(weights, f, indent=4)
    print(f"Saved weights to {SAVED_MODELS_DIR / f'{args.name}_weights.json'}")
