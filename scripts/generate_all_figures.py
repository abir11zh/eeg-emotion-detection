"""
scripts/generate_all_figures.py
--------------------------------
Small CLI to (re)run model evaluation loops and produce all figures
using `evaluation.evaluate` utilities. By default it runs classical models
on the full feature set and saves confusion matrices, ROC, feature
importance and t-SNE plots under `figures/`.

Usage (from project root):
  python scripts/generate_all_figures.py --data-dir data --models random_forest,knn,svm
  python scripts/generate_all_figures.py --quick --models random_forest
"""
import os
import sys
import argparse
import numpy as np

# Ensure non-interactive backend to avoid IPython / GUI hang when importing seaborn
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed
from dataset import iter_subjects, binarize_labels
from preprocessing.preprocessing import PreprocessConfig, run_pipeline
from features.feature_extraction import extract_features_dataset
from models.models import CLASSICAL_MODELS, DEEP_MODELS
from evaluation.evaluate import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_tsne,
    plot_training_curves,
)


def extract_all_features(data_dir, subjects, cfg, quick=False):
    feats_list, y_list, groups_list = [], [], []
    max_windows_per_subject = 200 if quick else None
    for eeg, labels, sid in iter_subjects(data_dir, subject_ids=list(range(1, subjects + 1))):
        windows = run_pipeline(eeg, cfg, use_ica=True)
        n_windows = windows.shape[1]
        feats = extract_features_dataset(windows)
        del windows
        label_dict = binarize_labels(labels)
        y = np.repeat(label_dict["valence"], n_windows)

        if max_windows_per_subject is not None and feats.shape[0] > max_windows_per_subject:
            feats = feats[:max_windows_per_subject]
            y = y[:max_windows_per_subject]

        feats_list.append(feats)
        y_list.append(y)
        groups_list.append(np.full(feats.shape[0], sid, dtype=int))
        print(f"  subject {sid:02d}: features {feats.shape}")

    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    return X, y, groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--models", default=",".join(CLASSICAL_MODELS.keys()),
                   help="comma-separated list of models to evaluate (classical + deep names)")
    p.add_argument("--subjects", type=int, default=32)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    set_seed(42)
    cfg = PreprocessConfig()

    if args.quick:
        args.subjects = min(args.subjects, 4)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Ensure figures dir exists (evaluate._savefig also creates it, but keep local copy)
    os.makedirs(args.out_dir, exist_ok=True)

    # Extract features once (classical models use features)
    print("Extracting features for classical models...")
    X, y, groups = extract_all_features(args.data_dir, args.subjects, cfg, quick=args.quick)
    print("Features shape:", X.shape, "Labels:", y.shape)

    # Use simple StratifiedKFold (10 folds) consistent with training/train.py default
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    n_splits = 2 if args.quick else 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name in models:
        print(f"Processing model: {model_name}")
        if model_name in CLASSICAL_MODELS:
            fold = 0
            for train_idx, test_idx in skf.split(X, y):
                print(f"  fold {fold+1}/{n_splits} - training {model_name}")
                clf = CLASSICAL_MODELS[model_name]()
                clf.fit(X[train_idx], y[train_idx])
                preds = clf.predict(X[test_idx])

                # filenames include model and fold
                base = f"{model_name}_valence_fold{fold}"
                plot_confusion_matrix(y[test_idx], preds, class_names=["low", "high"], title=f"{model_name} - confusion (fold {fold})", name=f"{base}_confusion.png")
                # roc: need probs or decision function
                try:
                    probs = clf.predict_proba(X[test_idx])
                except Exception:
                    # fallback: use decision_function if available
                    try:
                        scores = clf.decision_function(X[test_idx])
                        plot_roc_curve(y[test_idx], scores, class_names=["low", "high"], title=f"{model_name} ROC (fold {fold})", name=f"{base}_roc.png")
                    except Exception:
                        print(f"No probability/score available for {model_name}; skipping ROC")
                else:
                    plot_roc_curve(y[test_idx], probs, class_names=["low", "high"], title=f"{model_name} ROC (fold {fold})", name=f"{base}_roc.png")

                # feature importance (per-fold)
                try:
                    plot_feature_importance(clf, feature_names=None, X=X[test_idx], y=y[test_idx], title=f"{model_name} feature importance (fold {fold})", name=f"{base}_featimp.png")
                except Exception as e:
                    print(f"Feature importance failed for {model_name} fold {fold}: {e}")

                # t-SNE on test set features (small sample for speed)
                sample_idx = np.random.RandomState(42 + fold).choice(len(test_idx), size=min(2000, len(test_idx)), replace=False)
                try:
                    plot_tsne(X[test_idx][sample_idx], y[test_idx][sample_idx], class_names=["low", "high"], title=f"{model_name} t-SNE (fold {fold})", name=f"{base}_tsne.png")
                except Exception as e:
                    print(f"t-SNE failed for {model_name} fold {fold}: {e}")

                fold += 1

        elif model_name in DEEP_MODELS:
            print(f"Deep model {model_name} requested: running deep-model evaluation is expensive. Consider using --quick or reduce subjects/epochs in `training/train.py`. Skipping by default.")
        else:
            print(f"Unknown model: {model_name}; skipping")

    print("Finished generating figures.")


if __name__ == "__main__":
    main()
