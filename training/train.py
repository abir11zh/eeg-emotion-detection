"""
training/train.py
-------------------
CLI entry point that reproduces the experimental protocol:

    1. Load DEAP preprocessed subjects (dataset.py)
    2. Run the preprocessing pipeline (preprocessing/preprocessing.py)
    3. Extract features (features/feature_extraction.py) for classical models,
       or use raw windows directly for deep models
    4. Binarize valence/arousal labels (dataset.binarize_labels)
    5. Cross-validate:
         --cv subject_dependent : 10-fold CV pooling all subjects' windows (default,
              matches the "10-fold subject-dependent CV" protocol used across DEAP papers)
         --cv loso              : Leave-One-Subject-Out (subject-independent evaluation)
    6. Train the requested model, save metrics + model checkpoint to results/

Usage
-----
    python training/train.py --data-dir data/ --model random_forest --task valence --cv subject_dependent
    python training/train.py --data-dir data/ --model bilstm --task arousal --cv loso --epochs 30
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import set_seed, get_device, save_json
from dataset import iter_subjects, binarize_labels
from preprocessing.preprocessing import PreprocessConfig, run_pipeline
from features.feature_extraction import extract_features_dataset
from models.models import CLASSICAL_MODELS, DEEP_MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", required=True,
                    choices=list(CLASSICAL_MODELS) + list(DEEP_MODELS))
    p.add_argument("--task", default="valence", choices=["valence", "arousal", "quadrant"])
    p.add_argument("--cv", default="subject_dependent", choices=["subject_dependent", "loso"])
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-ica", action="store_true", help="skip ICA step (faster; DEAP-preprocessed data is already EOG-cleaned)")
    p.add_argument("--subjects", type=int, default=32, help="number of subjects to load (use fewer for a quick smoke test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--quick", action="store_true", help="run a fast smoke test with fewer subjects/folds/epochs")
    return p.parse_args()


def extract_all_features(args, cfg: PreprocessConfig):
    """Memory-safe path for classical models (KNN/SVM/DT/RF): process ONE
    subject at a time (load -> preprocess -> window -> extract features),
    keeping only the small resulting feature matrix and discarding that
    subject's raw EEG / windows before moving to the next subject. Peak
    memory is therefore bounded by a single subject, not all 32 at once.
    """
    feats_list, y_list, groups_list = [], [], []
    subject_ids = list(range(1, args.subjects + 1))

    for eeg, labels, sid in iter_subjects(args.data_dir, subject_ids=subject_ids):
        windows = run_pipeline(eeg, cfg, use_ica=not args.no_ica)  # (40, n_windows, 32, wlen)
        n_windows = windows.shape[1]
        feats = extract_features_dataset(windows)  # (40*n_windows, n_features) -- small
        del windows  # free before next subject's raw EEG is loaded

        label_dict = binarize_labels(labels)
        y = np.repeat(label_dict[args.task], n_windows)

        feats_list.append(feats)
        y_list.append(y)
        groups_list.append(np.full(feats.shape[0], sid, dtype=int))
        print(f"  subject {sid:02d}: features {feats.shape}")

    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    return X, y, groups


def collect_all_windows(args, cfg: PreprocessConfig):
    """Memory-conscious path for deep models: process one subject at a time
    through preprocessing, then keep only that subject's (float32) windows.
    This still needs all windows resident for the CV loop below (deep models
    train on raw windows, not compact features), so it uses more RAM than
    `extract_all_features` -- if you run out of memory here, lower
    `--subjects` (e.g. 10-15) rather than the full 32.
    """
    windows_list, y_list, groups_list = [], [], []
    subject_ids = list(range(1, args.subjects + 1))

    for eeg, labels, sid in iter_subjects(args.data_dir, subject_ids=subject_ids):
        windows = run_pipeline(eeg, cfg, use_ica=not args.no_ica)  # (40, n_windows, 32, wlen)
        n_windows = windows.shape[1]
        flat = windows.reshape(-1, *windows.shape[2:])  # (40*n_windows, 32, wlen)
        del windows

        label_dict = binarize_labels(labels)
        y = np.repeat(label_dict[args.task], n_windows)

        windows_list.append(flat)
        y_list.append(y)
        groups_list.append(np.full(y.shape[0], sid, dtype=int))
        print(f"  subject {sid:02d}: windows {flat.shape}")

    flat_windows = np.concatenate(windows_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    return flat_windows, y, groups


def train_classical(model_name, X, y, groups, args):
    splitter = (
        LeaveOneGroupOut()
        if args.cv == "loso"
        else StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    )
    split_iter = splitter.split(X, y, groups) if args.cv == "loso" else splitter.split(X, y)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(split_iter):
        clf = CLASSICAL_MODELS[model_name]()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        f1 = f1_score(y[test_idx], preds, average="macro")
        fold_metrics.append({"fold": fold, "accuracy": acc, "f1_macro": f1, "n_test": len(test_idx)})
        print(f"[{model_name}] fold {fold}: acc={acc:.4f} f1={f1:.4f}")

    return fold_metrics


def train_deep(model_name, flat_windows, y, groups, args):
    """`flat_windows`: ndarray (n_samples, n_channels, window_len), already
    flattened across trials/windows (see `collect_all_windows`)."""
    print(f"Starting deep training with {flat_windows.shape[0]} samples, {args.epochs} epochs, {args.n_folds} folds")
    device = get_device()
    n_channels = flat_windows.shape[1]
    n_classes = len(np.unique(y))

    splitter = (
        LeaveOneGroupOut()
        if args.cv == "loso"
        else StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    )
    split_iter = (
        splitter.split(flat_windows, y, groups)
        if args.cv == "loso"
        else splitter.split(flat_windows, y)
    )

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(split_iter):
        model_cls = DEEP_MODELS[model_name]
        model = model_cls(n_channels=n_channels, n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        X_train = torch.tensor(flat_windows[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_test = torch.tensor(flat_windows[test_idx], dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

        history = {"loss": [], "train_acc": []}
        model.train()
        for epoch in range(args.epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total += xb.size(0)
            history["loss"].append(epoch_loss / total)
            history["train_acc"].append(correct / total)

        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(device))
            preds = logits.argmax(1).cpu().numpy()
        acc = accuracy_score(y[test_idx], preds)
        f1 = f1_score(y[test_idx], preds, average="macro")
        fold_metrics.append({"fold": fold, "accuracy": acc, "f1_macro": f1, "history": history})
        print(f"[{model_name}] fold {fold}: acc={acc:.4f} f1={f1:.4f}")

        # only run a single fold for LOSO-32 by default unless the user wants the full sweep;
        # comment this break out to run the exhaustive 32-fold LOSO.
        if args.cv == "loso" and fold >= args.n_folds - 1:
            break

    return fold_metrics


def main():
    args = parse_args()
    if args.quick:
        args.subjects = min(args.subjects, 4)
        args.n_folds = min(args.n_folds, 2)
        args.epochs = min(args.epochs, 3)
        print(f"Quick mode enabled: subjects={args.subjects}, folds={args.n_folds}, epochs={args.epochs}")

    set_seed(args.seed)
    cfg = PreprocessConfig()

    t0 = time.time()
    if args.model in CLASSICAL_MODELS:
        X, y, groups = extract_all_features(args, cfg)
        print(f"Loaded + preprocessed + extracted features: {X.shape} in {time.time()-t0:.1f}s")
        metrics = train_classical(args.model, X, y, groups, args)
    else:
        flat_windows, y, groups = collect_all_windows(args, cfg)
        print(f"Loaded + preprocessed windows: {flat_windows.shape} in {time.time()-t0:.1f}s")
        if flat_windows.shape[0] > 20000 or args.n_folds > 3 or args.epochs > 5:
            print("Deep-model training can take a long time on the full 32-subject DEAP set. Consider using --quick or lowering --subjects/--n-folds/--epochs for testing.")
        metrics = train_deep(args.model, flat_windows, y, groups, args)

    accs = [m["accuracy"] for m in metrics]
    summary = {
        "model": args.model,
        "task": args.task,
        "cv": args.cv,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "folds": metrics,
    }
    out_path = os.path.join(args.out_dir, f"{args.model}_{args.task}_{args.cv}.json")
    save_json(summary, out_path)
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
