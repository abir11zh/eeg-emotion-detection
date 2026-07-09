"""
evaluation/evaluate.py
------------------------
All the diagnostic/analysis visualizations requested for scientific reporting:
    - confusion matrix
    - ROC curve(s) (one-vs-rest for multi-class)
    - training curves (loss / accuracy vs. epoch, for deep models)
    - t-SNE visualization of the feature / embedding space
    - feature importance (native for RF/DT; permutation importance otherwise)
    - SHAP explanations (TreeExplainer for RF/DT; KernelExplainer fallback)

Every function saves a figure to `figures/` and returns the matplotlib Figure
so it can also be displayed inline in a notebook.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {path}")
    return path


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", name="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    else:
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        if class_names is not None:
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    _savefig(fig, name)
    return fig


def plot_roc_curve(y_true, y_score, class_names=None, title="ROC Curve", name="roc_curve.png"):
    """y_score: (n_samples,) for binary, or (n_samples, n_classes) for multi-class probabilities."""
    fig, ax = plt.subplots(figsize=(5, 4))
    y_score = np.asarray(y_score)

    if y_score.ndim == 1 or y_score.shape[1] == 2:
        scores = y_score if y_score.ndim == 1 else y_score[:, 1]
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        n_classes = y_score.shape[1]
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve((np.array(y_true) == c).astype(int), y_score[:, c])
            label = class_names[c] if class_names else f"class {c}"
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    _savefig(fig, name)
    return fig


def plot_training_curves(history: dict, title="Training Curves", name="training_curves.png"):
    """history: dict with keys like 'loss', 'train_acc' (as produced by training/train.py)."""
    fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    if len(history) == 1:
        axes = [axes]
    for ax, (key, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs epoch")
    fig.suptitle(title)
    _savefig(fig, name)
    return fig


def plot_tsne(features: np.ndarray, labels: np.ndarray, class_names=None, title="t-SNE", name="tsne.png", perplexity=30):
    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(features) // 4)), random_state=42, init="pca")
    embedding = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    if class_names:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, class_names, title="class", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    _savefig(fig, name)
    return fig


def plot_feature_importance(model, feature_names=None, X=None, y=None, title="Feature Importance", name="feature_importance.png", top_k=20):
    """Uses the model's native `.feature_importances_` when available (RF/DT);
    otherwise falls back to permutation importance (requires X, y)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        if X is None or y is None:
            raise ValueError("X and y are required for permutation importance on models without .feature_importances_")
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean

    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] if feature_names else f"f{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(6, max(4, top_k * 0.25)))
    ax.barh(range(len(idx)), importances[idx][::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    _savefig(fig, name)
    return fig


def plot_shap_summary(model, X, feature_names=None, model_type="tree", max_display=20, name="shap_summary.png"):
    """SHAP explanations, as used in the paper for interpretability.

    model_type: 'tree' for RF/DT (fast, exact TreeExplainer), 'kernel' for
    any other model (SVM/KNN/MLP; slower, uses a background sample).
    """
    import shap

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        background = shap.sample(X, min(100, len(X)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X[: min(200, len(X))], nsamples=100)
        X = X[: min(200, len(X))]

    fig = plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display, show=False)
    _savefig(fig, name)
    return fig


def compare_with_paper(reproduced: dict, paper_reported: dict, name="paper_comparison.png"):
    """Bar chart comparing our reproduced accuracy against the numbers
    reported in Mouazen et al. 2025 (Sensors 25(6):1827) for the same task/model.

    reproduced / paper_reported: {"model_name": accuracy_percent, ...}
    """
    labels = list(reproduced.keys())
    repro_vals = [reproduced[k] for k in labels]
    paper_vals = [paper_reported.get(k, np.nan) for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(x - width / 2, repro_vals, width, label="Reproduced")
    ax.bar(x + width / 2, paper_vals, width, label="Paper-reported")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Reproduced vs. paper-reported accuracy")
    ax.legend()
    _savefig(fig, name)
    return fig
