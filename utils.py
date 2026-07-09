"""
utils.py
--------
Shared utilities: reproducibility (seed fixing), device selection, DEAP channel
metadata, and small I/O helpers used across the project.

Reproducing: Mouazen et al. 2025, "Enhancing EEG-Based Emotion Detection with
Hybrid Models: Insights from DEAP Dataset Applications", Sensors 25(6):1827.
"""
import os
import json
import random
import numpy as np

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix every source of randomness we know about.

    Called at the top of every script (train.py, evaluate.py, notebooks) so
    that a given run is bit-for-bit comparable across executions, and so that
    reported numbers can be independently checked.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic (slower) cuDNN kernels -> exact reproducibility over speed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """Return a torch.device, preferring CUDA if available."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# DEAP dataset metadata
# ----------------------------------------------------------------------------
# Channel order for the *preprocessed* DEAP .dat files (data_preprocessed_python).
# Channels 0-31 are EEG (10-20 system, Biosemi montage), 32-39 are peripheral
# (EOG/EMG/GSR/Resp/Temp/Plethysmograph) and are dropped for EEG-only pipelines.
DEAP_EEG_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
    "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
    "PO4", "O2",
]

N_EEG_CHANNELS = 32
N_TRIALS_PER_SUBJECT = 40
N_SUBJECTS = 32
ORIGINAL_SAMPLING_RATE = 512  # raw BioSemi acquisition rate
PREPROCESSED_SAMPLING_RATE = 128  # rate of the distributed data_preprocessed_python files
BASELINE_SECONDS = 3  # first 3s of every trial is a pre-stimulus baseline
TRIAL_SECONDS = 63  # 3s baseline + 60s stimulus
LABEL_NAMES = ["valence", "arousal", "dominance", "liking"]

# EEG frequency bands (Hz) used throughout feature_extraction.py
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)
