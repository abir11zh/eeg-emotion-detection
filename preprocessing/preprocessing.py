"""
preprocessing/preprocessing.py
-------------------------------
Implements, step by step, the preprocessing pipeline described in the paper:

    "band-pass filtering to remove low-frequency drifts and high-frequency
    noise, independent component analysis (ICA) to separate and eliminate
    ocular and muscular artifacts, and signal normalization to reduce
    inter-subject variability. Additionally, the EEG signals are segmented
    into meaningful time windows"

Pipeline order (per trial, per subject):
    1. band_pass_filter      : 4-45 Hz Butterworth, zero-phase
    2. remove_ica_artifacts  : ICA decomposition + automatic EOG-component rejection
    3. remove_baseline       : subtract mean of the 3s pre-stimulus baseline
    4. zscore_normalize      : per-channel, per-trial z-score
    5. segment_windows       : sliding window segmentation (default 4s / 50% overlap)

Assumption flagged explicitly (paper does not give exact numbers):
    - band-pass range 4-45 Hz: matches both the DEAP-preprocessed release and
      the frequency bands (theta/alpha/beta/gamma) used for feature extraction.
    - window length 4s, 50% overlap: the most common choice in DEAP-based
      emotion-recognition literature (e.g. Koelstra et al. baselines, and the
      Chabachib/Bhardwaj DEAP repos this project reproduces).
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.signal import butter, filtfilt

from utils import BASELINE_SECONDS, PREPROCESSED_SAMPLING_RATE


@dataclass
class PreprocessConfig:
    fs: int = PREPROCESSED_SAMPLING_RATE
    band_low: float = 4.0
    band_high: float = 45.0
    filter_order: int = 4
    window_seconds: float = 4.0
    overlap: float = 0.5
    n_ica_components: int = 15
    baseline_seconds: int = BASELINE_SECONDS


# ----------------------------------------------------------------------------
# 1. Band-pass filtering
# ----------------------------------------------------------------------------
def band_pass_filter(eeg: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Zero-phase Butterworth band-pass, applied independently per channel.

    Parameters
    ----------
    eeg : ndarray (..., n_channels, n_samples)
    """
    nyq = cfg.fs / 2.0
    b, a = butter(cfg.filter_order, [cfg.band_low / nyq, cfg.band_high / nyq], btype="band")
    # filtfilt computes in float64 internally regardless of input dtype; cast
    # back down immediately so memory doesn't silently double here.
    return filtfilt(b, a, eeg, axis=-1).astype(np.float32)


# ----------------------------------------------------------------------------
# 2. ICA-based artifact removal
# ----------------------------------------------------------------------------
def remove_ica_artifacts(eeg: np.ndarray, cfg: PreprocessConfig, eog_corr_threshold: float = 0.6):
    """Run ICA per trial and zero out components highly correlated with the
    outer/frontal channels (Fp1/Fp2), which is the standard automatic proxy
    for ocular (EOG) artifacts when true EOG channels are not passed in.

    Because the DEAP-preprocessed release has *already* had EOG artifacts
    removed by the dataset authors, this step is close to a no-op there; it
    is included so the pipeline is complete and directly usable on raw data
    (e.g. loaded via `dataset.load_raw_subject`).

    Parameters
    ----------
    eeg : ndarray (n_trials, n_channels, n_samples)
    """
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning

    n_trials, n_channels, n_samples = eeg.shape
    cleaned = np.empty_like(eeg)
    n_comp = min(cfg.n_ica_components, n_channels)

    # Fp1/Fp2 are channel indices 0 and 16 in the DEAP 10-20 montage order.
    frontal_idx = [0, 16]

    for t in range(n_trials):
        X = eeg[t].T  # (n_samples, n_channels)
        ica = FastICA(
            n_components=n_comp,
            random_state=42,
            max_iter=1000,
            tol=1e-4,
            whiten="unit-variance",
            algorithm="parallel",
            fun="logcosh",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            try:
                sources = ica.fit_transform(X)  # (n_samples, n_comp)
            except Exception:
                cleaned[t] = eeg[t]
                continue

        frontal_ref = X[:, frontal_idx].mean(axis=1)
        corrs = np.array([
            np.abs(np.corrcoef(sources[:, c], frontal_ref)[0, 1]) for c in range(n_comp)
        ])
        bad_components = np.where(corrs > eog_corr_threshold)[0]

        sources_clean = sources.copy()
        sources_clean[:, bad_components] = 0.0
        X_clean = ica.inverse_transform(sources_clean)
        cleaned[t] = X_clean.T

    return cleaned


# ----------------------------------------------------------------------------
# 3. Baseline removal
# ----------------------------------------------------------------------------
def remove_baseline(eeg: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Subtract the mean of the pre-stimulus baseline from the stimulus signal.

    Parameters
    ----------
    eeg : ndarray (n_trials, n_channels, n_samples) where n_samples includes
        the leading `baseline_seconds` of pre-stimulus recording.
    """
    n_baseline = int(cfg.baseline_seconds * cfg.fs)
    baseline_mean = eeg[..., :n_baseline].mean(axis=-1, keepdims=True)
    stimulus = eeg[..., n_baseline:] - baseline_mean
    return stimulus


# ----------------------------------------------------------------------------
# 4. Normalization
# ----------------------------------------------------------------------------
def zscore_normalize(eeg: np.ndarray) -> np.ndarray:
    """Per-trial, per-channel z-score normalization (reduces inter-subject /
    inter-trial amplitude variability, as described in the paper)."""
    mean = eeg.mean(axis=-1, keepdims=True)
    std = eeg.std(axis=-1, keepdims=True) + 1e-8
    return (eeg - mean) / std


# ----------------------------------------------------------------------------
# 5. Windowing / segmentation
# ----------------------------------------------------------------------------
def segment_windows(eeg: np.ndarray, cfg: PreprocessConfig):
    """Slide a fixed-length window over each trial.

    Parameters
    ----------
    eeg : ndarray (n_trials, n_channels, n_samples)

    Returns
    -------
    windows : ndarray (n_trials, n_windows, n_channels, window_len)
    """
    window_len = int(cfg.window_seconds * cfg.fs)
    step = int(window_len * (1 - cfg.overlap))
    n_trials, n_channels, n_samples = eeg.shape

    starts = list(range(0, n_samples - window_len + 1, step))
    n_windows = len(starts)
    out = np.empty((n_trials, n_windows, n_channels, window_len), dtype=np.float32)
    for w, s in enumerate(starts):
        out[:, w] = eeg[:, :, s : s + window_len]
    return out


def run_pipeline(eeg: np.ndarray, cfg: PreprocessConfig = None, use_ica: bool = True):
    """Full preprocessing pipeline, steps 1 -> 5, run in order.

    Parameters
    ----------
    eeg : ndarray (n_trials, n_channels, n_samples), raw amplitude (uV)

    Returns
    -------
    windows : ndarray (n_trials, n_windows, n_channels, window_len)
    """
    cfg = cfg or PreprocessConfig()
    x = band_pass_filter(eeg, cfg)
    if use_ica:
        x = remove_ica_artifacts(x, cfg)
    x = remove_baseline(x, cfg)
    x = zscore_normalize(x)
    windows = segment_windows(x, cfg)
    return windows
