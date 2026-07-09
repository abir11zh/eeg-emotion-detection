"""
features/feature_extraction.py
--------------------------------
Feature engineering matching the paper's description:

    "transformed into the frequency domain using spectral analysis techniques
    such as Short-Time Fourier Transform (STFT) and Wavelet Transform to
    capture temporal dynamics"

We additionally extract Power Spectral Density (PSD) band powers, Differential
Entropy (DE, the de-facto standard feature for DEAP/SEED emotion recognition
in the wider literature that this paper's models are benchmarked against) and
Hjorth parameters, since "PSD, Differential Entropy, Hjorth, Wavelet" are the
canonical feature families for this task and the ablation in Section 4 of the
paper compares model *architectures* rather than feature families in isolation
-- we therefore extract all of them and let `training/train.py` select/ablate.

All functions operate on a single window: ndarray (n_channels, window_len).
Batch helpers operate on ndarray (n_trials, n_windows, n_channels, window_len).
"""
import numpy as np
from scipy.signal import welch, stft
import pywt  # type: ignore

from utils import BANDS, PREPROCESSED_SAMPLING_RATE

# NumPy >=2.0 renamed trapz -> trapezoid; keep this file working on either version.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------
# Power Spectral Density (band power)
# ----------------------------------------------------------------------------
def band_power(window: np.ndarray, fs: int = PREPROCESSED_SAMPLING_RATE, bands=BANDS):
    """Welch PSD band power per channel per band.

    Returns
    -------
    ndarray (n_channels, n_bands)
    """
    n_channels = window.shape[0]
    powers = np.zeros((n_channels, len(bands)))
    for c in range(n_channels):
        freqs, psd = welch(window[c], fs=fs, nperseg=min(256, window.shape[-1]))
        for b, (lo, hi) in enumerate(bands.values()):
            mask = (freqs >= lo) & (freqs <= hi)
            powers[c, b] = _trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
    return powers


# ----------------------------------------------------------------------------
# Differential Entropy
# ----------------------------------------------------------------------------
def differential_entropy(window: np.ndarray, fs: int = PREPROCESSED_SAMPLING_RATE, bands=BANDS):
    """DE per channel per band. For a signal that is approximately Gaussian
    within a narrow band, DE = 0.5 * log(2*pi*e*sigma^2), computed here from
    the band-limited variance obtained via Welch PSD integration (equivalent
    to the standard STFT/band-energy-based DE formulation used in SEED/DEAP
    emotion-recognition papers).

    Returns
    -------
    ndarray (n_channels, n_bands)
    """
    powers = band_power(window, fs, bands)  # proportional to band variance
    return 0.5 * np.log(2 * np.pi * np.e * (powers + 1e-12))


# ----------------------------------------------------------------------------
# Hjorth parameters
# ----------------------------------------------------------------------------
def hjorth_parameters(window: np.ndarray):
    """Activity, Mobility, Complexity per channel.

    Returns
    -------
    ndarray (n_channels, 3)
    """
    n_channels = window.shape[0]
    out = np.zeros((n_channels, 3))
    for c in range(n_channels):
        x = window[c]
        dx = np.diff(x)
        ddx = np.diff(dx)

        activity = np.var(x)
        mobility = np.sqrt(np.var(dx) / (activity + 1e-12))
        complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (mobility + 1e-12)

        out[c] = [activity, mobility, complexity]
    return out


# ----------------------------------------------------------------------------
# Discrete Wavelet Transform energy features
# ----------------------------------------------------------------------------
def wavelet_features(window: np.ndarray, wavelet: str = "db4", level: int = 4):
    """DWT sub-band energy per channel. db4 / 4 levels on 128 Hz data splits
    roughly into gamma/beta/alpha/theta-delta bands, mirroring the Wavelet
    Transform branch described in the paper.

    Returns
    -------
    ndarray (n_channels, level + 1)
    """
    n_channels = window.shape[0]
    out = np.zeros((n_channels, level + 1))
    for c in range(n_channels):
        coeffs = pywt.wavedec(window[c], wavelet=wavelet, level=level)
        for i, coeff in enumerate(coeffs):
            out[c, i] = np.sum(coeff ** 2)
    return out


# ----------------------------------------------------------------------------
# Short-Time Fourier Transform features (spectrogram summary)
# ----------------------------------------------------------------------------
def stft_features(window: np.ndarray, fs: int = PREPROCESSED_SAMPLING_RATE, nperseg: int = 64):
    """Mean spectrogram magnitude per band per channel (a compact summary of
    the STFT branch mentioned in the paper; the full time-frequency map is
    also returned for use as CNN input, see `stft_image`).

    Returns
    -------
    ndarray (n_channels, n_bands)
    """
    n_channels = window.shape[0]
    out = np.zeros((n_channels, len(BANDS)))
    for c in range(n_channels):
        f, t, Zxx = stft(window[c], fs=fs, nperseg=min(nperseg, window.shape[-1]))
        mag = np.abs(Zxx)
        for b, (lo, hi) in enumerate(BANDS.values()):
            mask = (f >= lo) & (f <= hi)
            out[c, b] = mag[mask].mean() if mask.any() else 0.0
    return out


def stft_image(window: np.ndarray, fs: int = PREPROCESSED_SAMPLING_RATE, nperseg: int = 64):
    """Full multi-channel spectrogram, stacked as an image for CNN / hybrid
    AlexNet-DenseNet style pipelines (see models/hybrid_alexnet_densenet.py).

    Returns
    -------
    ndarray (n_channels, n_freq_bins, n_time_bins)
    """
    specs = []
    for c in range(window.shape[0]):
        f, t, Zxx = stft(window[c], fs=fs, nperseg=min(nperseg, window.shape[-1]))
        specs.append(np.abs(Zxx))
    return np.stack(specs, axis=0)


# ----------------------------------------------------------------------------
# Combined feature vector
# ----------------------------------------------------------------------------
def extract_features(window: np.ndarray, fs: int = PREPROCESSED_SAMPLING_RATE) -> np.ndarray:
    """Concatenate PSD + DE + Hjorth + Wavelet + STFT summary into a single
    flat feature vector for classical ML models (KNN/SVM/DT/RF).

    Returns
    -------
    ndarray, shape (n_channels * (4 + 4 + 3 + 5 + 4),) for 32 channels ~= 32*20 = 640-d
    """
    feats = [
        band_power(window, fs),
        differential_entropy(window, fs),
        hjorth_parameters(window),
        wavelet_features(window),
        stft_features(window, fs),
    ]
    return np.concatenate([f.reshape(-1) for f in feats])


def extract_features_dataset(windows: np.ndarray, fs: int = PREPROCESSED_SAMPLING_RATE):
    """Vectorized wrapper over `extract_features` for a full windows array.

    Parameters
    ----------
    windows : ndarray (n_trials, n_windows, n_channels, window_len)

    Returns
    -------
    ndarray (n_trials * n_windows, n_features)
    """
    n_trials, n_windows = windows.shape[:2]
    flat = windows.reshape(n_trials * n_windows, *windows.shape[2:])
    feats = np.stack([extract_features(w, fs) for w in flat], axis=0)
    return feats
