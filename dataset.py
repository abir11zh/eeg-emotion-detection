"""
dataset.py
----------
Loading utilities for the DEAP dataset.

The paper (and the Chabachib/EEG-Based-Emotion-Detection repo it documents)
works from the *preprocessed* DEAP release (`data_preprocessed_python/s01.dat`
... `s32.dat`, the same files distributed in the linked Kaggle mirror
"manh123df/deap-dataset"). Each file is a Python pickle:

    {
      'data':   ndarray (40 trials, 40 channels, 8064 samples)
      'labels': ndarray (40 trials, 4)  -> [valence, arousal, dominance, liking], 1-9 scale
    }

Channels 0-31 are EEG (10-20 montage, see utils.DEAP_EEG_CHANNELS), channels
32-39 are peripheral (EOG x2, EMG x2, GSR, Respiration, Plethysmograph, Temp).
The data is already downsampled to 128 Hz, band-pass filtered 4.0-45.0 Hz and
had EOG artifacts removed by the DEAP authors themselves. Because of that,
`preprocessing.py` re-applies (rather than skips) band-pass filtering and ICA
so the *full* pipeline described in the paper is exercised end-to-end; on
already-clean data this step is idempotent (near no-op) but is kept for
faithfulness to the described methodology and for use with raw .bdf files.

If you instead only have raw BioSemi `.bdf` files, use `load_raw_subject`,
which relies on MNE-Python for I/O and the 10-20 montage.
"""
import os
import pickle
import numpy as np
import mne

from utils import N_EEG_CHANNELS, N_TRIALS_PER_SUBJECT


def _subject_path(data_dir: str, subject_id: int) -> str:
    fname = f"s{subject_id:02d}.dat"
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the DEAP 'data_preprocessed_python' "
            f"archive (see README.md -> Dataset preparation) and place s01.dat.."
            f"s32.dat under {data_dir}/"
        )
    return path


def load_preprocessed_subject(data_dir: str, subject_id: int):
    """Load one subject's preprocessed DEAP file.

    Returns
    -------
    eeg : ndarray, shape (40, 32, 8064)
        EEG-only channels, raw amplitude in microvolts.
    labels : ndarray, shape (40, 4)
        [valence, arousal, dominance, liking] in [1, 9].
    """
    path = _subject_path(data_dir, subject_id)
    with open(path, "rb") as f:
        # DEAP .dat files were pickled under Python 2 -> latin1 encoding required.
        obj = pickle.load(f, encoding="latin1")
    data = obj["data"]      # (40, 40, 8064)
    labels = obj["labels"]  # (40, 4)
    eeg = data[:, :N_EEG_CHANNELS, :]
    assert eeg.shape == (N_TRIALS_PER_SUBJECT, N_EEG_CHANNELS, eeg.shape[-1])
    # float32 (not float64): halves memory everywhere downstream (filtering,
    # ICA, windowing all copy the array at least once). DEAP's amplitude
    # precision does not need float64.
    return eeg.astype(np.float32), labels.astype(np.float32)


def iter_subjects(data_dir: str, subject_ids=None):
    """Generator: yields (eeg, labels, subject_id) one subject at a time.

    Use this instead of `load_all_subjects` whenever you can process each
    subject immediately (e.g. preprocess + extract features), since it never
    holds more than one subject's raw EEG in memory at once. This is what
    `training/train.py` now uses by default -- see `get_windows_and_labels`.
    """
    if subject_ids is None:
        subject_ids = list(range(1, 33))
    for sid in subject_ids:
        eeg, labels = load_preprocessed_subject(data_dir, sid)
        yield eeg, labels, sid


def load_all_subjects(data_dir: str, subject_ids=None):
    """Load and concatenate EEG + labels for a list of subjects.

    WARNING: this holds *all* requested subjects' raw EEG in memory at once
    (float32: ~1.2 GB for all 32 subjects at 8064 samples/trial). On machines
    with limited RAM, prefer `iter_subjects` + per-subject processing instead
    (this is what `training/train.py` does by default). Kept for convenience
    / small subject counts / notebook exploration.

    Returns
    -------
    eeg : ndarray (n_subjects*40, 32, 8064)
    labels : ndarray (n_subjects*40, 4)
    subject_idx : ndarray (n_subjects*40,) subject id for each trial (for LOSO CV / grouping)
    """
    eeg_all, labels_all, subj_all = [], [], []
    for eeg, labels, sid in iter_subjects(data_dir, subject_ids):
        eeg_all.append(eeg)
        labels_all.append(labels)
        subj_all.append(np.full(eeg.shape[0], sid, dtype=int))

    return (
        np.concatenate(eeg_all, axis=0),
        np.concatenate(labels_all, axis=0),
        np.concatenate(subj_all, axis=0),
    )


def binarize_labels(labels: np.ndarray, threshold: float = 5.0):
    """Turn continuous 1-9 valence/arousal/dominance ratings into class labels.

    Two supervised tasks are reproduced, matching common practice in DEAP
    papers (including the source repo):
      - binary valence:  low (<=5) vs high (>5)
      - binary arousal:  low (<=5) vs high (>5)
      - 4-class quadrant: combination of the two -> {LVLA, LVHA, HVLA, HVHA}

    Parameters
    ----------
    labels : ndarray (n_trials, 4) columns = [valence, arousal, dominance, liking]
    """
    valence, arousal = labels[:, 0], labels[:, 1]
    val_bin = (valence > threshold).astype(int)
    aro_bin = (arousal > threshold).astype(int)
    quadrant = val_bin * 2 + aro_bin  # 0:LVLA 1:LVHA 2:HVLA 3:HVHA
    return {
        "valence": val_bin,
        "arousal": aro_bin,
        "quadrant": quadrant,
    }


def load_raw_subject(bdf_path: str, montage: str = "biosemi32"):
    """Load a *raw* BioSemi .bdf recording with MNE, for use when working
    from the original (unprocessed) DEAP distribution rather than the
    preprocessed .dat mirror. Requires `pip install mne`.
    """
    import mne

    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    raw.set_montage(montage, on_missing="warn")
    return raw
