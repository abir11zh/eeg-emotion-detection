import os
import pickle
import warnings
import unittest

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from preprocessing.preprocessing import PreprocessConfig, run_pipeline


class PreprocessingRegressionTest(unittest.TestCase):
    def test_run_pipeline_ica_does_not_emit_convergence_warning(self):
        path = os.path.join("data", "s01.dat")
        with open(path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        eeg = obj["data"][:, :32, :].astype(np.float32)
        cfg = PreprocessConfig()

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            windows = run_pipeline(eeg, cfg, use_ica=True)

        self.assertEqual(windows.shape[0], eeg.shape[0])
        self.assertEqual(windows.shape[2], eeg.shape[1])
        self.assertEqual(windows.shape[3], int(cfg.window_seconds * cfg.fs))


if __name__ == "__main__":
    unittest.main()
