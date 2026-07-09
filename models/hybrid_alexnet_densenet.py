"""
models/hybrid_alexnet_densenet.py
-----------------------------------
Reproduces the specific "hybrid deep learning model" the paper highlights as
its strongest comparison point:

    "a hybrid deep learning model that extracts features using AlexNet and
    DenseNet models, followed by feature fusion and dimensionality reduction
    via Principal Component Analysis (PCA). The reduced features are then
    classified using a multi-class Support Vector Machine (SVM)"
    -> reported 95.54% (valence) / 97.26% (arousal) on DEAP.

AlexNet/DenseNet are 2D image-classification CNNs pretrained on ImageNet, so
EEG windows must first be converted into image-like inputs. We use the
multi-channel STFT spectrogram (`features.feature_extraction.stft_image`),
average/stack it into a 3-channel "image" (replicating to 3 channels the way
grayscale-to-RGB adaptation is standardly done for pretrained CNNs), and feed
it through frozen AlexNet/DenseNet backbones as fixed feature extractors.

ASSUMPTIONS (paper does not specify): exact image construction from EEG
(we use per-channel-averaged STFT log-magnitude resized to 224x224),
PCA target dimensionality (we default to 128, tunable), and SVM
hyperparameters (RBF, C=10, matching models.build_svm defaults).
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class FrozenBackboneExtractor(nn.Module):
    """Wraps a torchvision backbone (AlexNet or DenseNet), pretrained on
    ImageNet, as a frozen feature extractor returning a flat pooled vector."""

    def __init__(self, name: str = "alexnet"):
        super().__init__()
        if name == "alexnet":
            net = tvm.alexnet(weights=tvm.AlexNet_Weights.IMAGENET1K_V1)
            self.backbone = net.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 256
        elif name == "densenet":
            net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
            self.backbone = net.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 1024
        else:
            raise ValueError(f"Unknown backbone: {name}")

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x):
        feats = self.backbone(x)
        feats = self.pool(feats)
        return torch.flatten(feats, 1)


_IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def eeg_window_to_image(spec: np.ndarray) -> torch.Tensor:
    """Convert a multi-channel STFT spectrogram (n_channels, n_freq, n_time)
    into a 3-channel pseudo-RGB image tensor suitable for ImageNet backbones.
    """
    avg = spec.mean(axis=0)  # average across EEG channels -> (n_freq, n_time)
    avg = np.log1p(avg)
    avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
    img = np.stack([avg, avg, avg], axis=0)  # replicate to 3 "RGB" channels
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,3,H,W)
    return _IMAGE_TRANSFORM(tensor)


class HybridAlexNetDenseNetPCASVM:
    """End-to-end hybrid pipeline: AlexNet + DenseNet feature fusion -> PCA -> SVM."""

    def __init__(self, pca_components: int = 128, svm_kwargs=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alexnet = FrozenBackboneExtractor("alexnet").to(self.device)
        self.densenet = FrozenBackboneExtractor("densenet").to(self.device)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=42)
        svm_kwargs = svm_kwargs or dict(kernel="rbf", C=10.0, gamma="scale")
        self.svm = SVC(probability=True, random_state=42, **svm_kwargs)

    @torch.no_grad()
    def _fuse_features(self, spectrograms: np.ndarray) -> np.ndarray:
        """spectrograms: (n_samples, n_channels, n_freq, n_time) -> fused feature matrix."""
        feats = []
        for spec in spectrograms:
            img = eeg_window_to_image(spec).to(self.device)
            f1 = self.alexnet(img).cpu().numpy().reshape(-1)
            f2 = self.densenet(img).cpu().numpy().reshape(-1)
            feats.append(np.concatenate([f1, f2]))  # feature fusion (concatenation)
        return np.stack(feats, axis=0)

    def fit(self, spectrograms: np.ndarray, y: np.ndarray):
        fused = self._fuse_features(spectrograms)
        fused = self.scaler.fit_transform(fused)
        reduced = self.pca.fit_transform(fused)
        self.svm.fit(reduced, y)
        return self

    def predict(self, spectrograms: np.ndarray):
        fused = self._fuse_features(spectrograms)
        fused = self.scaler.transform(fused)
        reduced = self.pca.transform(fused)
        return self.svm.predict(reduced)

    def predict_proba(self, spectrograms: np.ndarray):
        fused = self._fuse_features(spectrograms)
        fused = self.scaler.transform(fused)
        reduced = self.pca.transform(fused)
        return self.svm.predict_proba(reduced)
