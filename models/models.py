"""
models/models.py
-----------------
Every model family named in the paper:

    "K-Nearest Neighbors (KNN), Support Vector Machines (SVMs), Decision Tree
    (DT), Random Forest (RF), Bidirectional Long Short-Term Memory (BiLSTM),
    Gated Recurrent Units (GRUs), Convolutional Neural Networks (CNNs),
    autoencoders, and transformers"

Classical models (KNN/SVM/DT/RF) operate on the flat feature vectors from
`features/feature_extraction.py`. Deep models operate on raw (preprocessed)
window tensors of shape (n_channels, window_len) and are implemented in
PyTorch. Hyperparameters are not given in the paper; the values below are
reasonable, literature-typical defaults (flagged as ASSUMPTION) meant to be
tuned via the `--grid-search` flag in `training/train.py`.
"""
import torch
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------------------------------------
# Classical ML (operate on flat feature vectors)
# ----------------------------------------------------------------------------
def build_knn(n_neighbors: int = 5):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")


def build_svm(kernel: str = "rbf", C: float = 10.0, gamma: str = "scale"):
    # ASSUMPTION: kernel/C/gamma not specified in the paper; RBF-SVM with
    # moderate regularization is the standard choice across DEAP literature.
    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)


def build_decision_tree(max_depth: int = 12):
    return DecisionTreeClassifier(max_depth=max_depth, random_state=42)


def build_random_forest(n_estimators: int = 300, max_depth: int = None):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
    )


CLASSICAL_MODELS = {
    "knn": build_knn,
    "svm": build_svm,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
}


# ----------------------------------------------------------------------------
# Deep learning models (PyTorch). Input: (batch, n_channels, window_len)
# ----------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM over the time axis, channels treated as the input
    feature dimension at each time step."""

    def __init__(self, n_channels=32, hidden_size=128, num_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x):  # x: (batch, n_channels, window_len)
        x = x.permute(0, 2, 1)  # -> (batch, time, channels)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)  # temporal average pooling
        return self.classifier(pooled)


class GRUClassifier(nn.Module):
    def __init__(self, n_channels=32, hidden_size=128, num_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


class EEGCNNClassifier(nn.Module):
    """1D-CNN over the temporal axis, with channels as input feature maps."""

    def __init__(self, n_channels=32, n_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):  # x: (batch, n_channels, window_len)
        return self.classifier(self.features(x))


class EEGAutoencoder(nn.Module):
    """Convolutional autoencoder for unsupervised representation learning.
    Train unsupervised (MSE reconstruction), then reuse `.encode()` output as
    a feature extractor for a downstream classifier (e.g. SVM / linear head),
    matching the "autoencoders" branch named in the paper."""

    def __init__(self, n_channels=32, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 32 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, n_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder_fc(z).view(z.size(0), 32, 8)
        recon = self.decoder(h)
        # Crop/pad to match input length (stride-2 convs can shift length by a few samples).
        if recon.shape[-1] != x.shape[-1]:
            recon = nn.functional.interpolate(recon, size=x.shape[-1], mode="linear", align_corners=False)
        return recon, z


class AutoencoderClassifierHead(nn.Module):
    """Linear/MLP classifier trained on top of a *frozen* pretrained encoder."""

    def __init__(self, autoencoder: EEGAutoencoder, latent_dim=64, n_classes=2, dropout=0.3):
        super().__init__()
        self.autoencoder = autoencoder
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            z = self.autoencoder.encode(x)
        return self.classifier(z)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # x: (batch, seq, d_model)
        return x + self.pe[:, : x.size(1)]


class EEGTransformerClassifier(nn.Module):
    """Transformer encoder over down-sampled time steps (channels as the
    per-token feature dimension, projected to d_model)."""

    def __init__(self, n_channels=32, d_model=64, nhead=4, num_layers=2, n_classes=2, dropout=0.3, pool_stride=4):
        super().__init__()
        self.pool = nn.AvgPool1d(pool_stride)
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):  # x: (batch, n_channels, window_len)
        x = self.pool(x)              # downsample time axis for tractable sequence length
        x = x.permute(0, 2, 1)        # (batch, time, channels)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


DEEP_MODELS = {
    "bilstm": BiLSTMClassifier,
    "gru": GRUClassifier,
    "cnn": EEGCNNClassifier,
    "transformer": EEGTransformerClassifier,
}
