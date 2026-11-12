import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from active_testing_benchmark.modeling.models import FrozenImageNetFeatures


class SpatialTextureFeatures(nn.Module):
    """Summarize intensity, contrast, and edges at two spatial resolutions.

    Inputs are floating point BCHW images, including the normalized images used
    by the target model. No labels, fitted statistics, or pretrained weights are
    needed. First pool to 64x64 to bound convolution cost independently of input
    size. At each of the 4x4 and 8x8 grids, concatenate per-channel local mean,
    population standard deviation, and mean absolute x/y Sobel responses.
    RGB inputs produce 960 features. Channel order and normalization are kept
    intact, so training, prediction, and online updates use the same transform.

    These handcrafted descriptors add coarse texture information but are not
    learned semantic features. Their benefit must be measured on held-out data.
    """

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8
        self.register_buffer("sobel", torch.stack((sobel_x, sobel_x.T)).unsqueeze(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or not images.is_floating_point():
            raise ValueError("Expected floating point images with shape (batch, channels, H, W)")
        if images.device.type == "mps" and any(size % 64 for size in images.shape[-2:]):
            # MPS cannot pool non-divisible sizes (e.g. 224 -> 64). Preserve
            # identical pooling semantics on CPU, then resume extraction on MPS.
            # https://github.com/pytorch/pytorch/issues/96056
            images = F.adaptive_avg_pool2d(images.cpu(), (64, 64)).to(images.device)
        else:
            images = F.adaptive_avg_pool2d(images, (64, 64))
        batch, channels, height, width = images.shape
        # Replicate padding avoids introducing artificial edges around constant images.
        padded = F.pad(
            images.reshape(batch * channels, 1, height, width), (1, 1, 1, 1), mode="replicate"
        )
        edges = F.conv2d(padded, self.sobel.to(images)).abs()
        edges = edges.reshape(batch, channels, 2, height, width)
        features = []
        for grid in (4, 8):
            mean = F.adaptive_avg_pool2d(images, grid)
            # Center before squaring to avoid cancellation for nearly constant images.
            centered = images - F.interpolate(mean, size=(height, width), mode="nearest")
            std = F.adaptive_avg_pool2d(centered.square(), grid).sqrt()
            features.extend(
                statistic.flatten(1)
                for statistic in (
                    mean,
                    std,
                    F.adaptive_avg_pool2d(edges[:, :, 0], grid),
                    F.adaptive_avg_pool2d(edges[:, :, 1], grid),
                )
            )
        return torch.cat(features, dim=1)


class MeanProbabilityEnsemble(nn.Module):
    """Average member class probabilities and return their logarithms as logits."""

    def __init__(self, members: list[nn.Module] | tuple[nn.Module, ...]) -> None:
        super().__init__()
        if not members:
            raise ValueError("An ensemble requires at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = torch.stack(
            [member(inputs).softmax(dim=-1) for member in self.members]
        ).mean(dim=0)
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


class RandomForestModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_extractor: nn.Module | None = None,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        super().__init__()

        if feature_extractor is None:
            feature_extractor = FrozenImageNetFeatures()

        self.num_classes = num_classes
        self.feature_extractor = feature_extractor

        # Frozen feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.forest = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self._development_features: np.ndarray | None = None
        self._development_labels: np.ndarray | None = None

    @torch.no_grad()
    def _extract_features(self, images):
        self.feature_extractor.eval()
        features = self.feature_extractor(images)
        return features.flatten(1).cpu().numpy()

    def fit(self, dataloader, device: torch.device | str = "cpu"):
        self.feature_extractor.to(device).eval()

        X, y = [], []

        for batch in tqdm(dataloader, desc="Extracting features for Random Forest training"):
            images = batch["image"]
            labels = batch["label"]

            features = self._extract_features(images.to(device))
            X.append(features)
            y.append(labels.cpu().numpy())

        X = np.concatenate(X)
        y = np.concatenate(y)

        self._development_features = X
        self._development_labels = y
        self.forest.fit(X, y)
        return self

    @torch.no_grad()
    def forward(self, images):
        features = self._extract_features(images)
        probs = self.forest.predict_proba(features)

        # Maintain a consistent output dimension
        full_probs = np.zeros(
            (len(probs), self.num_classes),
            dtype=np.float32,
        )
        full_probs[:, self.forest.classes_.astype(int)] = probs

        # Return log-probabilities as logits
        return torch.from_numpy(full_probs).to(images.device).clamp_min(1e-8).log()


def update_random_forest_surrogates(
    surrogates: tuple[RandomForestModule, ...] | list[RandomForestModule],
    X_observed: torch.Tensor,
    y_observed: torch.Tensor,
) -> None:
    """Refit each forest on development data plus the full observed history."""
    labels = y_observed.detach().cpu().numpy()

    for surrogate in surrogates:
        if surrogate._development_features is None or surrogate._development_labels is None:
            raise ValueError("RandomForestModule must be fitted before it can be updated")

        try:
            feature_device = next(surrogate.feature_extractor.parameters()).device
        except StopIteration:
            try:
                feature_device = next(surrogate.feature_extractor.buffers()).device
            except StopIteration:
                feature_device = X_observed.device

        observed_feature_batches = []
        for start in range(0, len(X_observed), 256):
            images = X_observed[start : start + 256].to(feature_device)
            observed_feature_batches.append(surrogate._extract_features(images))

        observed_features = np.concatenate(observed_feature_batches)
        combined_features = np.concatenate(
            (surrogate._development_features, observed_features), axis=0
        )
        combined_labels = np.concatenate((surrogate._development_labels, labels), axis=0)
        surrogate.forest.fit(combined_features, combined_labels)
