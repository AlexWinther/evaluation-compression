import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
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


class ConstantClassSurrogate(nn.Module):
    """Predict one class with certainty for every input, ignoring the image.

    This is a baseline that needs no training. Under it, the Kossen score of an
    image is the model's cross-entropy if the label were `target_class`, so it ranks
    images by how strongly the model rejects that class. With the positive class,
    the score is highest for the images the model is most sure are negative, which
    includes the confident misses that hold most of the loss on FairVision DR.
    """

    def __init__(self, num_classes: int, target_class: int) -> None:
        super().__init__()
        if not 0 <= target_class < num_classes:
            raise ValueError(f"target_class must be in [0, {num_classes})")
        self.num_classes = num_classes
        self.target_class = target_class

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Log-probabilities of a one-hot distribution. The tiny floor keeps the other
        # classes finite, as in RandomForestModule.
        logits = torch.full((len(images), self.num_classes), math.log(1e-8))
        logits[:, self.target_class] = 0.0
        return logits.to(images.device)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return the temperature T that minimises the NLL of softmax(logits / T).

    Temperature scaling (Guo et al., 2017) rescales confidence without changing the
    predicted class. ASE calibrates its deep-ensemble surrogates this way on a
    validation set (Kossen et al., 2022, app. D.3). Fit it on held-out labelled data,
    never on the evaluation pool. The search is over log T in [-4, 4], so T stays
    between about 0.02 and 55.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or len(logits) != len(labels) or len(labels) == 0:
        raise ValueError("Expected [n, classes] logits and n labels")
    if not np.isfinite(logits).all():
        raise ValueError("Expected finite logits")

    def negative_log_likelihood(log_temperature: float) -> float:
        log_probabilities = log_softmax(logits / np.exp(log_temperature), axis=1)
        return float(-log_probabilities[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(negative_log_likelihood, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


@torch.inference_mode()
def predict_logits(
    module: nn.Module, dataset, device: torch.device | str, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return a module's outputs and the labels for a labelled dataset, in order."""
    module.eval()
    outputs, labels = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        outputs.append(module(batch["image"].to(device)).cpu().double().numpy())
        labels.append(torch.as_tensor(batch["label"]).long().numpy())
    return np.concatenate(outputs), np.concatenate(labels)


class TemperatureScaled(nn.Module):
    """Divide a classifier's logits by a fixed, fitted temperature."""

    def __init__(self, model: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs) / self.temperature.to(inputs.device)


class RandomForestModule(nn.Module):
    """A random forest on frozen image features that returns log-probabilities.

    The forest models p(y | x) at the prevalence of its training data, so it uses no
    class reweighting. Reweighting would shift its probabilities towards the minority
    class and bias every expected loss computed from them, which ASE uses directly as
    its estimate. The entropy criterion and square-root feature sampling follow the
    forest surrogate of Kossen et al. (2021, app. C).

    With `bootstrap_development=True`, `fit` draws one bootstrap resample of the
    development data from `random_state` and keeps it for all later refits. Several
    such forests form an ensemble whose members differ because of the finite training
    sample. Without the resample, the forests of an ensemble differ only by the Monte
    Carlo noise of averaging their trees, which is too small to rank for XWED.

    `calibrate` fits a temperature on held-out data. It applies to the log-probabilities
    and is kept fixed across refits, since no new held-out labels arrive during
    acquisition.
    """

    def __init__(
        self,
        num_classes: int,
        feature_extractor: nn.Module | None = None,
        n_estimators: int = 200,
        random_state: int = 42,
        *,
        bootstrap_development: bool = False,
    ):
        super().__init__()

        if feature_extractor is None:
            feature_extractor = FrozenImageNetFeatures()

        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.random_state = random_state
        self.bootstrap_development = bootstrap_development

        # Frozen feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.forest = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion="entropy",
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        self.temperature = 1.0
        self._development_features: np.ndarray | None = None
        self._development_labels: np.ndarray | None = None

    @torch.no_grad()
    def _extract_features(self, images):
        self.feature_extractor.eval()
        features = self.feature_extractor(images)
        return features.flatten(1).cpu().numpy()

    def _features_and_labels(self, dataloader, device, description: str):
        self.feature_extractor.to(device).eval()
        X, y = [], []
        for batch in tqdm(dataloader, desc=description, leave=False):
            X.append(self._extract_features(batch["image"].to(device)))
            y.append(torch.as_tensor(batch["label"]).long().cpu().numpy())
        return np.concatenate(X), np.concatenate(y)

    def fit(self, dataloader, device: torch.device | str = "cpu"):
        X, y = self._features_and_labels(
            dataloader, device, "Extracting features for Random Forest training"
        )
        if self.bootstrap_development:
            rng = np.random.default_rng(self.random_state)
            sample = rng.integers(0, len(y), size=len(y))
            X, y = X[sample], y[sample]

        self._development_features = X
        self._development_labels = y
        self.forest.fit(X, y)
        return self

    def calibrate(self, dataloader, device: torch.device | str = "cpu"):
        """Fit the temperature of the forest's log-probabilities on held-out data."""
        X, y = self._features_and_labels(dataloader, device, "Calibrating Random Forest")
        self.temperature = 1.0
        self.temperature = fit_temperature(self._log_probabilities(X), y)
        return self

    def _log_probabilities(self, features: np.ndarray) -> np.ndarray:
        probs = self.forest.predict_proba(features)
        # Maintain a consistent output dimension
        full_probs = np.zeros((len(probs), self.num_classes), dtype=np.float64)
        full_probs[:, self.forest.classes_.astype(int)] = probs
        return np.log(np.maximum(full_probs, 1e-8)) / self.temperature

    @torch.no_grad()
    def forward(self, images):
        log_probabilities = self._log_probabilities(self._extract_features(images))
        # Return calibrated log-probabilities as logits.
        return torch.from_numpy(log_softmax(log_probabilities, axis=1).astype(np.float32)).to(
            images.device
        )


def update_random_forest_surrogates(
    surrogates: tuple[RandomForestModule, ...] | list[RandomForestModule],
    X_observed: torch.Tensor,
    y_observed: torch.Tensor,
) -> None:
    """Refit each forest on its development data plus the full observed history.

    A bootstrapped forest keeps its own development resample, so the members of an
    ensemble stay distinct after every refit. The observed labels are added once each.
    """
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


class FrozenFeatureProbe(nn.Module):
    """A logistic-regression probe on frozen image features, returning log-probabilities.

    This is the PPAT-style proxy built from a pretrained encoder: only the linear
    probe is trained, on the development split, and the encoder is never fine-tuned.
    Features are standardised before the probe. The probe has no class reweighting, so
    its probabilities match the development prevalence.
    """

    def __init__(self, num_classes: int, feature_extractor: nn.Module, max_iter: int = 2000):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor.requires_grad_(False)
        self.probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        self.feature_extractor.eval()
        return self.feature_extractor(images).flatten(1).cpu().double().numpy()

    def fit_features(self, features: np.ndarray, labels: np.ndarray) -> "FrozenFeatureProbe":
        self.probe.fit(features, labels)
        return self

    def fit(self, dataloader, device: torch.device | str = "cpu") -> "FrozenFeatureProbe":
        self.feature_extractor.to(device)
        features, labels = [], []
        for batch in tqdm(dataloader, desc="Extracting features for the probe", leave=False):
            features.append(self.extract_features(batch["image"].to(device)))
            labels.append(torch.as_tensor(batch["label"]).long().numpy())
        return self.fit_features(np.concatenate(features), np.concatenate(labels))

    def log_probabilities(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((len(features), self.num_classes), dtype=np.float64)
        probabilities[:, self.probe.classes_.astype(int)] = self.probe.predict_proba(features)
        return np.log(np.maximum(probabilities, 1e-8))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        log_probabilities = self.log_probabilities(self.extract_features(images))
        return torch.from_numpy(log_probabilities.astype(np.float32)).to(images.device)
