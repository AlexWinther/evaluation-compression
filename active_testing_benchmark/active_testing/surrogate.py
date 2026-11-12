import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from tqdm.auto import tqdm


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
            feature_extractor = nn.Sequential(
                nn.AdaptiveAvgPool2d((16, 16)),
                nn.Flatten(),
            )

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
