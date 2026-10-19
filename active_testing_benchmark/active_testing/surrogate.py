import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from tqdm.auto import tqdm


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

    @torch.no_grad()
    def _extract_features(self, images):
        self.feature_extractor.eval()
        features = self.feature_extractor(images)
        return features.flatten(1).cpu().numpy()

    def fit(self, dataloader, device="cpu"):
        self.feature_extractor.to(device).eval()

        X, y = [], []

        for batch in tqdm(dataloader, desc="Extracting features for Random Forest training"):
            images = batch["image"]
            labels = batch["target"]

            features = self._extract_features(images.to(device))
            X.append(features)
            y.append(labels.cpu().numpy())

        X = np.concatenate(X)
        y = np.concatenate(y)

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
