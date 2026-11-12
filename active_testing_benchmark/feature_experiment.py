"""Compare random forest image features on a held-out part of FairVision training data."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlflow
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from active_testing_benchmark.active_testing.surrogate import (
    RandomForestModule,
    SpatialTextureFeatures,
)
from active_testing_benchmark.config import RAW_DATA_DIR
from active_testing_benchmark.experiment import evaluate_model
from active_testing_benchmark.modeling.data import FairVisionData
from active_testing_benchmark.modeling.models import create_image_transform
from active_testing_benchmark.modeling.provenance import log_dataset_provenance
from active_testing_benchmark.modeling.registry import experiment_run, log_metrics
from active_testing_benchmark.modeling.train import split_dataset


class PooledPixels(nn.Module):
    """The original 16x16 pooled-pixel baseline, including its MPS workaround."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        if device.type == "mps" and any(size % 16 for size in images.shape[-2:]):
            images = images.cpu()
        return F.adaptive_avg_pool2d(images, (16, 16)).flatten(1).to(device)


@dataclass(frozen=True)
class FeatureExperimentConfig:
    model_type: str
    random_state: int
    n_estimators: int
    batch_size: int
    image_size: int
    validation_fraction: float = 0.2


def run_feature_experiment(
    disease: str = "dr",
    seed: int = 42,
    n_estimators: int = 200,
    batch_size: int = 32,
    image_size: int = 224,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    track: bool = True,
) -> pd.DataFrame:
    """Compare descriptors using the same forest and stratified training holdout.

    The official validation and test partitions are not used for feature selection.
    """
    if n_estimators <= 0 or batch_size <= 0 or image_size <= 0:
        raise ValueError("n_estimators, batch_size, and image_size must be positive")
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dataset = FairVisionData(
        disease=disease,
        split="train",
        transform=create_image_transform(image_size, training=False),
        fairvision_data_root=data_root,
    )
    fit, holdout = next(
        split_dataset(dataset, dataset.labels, validation_fraction=0.2, random_state=seed)
    )
    loader = DataLoader(fit, batch_size=batch_size, shuffle=False)
    rows = []
    for name, extractor in (
        ("pooled_pixels", PooledPixels()),
        ("spatial_texture", SpatialTextureFeatures()),
    ):
        config = FeatureExperimentConfig(name, seed, n_estimators, batch_size, image_size)
        with experiment_run("feature_extractors", config, disease, device, enabled=track) as run:
            if run is not None:
                log_dataset_provenance(train=fit, validation=holdout)
                mlflow.log_params({"train_size": len(fit), "validation_size": len(holdout)})
                mlflow.set_tag("stage", "training")
            model = RandomForestModule(
                num_classes=len(dataset.classes),
                feature_extractor=extractor,
                n_estimators=n_estimators,
                random_state=seed,
            )
            start = perf_counter()
            model.fit(loader, device=device)
            fit_seconds = perf_counter() - start
            if run is not None:
                mlflow.set_tag("stage", "evaluation")
            start = perf_counter()
            metrics = evaluate_model(model, holdout, batch_size, device)
            evaluation_seconds = perf_counter() - start
            overall = {key: metrics[key] for key in ("loss", "accuracy", "f1", "auc")}
            log_metrics(overall, prefix="evaluation")
            log_metrics({"fit_seconds": fit_seconds, "evaluation_seconds": evaluation_seconds})
            if run is not None:
                mlflow.log_param("feature_count", model.forest.n_features_in_)
            rows.append(
                {
                    "extractor": name,
                    "train_size": len(fit),
                    "validation_size": len(holdout),
                    "feature_count": model.forest.n_features_in_,
                    **overall,
                    "fit_seconds": fit_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "mlflow_run_id": run.info.run_id if run else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=("dr", "amd", "glaucoma"), default="dr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR / "fairvision")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    results = run_feature_experiment(
        disease=args.disease,
        seed=args.seed,
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        data_root=args.data_root,
        track=not args.no_mlflow,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
