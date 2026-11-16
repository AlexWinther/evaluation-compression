"""Compare candidate PPAT proxies on the holdout of an AT checkpoint, before any benchmark.

PPAT reduces variance only when the proxy loss L(f(x), g(x)) correlates with the true
loss of the model under test f (Ashouritaklimi et al., 2026, Prop. 3.2). Fixed
lambda = 1/2 helps when that correlation exceeds about 1/4. This experiment measures it
for proxies that need no fine-tuning, so the proxy can be chosen up front:

    imagenet_probe   logistic regression on frozen ImageNet ResNet-18 features
    retfound_probe   logistic regression on frozen RETFound features
    flair_probe      logistic regression on frozen FLAIR image embeddings
    flair_zero_shot  FLAIR zero-shot with the prompts in `modeling.foundation`

Probes are trained on the checkpoint's 80 % development split and every candidate is
scored on its 20 % holdout, which contains no image the classifier was trained on. The
official validation and test partitions are not used. Frozen features are cached in
`data/interim/proxy_features/<disease>/`, keyed by sample ID.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing.diagnostics import proxy_diagnostics
from active_testing_benchmark.active_testing.surrogate import FrozenFeatureProbe, predict_logits
from active_testing_benchmark.config import CLASS_TO_IDX, INTERIM_DATA_DIR, RAW_DATA_DIR
from active_testing_benchmark.experiments.utils import (
    load_checkpoint,
    model_from_checkpoint,
    reused_checkpoint_seed,
)
from active_testing_benchmark.modeling.data import FairVisionData
from active_testing_benchmark.modeling.foundation import (
    FLAIR_PROMPTS,
    FlairImageEncoder,
    FlairZeroShot,
    RetFoundFeatures,
    flair_image_transform,
)
from active_testing_benchmark.modeling.models import FrozenImageNetFeatures, create_image_transform
from active_testing_benchmark.modeling.provenance import log_dataset_provenance
from active_testing_benchmark.modeling.registry import experiment_run, log_metrics
from active_testing_benchmark.modeling.train import split_dataset

CANDIDATES = ("imagenet_probe", "retfound_probe", "flair_probe", "flair_zero_shot")


@dataclass(frozen=True)
class ProxyExperimentConfig:
    model_type: str
    random_state: int
    batch_size: int
    image_size: int
    validation_fraction: float = 0.2


def cached_features(
    name: str,
    make_extractor: Callable[[], nn.Module],
    dataset: FairVisionData,
    device: torch.device | str,
    batch_size: int,
    cache_dir: Path,
) -> np.ndarray:
    """Return frozen features for every image of `dataset`, in dataset order.

    Extraction is the expensive step for large encoders, so features are saved with
    their sample IDs and reused only when the IDs match exactly. The extractor is built
    only when the cache cannot be used, so cached runs never load large weights.
    """
    path = Path(cache_dir) / f"{name}.npz"
    sample_ids = np.asarray(dataset.sample_ids)
    if path.exists():
        stored = np.load(path)
        if np.array_equal(stored["sample_ids"], sample_ids):
            logger.info(f"Loaded cached {name} features from {path}")
            return stored["features"]
        logger.warning(f"Ignoring {path}: its sample IDs do not match the dataset")
    extractor = make_extractor().to(device).eval()
    batches = []
    with torch.inference_mode():
        for batch in tqdm(
            DataLoader(dataset, batch_size=batch_size, shuffle=False),
            desc=f"Extracting {name} features",
        ):
            batches.append(extractor(batch["image"].to(device)).flatten(1).cpu().double().numpy())
    features = np.concatenate(batches)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, sample_ids=sample_ids)
    return features


def probe_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Fit the benchmark's logistic-regression probe and predict the holdout."""
    probe = FrozenFeatureProbe(num_classes, nn.Identity())
    probe.fit_features(features[fit_indices], labels[fit_indices])
    return np.exp(probe.log_probabilities(features[holdout_indices]))


def run_proxy_experiment(
    target_checkpoint: Path,
    disease: str = "dr",
    seed: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    cache_dir: Path | None = None,
    track: bool = True,
    candidates: tuple[str, ...] = CANDIDATES,
) -> pd.DataFrame:
    """Score each candidate proxy against the checkpoint's losses on its own holdout."""
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Choose at least one candidate without duplicates")
    if set(candidates).difference(CANDIDATES):
        raise ValueError(f"Unknown candidate, choose from {CANDIDATES}")
    seed = reused_checkpoint_seed(target_checkpoint, seed)
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint = load_checkpoint(target_checkpoint)
    image_size = checkpoint["config"]["image_size"]
    cache_dir = Path(cache_dir or INTERIM_DATA_DIR / "proxy_features" / disease.lower())

    def load(transform) -> FairVisionData:
        return FairVisionData(
            disease=disease,
            split="train",
            transform=transform,
            class_to_idx=CLASS_TO_IDX[disease.lower()],
            fairvision_data_root=data_root,
        )

    standard = load(create_image_transform(image_size, training=False))
    if checkpoint["classes"] != standard.classes:
        raise ValueError(f"Checkpoint classes do not match the data: {target_checkpoint}")
    # The checkpoint's seed reproduces its 80/20 split, so the holdout is unseen by it.
    fit, holdout = next(
        split_dataset(
            standard,
            standard.labels,
            # Checkpoints store the split fraction of their TrainingConfig, 0.2 by default.
            validation_fraction=checkpoint["config"].get("validation_fraction", 0.2),
            random_state=seed,
        )
    )
    fit_indices = np.asarray(fit.indices)
    holdout_indices = np.asarray(holdout.indices)
    labels = standard.labels.astype(np.int64)
    num_classes = len(standard.classes)

    target = model_from_checkpoint(checkpoint, device)
    logits, holdout_labels = predict_logits(target, holdout, device, batch_size)
    class_losses = -torch.from_numpy(logits).log_softmax(dim=1).numpy()
    if not np.array_equal(holdout_labels, labels[holdout_indices]):
        raise ValueError("Holdout labels are not in dataset order")

    uses_flair = any(candidate.startswith("flair") for candidate in candidates)
    flair_dataset = load(flair_image_transform()) if uses_flair else None
    flair_model = None

    def flair() -> nn.Module:
        nonlocal flair_model
        if flair_model is None:
            flair_model = FlairImageEncoder().flair.to(device)
        return flair_model

    def score(candidate: str) -> np.ndarray:
        if candidate == "imagenet_probe":
            features = cached_features(
                "imagenet_resnet18",
                FrozenImageNetFeatures,
                standard,
                device,
                batch_size,
                cache_dir,
            )
        elif candidate == "retfound_probe":
            features = cached_features(
                "retfound_mae_cfp", RetFoundFeatures, standard, device, batch_size, cache_dir
            )
        elif candidate == "flair_probe":
            features = cached_features(
                "flair",
                lambda: FlairImageEncoder(flair()),
                flair_dataset,
                device,
                batch_size,
                cache_dir,
            )
        else:
            zero_shot = FlairZeroShot(FLAIR_PROMPTS[disease.lower()], flair()).to(device)
            log_probabilities, _ = predict_logits(
                zero_shot, Subset(flair_dataset, holdout_indices.tolist()), device, batch_size
            )
            return np.exp(log_probabilities)
        return probe_probabilities(features, labels, fit_indices, holdout_indices, num_classes)

    rows = []
    parent_config = ProxyExperimentConfig("comparison", seed, batch_size, image_size)
    with experiment_run(
        "proxy_candidates", parent_config, disease, device, enabled=track, role="parent"
    ) as parent:
        if parent is not None:
            mlflow.log_params(
                {
                    "candidates": list(candidates),
                    "target_checkpoint": str(target_checkpoint),
                    "train_size": len(fit),
                    "validation_size": len(holdout),
                }
            )
            log_dataset_provenance(train=fit, validation=holdout)
        for candidate in candidates:
            config = ProxyExperimentConfig(candidate, seed, batch_size, image_size)
            with experiment_run(
                "proxy_candidates", config, disease, device, enabled=track, nested=True
            ) as run:
                start = perf_counter()
                probabilities = score(candidate)
                seconds = perf_counter() - start
                metrics = proxy_diagnostics(class_losses, holdout_labels, probabilities)
                log_metrics({**metrics, "seconds": seconds}, prefix="proxy")
                logger.info(f"{candidate}: {metrics} ({seconds:.0f} s)")
                rows.append(
                    {
                        "candidate": candidate,
                        "train_size": len(fit),
                        "holdout_size": len(holdout),
                        "holdout_prevalence": float(np.mean(holdout_labels == 1))
                        if num_classes == 2
                        else None,
                        "proxy_positive_rate": float(np.mean(probabilities.argmax(1) == 1))
                        if num_classes == 2
                        else None,
                        **metrics,
                        "seconds": seconds,
                        "mlflow_run_id": run.info.run_id if run else None,
                    }
                )
        results = pd.DataFrame(rows)
        if parent is not None:
            mlflow.log_text(results.to_csv(index=False), "comparison.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        required=True,
        help="AT_dev classifier checkpoint, for example models/experiment/AT_dev/dr/resnet.pt",
    )
    parser.add_argument("--disease", choices=("dr", "amd", "glaucoma"), default="dr")
    parser.add_argument("--seed", type=int, help="Must match the checkpoint seed if given")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR / "fairvision")
    parser.add_argument("--cache-dir", type=Path, help="Default: data/interim/proxy_features")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATES,
        default=CANDIDATES,
        help="Proxies to compare (default: all). FLAIR and RETFound need their weights",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output", type=Path, help="Also save the comparison as CSV")
    args = parser.parse_args()
    try:
        reused_checkpoint_seed(args.target_checkpoint, args.seed)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    results = run_proxy_experiment(
        args.target_checkpoint,
        disease=args.disease,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        track=not args.no_mlflow,
        candidates=tuple(args.candidates),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
