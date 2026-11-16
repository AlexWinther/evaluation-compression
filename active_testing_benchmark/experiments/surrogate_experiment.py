"""Fit surrogate conditions for an AT checkpoint and check them on its holdout first.

This is the pre-benchmark gate of `specs/surrogate-pipeline-review.md` (Sec. 7.7). For
each surrogate condition it fits exactly what an AT run with the same checkpoint
would fit (`fit_auxiliary_models`, same split, seeds and ensemble cache) and reports
`surrogate_diagnostics` on the checkpoint-selection holdout. The holdout contains no
image the classifier was trained on, and the official validation and test partitions
are not used. A deep ensemble trained here is cached and reused by later AT runs.

The holdout also early-stops and calibrates the surrogates, so the diagnostics are
slightly optimistic.
"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

from loguru import logger
import mlflow
import pandas as pd
import torch

from active_testing_benchmark.active_testing.diagnostics import surrogate_diagnostics
from active_testing_benchmark.config import CLASS_TO_IDX, RAW_DATA_DIR
from active_testing_benchmark.experiments.utils import (
    ensemble_cache_dir,
    load_checkpoint,
    model_from_checkpoint,
    reused_checkpoint_seed,
)
from active_testing_benchmark.modeling.data import FairVisionData
from active_testing_benchmark.modeling.models import create_image_transform
from active_testing_benchmark.modeling.provenance import log_dataset_provenance
from active_testing_benchmark.modeling.registry import experiment_run, log_metrics
from active_testing_benchmark.modeling.train import (
    PROXY_KINDS,
    SURROGATE_KINDS,
    TrainingConfig,
    fit_auxiliary_models,
    split_dataset,
)


@dataclass(frozen=True)
class SurrogateExperimentConfig:
    model_type: str
    random_state: int
    proxy_kind: str


def checkpoint_training_config(checkpoint: dict) -> TrainingConfig:
    """Rebuild the classifier's TrainingConfig, with defaults for fields added later."""
    names = {field.name for field in fields(TrainingConfig)}
    return TrainingConfig(**{k: v for k, v in checkpoint["config"].items() if k in names})


def run_surrogate_experiment(
    target_checkpoint: Path,
    disease: str = "dr",
    surrogate_kinds: tuple[str, ...] = SURROGATE_KINDS,
    proxy_kind: str = "resnet_ensemble",
    seed: int | None = None,
    device: str | None = None,
    data_root: Path = RAW_DATA_DIR / "fairvision",
    track: bool = True,
) -> pd.DataFrame:
    """Return one row of holdout diagnostics per surrogate condition."""
    if not surrogate_kinds or set(surrogate_kinds).difference(SURROGATE_KINDS):
        raise ValueError(f"Choose surrogate kinds from {SURROGATE_KINDS}")
    if proxy_kind not in PROXY_KINDS:
        raise ValueError(f"proxy_kind must be one of {PROXY_KINDS}")
    seed = reused_checkpoint_seed(target_checkpoint, seed)
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint = load_checkpoint(target_checkpoint)
    config = checkpoint_training_config(checkpoint)
    development = FairVisionData(
        disease=disease,
        split="train",
        transform=create_image_transform(config.image_size, training=False),
        class_to_idx=CLASS_TO_IDX[disease.lower()],
        fairvision_data_root=data_root,
    )
    if checkpoint["classes"] != development.classes:
        raise ValueError(f"Checkpoint classes do not match the data: {target_checkpoint}")
    # The checkpoint's seed reproduces its split, so the holdout is unseen by the model.
    fit, holdout = next(
        split_dataset(
            development,
            development.labels,
            validation_fraction=config.validation_fraction,
            random_state=seed,
        )
    )
    model = model_from_checkpoint(checkpoint, device)
    ensemble_dir = ensemble_cache_dir(target_checkpoint, seed)

    rows = []
    parent_config = SurrogateExperimentConfig("comparison", seed, proxy_kind)
    with experiment_run(
        "surrogate_candidates", parent_config, disease, device, enabled=track, role="parent"
    ) as parent:
        if parent is not None:
            mlflow.log_params(
                {
                    "target_checkpoint": str(target_checkpoint),
                    "surrogate_kinds": list(surrogate_kinds),
                    "ensemble_dir": str(ensemble_dir),
                    "train_size": len(fit),
                    "validation_size": len(holdout),
                }
            )
            log_dataset_provenance(train=fit, validation=holdout)
        for kind in surrogate_kinds:
            run_config = SurrogateExperimentConfig(kind, seed, proxy_kind)
            with experiment_run(
                "surrogate_candidates", run_config, disease, device, enabled=track, nested=True
            ) as run:
                start = perf_counter()
                auxiliary = fit_auxiliary_models(
                    fit,
                    holdout,
                    development.classes,
                    config,
                    device,
                    surrogate_kind=kind,
                    proxy_kind=proxy_kind,
                    ensemble_dir=ensemble_dir,
                )
                fit_seconds = perf_counter() - start
                diagnostics = surrogate_diagnostics(
                    model,
                    auxiliary.surrogates,
                    auxiliary.proxy,
                    holdout,
                    device,
                    config.batch_size,
                )
                log_metrics({**diagnostics, "fit_seconds": fit_seconds}, prefix="diagnostics")
                logger.info(f"{kind} surrogate with {proxy_kind} proxy: {diagnostics}")
                rows.append(
                    {
                        "surrogate_kind": kind,
                        "proxy_kind": proxy_kind,
                        "train_size": len(fit),
                        "holdout_size": len(holdout),
                        "member_seeds": list(auxiliary.seeds),
                        **diagnostics,
                        "fit_seconds": fit_seconds,
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
    parser.add_argument(
        "--surrogates", nargs="+", choices=SURROGATE_KINDS, default=list(SURROGATE_KINDS)
    )
    parser.add_argument("--proxy", choices=PROXY_KINDS, default="resnet_ensemble")
    parser.add_argument("--seed", type=int, help="Must match the checkpoint seed if given")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR / "fairvision")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output", type=Path, help="Also save the comparison as CSV")
    args = parser.parse_args()
    try:
        reused_checkpoint_seed(args.target_checkpoint, args.seed)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    results = run_surrogate_experiment(
        args.target_checkpoint,
        disease=args.disease,
        surrogate_kinds=tuple(args.surrogates),
        proxy_kind=args.proxy,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        track=not args.no_mlflow,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
