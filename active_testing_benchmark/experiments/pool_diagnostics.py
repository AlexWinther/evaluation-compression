"""Compute surrogate diagnostics for a finished benchmark run, including on its evaluation pool.

The benchmark itself measures the surrogates only on the checkpoint-selection
holdout, so that no evaluation label is used before acquisition. To explain a
finished run, this script also measures them on the evaluation pool. This is a
post hoc analysis: it uses the pool labels and must not inform choices made
before the benchmark.

The script refits the surrogates and proxy exactly as the run did, from the run's
seed, and saves `<run>-diagnostics.json` next to the run with:

    holdout   diagnostics on the holdout of the classifier's own training split
    pool      diagnostics on the evaluation pool, with the run's surrogates
    pool_loss how concentrated the classifier's loss is on the evaluation pool

Runs made before the checkpoint seed was reused split the development data with a
different seed than the classifier was trained with. Their holdout would then
contain training images of the classifier, so the holdout diagnostics use
surrogates fitted on the classifier's own split instead.
"""

import argparse
import json
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from active_testing_benchmark.active_testing.diagnostics import surrogate_diagnostics
from active_testing_benchmark.config import CLASS_TO_IDX
from active_testing_benchmark.experiments.utils import (
    ensemble_cache_dir,
    load_checkpoint,
    model_from_checkpoint,
)
from active_testing_benchmark.modeling.data import FairVisionData
from active_testing_benchmark.modeling.models import create_image_transform
from active_testing_benchmark.modeling.train import (
    TrainingConfig,
    fit_auxiliary_models,
    split_dataset,
)
from active_testing_benchmark.reporting import default_diagnostics_path, run_paths

# The evaluation pool of each benchmark experiment, see `experiments.at_benchmarking`.
POOL_SPLITS = {"AT_dev": "validation", "AT_final": "test"}


@torch.inference_mode()
def loss_concentration(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device | str,
    batch_size: int = 64,
    top_fraction: float = 0.05,
) -> dict[str, float]:
    """Describe how much of the pool's cross-entropy loss a few images hold.

    Active testing gains most when the loss is concentrated, but only if the
    proposal finds the images that hold it. Confident errors are hard to find for
    any proposal built from the classifier's confidence.
    """
    model.eval()
    log_probabilities, labels = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        log_probabilities.append(model(batch["image"].to(device)).cpu().double().log_softmax(1))
        labels.append(torch.as_tensor(batch["label"]).long())
    log_probabilities = torch.cat(log_probabilities).numpy()
    labels = torch.cat(labels).numpy()
    losses = -log_probabilities[np.arange(len(labels)), labels]
    misclassified = log_probabilities.argmax(1) != labels
    top = np.argsort(-losses)[: max(1, round(top_fraction * len(losses)))]
    return {
        "risk": float(losses.mean()),
        "top_fraction": top_fraction,
        "top_loss_share": float(losses[top].sum() / losses.sum()),
        "misclassified": int(misclassified.sum()),
        "misclassified_loss_share": float(losses[misclassified].sum() / losses.sum()),
        "top_misclassified": int(misclassified[top].sum()),
        # Confidence of the predicted class, high for confident errors.
        "top_median_confidence": float(np.median(np.exp(log_probabilities[top].max(1)))),
        "class_counts": np.bincount(labels).tolist(),
    }


def run_pool_diagnostics(run: Path, device: str | None = None, output: Path | None = None) -> dict:
    """Refit the run's surrogates and save holdout, pool, and loss diagnostics."""
    _, metadata_path = run_paths(run)
    metadata = json.loads(metadata_path.read_text())
    configuration = metadata["configuration"]
    disease = configuration["disease"]
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_path = Path(metadata["training"]["checkpoint_path"])
    checkpoint = load_checkpoint(checkpoint_path)
    model = model_from_checkpoint(checkpoint, device)
    run_config = TrainingConfig(**configuration["training"])
    checkpoint_seed = checkpoint["config"]["random_state"]

    transform = create_image_transform(run_config.image_size, training=False)
    data_root = Path(configuration["data_root"])

    def load(split: str) -> FairVisionData:
        return FairVisionData(
            disease=disease,
            split=split,
            transform=transform,
            class_to_idx=CLASS_TO_IDX[disease],
            fairvision_data_root=data_root,
        )

    development = load("train")
    pool = load(POOL_SPLITS[configuration["experiment"]])

    def fit_surrogates(seed: int):
        fit, holdout = next(
            split_dataset(
                development,
                development.labels,
                validation_fraction=run_config.validation_fraction,
                random_state=seed,
            )
        )
        auxiliary = fit_auxiliary_models(
            fit,
            holdout,
            pool.classes,
            TrainingConfig(**{**configuration["training"], "random_state": seed}),
            device,
            surrogate_kind=configuration.get("surrogate_kind", "forest"),
            proxy_kind=configuration.get("proxy_kind", "resnet_ensemble"),
            ensemble_dir=ensemble_cache_dir(checkpoint_path, seed),
            oracle_noise=configuration.get("oracle_noise", 0.0),
        )
        return auxiliary, holdout

    run_seed = run_config.random_state
    logger.info(f"Refitting the run's surrogates with seed {run_seed}")
    auxiliary, run_holdout = fit_surrogates(run_seed)
    if list(auxiliary.seeds) != metadata["auxiliary_seeds"]:
        raise ValueError(
            f"Refitted surrogate seeds {auxiliary.seeds} differ from the run's "
            f"{metadata['auxiliary_seeds']}. Runs made before the surrogate rework "
            "cannot be refitted with the current pipeline."
        )
    batch_size = run_config.batch_size
    result = {
        "run": str(metadata_path),
        "checkpoint": str(checkpoint_path),
        "run_seed": run_seed,
        "checkpoint_seed": checkpoint_seed,
        "pool_split": POOL_SPLITS[configuration["experiment"]],
        "pool": surrogate_diagnostics(
            model, auxiliary.surrogates, auxiliary.proxy, pool, device, batch_size
        ),
        "pool_loss": loss_concentration(model, pool, device, batch_size),
    }
    if run_seed != checkpoint_seed:
        logger.info(f"Run seed differs from checkpoint seed {checkpoint_seed}, refitting holdout")
        auxiliary, run_holdout = fit_surrogates(checkpoint_seed)
    result["holdout"] = surrogate_diagnostics(
        model, auxiliary.surrogates, auxiliary.proxy, run_holdout, device, batch_size
    )
    result["holdout_surrogate_seeds"] = list(auxiliary.seeds)

    output = output or default_diagnostics_path(run)
    output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(f"Saved diagnostics to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="Benchmark run: its CSV, JSON, or their stem")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--output", type=Path, help="Default: <run>-diagnostics.json")
    args = parser.parse_args()
    result = run_pool_diagnostics(args.run, args.device, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
