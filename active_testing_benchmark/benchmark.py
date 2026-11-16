"""Benchmark method configuration and repeated finite-pool evaluation."""

from copy import deepcopy

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing import (
    ActiveSurrogateEstimator,
    KossenActiveTesting,
    PredictionPoweredActiveTesting,
    StratifiedSampleMean,
    UniformSampleMean,
    run_active_evaluation,
)
from active_testing_benchmark.active_testing.losses import ZeroOneLoss
from active_testing_benchmark.active_testing.prediction_cache import (
    cache_pool_predictions,
    predict_in_batches,
)
from active_testing_benchmark.active_testing.runner import (
    EvaluationPool,
    prepare_evaluation_pool,
    reporting_memberships,
    subset_evaluation_pool,
)
from active_testing_benchmark.active_testing.sota import KOSSEN_RETRAINING_SCHEDULE
from active_testing_benchmark.active_testing.surrogate import (
    MeanProbabilityEnsemble,
    update_random_forest_surrogates,
)
from active_testing_benchmark.modeling.registry import log_metrics

# Every remaining item keeps at least this fraction of its uniform probability, as in
# the minimum proposal probability of Kossen et al. (2021, app. B.1). AT and PPAT share
# it, so their comparison isolates the estimator and the proposal score.
EXPLORATION = 0.2
# PPAT's conservative fixed coefficient (Ashouritaklimi et al., 2026, Sec. 3.3). It
# reduces variance relative to LURE when the proxy correlation exceeds about 1/4.
PPAT_LAMBDA = 0.5
# Resampled pools come from their own random stream, keyed by (seed, repetition, stream),
# so the pool of a repetition never depends on the method seeds or on other repetitions.
POOL_STREAM = 1
# Labels acquired per step by every method. Proposals are recomputed between steps, so
# smaller batches follow the surrogate more closely at a higher computational cost.
DEFAULT_ACQUISITION_BATCH_SIZE = 25


def pool_rng(seed: int, repetition: int) -> np.random.Generator:
    """Random generator that draws the evaluation pool of one repetition."""
    return np.random.default_rng((int(seed), int(repetition), POOL_STREAM))


def class_stratified_positions(
    labels: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw `size` distinct positions that keep each class's share of the source.

    Kossen et al. (2021, app. C.5) evaluate on a new stratified random subset of the
    test set in every run. Each class receives its proportional quota rounded down,
    and the remaining places go to the classes with the largest remainders (Hamilton's
    method). Within a class, positions are drawn uniformly without replacement. They
    are returned sorted, so the pool keeps the order of the source split.
    """
    labels = np.asarray(labels)
    if not 1 <= size <= len(labels):
        raise ValueError("The pool size must be between 1 and the source size")
    classes, counts = np.unique(labels, return_counts=True)
    quotas = size * counts / len(labels)
    allocation = np.floor(quotas).astype(int)
    largest_remainders = np.argsort(-(quotas - allocation), kind="stable")
    allocation[largest_remainders[: size - allocation.sum()]] += 1
    positions = [
        rng.choice(np.flatnonzero(labels == label), size=count, replace=False)
        for label, count in zip(classes, allocation)
    ]
    return np.sort(np.concatenate(positions))


def pool_reference_risks(
    pool: EvaluationPool, logits: torch.Tensor, loss_fn: nn.Module
) -> dict[str, float]:
    """Exact risk of the model on one pool, overall and for every reporting group.

    This is the finite-pool risk that every method estimates, so it is the reference
    for that pool's estimation errors. Each item uses the runner's per-item loss.
    Keys are the curve columns `reference_risk` and `reference_risk <group>`.
    """
    loss_fn = deepcopy(loss_fn).to("cpu")
    logits = logits.cpu()
    labels = pool.y_pool.cpu()
    with torch.inference_mode():
        losses = np.array(
            [loss_fn(logits[i : i + 1], labels[i : i + 1]).item() for i in range(len(labels))]
        )
    references = {"reference_risk": float(losses.mean())}
    for group, indices in reporting_memberships(pool).items():
        references[f"reference_risk {group}"] = float(losses[list(indices)].mean())
    return references


def build_method_configs(
    auxiliary,
    num_classes: int,
    batch_size: int = DEFAULT_ACQUISITION_BATCH_SIZE,
    *,
    ase_update_every: int | None = None,
):
    """Construct the benchmark's six method configurations around one shared surrogate.

    All active methods use the same surrogate ensemble, as in PPAT (Ashouritaklimi et
    al., 2026, p.9). AT averages the members, ASE and PPAT use them as members.

    Only the cheap forest surrogate is refitted during acquisition, by ASE. It refits
    on the retraining schedule of Kossen et al. (2021), or every `ase_update_every`
    labels when that is given. The deep ensemble stays fixed, as in the large-scale
    experiments of Kossen et al. (2021, 2022).

    XWED needs at least two members. A single surrogate, such as a `LabelOracle`
    control, makes ASE acquire by expected loss instead. A fixed surrogate gives the
    same ASE estimate whatever is acquired, so this only changes which labels it reads.
    """
    if ase_update_every is not None and ase_update_every < 1:
        raise ValueError("ase_update_every must be positive")
    surrogate_kind = getattr(auxiliary, "surrogate_kind", "forest")
    ase_kwargs = {}
    if surrogate_kind == "forest":
        ase_kwargs["surrogate_update"] = update_random_forest_surrogates
        if ase_update_every is None:
            ase_kwargs["update_schedule"] = KOSSEN_RETRAINING_SCHEDULE
        else:
            ase_kwargs["update_every"] = ase_update_every
    method_configs = [
        (UniformSampleMean, {"batch_size": batch_size}),
        (StratifiedSampleMean, {"allocation": "balanced", "batch_size": batch_size}),
        (StratifiedSampleMean, {"allocation": "proportional", "batch_size": batch_size}),
        (
            KossenActiveTesting,
            {
                "surrogate": MeanProbabilityEnsemble(auxiliary.surrogates),
                "num_classes": num_classes,
                "exploration": EXPLORATION,
                "batch_size": batch_size,
            },
        ),
        (
            ActiveSurrogateEstimator,
            {
                "surrogate": list(auxiliary.surrogates),
                "num_classes": num_classes,
                "acquisition": "xwed" if len(auxiliary.surrogates) > 1 else "expected_loss",
                "loss_type": "cross_entropy",
                "surrogate_output": "logits",
                "batch_size": batch_size,
                **ase_kwargs,
            },
        ),
        (
            PredictionPoweredActiveTesting,
            {
                "num_classes": num_classes,
                "proxy": auxiliary.proxy,
                "surrogate": list(auxiliary.surrogates),
                "lambda_value": PPAT_LAMBDA,
                "exploration": EXPLORATION,
                "loss_type": "cross_entropy",
                "batch_size": batch_size,
            },
        ),
    ]

    return method_configs


def run_repetition(
    *,
    seed,
    device,
    dataset,
    model,
    method_configs,
    repetition=0,
    budget=1000,
    class_weights=None,
    protected_columns=("gender",),
    include_accuracy=False,
    progress_position=0,
    stratification_attribute=None,
):
    """Share one materialized pool across objectives, using fresh method instances.

    Every row also stores the pool size, the size of each reporting group in the pool,
    and the exact risk of the pool for its objective (`reference_risk` columns).
    """
    has_stratified = any(issubclass(cls, StratifiedSampleMean) for cls, _ in method_configs)
    if has_stratified and stratification_attribute is None:
        if len(protected_columns) != 1:
            raise ValueError("Stratified methods require an explicit stratification attribute")
        stratification_attribute = protected_columns[0]
    attribute = stratification_attribute or (
        protected_columns[0] if len(protected_columns) == 1 else "shared"
    )
    dataset = prepare_evaluation_pool(dataset, device, list(protected_columns))
    objectives = [("unweighted_cross_entropy", nn.CrossEntropyLoss(reduction="sum"))]
    if class_weights is not None:
        objectives.append(
            (
                "class_balanced_cross_entropy",
                nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="sum"),
            )
        )
    if include_accuracy:
        objectives.append(("error_rate", ZeroOneLoss()))
    pool_logits = predict_in_batches(model, dataset.X_pool, batch_size=256)
    pool_counts = {
        f"pool_n_{group}": len(indices)
        for group, indices in reporting_memberships(dataset).items()
    }
    frames = []
    with tqdm(
        objectives,
        desc="Risk types",
        unit="risk type",
        position=progress_position,
        leave=False,
    ) as risk_progress:
        for objective, loss_fn in risk_progress:
            risk_progress.set_postfix_str(objective)
            references = pool_reference_risks(dataset, pool_logits, loss_fn)
            with tqdm(
                method_configs,
                desc="Methods",
                unit="method",
                position=progress_position + 1,
                leave=False,
            ) as method_progress:
                for method_class, kwargs in method_progress:
                    method_kwargs = dict(kwargs)
                    if method_kwargs.get("surrogate_update") is not None:
                        # Refits must not affect other objectives, methods or repetitions.
                        # A fixed surrogate is shared without copying.
                        method_kwargs["surrogate"] = deepcopy(method_kwargs["surrogate"])
                    if objective == "error_rate":
                        if method_class is KossenActiveTesting:
                            method_kwargs["acquisition"] = "error_rate"
                        elif method_class in {
                            ActiveSurrogateEstimator,
                            PredictionPoweredActiveTesting,
                        }:
                            method_kwargs["loss_type"] = "error_rate"
                    method = method_class(**method_kwargs, seed=seed)
                    name = method_class.__name__
                    if isinstance(method, StratifiedSampleMean):
                        name += method.allocation.capitalize()
                    method_progress.set_postfix_str(name)
                    if mlflow.active_run() is not None:
                        mlflow.set_tags(
                            {
                                "stage": "AT",
                                "AT.method": name,
                                "AT.repetition": repetition + 1,
                                "AT.protected_attribute": attribute,
                            }
                        )
                    history = run_active_evaluation(
                        method=method,
                        model=model,
                        dataset=dataset,
                        device=device,
                        loss_fn=loss_fn,
                        protected_column=list(protected_columns),
                        budget=budget,
                        progress_position=progress_position + 2,
                        stratification_attribute=(
                            stratification_attribute
                            if isinstance(method, StratifiedSampleMean)
                            else None
                        ),
                    )
                    records = []
                    for record in history:
                        estimate = record["estimate"]
                        records.append(
                            {
                                "n_labels": estimate.n_labels,
                                "risk": estimate.risk,
                                "method": name,
                                "protected_attribute": attribute,
                                "risk_objective": objective,
                                "repetition": repetition + 1,
                                "seed": seed,
                                **{f"risk {g}": r for g, r in estimate.subgroup_risks.items()},
                                **{
                                    f"n_labels_{g}": n
                                    for g, n in estimate.subgroup_n_labels.items()
                                },
                                "pool_size": len(dataset),
                                **pool_counts,
                                **references,
                            }
                        )
                    frames.append(pd.DataFrame(records))
                    if records:
                        log_metrics(
                            {"risk": records[-1]["risk"], "n_labels": records[-1]["n_labels"]},
                            prefix=f"AT/{attribute}/{name}/{objective}",
                            step=repetition + 1,
                        )
    return frames


def run_benchmark(
    *,
    model,
    dataset,
    auxiliary,
    num_classes,
    device,
    seed,
    repetitions=1,
    budget=1000,
    class_weights=None,
    protected_columns=("gender",),
    include_accuracy=False,
    ase_update_every: int | None = None,
    stratification_attribute: str | None = None,
    pool_size: int | None = None,
    prediction_batch_size: int = 256,
    acquisition_batch_size: int = DEFAULT_ACQUISITION_BATCH_SIZE,
):
    """Share non-stratified trajectories and run stratified methods per attribute.

    All rows report all requested marginal groups. Stratified trajectories differ
    only in their sampling partition. Repetition seeds are paired across trajectories.

    With `pool_size=None` the pool is frozen: every repetition evaluates the whole
    `dataset`, and only the acquisition randomness changes, as in PPAT
    (Ashouritaklimi et al., 2026, app. B.1). With a `pool_size`, every repetition
    draws a new class-stratified pool of that size from `dataset` without replacement,
    as in Kossen et al. (2021, app. C.5). Its draw depends only on `seed` and the
    repetition (see `pool_rng`), and all methods of a repetition share it.

    The model, surrogates and proxy are fixed in both settings, so their outputs on
    `dataset` are computed once (see `prediction_cache`). Each row stores the exact
    risk of its own pool as `reference_risk` columns.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if stratification_attribute is not None and stratification_attribute not in protected_columns:
        raise ValueError("Stratification attribute must be included in protected columns")
    source = prepare_evaluation_pool(dataset, device, list(protected_columns))
    size = len(source) if pool_size is None else int(pool_size)
    if not 1 <= size <= len(source):
        raise ValueError("pool_size must be between 1 and the size of the dataset")
    if not 1 <= budget <= size:
        raise ValueError("budget must be between 1 and the pool size")
    source, model, auxiliary = cache_pool_predictions(
        source, model, auxiliary, batch_size=prediction_batch_size
    )
    source_labels = source.y_pool.cpu().numpy()
    if acquisition_batch_size < 1:
        raise ValueError("acquisition_batch_size must be positive")
    methods = build_method_configs(
        auxiliary, num_classes, acquisition_batch_size, ase_update_every=ase_update_every
    )
    shared_methods = [
        (cls, kwargs) for cls, kwargs in methods if not issubclass(cls, StratifiedSampleMean)
    ]
    stratified_methods = [
        (cls, kwargs) for cls, kwargs in methods if issubclass(cls, StratifiedSampleMean)
    ]
    runs = [(None, shared_methods)] if shared_methods else []
    if stratified_methods:
        sampling_attributes = (
            (stratification_attribute,)
            if stratification_attribute is not None
            else protected_columns
        )
        runs.extend((attribute, stratified_methods) for attribute in sampling_attributes)
    frames = []

    for repetition in tqdm(range(repetitions), desc="Repetitions", unit="repetition", position=0):
        repetition_seed = int((seed + repetition) % 2**32)
        if pool_size is None:
            pool = source
        else:
            positions = class_stratified_positions(source_labels, size, pool_rng(seed, repetition))
            pool = subset_evaluation_pool(source, positions)
        for sampling_attribute, configs in runs:
            frames.extend(
                run_repetition(
                    seed=repetition_seed,
                    device=device,
                    dataset=pool,
                    model=model,
                    method_configs=configs,
                    repetition=repetition,
                    budget=budget,
                    class_weights=class_weights,
                    protected_columns=protected_columns,
                    include_accuracy=include_accuracy,
                    progress_position=1,
                    stratification_attribute=sampling_attribute,
                )
            )

    return pd.concat(frames, ignore_index=True)
