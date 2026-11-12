"""Benchmark method configuration and repeated finite-pool evaluation."""

from copy import deepcopy

import mlflow
import pandas as pd
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
from active_testing_benchmark.active_testing.runner import prepare_evaluation_pool
from active_testing_benchmark.active_testing.surrogate import (
    MeanProbabilityEnsemble,
    update_random_forest_surrogates,
)
from active_testing_benchmark.modeling.registry import log_metrics


def build_method_configs(
    auxiliary, num_classes: int, batch_size: int = 25, *, ase_update_every: int = 50
):
    """Construct the benchmark's six existing method configurations."""
    if ase_update_every <= 1:
        raise ValueError("ase_update_every must be positive")
    method_configs = [
        (UniformSampleMean, {"batch_size": batch_size}),
        (StratifiedSampleMean, {"allocation": "balanced", "batch_size": batch_size}),
        (StratifiedSampleMean, {"allocation": "proportional", "batch_size": batch_size}),
        (
            KossenActiveTesting,
            {
                "surrogate": MeanProbabilityEnsemble(auxiliary.surrogates),
                "num_classes": num_classes,
                "batch_size": batch_size,
            },
        ),
        (
            ActiveSurrogateEstimator,
            {
                "surrogate": list(auxiliary.surrogates),
                "num_classes": num_classes,
                "acquisition": "xwed",
                "surrogate_update": update_random_forest_surrogates,
                "update_every": ase_update_every,
                "loss_type": "cross_entropy",
                "surrogate_output": "logits",
                "batch_size": batch_size,
            },
        ),
        (
            PredictionPoweredActiveTesting,
            {
                "num_classes": num_classes,
                "proxy": auxiliary.proxy,
                "surrogate": list(auxiliary.surrogates),
                "lambda_value": 0.5,
                "exploration": 0.05,
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
    """Share one materialized pool across objectives, using fresh method instances."""
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
            with tqdm(
                method_configs,
                desc="Methods",
                unit="method",
                position=progress_position + 1,
                leave=False,
            ) as method_progress:
                for method_class, kwargs in method_progress:
                    method_kwargs = dict(kwargs)
                    if method_class is ActiveSurrogateEstimator:
                        # ASE refits must not affect other objectives, methods or repetitions.
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
    ase_update_every: int = 50,
    stratification_attribute: str | None = None,
):
    """Share non-stratified trajectories and run stratified methods per attribute.

    All rows report all requested marginal groups. Stratified trajectories differ
    only in their sampling partition. Repetition seeds are paired across trajectories.
    """
    if repetitions < 1 or not 1 <= budget <= len(dataset):
        raise ValueError("repetitions must be positive and budget must be within the pool size")
    if stratification_attribute is not None and stratification_attribute not in protected_columns:
        raise ValueError("Stratification attribute must be included in protected columns")
    dataset = prepare_evaluation_pool(dataset, device, list(protected_columns))
    methods = build_method_configs(auxiliary, num_classes, ase_update_every=ase_update_every)
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
        for sampling_attribute, configs in runs:
            frames.extend(
                run_repetition(
                    seed=repetition_seed,
                    device=device,
                    dataset=dataset,
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
