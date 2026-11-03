"""Benchmark method configuration and repeated finite-pool evaluation."""

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


def build_method_configs(auxiliary, num_classes: int):
    """Construct the benchmark's six existing method configurations."""
    method_configs = [
        (UniformSampleMean, {"batch_size": 10}),
        (StratifiedSampleMean, {"allocation": "balanced", "batch_size": 10}),
        (StratifiedSampleMean, {"allocation": "proportional", "batch_size": 10}),
        (
            KossenActiveTesting,
            {
                "surrogate": auxiliary.surrogates[0],
                "num_classes": num_classes,
                "batch_size": 10,
            },
        ),
        (
            ActiveSurrogateEstimator,
            {
                "surrogate": list(auxiliary.surrogates),
                "num_classes": num_classes,
                "acquisition": "xwed",
                "loss_type": "cross_entropy",
                "surrogate_output": "logits",
                "batch_size": 10,
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
                "batch_size": 10,
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
):
    """Share one materialized pool across objectives, using fresh method instances."""
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
                    history = run_active_evaluation(
                        method=method,
                        model=model,
                        dataset=dataset,
                        device=device,
                        loss_fn=loss_fn,
                        protected_column=list(protected_columns),
                        budget=budget,
                        progress_position=progress_position + 2,
                    )
                    records = []
                    for record in history:
                        estimate = record["estimate"]
                        records.append(
                            {
                                "n_labels": estimate.n_labels,
                                "risk": estimate.risk,
                                "method": name,
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
):
    """Return curves for all repetitions, loading the fixed evaluation pool only once."""
    if repetitions < 1 or not 1 <= budget <= len(dataset):
        raise ValueError("repetitions must be positive and budget must be within the pool size")
    dataset = prepare_evaluation_pool(dataset, device, list(protected_columns))
    methods = build_method_configs(auxiliary, num_classes)
    frames = []

    for repetition in tqdm(range(repetitions), desc="Repetitions", unit="repetition", position=0):
        repetition_seed = int((seed + repetition) % 2**32)
        frames.extend(
            run_repetition(
                seed=repetition_seed,
                device=device,
                dataset=dataset,
                model=model,
                method_configs=methods,
                repetition=repetition,
                budget=budget,
                class_weights=class_weights,
                protected_columns=protected_columns,
                include_accuracy=include_accuracy,
                progress_position=1,
            )
        )

    return pd.concat(frames, ignore_index=True)
