from collections import Counter
from typing import Callable

import numpy as np
import torch
from scipy.special import log_softmax, softmax
from torch import nn

from active_testing_benchmark.active_testing.interface import (
    ActiveEvaluationMethod,
    Estimate,
    SelectionPlan,
)


class KossenActiveTesting(ActiveEvaluationMethod):
    """Active testing with the LURE risk estimator (Kossen et al., 2021).

    For cross-entropy, the acquisition score is the expected loss
    under a surrogate distribution (Eq. 12 from the paper). For 0-1 loss, it is
    the surrogate probability of disagreeing with the evaluated model (Eq. 14).

    Sampling is without replacement, with a uniform exploration mixture to
    keep every conditional selection probability strictly positive. The
    actual conditional probabilities are retained for the LURE correction
    (Eqs. 3-4). `estimate()` uses the number of observed labels as the LURE 
    horizon, so every prefix has an unbiased risk estimate.

    Parameters
    ----------
    surrogate:
        Optional classifier with outputs for the same classes as `model`.
        Defaults to the evaluated model itself (the paper's inexpensive
        predictive-entropy / model-uncertainty baseline). For better
        acquisitions, pass a distinct, well-calibrated surrogate.
    acquisition:
        'cross_entropy' (default) or 'error_rate'. The `loss_fn` given to
        `run_active_evaluation` must compute the corresponding per-item loss:
        cross-entropy for the first, 0-1 error for the second. The method
        estimates the losses that the runner actually passes to `observe`.
    surrogate_output:
        Whether a distinct surrogate returns logits or class probabilities.
        The *evaluated* model is assumed to return logits, as required by
        the runner's default `nn.CrossEntropyLoss`.
    exploration:
        Fraction of each proposal mixed with uniform sampling (0 < e <= 1).
        Ensures all weights are finite, even for zero acquisition scores.
    surrogate_update:
        Optional callback `(surrogate, X_observed, y_observed) -> None`, called
        after every `observe`. It receives only labels acquired so far and
        may refit the surrogate, using original training data captured by a
        closure if desired. It must never modify the evaluated model.
    batch_size:
        Number to acquire before observing new labels. Each draw within a
        batch uses an updated *without-replacement* proposal, but the
        surrogate is only updated after the full batch is observed.
    """

    def __init__(
        self,
        surrogate: nn.Module | None = None,
        *,
        acquisition: str = "cross_entropy",
        surrogate_output: str = "logits",
        exploration: float = 0.05,
        surrogate_update: Callable[[nn.Module, torch.Tensor, torch.Tensor], None] | None = None,
        seed: int = 0,
        batch_size: int | None = 1,
        prediction_batch_size: int = 256,
    ) -> None:
        if acquisition not in {"cross_entropy", "error_rate"}:
            raise ValueError("acquisition must be 'cross_entropy' or 'error_rate'")
        if surrogate_output not in {"logits", "probabilities"}:
            raise ValueError("surrogate_output must be 'logits' or 'probabilities'")
        if not 0 < exploration <= 1:
            raise ValueError("exploration must be in (0, 1]")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        if surrogate_update is not None and surrogate is None:
            raise ValueError("surrogate_update requires a distinct surrogate")

        self.surrogate = surrogate
        self.acquisition = acquisition
        self.surrogate_output = surrogate_output
        self.exploration = float(exploration)
        self.surrogate_update = surrogate_update
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.prediction_batch_size = prediction_batch_size

    def _predict(self, module: nn.Module, X: torch.Tensor) -> np.ndarray:
        """Predict batched network outputs without accessing any pool labels."""
        module.eval()
        parameter = next(module.parameters(), None)
        device = parameter.device if parameter is not None else X.device
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(X), self.prediction_batch_size):
                inputs = X[start : start + self.prediction_batch_size].to(device)
                outputs = module(inputs)
                if outputs.ndim != 2:
                    raise ValueError("Classifier must return [batch, num_classes]")
                chunks.append(outputs.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def initialize(self, problem: dict) -> dict:
        N = int(problem["N"])
        budget = int(problem["budget"])
        if N < 1 or not 0 <= budget <= N:
            raise ValueError("Require N > 0 and 0 <= budget <= N")
        
        X = problem["X_pool"]
        if len(X) != N or not isinstance(X, torch.Tensor):
            raise ValueError("X_pool must be a tensor containing N inputs")

        model = problem["model"]
        if self.surrogate_update is not None and self.surrogate is model:
            raise ValueError("Cannot update the model being evaluated")

        # The runner's default loss is cross entropy on logits. A 0-1
        # acquisition requires supplying a matching 0-1 loss to the runner.
        if self.acquisition == "error_rate" and isinstance(
            problem.get("loss_fn"), nn.CrossEntropyLoss
        ):
            raise ValueError("error_rate requires a 0-1 loss_fn, not CrossEntropyLoss")

        logits = self._predict(model, X)
        if logits.shape[1] < 2 or not np.isfinite(logits).all():
            raise ValueError("Expected finite logits for at least two classes")

        log_probs = log_softmax(logits, axis=1)

        groups = list(problem.get("protected_attribute", []))
        has_groups = bool(groups)
        if has_groups and len(groups) != N:
            raise ValueError("Protected attributes must match pool length")
        if not has_groups:
            groups = [None] * N  # Expected by run_active_evaluation's history.
        sizes = dict(Counter(groups)) if has_groups else {}

        return {
            "N": N,
            "X_pool": X,
            "model": model,
            "log_probs": log_probs,
            "predicted_classes": np.argmax(logits, axis=1),
            "scores": None,  # Recompute after any surrogate refit.
            "remaining": set(range(N)),
            "remaining_budget": budget,
            "selections": set(),
            "selected_order": [],
            "selection_probs": [],
            "observed_losses": [],
            "observed_labels": [],
            "group_by_index": groups,
            "subgroup_sizes": sizes,
            "subgroup_counts": {g: 0 for g in sizes},
        }

    def _acquisition_scores(self, state: dict) -> np.ndarray:
        """Eq. 12 or 14, evaluated across the full (unlabeled) test pool."""
        if self.surrogate is None:
            surrogate_probs = np.exp(state["log_probs"])
        else:
            predictions = self._predict(self.surrogate, state["X_pool"])
            if predictions.shape != state["log_probs"].shape:
                raise ValueError("Surrogate and evaluated model need the same classes")
            if not np.isfinite(predictions).all():
                raise ValueError("Surrogate returned non-finite predictions")
            if self.surrogate_output == "logits":
                surrogate_probs = softmax(predictions.astype(np.float64), axis=1)
            else:
                if np.any(predictions < 0):
                    raise ValueError("Surrogate probabilities must be nonnegative")
                row_sums = predictions.sum(axis=1, keepdims=True)
                if np.any(row_sums <= 0):
                    raise ValueError("Surrogate probability rows must sum to > 0")
                surrogate_probs = predictions / row_sums

        if self.acquisition == "cross_entropy":
            scores = -np.sum(surrogate_probs * state["log_probs"], axis=1)
        else:
            scores = 1 - surrogate_probs[
                np.arange(state["N"]), state["predicted_classes"]
            ]
        if not np.isfinite(scores).all():
            raise ValueError("Acquisition scores must be finite")
        return np.maximum(scores, 0.0)

    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        count = min(
            self.batch_size or state["remaining_budget"],
            state["remaining_budget"],
            len(state["remaining"]),
        )
        if count < 1:
            raise ValueError("No remaining budget or unlabeled items")
        if state["scores"] is None:
            state["scores"] = self._acquisition_scores(state)

        candidates = np.array(sorted(state["remaining"]), dtype=np.int64)
        selected, selection_probs = [], []
        for _ in range(count):
            scores = state["scores"][candidates]
            score_sum = scores.sum()
            if score_sum > 0:
                q = (1 - self.exploration) * (scores / score_sum)
                q += self.exploration / len(candidates)
            else:
                q = np.full(len(candidates), 1 / len(candidates))

            position = int(self.rng.choice(len(candidates), p=q))
            selected.append(int(candidates[position]))
            selection_probs.append(float(q[position]))
            candidates = np.delete(candidates, position)

        return SelectionPlan(
            indices=selected,
            selection_probabilities=selection_probs,
            inclusion_probabilities=None,  # Adaptive design: no simple marginals.
            protected_attribute_group=[state["group_by_index"][i] for i in selected],
            metadata={
                "design": "kossen_active_testing_lure_without_replacement",
                "acquisition": self.acquisition,
                "exploration": self.exploration,
            },
        )

    def observe(
        self, state: dict, selection: SelectionPlan, labels, losses
    ) -> dict:
        indices = list(selection.indices)
        q = selection.selection_probabilities
        if (
            not indices
            or q is None
            or len(indices) != len(q)
            or len(indices) != len(losses)
            or len(indices) != len(labels)
            or len(indices) != len(set(indices))
            or len(indices) > state["remaining_budget"]
            or not set(indices).issubset(state["remaining"])
        ):
            raise ValueError("Invalid selection, labels, losses, or probabilities")
        q_array = np.asarray(q, dtype=np.float64)
        loss_array = np.asarray(losses, dtype=np.float64)
        if (
            not np.isfinite(q_array).all()
            or np.any(q_array <= 0)
            or np.any(q_array > 1)
            or not np.isfinite(loss_array).all()
        ):
            raise ValueError("Require finite losses and probabilities in (0, 1]")

        state["remaining"].difference_update(indices)
        state["remaining_budget"] -= len(indices)
        state["selections"].update(indices)
        state["selected_order"].extend(indices)
        state["selection_probs"].extend(q_array.tolist())
        state["observed_losses"].extend(loss_array.tolist())
        for index in indices:
            group = state["group_by_index"][index]
            if group in state["subgroup_counts"]:
                state["subgroup_counts"][group] += 1

        if self.surrogate_update is not None:

            state["observed_labels"].extend(
                torch.as_tensor(labels).detach().cpu().reshape(-1).tolist()
            )

            # Use *only* observed labels (and never update the target model).
            X_observed = state["X_pool"][state["selected_order"]]
            y_observed = torch.as_tensor(
                state["observed_labels"], device=X_observed.device, dtype=torch.long
            )
            self.surrogate_update(self.surrogate, X_observed, y_observed)
            state["scores"] = None
        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        n, N = len(state["observed_losses"]), state["N"]
        if n == 0:
            return Estimate(
                risk=float("nan"),
                n_labels=0,
                subgroup_risks={g: None for g in state["subgroup_sizes"]},
                subgroup_n_labels=dict(state["subgroup_counts"]),
            )

        q = np.asarray(state["selection_probs"], dtype=np.float64)
        losses = np.asarray(state["observed_losses"], dtype=np.float64)
        if n == N:
            weights = np.ones(n)  # Full pool: the empirical risk is exact.
        else:
            m = np.arange(1, n + 1, dtype=np.float64)
            # Eq. 4 with M = n (the *current* estimation horizon).
            weights = 1.0 + (N - n) / (N - m) * (
                1.0 / ((N - m + 1.0) * q) - 1.0
            )
        risk = float(np.dot(weights, losses) / n)

        # Apply the same LURE estimator to indicator-weighted losses for
        # each group, then rescale N/N_g. This is design-unbiased even for
        # adaptive, globally sampled proposals. A zero with count=0 is
        # the valid HT-style estimate, not an observed zero group loss.
        groups = state["group_by_index"]
        selected = state["selected_order"]
        weighted_losses = weights * losses
        subgroup_risks = {}
        for group, size in state["subgroup_sizes"].items():
            group_total = sum(
                weighted_losses[j]
                for j, index in enumerate(selected)
                if groups[index] == group
            )
            subgroup_risks[group] = float(N * group_total / (n * size))

        return Estimate(
            risk=risk,
            n_labels=n,
            subgroup_risks=subgroup_risks,
            subgroup_n_labels=dict(state["subgroup_counts"]),
        )