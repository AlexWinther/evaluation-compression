"""Active finite-pool risk estimation runner for FairVision image-classification."""

from collections import Counter
from collections.abc import Callable, Iterable
import statistics

import numpy as np
from scipy.special import softmax
import torch
from torch import nn

from active_testing_benchmark.active_testing.interface import (
    ActiveEvaluationMethod,
    Estimate,
    Problem,
    SelectionPlan,
)

# ---------------------------------------------------------------------------
# Shared helpers for the classification implementations of ASE and PPAT.
# ---------------------------------------------------------------------------


def _predict_logits(
    model: nn.Module, X: torch.Tensor, batch_size: int, num_classes: int
) -> np.ndarray:
    """Return [N, num_classes] logits without access to test labels."""
    if not isinstance(model, nn.Module):
        raise TypeError("Expected a torch.nn.Module")
    model.eval()

    # get model device else device of data
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else X.device

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            result = model(X[start : start + batch_size].to(device))
            if result.ndim != 2 or result.shape[1] != num_classes:
                raise ValueError(f"Classifier must return [batch, {num_classes}]")
            outputs.append(result.detach().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float64)

    if not np.isfinite(result).all():
        raise ValueError("Expected finite logits")

    return result


def _to_probabilities(output: np.ndarray, kind: str) -> np.ndarray:
    """Convert logits or probabilities to a valid last-axis class distribution."""
    values = np.asarray(output, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite surrogate predictions")

    if kind == "logits":
        return softmax(values, axis=1)

    if np.any(values < 0) or np.any(values.sum(axis=-1) <= 0):
        raise ValueError("Surrogate probabilities must be nonnegative with nonzero sums")

    return values / values.sum(axis=-1, keepdims=True)


def _predict_surrogate_members(
    surrogate: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
    X: torch.Tensor,
    batch_size: int,
    output_kind: str,
    num_classes: int,
) -> np.ndarray:
    """
    Posterior/ensemble class probabilities, shape [members, samples, classes].

    A module can return [batch,classes] or [batch,members,classes]. A list
    of modules is treated as a deep ensemble (one member per module).
    """
    modules = list(surrogate) if isinstance(surrogate, (list, tuple)) else [surrogate]
    if not modules or not all(isinstance(module, nn.Module) for module in modules):
        raise ValueError("surrogate must be a Module or a nonempty sequence of Modules")

    member_arrays = []

    for module in modules:
        module.eval()
        parameter = next(module.parameters(), None)
        device = parameter.device if parameter is not None else X.device

        chunks = []
        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                output = module(X[start : start + batch_size].to(device))

                # Add dimension if module is not ensemble
                if output.ndim == 2:
                    output = output.unsqueeze(1)

                if output.ndim != 3 or output.shape[2] != num_classes:
                    raise ValueError(
                        "Surrogate output must be "
                        f"[batch, {num_classes}] or [batch, members, {num_classes}]"
                    )
                chunks.append(output.detach().cpu().numpy())

        values = np.concatenate(chunks, axis=0)  # [N, M, C]
        member_arrays.append(np.transpose(_to_probabilities(values, output_kind), (1, 0, 2)))

    # Concat across model/modules to get ensemble collection
    result = np.concatenate(member_arrays, axis=0)

    if result.ndim != 3 or result.shape[1] != len(X):
        raise ValueError("Inconsistent surrogate predictions")

    return result


def _compute_hypothetical_class_losses(
    logits: np.ndarray, loss_fn: nn.Module, loss_type: str
) -> np.ndarray:
    """
    [N, num_classes] loss L(f(x_i), y), for every hypothetical class y.

    This is NOT based on actual test labels. For cross entropy the provided
    runner loss function determines per-item losses, including class weights
    and label smoothing. Only nn.CrossEntropyLoss is supported currently.
    """
    N, C = logits.shape

    if loss_type == "error_rate":
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            raise ValueError("error_rate requires a 0-1 runner loss_fn")
        return (np.argmax(logits, axis=1)[:, None] != np.arange(C)[None, :]).astype(float)

    if not isinstance(loss_fn, nn.CrossEntropyLoss):
        raise TypeError("cross_entropy requires nn.CrossEntropyLoss as the runner loss_fn")

    values = torch.as_tensor(logits, dtype=torch.float64)
    inputs = values.repeat_interleave(C, dim=0)
    classes = torch.arange(C).repeat(N)
    weights = None if loss_fn.weight is None else loss_fn.weight.detach().cpu().double()

    per_class = torch.nn.functional.cross_entropy(
        inputs,
        classes,
        weight=weights,
        label_smoothing=loss_fn.label_smoothing,
        reduction="none",
    )
    if loss_fn.reduction == "mean" and weights is not None:
        # The runner applies the loss to *one* item at a time. Its weighted
        # cross-entropy denominator is the weight of that single target.
        if torch.any(weights <= 0):
            raise ValueError("Weighted mean cross entropy needs positive class weights")
        per_class = per_class / weights[classes]

    return per_class.reshape(N, C).numpy()


def _initialize_common_state(
    problem: Problem, prediction_batch_size: int, loss_type: str, num_classes: int
) -> dict:
    """
    Helper function to initialize common state for ASE and PPAT, including
    frozen target logits, class-loss table and subgroup metadata
    """
    N = int(problem.N)
    budget = int(problem.budget)
    X = problem.X_pool
    if N < 1 or not 0 <= budget <= N or not isinstance(X, torch.Tensor) or len(X) != N:
        raise ValueError("Require a tensor X_pool of length N > 0 and 0 <= budget <= N")

    mut_logits = _predict_logits(problem.model, X, prediction_batch_size, num_classes)
    hypothetical_losses = _compute_hypothetical_class_losses(
        mut_logits, problem.loss_fn, loss_type
    )

    groups = problem.protected_attribute
    if groups and len(groups) != N:
        raise ValueError("Protected attributes must match pool length")
    sizes = dict(Counter(groups))

    return {
        "N": N,
        "X_pool": X,
        "model": problem.model,
        "target_logits": mut_logits,
        "hypothetical_losses": hypothetical_losses,
        "remaining": set(range(N)),
        "remaining_budget": budget,
        "selections": set(),
        "selected_order": [],
        "observed_labels": [],
        "observed_losses": [],
        "group_by_index": groups,
        "subgroup_sizes": sizes,
        "subgroup_counts": {g: 0 for g in sizes},
        "surrogate_members": None,
    }


def _validate_observation(
    state: dict, selection: SelectionPlan, labels, losses
) -> tuple[list[int], list[int], np.ndarray]:
    """
    Helper function to validate observations before updating method state
    """
    indices = list(selection.indices)
    observed_labels = torch.as_tensor(labels).detach().cpu().reshape(-1).tolist()
    observed_losses = np.asarray(losses, dtype=np.float64).reshape(-1)

    if (
        not indices
        or len(indices) != len(set(indices))
        or len(indices) != len(observed_labels)
        or len(indices) != len(observed_losses)
        or len(indices) > state["remaining_budget"]
        or not set(indices).issubset(state["remaining"])
        or not np.isfinite(observed_losses).all()
    ):
        raise ValueError("Invalid acquired indices, labels, or losses")

    classes = state["hypothetical_losses"].shape[1]
    if any(not isinstance(v, (int, np.integer)) or not 0 <= v < classes for v in observed_labels):
        raise ValueError("Observed labels must be integer IDs within the model's classes")

    expected = state["hypothetical_losses"][indices, observed_labels]
    if not np.allclose(observed_losses, expected, rtol=1e-4, atol=1e-4):
        raise ValueError(
            "Observed losses do not match the declared loss_type and runner loss_fn "
            "for the fixed evaluated model"
        )

    return indices, [int(v) for v in observed_labels], observed_losses


def _record_observation(
    state: dict, indices: list[int], labels: list[int], losses: np.ndarray
) -> dict:
    """Shared state updates across PPAT and ASE"""
    # Remove acquired indices from the unlabelled pool and reduce the available labeling budget
    state["remaining"].difference_update(indices)
    state["remaining_budget"] -= len(indices)
    state["selections"].update(indices)

    # Append chronologically aligned indices, ground-truth labels and observed losses
    state["selected_order"].extend(indices)
    state["observed_labels"].extend(labels)
    state["observed_losses"].extend(losses.tolist())

    # Record how many labels have been acquired within each known protected subgroup.
    for index in indices:
        group = state["group_by_index"][index]
        if group in state["subgroup_counts"]:
            state["subgroup_counts"][group] += 1

    return state


def _compute_lure_weights(N: int, q: list[float], horizon: int) -> np.ndarray:
    """LURE weights at the current horizon"""
    if horizon == N:
        return np.ones(N, dtype=np.float64)

    m = np.arange(1, horizon + 1, dtype=np.float64)
    weights = 1.0 + (N - horizon) / (N - m) * (1.0 / ((N - m + 1.0) * q) - 1.0)

    return weights


class ActiveSurrogateEstimator(ActiveEvaluationMethod):
    """Classification ASE with XWED (Kossen et al., 2022).

    The estimate is the mean surrogate-predicted loss over the entire pool,
    including observed indices. It is not a LURE or unbiased MC estimate.

    XWED (paper Eq. 6) requires >=2 ensemble / posterior member predictions
    to represent epistemic uncertainty. For one deterministic surrogate use
    acquisition='expected_loss' as a declared ablation, not as XWED.

    Surrogate models must be fitted without evaluation-pool labels in advance.
    With surrogate_update, newly acquired labels may be used after observation.
    num_classes is the number of classes expected from the target and surrogates.
    """

    def __init__(
        self,
        surrogate: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
        *,
        num_classes: int,
        acquisition: str = "xwed",
        loss_type: str = "cross_entropy",
        surrogate_output: str = "logits",
        surrogate_update: Callable | None = None,
        update_every: int = 1,
        seed: int = 0,
        batch_size: int | None = 1,
        prediction_batch_size: int = 256,
    ) -> None:
        if acquisition not in {"xwed", "expected_loss"}:
            raise ValueError("acquisition must be 'xwed' or 'expected_loss'")
        if loss_type not in {"cross_entropy", "error_rate"}:
            raise ValueError("loss_type must be 'cross_entropy' or 'error_rate'")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if surrogate_output not in {"logits", "probabilities"}:
            raise ValueError("surrogate_output must be 'logits' or 'probabilities'")
        if (
            update_every < 1
            or prediction_batch_size < 1
            or (batch_size is not None and batch_size < 1)
        ):
            raise ValueError("Batch sizes and update_every must be positive")

        self.surrogate = surrogate
        self.num_classes = num_classes
        self.acquisition = acquisition
        self.loss_type = loss_type
        self.surrogate_output = surrogate_output
        self.surrogate_update = surrogate_update
        self.update_every = update_every
        self.batch_size = batch_size
        self.prediction_batch_size = prediction_batch_size
        self.rng = np.random.default_rng(seed)

    def _refresh_predictions(self, state: dict) -> dict:
        members = _predict_surrogate_members(
            self.surrogate,
            state["X_pool"],
            self.prediction_batch_size,
            self.surrogate_output,
            self.num_classes,
        )
        if members.shape[2] != state["hypothetical_losses"].shape[1]:
            raise ValueError("Surrogate and target model must have matching class order and count")
        if self.acquisition == "xwed" and len(members) < 2:
            raise ValueError("XWED requires at least two distinct ensemble/posterior draws")

        state["surrogate_members"] = members

        return state

    def initialize(self, problem: Problem) -> dict:
        """
        Initialize state for ASE with class-loss matrix, initial surrogate predictions
        """
        modules = self.surrogate if isinstance(self.surrogate, (tuple, list)) else [self.surrogate]
        if self.surrogate_update is not None and any(m is problem.model for m in modules):
            raise ValueError("The surrogate callback may not update the evaluated model")

        state = _initialize_common_state(
            problem, self.prediction_batch_size, self.loss_type, self.num_classes
        )

        state = self._refresh_predictions(state)

        return state

    def _scores(self, state: dict) -> np.ndarray:
        """
        Compute the acquisition score per image for either XWED or surrogate-expected loss
        """
        surrogate_y_preds = state["surrogate_members"]  # [S,N,C]
        surrogate_y_pred_mean = surrogate_y_preds.mean(axis=0)

        hypothetical_losses = state["hypothetical_losses"]  # [N,C]
        if self.acquisition == "expected_loss":
            return (surrogate_y_pred_mean * hypothetical_losses).sum(axis=1)
        # XWED = sum_c L_c (mean_s[p_sc log p_sc] - mean_p_c log mean_p_c)
        # 0*log(0) is interpreted as zero for stability.
        member_p_log_p = np.where(
            surrogate_y_preds > 0,
            surrogate_y_preds * np.log(np.maximum(surrogate_y_preds, 1e-300)),
            0.0,
        ).mean(axis=0)
        mean_p_log_p = surrogate_y_pred_mean * np.log(np.maximum(surrogate_y_pred_mean, 1e-300))
        return np.maximum(((member_p_log_p - mean_p_log_p) * hypothetical_losses).sum(axis=1), 0.0)

    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        """Acquire highest scoring sample with random tie braking"""
        count = min(
            self.batch_size or state["remaining_budget"],
            state["remaining_budget"],
            len(state["remaining"]),
        )
        if count <= 0:
            raise ValueError("No remaining budget or unlabeled items")

        candidates = np.array(sorted(state["remaining"]), dtype=int)
        scores = self._scores(state)[candidates]

        # Deterministic XWED maximisation, randomness from tie-breaking only
        ties = self.rng.permutation(len(candidates))
        ordered = ties[np.argsort(-scores[ties], kind="stable")]

        indices = candidates[ordered[:count]].tolist()

        return SelectionPlan(
            indices=[int(i) for i in indices],
            selection_probabilities=None,
            inclusion_probabilities=None,
            protected_attribute_group=[state["group_by_index"][i] for i in indices],
            metadata={"design": "ase_surrogate_interpolation", "acquisition": self.acquisition},
        )

    def observe(self, state: dict, selection: SelectionPlan, new_labels, new_losses) -> dict:
        # Shared validation step with PPAT
        indices, y, l = _validate_observation(state, selection, new_labels, new_losses)

        # Update state
        previous_count = len(state["selections"])
        state = _record_observation(state, indices, y, l)

        # Update the surrogate if enabled
        if (
            self.surrogate_update is not None
            and len(state["selections"]) // self.update_every > previous_count // self.update_every
        ):
            X_seen = state["X_pool"][state["selected_order"]]
            y_seen = torch.as_tensor(
                state["observed_labels"], device=X_seen.device, dtype=torch.long
            )
            self.surrogate_update(self.surrogate, X_seen, y_seen)
            state = self._refresh_predictions(state)
        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        """Estimate the image's loss using the current surrogate's predictions as a proxy for labels"""

        # Expected losses with current surrogate
        expected_losses = (
            state["surrogate_members"].mean(axis=0) * state["hypothetical_losses"]
        ).sum(axis=1)
        group_risks = {}
        groups = state["group_by_index"]

        # Subgroup risk estimation
        for g in state["subgroup_sizes"]:
            mask = np.array([member == g for member in groups], dtype=bool)
            group_risks[g] = float(expected_losses[mask].mean())

        return Estimate(
            risk=float(expected_losses.mean()),
            n_labels=len(state["selections"]),
            subgroup_risks=group_risks,
            subgroup_n_labels=dict(state["subgroup_counts"]),
        )


class PredictionPoweredActiveTesting(ActiveEvaluationMethod):
    """Classification PPAT (Ashouritaklimi et al., 2026, Eqs. 8, 12-14).

    Supply ONE of proxy (a separate classifier), proxy_labels (its cached hard
    predictions) or proxy_losses (cached L(f(x), proxy(x))). Proxy values must
    not originate from *true test labels*. They are fixed and cheap on all N.

    Surrogate (independent from the proxy) models unknown label conditionals
    for RMS-residual acquisition. When omitted, the evaluated model's softmax
    is used as a weak acquisition surrogate only.

    lambda_value: fixed float gives design-unbiased prefix PPAT estimates.
    'online' uses the paper's plug-in coefficient (Eq. 14), which is generally
    *biased at finite sample sizes* but consistent under its assumptions.
    Fixed lambda=0 recovers LURE with the PPAT RMS-loss proposal.
    num_classes is the number of classes expected from the target, proxy and surrogates.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        proxy: nn.Module | None = None,
        proxy_labels: Iterable[int] | None = None,
        proxy_losses: Iterable[float] | None = None,
        surrogate: nn.Module | list[nn.Module] | tuple[nn.Module, ...] | None = None,
        loss_type: str = "cross_entropy",
        surrogate_output: str = "logits",
        lambda_value: float | str = 1.0,
        lambda_initial: float = 0.5,
        lambda_update_every: int = 100,
        exploration: float = 0.05,
        surrogate_update: Callable | None = None,
        update_every: int = 1,
        seed: int = 0,
        batch_size: int | None = 1,
        prediction_batch_size: int = 256,
        confidence_level: float = 0.95,
    ) -> None:
        if sum(v is not None for v in (proxy, proxy_labels, proxy_losses)) != 1:
            raise ValueError("Provide exactly one: proxy, proxy_labels, proxy_losses")

        if loss_type not in {"cross_entropy", "error_rate"} or surrogate_output not in {
            "logits",
            "probabilities",
        }:
            raise ValueError("Invalid loss_type or surrogate_output")

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        if lambda_value != "online" and (
            not isinstance(lambda_value, (int, float)) or not np.isfinite(lambda_value)
        ):
            raise ValueError("lambda_value must be a finite number or 'online'")

        if not np.isfinite(lambda_initial) or lambda_update_every < 1:
            raise ValueError("lambda_initial must be finite, lambda_update_every >= 1")

        if not (0 < exploration <= 1) or not (0 < confidence_level < 1):
            raise ValueError("exploration and confidence_level must be in (0,1)")

        if (
            update_every < 1
            or prediction_batch_size < 1
            or (batch_size is not None and batch_size < 1)
        ):
            raise ValueError("Batch sizes and update_every must be positive")

        if surrogate_update is not None and surrogate is None:
            raise ValueError("surrogate_update requires a distinct surrogate")

        self.proxy = proxy
        self.proxy_labels = proxy_labels
        self.proxy_losses = proxy_losses
        self.surrogate = surrogate
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.surrogate_output = surrogate_output
        self.lambda_value = lambda_value
        self.lambda_initial = float(lambda_initial)
        self.lambda_update_every = lambda_update_every
        self.exploration = float(exploration)
        self.surrogate_update = surrogate_update
        self.update_every = update_every
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.prediction_batch_size = prediction_batch_size
        self.confidence_level = confidence_level

    def _refresh_predictions(self, state: dict) -> dict:
        # Use target-model softmax as a weak label-distribution surrogate if no
        # independent surrogate was given
        if self.surrogate is None:
            members = _to_probabilities(state["target_logits"], "logits")[None, ...]
        else:
            # Surrogate predictions at members
            members = _predict_surrogate_members(
                self.surrogate,
                state["X_pool"],
                self.prediction_batch_size,
                self.surrogate_output,
                self.num_classes,
            )

        if members.shape[2] != state["hypothetical_losses"].shape[1]:
            raise ValueError(
                "Surrogate must use the same number of classes and class order as evaluated model"
            )

        # Store surrogate predictions for later scoring
        state["surrogate_members"] = members

        return state

    def initialize(self, problem: Problem) -> dict:
        # Stop surrogate refitting from affecting model under test
        if self.surrogate_update is not None:
            modules = (
                self.surrogate if isinstance(self.surrogate, (tuple, list)) else [self.surrogate]
            )
            if any(module is problem.model for module in modules):
                raise ValueError("Cannot update the evaluated target model")

        state = _initialize_common_state(
            problem, self.prediction_batch_size, self.loss_type, self.num_classes
        )
        N, C = state["hypothetical_losses"].shape

        # Use proxy losses if provided
        if self.proxy_losses is not None:
            losses = np.asarray(list(self.proxy_losses), dtype=np.float64)
            if losses.shape != (N,) or not np.isfinite(losses).all():
                raise ValueError("proxy_losses must contain N finite values")
        else:
            # predict proxy labels from a proxy model
            if self.proxy is not None:
                predictions = _predict_logits(
                    self.proxy,
                    state["X_pool"],
                    self.prediction_batch_size,
                    self.num_classes,
                )
                if predictions.shape != state["target_logits"].shape:
                    raise ValueError("Proxy and target must have identical class order and count")

                pseudo_labels = np.argmax(predictions, axis=1)
            else:
                # Or use proxy labels to compute losses
                pseudo_labels = np.asarray(list(self.proxy_labels))
            if (
                pseudo_labels.shape != (N,)
                or not np.issubdtype(pseudo_labels.dtype, np.integer)
                or (pseudo_labels < 0).any()
                or (pseudo_labels >= C).any()
            ):
                raise ValueError("proxy_labels must be N integer class IDs in [0,C)")

            # Compute proxy losses
            losses = state["hypothetical_losses"][np.arange(N), pseudo_labels]

        state["proxy_losses"] = losses
        state["proxy_mean"] = float(losses.mean())

        # Save loss mean for subgroups for subgroup estimations
        group_means = {}
        groups = state["group_by_index"]
        for g in state["subgroup_sizes"]:
            mask = np.array([member == g for member in groups], dtype=bool)
            group_means[g] = float(losses[mask].mean())
        state["proxy_group_means"] = group_means

        state["centered_proxy_losses"] = losses - state["proxy_mean"]
        state["centered_proxy_second_moment"] = float(np.mean(state["centered_proxy_losses"] ** 2))
        state["selection_probs"] = []
        state["lambda"] = (
            self.lambda_initial if self.lambda_value == "online" else float(self.lambda_value)
        )

        # refresh surrogate predictions
        state = self._refresh_predictions(state)

        return state

    def _scores(self, state: dict, candidates: np.ndarray) -> np.ndarray:
        """Compute the rooted mean squared loss of"""
        # Mean surrogate predictions
        p = state["surrogate_members"].mean(axis=0)[candidates]

        # Correction using centered proxy losses
        correction = (state["lambda"] * state["centered_proxy_losses"][candidates])[:, None]

        # Compute losses for all hypothetical classes
        residual_by_class = state["hypothetical_losses"][candidates] - correction

        # Take square root of surrogate-weighted squared residual
        # higher RMS -> higher acquisition weight
        return np.sqrt(np.maximum(np.sum(p * residual_by_class**2, axis=1), 0.0))

    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        """Sample next labels using proxy-corrected RMS loss scores + uniform exploration"""
        count = min(
            self.batch_size or state["remaining_budget"],
            state["remaining_budget"],
            len(state["remaining"]),
        )
        if count < 1:
            raise ValueError("No remaining budget or unlabeled items")

        # define remaining candidates
        candidates = np.array(sorted(state["remaining"]), dtype=int)
        indices, q_selected = [], []
        for _ in range(count):
            scores = self._scores(state, candidates)

            if not np.isfinite(scores).all():
                raise ValueError("Non-finite acquisition scores")

            # Mix RMS scores with exploration
            if scores.sum() > 0:
                q = (1 - self.exploration) * scores / scores.sum() + self.exploration / len(
                    candidates
                )
            else:
                # Full exploration if no positive scores
                q = np.full(len(candidates), 1 / len(candidates), dtype=float)

            # Sample using probability q
            position = int(self.rng.choice(len(candidates), p=q))
            indices.append(int(candidates[position]))
            q_selected.append(float(q[position]))

            # Update candidate set
            candidates = np.delete(candidates, position)

        return SelectionPlan(
            indices=indices,
            selection_probabilities=q_selected,
            inclusion_probabilities=None,
            protected_attribute_group=[state["group_by_index"][i] for i in indices],
            metadata={
                "design": "ppat_rms_residual_without_replacement",
                "lambda_at_acquisition": float(state["lambda"]),
                "exploration": self.exploration,
            },
        )

    def observe(self, state: dict, selection: SelectionPlan, new_labels, new_losses) -> dict:
        """Update the state of PPAT, refit the surrogate and update lambda if enabled"""
        # Shared validation with ASE
        indices, y, l = _validate_observation(state, selection, new_labels, new_losses)

        # Note down the selection probabilities
        q = selection.selection_probabilities
        if (
            q is None
            or len(q) != len(indices)
            or not np.isfinite(q).all()
            or not all(0 < p <= 1 for p in q)
        ):
            raise ValueError("PPAT needs positive conditional selection probabilities for LURE")
        state["selection_probs"].extend(float(p) for p in q)

        # Save the count before and after updating state
        previous_count = len(state["selections"])
        state = _record_observation(state, indices, y, l)
        current_count = len(state["selections"])

        # handle updating of surrogate
        if (
            self.surrogate_update is not None
            and current_count // self.update_every > previous_count // self.update_every
        ):
            X_seen = state["X_pool"][state["selected_order"]]
            y_seen = torch.as_tensor(
                state["observed_labels"], device=X_seen.device, dtype=torch.long
            )

            # refit surrogate with all now seen samples
            self.surrogate_update(self.surrogate, X_seen, y_seen)

            # refresh surrogate predictions on full pool
            state = self._refresh_predictions(state)

        # handle lambda update
        if (
            self.lambda_value == "online"
            and current_count // self.lambda_update_every
            > previous_count // self.lambda_update_every
        ):
            denom = state["centered_proxy_second_moment"]
            if denom > 0:
                # get current LURE weights
                v = _compute_lure_weights(state["N"], state["selection_probs"], current_count)

                new_losses = np.asarray(state["observed_losses"], dtype=float)
                correction = state["centered_proxy_losses"][state["selected_order"]]

                # lambda_hat=mean_m(v_m*ell_m*c_m) / mean_pool(c_i^2).
                state["lambda"] = float(
                    np.dot(v, new_losses * correction) / (current_count * denom)
                )
        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        """Compute proxy-centered/corrected losses and take their LURE weighted mean"""
        n, N = len(state["selected_order"]), state["N"]
        if n == 0:
            return Estimate(
                risk=float("nan"),
                n_labels=0,
                subgroup_risks={g: None for g in state["subgroup_sizes"]},
                subgroup_n_labels=dict(state["subgroup_counts"]),
            )

        lure_weights = _compute_lure_weights(N, state["selection_probs"], n)
        selected = np.asarray(state["selected_order"], dtype=int)
        observed_losses = np.asarray(state["observed_losses"], dtype=np.float64)
        lambda_ = state["lambda"]

        # Compute corrected losses and their LURE mean
        centered_losses = observed_losses - lambda_ * state["centered_proxy_losses"][selected]
        risk = float(np.dot(lure_weights, centered_losses) / n)

        # Compute subgroup risk
        subgroup_risks = {}
        groups = state["group_by_index"]
        # TODO: Currently if no point from subgroup g has been acquired, this indicator-weighted
        # LURE subgroup estimate evaluates to 0. This is not an observed zero loss.
        for g, size in state["subgroup_sizes"].items():
            group_mask = np.array([groups[i] == g for i in selected], dtype=bool)
            # Subgroup controls must be centered at their own
            # subgroup means for the indicator-weighted LURE sum to be unbiased.
            centered_losses_g = observed_losses[group_mask] - lambda_ * (
                state["proxy_losses"][selected[group_mask]] - state["proxy_group_means"][g]
            )
            # subgroup LURE means
            subgroup_risks[g] = float(
                N * np.dot(lure_weights[group_mask], centered_losses_g) / (n * size)
            )

        # CI computation
        standard_error = None
        ci = None
        if n == N:
            standard_error, ci = 0.0, (risk, risk)

        elif n >= 2:
            # Theorem 4.1, Eq. 15/16: asymptotic variance and normal CI from the PPAT paper.
            # Requires sufficient budget and paper's overlap/moment assumptions.
            m = np.arange(1, n + 1, dtype=np.float64)
            gamma = n * (N - n) / ((N - m) * (N - m + 1))
            q = np.asarray(state["selection_probs"], dtype=np.float64)

            A = (
                centered_losses / q + np.concatenate(([0.0], np.cumsum(centered_losses)[:-1]))
            ) / N
            variance = float(np.mean(gamma**2 * (A - risk) ** 2))

            if np.isfinite(variance):
                standard_error = float(np.sqrt(max(variance, 0.0) / n))
                z_critical = statistics.NormalDist().inv_cdf((1 + self.confidence_level) / 2)
                ci = (
                    float(risk - z_critical * standard_error),
                    float(risk + z_critical * standard_error),
                )

        return Estimate(
            risk=risk,
            n_labels=n,
            subgroup_risks=subgroup_risks,
            subgroup_n_labels=dict(state["subgroup_counts"]),
            std_error=standard_error,
            confidence_interval=ci,
        )
