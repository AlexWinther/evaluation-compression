"""Check whether the surrogates and proxy can help before running the benchmark.

Active testing only beats uniform sampling when its acquisition scores track the
true per-item loss (Kossen et al., 2021, Sec. 3). PPAT additionally needs proxy
losses that correlate with the true losses (Ashouritaklimi et al., 2026, Prop. 3.2).
These diagnostics measure both on a labelled holdout that is never the evaluation
pool, so they can be reported without touching evaluation labels.
"""

import math

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def variance_gain(scores: np.ndarray, losses: np.ndarray, exploration: float = 0.05) -> float:
    """Return Var(uniform) / Var(importance sampling) for one draw from the proposal.

    The proposal is proportional to `scores`, mixed with uniform sampling as in
    `KossenActiveTesting`. Values above 1 mean the proposal needs fewer labels than
    uniform sampling for the same variance. Proposal q gives the one-draw estimate
    loss_i / (N q_i), whose variance is sum_i q_i (loss_i / (N q_i) - risk)^2.
    """
    N = len(losses)
    risk = losses.mean()
    q = (1 - exploration) * scores / scores.sum() + exploration / N
    importance_variance = np.sum(q * (losses / (N * q) - risk) ** 2)
    uniform_variance = np.mean((losses - risk) ** 2)
    return float(uniform_variance / importance_variance)


def _finite_or_none(value: float) -> float | None:
    """JSON metadata rejects NaN, which a constant input gives for correlations."""
    return float(value) if math.isfinite(value) else None


@torch.inference_mode()
def surrogate_diagnostics(
    model: nn.Module,
    surrogates: tuple[nn.Module, ...],
    proxy: nn.Module,
    dataset: Dataset,
    device: torch.device | str,
    batch_size: int = 64,
) -> dict[str, float | None]:
    """Measure, for cross-entropy, how well the auxiliary models track the model's losses.

    Returns:
        kossen_score_spearman: rank correlation of the surrogate-expected loss
            (the Kossen acquisition score, Eq. 12) with the true loss.
        kossen_variance_gain: `variance_gain` of the Kossen proposal.
        oracle_variance_gain: `variance_gain` of a proposal proportional to the
            true loss. This is the upper limit for this model and holdout.
        ase_relative_bias: relative error of the ASE estimate before any labels,
            (mean expected loss - true risk) / true risk.
        ppat_proxy_correlation: Pearson correlation of the proxy loss
            L(f(x), argmax proxy(x)) with the true loss.
    """
    model.eval()
    logits, surrogate_probabilities, proxy_predictions, labels = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images = batch["image"].to(device)
        logits.append(model(images).cpu())
        # Softmax turns both logits and log-probabilities (the forests' outputs) into
        # probabilities. Members are averaged as in MeanProbabilityEnsemble.
        member_probabilities = [member(images).softmax(dim=1).cpu() for member in surrogates]
        surrogate_probabilities.append(torch.stack(member_probabilities).mean(dim=0))
        proxy_predictions.append(proxy(images).argmax(dim=1).cpu())
        labels.append(torch.as_tensor(batch["label"]).long())

    # class_losses[i, c] is the cross-entropy of item i if its label were class c.
    class_losses = -torch.cat(logits).double().log_softmax(dim=1).numpy()
    rows = np.arange(len(class_losses))
    losses = class_losses[rows, torch.cat(labels).numpy()]
    expected_losses = (torch.cat(surrogate_probabilities).double().numpy() * class_losses).sum(1)
    proxy_losses = class_losses[rows, torch.cat(proxy_predictions).numpy()]

    risk = losses.mean()
    return {
        "kossen_score_spearman": _finite_or_none(spearmanr(expected_losses, losses)[0]),
        "kossen_variance_gain": variance_gain(expected_losses, losses),
        "oracle_variance_gain": variance_gain(losses, losses),
        "ase_relative_bias": float((expected_losses.mean() - risk) / risk),
        "ppat_proxy_correlation": _finite_or_none(np.corrcoef(proxy_losses, losses)[0, 1]),
    }
