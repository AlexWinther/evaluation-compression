"""Predictions of fixed models, computed once per source split and looked up by index.

The benchmark evaluates one fixed model under test, with fixed surrogates and a fixed
proxy, on many evaluation pools. Every pool is a subset of one source split (or the
whole split when the pool is frozen), so each image needs to pass through each network
only once. `cache_pool_predictions` computes those outputs and returns an index pool,
whose `X_pool` holds source indices instead of images, together with modules that look
the outputs up by index. The active-testing methods are unchanged: they still call
`module(X_pool[batch])`, which now indexes the cache.

Random forest surrogates can be refitted during acquisition (ASE). For them the cache
holds the frozen image features instead of the forest's outputs, so a refitted forest
still predicts from the same features it would have extracted from the images.

A `LabelOracle` control cannot predict from images. Its cached outputs are computed
from the pool labels and the cached outputs of the model under test.
"""

from copy import copy, deepcopy
from dataclasses import replace

import torch
from torch import nn

from active_testing_benchmark.active_testing.runner import EvaluationPool
from active_testing_benchmark.active_testing.surrogate import LabelOracle, RandomForestModule


class CachedOutputs(nn.Module):
    """Return precomputed outputs for a batch of source indices.

    Outputs live on the CPU and are returned on the device of the indices, so that
    callers see the same devices as with the original module.
    """

    def __init__(self, outputs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("outputs", outputs.detach().cpu())

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("CachedOutputs expects a 1-D tensor of source indices")
        return self.outputs[indices.to(self.outputs.device)].to(indices.device)


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    """Return the device of a module's first parameter or buffer, else `fallback`."""
    for tensor in (*module.parameters(), *module.buffers()):
        return tensor.device
    return fallback


@torch.inference_mode()
def predict_in_batches(module: nn.Module, X: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a module's outputs for all inputs, in order, on the CPU."""
    module.eval()
    device = _module_device(module, X.device)
    outputs = [
        module(X[start : start + batch_size].to(device)).detach().cpu()
        for start in range(0, len(X), batch_size)
    ]
    return torch.cat(outputs)


def forest_with_feature_lookup(
    forest: RandomForestModule, lookup: CachedOutputs
) -> RandomForestModule:
    """Copy a fitted forest, replacing its feature extractor by a cached-feature lookup.

    The deep copy keeps the fitted trees, the development data and the temperature.
    Passing the extractor in the memo makes the copy use `lookup` in its place, so the
    backbone is neither copied nor changed.
    """
    return deepcopy(forest, memo={id(forest.feature_extractor): lookup})


def cache_pool_predictions(
    pool: EvaluationPool,
    model: nn.Module,
    auxiliary,
    batch_size: int = 256,
) -> tuple[EvaluationPool, CachedOutputs, object]:
    """Compute fixed-model outputs on a pool once and return index-based replacements.

    Returns the index pool, a lookup for the model under test, and a shallow copy of
    `auxiliary` whose `surrogates` and `proxy` read from the cache. Forest surrogates
    share one feature lookup per distinct feature extractor. The original model and
    auxiliary models are not modified, apart from being put in evaluation mode.
    """
    X = pool.X_pool
    index_pool = replace(pool, X_pool=torch.arange(len(X), device=X.device))
    cached_model = CachedOutputs(predict_in_batches(model, X, batch_size))

    def oracle_lookup(oracle: LabelOracle) -> CachedOutputs:
        log_probabilities = oracle.log_probabilities(
            pool.y_pool.cpu().numpy(), cached_model.outputs.double().numpy()
        )
        return CachedOutputs(torch.from_numpy(log_probabilities))

    cached_auxiliary = copy(auxiliary)
    surrogates = getattr(auxiliary, "surrogates", None)
    if surrogates is not None:
        feature_lookups: dict[int, CachedOutputs] = {}
        cached_surrogates = []
        for surrogate in surrogates:
            if isinstance(surrogate, LabelOracle):
                cached_surrogates.append(oracle_lookup(surrogate))
            elif isinstance(surrogate, RandomForestModule):
                key = id(surrogate.feature_extractor)
                if key not in feature_lookups:
                    features = predict_in_batches(surrogate.feature_extractor, X, batch_size)
                    feature_lookups[key] = CachedOutputs(features.flatten(1))
                cached_surrogates.append(
                    forest_with_feature_lookup(surrogate, feature_lookups[key])
                )
            else:
                cached_surrogates.append(
                    CachedOutputs(predict_in_batches(surrogate, X, batch_size))
                )
        cached_auxiliary.surrogates = tuple(cached_surrogates)
    proxy = getattr(auxiliary, "proxy", None)
    if isinstance(proxy, LabelOracle):
        cached_auxiliary.proxy = oracle_lookup(proxy)
    elif proxy is not None:
        cached_auxiliary.proxy = CachedOutputs(predict_in_batches(proxy, X, batch_size))
    return index_pool, cached_model, cached_auxiliary
