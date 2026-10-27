"""Permutation importance: how much a trained model relies on each input, on unseen dates.

Train once; then score validation snapshots normally and again with one input scrambled,
using the same frozen model. Importance = IC(real) - IC(scrambled), averaged over repeats.
Scrambling stays within each date and each node type, so every scrambled snapshot keeps
that date's exact set of values; only which node holds which value changes.
"""

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from gnns.eval.metrics import daily_ic, ic_summary
from gnns.train.trainer import predict


def _ic(model: nn.Module, samples: list[Data], device: str) -> float:
    return ic_summary(daily_ic(predict(model, samples, device)))["mean_ic"]


def _permutation(node_type: Tensor, gen: torch.Generator) -> Tensor:
    """A random reordering of nodes that only swaps nodes of the same type."""
    perm = torch.arange(len(node_type))
    for t in node_type.unique():
        idx = (node_type == t).nonzero().squeeze(1).cpu()
        perm[idx] = idx[torch.randperm(len(idx), generator=gen)]
    return perm.to(node_type.device)


def permutation_importance(
    model: nn.Module, samples: list[Data], names: list[str], repeats: int, device: str
) -> dict[str, float]:
    """IC drop per model input column (by name) and for "edges" (rewire who links to whom,
    keeping each edge's weight). Negative means scrambling helped: the input hurts out of
    sample, or (near zero) the model does not use it."""
    gen = torch.Generator().manual_seed(0)
    base = _ic(model, samples, device)

    def drop(scramble) -> float:
        drops = []
        for _ in range(repeats):
            scrambled = []
            for d in samples:
                d2 = d.clone()
                scramble(d2, _permutation(d2["node_type"], gen))
                scrambled.append(d2)
            drops.append(base - _ic(model, scrambled, device))
        return sum(drops) / len(drops)

    def column(j: int):
        def scramble(d: Data, perm: Tensor) -> None:
            x = d["x"].clone()
            x[:, j] = x[perm, j]
            d.x = x

        return scramble

    def edges(d: Data, perm: Tensor) -> None:
        d.edge_index = perm[d["edge_index"]]

    out = {name: drop(column(j)) for j, name in enumerate(names)}
    out["edges"] = drop(edges)
    return out
