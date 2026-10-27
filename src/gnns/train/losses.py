"""LossFactory: config name -> ranking loss over each snapshot's masked nodes."""

import math
from collections.abc import Callable

import torch


def listmle(
    scores: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """ListMLE per snapshot: -log P(true order of y | scores), averaged per position, then
    over snapshots. Only `mask` nodes count; snapshots in a batch never mix."""
    losses = []
    for g in torch.unique(batch[mask]):
        sel = mask & (batch == g)
        s = scores[sel][torch.argsort(y[sel], descending=True, stable=True)]
        if len(s) >= 2:
            losses.append((torch.logcumsumexp(s.flip(0), 0).flip(0) - s).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0


def listmle_no_info(n: int) -> float:
    """ListMLE when every score is equal: log(n!) / n, the loss of pure guessing. Any model
    whose scores ignore the returns averages at least this."""
    return math.lgamma(n + 1) / n


LOSSES: dict[str, Callable[..., torch.Tensor]] = {"listmle": listmle}
# Per loss: its value at equal scores for a snapshot of n labelled nodes (a chart reference).
NO_INFO: dict[str, Callable[[int], float]] = {"listmle": listmle_no_info}


def no_info_loss(name: str, samples: list) -> float:
    """Average no-information loss over snapshots, as the training loss averages them."""
    counts = [int(d["label_mask"].sum()) for d in samples]
    values = [NO_INFO[name](n) for n in counts if n >= 2]
    return sum(values) / len(values) if values else math.nan
