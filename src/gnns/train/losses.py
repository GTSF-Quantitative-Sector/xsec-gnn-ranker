"""LossFactory: config name -> ranking loss over each snapshot's masked nodes."""

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


LOSSES: dict[str, Callable[..., torch.Tensor]] = {"listmle": listmle}
