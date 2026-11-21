"""Trainer: fit one model on one fold's training samples, early-stopping on validation IC."""

import copy
import math

import polars as pl
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnns.eval.metrics import daily_ic, ic_summary


def train_fold(
    model: nn.Module, train: list[Data], val: list[Data], cfg: dict, loss_fn, device: str
) -> tuple[nn.Module, list[dict]]:
    """Adam for up to `max_epochs`; stop after `patience` epochs without a better mean
    validation IC and restore the best weights. Returns (model, per-epoch history)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(
        train,  # type: ignore[arg-type]
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    best_ic, best_state, wait, history = -math.inf, copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y, batch.label_mask, batch.batch)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
        val_ic = ic_summary(daily_ic(predict(model, val, device)))["mean_ic"]
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_ic": val_ic})
        if val_ic > best_ic:
            best_ic, best_state, wait = val_ic, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model: nn.Module, samples: list[Data], device: str) -> pl.DataFrame:
    """(timestamp, ticker, score, fwd_return) for every ranked (target) node at every date
    whose label window is complete; `fwd_return` is null where a node has no label.

    Dates in the last `horizon_bars + 1` bars (no labels yet) give no rows, so a consumer
    never sees a null return there. Metrics drop null returns; the backtest raises if it
    holds one (docs/CONTRACTS.md, "Labels").
    """
    model.eval()
    frames = []
    for d in samples:
        if not d["labeled"]:
            continue
        d = d.to(device)
        keep = d["target_mask"].cpu().numpy()
        labeled = d["label_mask"].cpu().numpy()[keep]
        y = d["y"].cpu().numpy()[keep]
        scores = model(d).cpu().numpy()
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": [d["timestamp"]] * int(keep.sum()),
                    "ticker": [t for t, m in zip(d["node_ids"], keep, strict=True) if m],
                    "score": scores[keep],
                    "fwd_return": [
                        float(v) if ok else None for v, ok in zip(y, labeled, strict=True)
                    ],
                },
                schema_overrides={
                    "ticker": pl.String,
                    "score": pl.Float32,
                    "fwd_return": pl.Float32,
                },
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "America/New_York"),
                "ticker": pl.String,
                "score": pl.Float32,
                "fwd_return": pl.Float32,
            }
        )
    return pl.concat(frames)
