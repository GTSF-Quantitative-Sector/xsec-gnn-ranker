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
    """(timestamp, ticker, score, fwd_return) for every labeled, ranked node."""
    model.eval()
    frames = []
    for d in samples:
        d = d.to(device)
        mask = d["label_mask"].cpu().numpy()
        scores = model(d).cpu().numpy()
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": [d["timestamp"]] * int(mask.sum()),
                    "ticker": [t for t, m in zip(d["node_ids"], mask, strict=True) if m],
                    "score": scores[mask],
                    "fwd_return": d["y"].cpu().numpy()[mask],
                }
            )
        )
    return pl.concat(frames)
