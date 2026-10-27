"""Audit saved results and replay representative checkpoints without retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from macrograph import ROOT
from graph_data import load_development, fold_indices, prepare_arrays
from graph_models import make_model, edge_mode
from train_graphs import predict


def verify(out):
    meta = json.loads((out / "run.json").read_text())
    assert meta["status"] == "complete" and not meta["holdout_loaded"]
    for path, digest in meta["hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, f"Changed source: {path}"
    settings = meta["experiment"]
    scores = pd.read_csv(out / "predictions.csv", parse_dates=["date"])
    keys = ["fold", "model", "seed", "date", "ticker"]
    assert not scores.duplicated(keys).any()
    assert np.isfinite(scores[["prediction", "target"]]).all().all()
    assert scores.date.max() <= pd.Timestamp(meta["data_config"]["validation_end"])
    histories = pd.read_csv(out / "training_history.csv")
    groups = histories.groupby(["fold", "model", "seed"])
    expected = len(settings["folds"]) * len(settings["models"]) * len(settings["seeds"])
    assert len(groups) == expected
    assert (groups.size() == settings["epochs"]).all()
    for fold, g in scores.groupby("fold"):
        pairs = [set(zip(x.date, x.ticker)) for _, x in g.groupby(["model", "seed"])]
        assert all(p == pairs[0] for p in pairs), f"Unmatched samples in {fold}"
    attention = pd.read_csv(out / "attention.csv")
    assert np.isfinite(attention.attention).all()
    sums = attention.groupby(["fold", "model", "seed", "date", "ticker"]).attention.sum()
    np.testing.assert_allclose(sums, 1., atol=1e-6)
    data = load_development(meta["data_config"])
    torch.set_num_threads(2)
    replays = 0
    for fold in settings["folds"]:
        train, evaluate = fold_indices(data, settings["sequence_length"], fold["train_end"], fold["eval_end"])
        for name in ["beta_mlp_magnitude", "gat_magnitude", "temporal_magnitude"]:
            if name not in settings["models"]:
                continue
            seed = settings["seeds"][0]
            base = out / "checkpoints" / fold["name"] / f"{name}_{seed}"
            saved = json.loads(base.with_suffix(".json").read_text())
            arrays, stats = prepare_arrays(data, train, settings["sequence_length"], edge_mode(name), seed)
            assert stats == saved["stats"], "Saved preprocessing differs"
            model = make_model(name, len(data.assets), settings["hidden_dim"], settings["heads"])
            model.load_state_dict(torch.load(base.with_suffix(".pt"), map_location="cpu", weights_only=True))
            predicted, _ = predict(model, arrays, evaluate[:2], settings, edge_mode(name))
            prior = scores[(scores.fold == fold["name"]) & (scores.model == name) & (scores.seed == seed)]
            ordered = prior.pivot(index="date", columns="ticker", values="prediction").reindex(
                index=data.dates[evaluate[:2]], columns=data.assets)
            np.testing.assert_allclose(predicted * stats["target_scale"], ordered, atol=1e-7, rtol=1e-5)
            replays += 1
    print(f"PASS: {expected} fits, identical comparison samples, normalized attention, unchanged inputs/code, {replays} checkpoint replays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    verify(parser.parse_args().run_directory)
