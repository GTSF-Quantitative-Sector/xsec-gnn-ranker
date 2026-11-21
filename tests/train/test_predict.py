"""predict (#3): every ranked node at dates with a complete label window, null fwd_return
where a node has no label; no rows at trailing dates."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from gnns.eval.metrics import daily_ic, top_k_returns
from gnns.train.trainer import predict

T = datetime(2024, 1, 2, 16, tzinfo=ZoneInfo("America/New_York"))


class FirstFeature(nn.Module):
    def forward(self, d):
        return d.x[:, 0]


def _sample(labels: list[float | None], target: list[bool], labeled: bool, day: int = 2) -> Data:
    n = len(labels)
    y = torch.tensor([0.0 if v is None else v for v in labels])
    has = torch.tensor([v is not None for v in labels])
    tm = torch.tensor(target)
    return Data(
        x=torch.arange(n, dtype=torch.float32).reshape(-1, 1),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        y=y,
        label_mask=has & tm,
        target_mask=tm,
        node_ids=[f"S{i}" for i in range(n)],
        timestamp=T.replace(day=day),
        labeled=labeled,
    )


def test_unlabeled_target_nodes_are_kept_with_a_null_return():
    s = _sample([0.1, None, 0.3, 0.2], [True, True, True, False], labeled=True)
    out = predict(FirstFeature(), [s], "cpu")
    assert out["ticker"].to_list() == ["S0", "S1", "S2"]  # S3 is not ranked
    assert out["fwd_return"].to_list()[1] is None
    assert out["score"].to_list() == [0.0, 1.0, 2.0]


def test_dates_without_a_complete_label_window_give_no_rows():
    ok = _sample([0.1, 0.2], [True, True], labeled=True, day=2)
    trailing = _sample([None, None], [True, True], labeled=False, day=3)
    out = predict(FirstFeature(), [ok, trailing], "cpu")
    assert out["timestamp"].unique().to_list() == [ok.timestamp]
    assert predict(FirstFeature(), [trailing], "cpu").is_empty()


def test_metrics_ignore_the_null_returns():
    s = _sample([0.1, None, 0.3], [True, True, True], labeled=True)
    out = predict(FirstFeature(), [s], "cpu")
    assert daily_ic(out)["ic"].item() == 1.0  # ranks of S0, S2 only
    assert top_k_returns(out, 1)["ret"].item() == pytest.approx(0.3)
