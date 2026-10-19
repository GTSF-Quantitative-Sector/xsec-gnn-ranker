import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnns.train.losses import listmle
from gnns.train.models import MODELS, build_model


def _sample(n: int = 4) -> Data:
    ei = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j]).T
    return Data(
        x=torch.randn(n, 3),
        edge_index=ei,
        edge_weight=torch.rand(ei.shape[1]),
        edge_attr=torch.zeros(ei.shape[1], 0),
        y=torch.randn(n),
        label_mask=torch.ones(n, dtype=torch.bool),
    )


def test_listmle_matches_hand_calculation_and_prefers_correct_order():
    y, mask, batch = (
        torch.tensor([3.0, 2.0, 1.0]),
        torch.ones(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
    )
    s = torch.tensor([1.0, 0.5, 0.0])
    # -log P(order) = sum_i log(sum_{j>=i} e^{s_j}) - s_i, averaged over 3 positions.
    expected = (
        sum(math.log(sum(math.exp(v) for v in s[i:].tolist())) - s[i].item() for i in range(3)) / 3
    )
    assert listmle(s, y, mask, batch).item() == pytest.approx(expected)
    assert listmle(s, y, mask, batch) < listmle(s.flip(0), y, mask, batch)


def test_listmle_ignores_masked_nodes_and_keeps_snapshots_separate():
    s, y = torch.tensor([1.0, 0.0, 9.0, 1.0, 0.0]), torch.tensor([2.0, 1.0, -5.0, 2.0, 1.0])
    mask = torch.tensor([True, True, False, True, True])
    batch = torch.tensor([0, 0, 0, 1, 1])
    one = listmle(s[:2], y[:2], mask[:2], torch.zeros(2, dtype=torch.long))
    assert listmle(s, y, mask, batch).item() == pytest.approx(one.item())


@pytest.mark.parametrize("name", sorted(MODELS))
def test_models_score_every_node_and_learn(name):
    torch.manual_seed(0)
    batch = Batch.from_data_list([_sample(), _sample()])
    model = build_model({"name": name, "hidden": 8}, _sample())
    scores = model(batch)
    assert scores.shape == (8,)
    listmle(scores, batch["y"], batch["label_mask"], batch["batch"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_mlp_ignores_edges():
    torch.manual_seed(0)
    d = _sample()
    model = build_model({"name": "mlp"}, d).eval()
    cut = d.clone()
    cut.edge_index = cut["edge_index"][:, :0]
    assert torch.equal(model(d), model(cut))


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"name": "gcn"}, _sample())
