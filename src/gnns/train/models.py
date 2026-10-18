"""ModelFactory: config name -> ranking model. Every model maps a snapshot to one score per node."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, SAGEConv


class Ranker(nn.Module):
    """`layers` message-passing (or per-node) layers, then a linear scoring head."""

    def __init__(self, layers: Sequence[nn.Module], width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, data: Data) -> Tensor:
        h: Tensor = data.x  # type: ignore[assignment]
        for layer in self.layers:
            h = self.dropout(nn.functional.elu(self.layer_forward(layer, h, data)))
        return self.head(h).squeeze(-1)

    def layer_forward(self, layer: nn.Module, h: Tensor, data: Data) -> Tensor:
        return layer(h)


class GATv2Ranker(Ranker):
    """GATv2 attention; edge weight (plus any model edge columns) enters as edge features."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden: int = 32,
        layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        dims = [in_dim] + [hidden * heads] * layers
        convs = [
            GATv2Conv(dims[i], hidden, heads=heads, edge_dim=edge_dim, dropout=dropout)
            for i in range(layers)
        ]
        super().__init__(convs, hidden * heads, dropout)

    def layer_forward(self, layer: nn.Module, h: Tensor, data: Data) -> Tensor:
        weight: Tensor = data.edge_weight  # type: ignore[assignment]
        attr: Tensor = data.edge_attr  # type: ignore[assignment]
        return layer(h, data.edge_index, torch.cat([weight.unsqueeze(-1), attr], dim=-1))


class SAGERanker(Ranker):
    """GraphSAGE mean aggregation; ignores edge weights (graph baseline)."""

    def __init__(
        self, in_dim: int, edge_dim: int, hidden: int = 32, layers: int = 2, dropout: float = 0.1
    ) -> None:
        dims = [in_dim] + [hidden] * layers
        super().__init__([SAGEConv(dims[i], hidden) for i in range(layers)], hidden, dropout)

    def layer_forward(self, layer: nn.Module, h: Tensor, data: Data) -> Tensor:
        return layer(h, data.edge_index)


class MLPRanker(Ranker):
    """Same depth and width, no edges: the matched no-graph baseline."""

    def __init__(
        self, in_dim: int, edge_dim: int, hidden: int = 32, layers: int = 2, dropout: float = 0.1
    ) -> None:
        dims = [in_dim] + [hidden] * layers
        super().__init__([nn.Linear(dims[i], hidden) for i in range(layers)], hidden, dropout)


MODELS: dict[str, Callable[..., Ranker]] = {
    "gatv2": GATv2Ranker,
    "graphsage": SAGERanker,
    "mlp": MLPRanker,
}


def build_model(spec: dict, sample: Data) -> Ranker:
    """Instantiate `spec["name"]` with the other keys as arguments, sized from one sample."""
    if spec["name"] not in MODELS:
        raise ValueError(f"unknown model {spec['name']!r}; valid: {sorted(MODELS)}")
    params = {k: v for k, v in spec.items() if k != "name"}
    x: Tensor = sample.x  # type: ignore[assignment]
    attr: Tensor = sample.edge_attr  # type: ignore[assignment]
    return MODELS[spec["name"]](x.shape[1], 1 + attr.shape[1], **params)
