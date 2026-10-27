"""PyG GATv2 and a factor-typed GATv2+GRU temporal adaptation (not original THGNN)."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv


def bipartite_edges(batch, n_assets, n_factors, device):
    # Disjoint source and target index spaces; no accidental macro/equity self-loops.
    source = torch.arange(n_factors, device=device).repeat(n_assets)
    target = torch.arange(n_assets, device=device).repeat_interleave(n_factors)
    offsets = torch.arange(batch, device=device)[:, None]
    return torch.stack([(source + offsets * n_factors).reshape(-1),
                        (target + offsets * n_assets).reshape(-1)])


class MacroGAT(nn.Module):
    def __init__(self, n_assets, n_factors=5, equity_dim=5, hidden=16, heads=2, temporal=False):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.n_assets, self.n_factors, self.temporal = n_assets, n_factors, temporal
        self.equity_encoder = nn.Linear(equity_dim, hidden)
        self.asset_embedding = nn.Embedding(n_assets, hidden)
        # Each macro factor is an explicit relation type with its own level/change encoder.
        self.factor_encoders = nn.ModuleList([nn.Linear(2, hidden) for _ in range(n_factors)])
        self.gat = GATv2Conv((hidden, hidden), hidden // heads, heads=heads,
                             edge_dim=1, add_self_loops=False, dropout=0.)
        self.norm = nn.LayerNorm(hidden)
        if temporal:
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, equity, macro, edge):
        if not self.temporal:
            equity, macro, edge = equity[:, -1:], macro[:, -1:], edge[:, -1:]
        b, length, n, _ = equity.shape
        k = self.n_factors
        eq = self.equity_encoder(equity) + self.asset_embedding(torch.arange(n, device=equity.device))
        mac = torch.stack([encoder(macro[:, :, i]) for i, encoder in enumerate(self.factor_encoders)], dim=2)
        edges = bipartite_edges(b * length, n, k, equity.device)
        messages, (_, alpha) = self.gat((mac.reshape(-1, mac.shape[-1]), eq.reshape(-1, eq.shape[-1])),
                                      edges, edge.reshape(-1, 1), return_attention_weights=True)
        h = F.elu(self.norm(eq + messages.reshape_as(eq)))
        if self.temporal:
            seq = h.permute(0, 2, 1, 3).reshape(b * n, length, -1)
            _, last = self.gru(seq)
            h = last[-1].reshape(b, n, -1)
        else:
            h = h[:, -1]
        # Last-snapshot attention is descriptive, not temporal attribution of the GRU's output.
        attention = alpha.reshape(b, length, n, k, -1)[:, -1].mean(-1)
        return self.head(h).squeeze(-1), attention


class TabularNet(nn.Module):
    def __init__(self, n_assets, n_factors=5, equity_dim=5, hidden=16, inputs="beta", temporal=False):
        super().__init__()
        self.inputs, self.temporal = inputs, temporal
        self.asset_embedding = nn.Embedding(n_assets, 4)
        dim = equity_dim + 4
        if inputs != "equity":
            dim += 2 * n_factors
        if inputs == "beta":
            dim += 2 * n_factors  # Edges and their interaction with scaled macro changes.
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        if temporal:
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, equity, macro, edge):
        if not self.temporal:
            equity, macro, edge = equity[:, -1:], macro[:, -1:], edge[:, -1:]
        b, length, n, _ = equity.shape
        identity = self.asset_embedding(torch.arange(n, device=equity.device)).expand(b, length, -1, -1)
        parts = [equity, identity]
        if self.inputs != "equity":
            parts.append(macro.flatten(-2).unsqueeze(2).expand(-1, -1, n, -1))
        if self.inputs == "beta":
            parts.extend([edge, edge * macro[..., 1].unsqueeze(2)])
        h = self.encoder(torch.cat(parts, dim=-1))
        if self.temporal:
            _, last = self.gru(h.permute(0, 2, 1, 3).reshape(b * n, length, -1))
            h = last[-1].reshape(b, n, -1)
        else:
            h = h[:, -1]
        return self.head(h).squeeze(-1), None


def make_model(name, n_assets, hidden, heads):
    if name.startswith("gat_") or name.startswith("temporal_"):
        return MacroGAT(n_assets, hidden=hidden, heads=heads, temporal=name.startswith("temporal_"))
    inputs = "equity" if name == "equity_mlp" else "macro" if name == "macro_mlp" else "beta"
    return TabularNet(n_assets, hidden=hidden, inputs=inputs, temporal=name == "sequence_mlp")


def edge_mode(name):
    for mode in ["signed", "constant", "shuffled", "frozen"]:
        if name.endswith("_" + mode):
            return mode
    return "magnitude"
