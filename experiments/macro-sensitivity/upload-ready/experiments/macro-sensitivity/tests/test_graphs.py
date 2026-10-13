import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from graph_data import GraphPanel, eligible_ends, fold_indices, prepare_arrays, sequence_batch, edge_values, load_development
from graph_evaluation import block_interval
from graph_models import MacroGAT, TabularNet, bipartite_edges


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.eq = torch.randn(2, 4, 3, 5)
        self.macro = torch.randn(2, 4, 5, 2)
        self.edge = torch.rand(2, 4, 3, 5)

    def test_direction_and_disjoint_batches(self):
        edge = bipartite_edges(2, 3, 5, "cpu")
        self.assertEqual(tuple(edge.shape), (2, 30))
        self.assertTrue(torch.equal(edge[0] // 5, edge[1] // 3))
        self.assertTrue(torch.equal(torch.bincount(edge[1]), torch.full((6,), 5)))

    def test_attention_normalizes_and_graphs_are_isolated(self):
        for temporal in [False, True]:
            model = MacroGAT(3, temporal=temporal).eval()
            p, a = model(self.eq, self.macro, self.edge)
            self.assertEqual(tuple(p.shape), (2, 3))
            torch.testing.assert_close(a.sum(-1), torch.ones(2, 3))
            altered = self.macro.clone()
            altered[1] += 100
            p2, _ = model(self.eq, altered, self.edge)
            torch.testing.assert_close(p[0], p2[0])

    def test_temporal_uses_past_snapshot_does_not(self):
        altered = self.eq.clone()
        altered[:, :-1] += 20
        snapshot = MacroGAT(3).eval()
        temporal = MacroGAT(3, temporal=True).eval()
        torch.testing.assert_close(snapshot(self.eq, self.macro, self.edge)[0], snapshot(altered, self.macro, self.edge)[0])
        self.assertFalse(torch.allclose(temporal(self.eq, self.macro, self.edge)[0], temporal(altered, self.macro, self.edge)[0]))

    def test_edges_and_gru_receive_gradients(self):
        model = MacroGAT(3, temporal=True)
        edge = self.edge.clone().requires_grad_(True)
        model(self.eq, self.macro, edge)[0].square().mean().backward()
        self.assertGreater(edge.grad.abs().sum().item(), 0)
        self.assertGreater(model.gru.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_model_can_fit_synthetic_target(self):
        model = TabularNet(3, hidden=16, inputs="equity")
        target = self.eq[:, -1, :, 0]
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        initial = ((model(self.eq, self.macro, self.edge)[0] - target) ** 2).mean().item()
        for _ in range(80):
            opt.zero_grad()
            loss = ((model(self.eq, self.macro, self.edge)[0] - target) ** 2).mean()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), initial * .1)


class GraphDataTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2024-01-01", periods=100)
        end = pd.Series(dates).shift(-6).to_numpy()
        self.data = GraphPanel(dates, ["A", "B", "C"], rng.normal(size=(100, 3, 5)),
                               rng.normal(size=(100, 5, 2)), rng.normal(size=(100, 3, 5)),
                               rng.normal(size=(100, 3)), np.tile(end[:, None], (1, 3)), np.ones(100, dtype=bool))

    def test_missing_session_is_not_compressed_away(self):
        self.data.feature_valid[50] = False
        ends = eligible_ends(self.data, 4)
        self.assertFalse(any(i in ends for i in range(50, 54)))
        self.assertIn(54, ends)

    def test_future_data_cannot_change_training_scalers_or_sequences(self):
        train = np.arange(4, 40)
        a, stats_a = prepare_arrays(self.data, train, 4, "magnitude", 7)
        changed = copy.deepcopy(self.data)
        changed.equity[40:] *= 100
        changed.macro[40:] += 100
        changed.beta[40:] *= 100
        changed.target[40:] += 100
        b, stats_b = prepare_arrays(changed, train, 4, "magnitude", 7)
        self.assertEqual(stats_a, stats_b)
        ba, bb = sequence_batch(a, train, 4), sequence_batch(b, train, 4)
        for key in ba:
            np.testing.assert_array_equal(ba[key], bb[key])

    def test_signed_and_magnitude_controls(self):
        beta = self.data.beta
        shuffled = edge_values(beta, "shuffled", 7)
        np.testing.assert_allclose(np.sort(shuffled, axis=1), np.sort(np.abs(beta), axis=1))
        self.assertFalse(np.allclose(shuffled, abs(beta)))
        np.testing.assert_array_equal(edge_values(beta, "constant", 7), np.ones_like(beta))
        arrays, _ = prepare_arrays(self.data, np.arange(4, 40), 4, "magnitude", 7)
        self.assertTrue((arrays["edge"] >= 0).all())
        signed, _ = prepare_arrays(self.data, np.arange(4, 40), 4, "signed", 7)
        np.testing.assert_array_equal(np.sign(signed["edge"]), np.sign(beta))

    def test_sequence_end_and_frozen_control(self):
        arrays, _ = prepare_arrays(self.data, np.arange(4, 40), 4, "magnitude", 7)
        batch = sequence_batch(arrays, np.array([50]), 4)
        np.testing.assert_array_equal(batch["equity"][0], arrays["equity"][47:51])
        frozen = sequence_batch(arrays, np.array([50]), 4, frozen=True)
        np.testing.assert_array_equal(frozen["edge"][0], np.repeat(arrays["edge"][47:48], 4, axis=0))

    def test_fold_labels_cannot_overlap_evaluation(self):
        boundary = str(self.data.dates[60].date())
        train, evaluate = fold_indices(self.data, 4, boundary, str(self.data.dates[90].date()))
        self.assertLess(self.data.label_end[train].max(), self.data.dates[evaluate[0]].to_datetime64())
        self.assertLessEqual(self.data.label_end[evaluate].max(), self.data.dates[90].to_datetime64())

    def test_identical_paired_metrics_have_zero_interval(self):
        frame = pd.DataFrame({"fold": ["a"] * 20, "date": self.data.dates[:20], "difference": np.zeros(20)})
        self.assertEqual(block_interval(frame, "difference", repeats=20), (0., 0.))

    def test_raw_holdout_values_are_excluded_before_features(self):
        from macrograph import config, FACTORS
        c = config()
        dates = pd.bdate_range("2020-01-01", periods=350)
        rng = np.random.default_rng(9)
        source = {name: pd.Series(100 + np.cumsum(rng.normal(0, .1, len(dates))), index=dates, name=name)
                  for name in c["assets"] + list(c["fred"]) + [c["gold_proxy"]]}
        c.update(validation_end=str(dates[300].date()), beta_window=30, beta_min_obs=25)
        with patch("graph_data.read_series", side_effect=lambda name: source[name]):
            before = load_development(c)
        for series in source.values():
            series.iloc[301:] *= 100
        with patch("graph_data.read_series", side_effect=lambda name: source[name]):
            after = load_development(c)
        self.assertEqual(before.dates[-1], dates[300])
        for name in ["equity", "macro", "beta", "target"]:
            np.testing.assert_allclose(getattr(before, name), getattr(after, name), equal_nan=True)
        self.assertTrue(np.isnan(before.target[-6:]).all())


if __name__ == "__main__":
    unittest.main()
