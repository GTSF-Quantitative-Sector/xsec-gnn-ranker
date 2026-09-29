import unittest

import numpy as np
import pandas as pd

from macrograph import FACTORS, build_panel, changes, config, rolling_betas, split_panel


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.c.update(beta_window=30, beta_min_obs=25, macro_lag_sessions=5)
        self.dates = pd.bdate_range("2020-01-01", periods=240)
        rng = np.random.default_rng(42)
        self.prices = pd.DataFrame({"TEST": 100 * np.exp(np.cumsum(rng.normal(0, .01, 240)))}, index=self.dates)
        self.levels = pd.DataFrame(100 + np.cumsum(rng.normal(0, .1, (240, 5)), axis=0), index=self.dates, columns=FACTORS)

    def test_future_changes_do_not_change_features(self):
        original = build_panel(self.prices, self.levels, self.c)
        p, m = self.prices.copy(), self.levels.copy()
        p.iloc[190:] *= 2
        m.iloc[190:] *= 3
        changed = build_panel(p, m, self.c)
        features = [x for x in original if x not in ["target", "entry_date", "label_end"]]
        pd.testing.assert_frame_equal(original.loc[:189, features], changed.loc[:189, features])

    def test_target_has_next_close_execution(self):
        panel = build_panel(self.prices, self.levels, self.c)
        expected = self.prices.iloc[156, 0] / self.prices.iloc[151, 0] - 1
        self.assertAlmostEqual(panel.loc[150, "target"], expected)
        self.assertEqual(panel.loc[150, "label_end"], self.dates[156])
        self.assertTrue(panel.tail(6).target.isna().all())

    def test_split_purges_overlapping_labels(self):
        panel = build_panel(self.prices, self.levels, self.c)
        self.c.update(train_end=str(self.dates[170].date()), validation_end=str(self.dates[220].date()))
        train, validation = split_panel(panel, self.c)
        self.assertLess(train.label_end.max(), validation.date.min())
        self.assertLessEqual(validation.label_end.max(), self.dates[220])

    def test_known_signed_exposure_is_recovered(self):
        rng = np.random.default_rng(1)
        x = pd.DataFrame(rng.normal(size=(100, 5)), columns=FACTORS)
        true_beta = np.array([.1, -.2, .3, -.4, .5])
        y = pd.Series(x.to_numpy() @ true_beta)
        estimated = rolling_betas(y, x, 100, 100, 0).iloc[-1].to_numpy()
        np.testing.assert_allclose(estimated, true_beta * x.std(ddof=0).to_numpy(), atol=1e-12)

    def test_pre_inception_data_does_not_get_betas(self):
        p = self.prices.copy()
        p.iloc[:100] = np.nan
        panel = build_panel(p, self.levels, self.c)
        self.assertTrue(panel.loc[:128, "beta_rates"].isna().all())

    def test_yield_changes_are_basis_points(self):
        delta = changes(self.levels)
        self.assertAlmostEqual(delta.rates.iloc[1], 100 * (self.levels.rates.iloc[1] - self.levels.rates.iloc[0]))


if __name__ == "__main__":
    unittest.main()
