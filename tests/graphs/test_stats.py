import numpy as np
import pytest

from gnns.graphs.stats import pairwise_corr


def test_matches_pairwise_corrcoef_with_gaps():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, (60, 8))
    rets[:, 1] += rets[:, 0]  # some real correlation
    rets[rng.random(rets.shape) < 0.15] = np.nan

    corr, n = pairwise_corr(rets)
    for i in range(8):
        for j in range(8):
            ok = np.isfinite(rets[:, i]) & np.isfinite(rets[:, j])
            assert n[i, j] == ok.sum()
            if i != j:
                expected = np.corrcoef(rets[ok, i], rets[ok, j])[0, 1]
                assert corr[i, j] == pytest.approx(expected, abs=1e-10)


def test_undefined_pairs_are_nan():
    rets = np.array([[0.01, 0.0, np.nan], [0.02, 0.0, 0.01], [-0.01, 0.0, np.nan]])
    corr, _ = pairwise_corr(rets)
    assert np.isnan(corr[0, 1])  # column 1 has no variance
    assert np.isnan(corr[0, 2])  # only one shared row
