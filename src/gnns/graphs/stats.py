"""Vectorized statistics shared by graph builders."""

import numpy as np


def pairwise_corr(rets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of every column pair over the rows where both are finite.

    `rets` is [bars, nodes] with NaN gaps. Returns (corr [N, N], n_obs [N, N]); corr is NaN
    where undefined (fewer than 2 shared rows, or no variance).
    """
    m = np.isfinite(rets).astype(np.float64)
    x = np.where(m > 0, rets, 0.0)
    n = m.T @ m
    sx = x.T @ m  # sx[i, j]: sum of column i over rows where column j is present
    sxx = (x * x).T @ m
    sxy = x.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        var = sxx - sx**2 / n
        corr = np.clip(cov / np.sqrt(var * var.T), -1.0, 1.0)
    corr[n < 2] = np.nan
    return corr, n
