"""Research prototype. Latest-vintage data; not a point-in-time trading backtest."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FACTORS = ["rates", "credit", "vix", "oil", "gold"]
EQUITY = ["momentum_21", "momentum_63", "momentum_126", "volatility_21", "volatility_63"]


def config():
    return json.loads((ROOT / "config.json").read_text())


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=45) as response:
        return response.read()


def download(c):
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = []
    series = [(s, "yahoo", s) for s in c["assets"] + [c["gold_proxy"]]]
    series += [(name, "fred", code) for name, code in c["fred"].items()]
    for name, provider, code in series:
        path = raw / f"{name}.csv"
        provenance = raw / f"{name}.json"
        if path.exists() and provenance.exists():
            entry = json.loads(provenance.read_text())
            if entry["requested_start"] == c["start"] and entry["requested_end"] == c["end"]:
                manifest.append(entry)
                print(f"Cached {name}", flush=True)
                continue
        try:
            if provider == "yahoo":
                start = int(pd.Timestamp(c["start"], tz="UTC").timestamp())
                end = int((pd.Timestamp(c["end"], tz="UTC") + pd.Timedelta(days=1)).timestamp())
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?" + urlencode(
                    {"period1": start, "period2": end, "interval": "1d"})
                payload = fetch(url)
                obj = json.loads(payload)["chart"]["result"][0]
                dates = pd.to_datetime(obj["timestamp"], unit="s", utc=True).tz_convert(
                    "America/New_York").tz_localize(None).normalize()
                values = obj["indicators"]["adjclose"][0]["adjclose"]
                df = pd.DataFrame({"date": dates, "value": values})
            else:
                url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode(
                    {"id": code, "cosd": c["start"], "coed": c["end"]})
                payload = fetch(url)
                df = pd.read_csv(io.BytesIO(payload), na_values=".")
                df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().sort_values("date")
            df = df[df.date.between(c["start"], c["end"])]
            if df.empty or df.date.duplicated().any():
                raise ValueError("Empty or duplicate-date source")
            df.to_csv(path, index=False)
            entry = {"name": name, "provider": provider, "series": code, "url": url,
                     "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                     "requested_start": c["start"], "requested_end": c["end"],
                     "first": str(df.date.min().date()), "last": str(df.date.max().date()),
                     "rows": len(df), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            provenance.write_text(json.dumps(entry, indent=2))
            manifest.append(entry)
            print(f"Downloaded {name}: {entry['rows']} rows, {entry['first']} to {entry['last']}", flush=True)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"FAILED {failures[-1]}", flush=True)
    (raw / "manifest.json").write_text(json.dumps({"sources": manifest, "failures": failures}, indent=2))
    if failures:
        raise RuntimeError("Some downloads failed; successful sources are cached. " + "; ".join(failures))


def read_series(name):
    df = pd.read_csv(ROOT / "data" / "raw" / f"{name}.csv", parse_dates=["date"])
    if df.date.duplicated().any():
        raise ValueError(f"Duplicate dates: {name}")
    return df.set_index("date").value.sort_index().rename(name)


def changes(levels):
    out = levels.diff()
    out[["rates", "credit"]] *= 100  # Source yields/spreads in percent -> basis points.
    for name in ["oil", "gold"]:
        if (levels[name].dropna() <= 0).any():
            raise ValueError(f"Nonpositive {name} price: cannot use percentage returns")
        out[name] = levels[name].pct_change(fill_method=None)
    return out


def rolling_betas(returns, factor_changes, window, minimum, penalty):
    """Contemporaneous regression, trailing-only scaling; beta per one local factor SD."""
    rows = np.full((len(returns), len(FACTORS)), np.nan)
    x_all = factor_changes[FACTORS].to_numpy(dtype=float)
    y_all = returns.to_numpy(dtype=float)
    for t in range(window - 1, len(returns)):
        x = x_all[t - window + 1:t + 1]
        y = y_all[t - window + 1:t + 1]
        good = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if good.sum() < minimum:
            continue
        x, y = x[good], y[good]
        sd = x.std(axis=0)
        if (sd < 1e-12).any():
            continue
        x = (x - x.mean(axis=0)) / sd
        rows[t] = np.linalg.solve(x.T @ x + penalty * np.eye(len(FACTORS)), x.T @ (y - y.mean()))
    return pd.DataFrame(rows, index=returns.index, columns=[f"beta_{f}" for f in FACTORS])


def build_panel(prices, levels, c):
    if not prices.index.equals(levels.index):
        raise ValueError("Price and macro calendars differ")
    delta = changes(levels)
    lag = c["macro_lag_sessions"]
    macro = levels.add_prefix("level_").join(delta.add_prefix("change_")).shift(lag)
    # Standardized changes use the same trailing-only scale convention as the betas.
    scale = delta.rolling(c["beta_window"], min_periods=c["beta_min_obs"]).std(ddof=0)
    z = (delta / scale.replace(0, np.nan)).shift(lag)
    rows = []
    dates = pd.Series(prices.index, index=prices.index)
    for ticker in prices:
        p = prices[ticker]
        ret = p.pct_change(fill_method=None)
        frame = pd.DataFrame(index=prices.index)
        for n in [21, 63, 126]:
            frame[f"momentum_{n}"] = p.pct_change(n, fill_method=None)
        for n in [21, 63]:
            frame[f"volatility_{n}"] = ret.rolling(n).std() * np.sqrt(252)
        beta = rolling_betas(ret, delta, c["beta_window"], c["beta_min_obs"], c["beta_ridge"]).shift(lag)
        frame = frame.join(macro).join(beta)
        for f in FACTORS:
            frame[f"interaction_{f}"] = beta[f"beta_{f}"] * z[f]
            frame[f"magnitude_{f}"] = beta[f"beta_{f}"].abs()
        # Signal after close t; next-close execution, exit h sessions after execution.
        frame["target"] = p.shift(-(c["horizon"] + 1)) / p.shift(-1) - 1
        frame["entry_date"] = dates.shift(-1)
        frame["label_end"] = dates.shift(-(c["horizon"] + 1))
        frame["ticker"] = ticker
        rows.append(frame.rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True)


def ridge_predict(train, evaluate, columns, penalty):
    x = train[columns].to_numpy(float)
    xt = evaluate[columns].to_numpy(float)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-12] = 1
    x, xt = (x - mean) / sd, (xt - mean) / sd
    y = train.target.to_numpy()
    intercept = y.mean()
    coefficient = np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ (y - intercept))
    return xt @ coefficient + intercept


def split_panel(panel, c, final=False):
    boundary = pd.Timestamp(c["train_end"])
    val_end = pd.Timestamp(c["validation_end"])
    # First future date is beyond boundary; purge labels that reach it.
    train = panel[(panel.date <= boundary) & (panel.label_end <= boundary)]
    if final:
        train = panel[(panel.date <= val_end) & (panel.label_end <= val_end)]
        evaluate = panel[panel.date > val_end]
    else:
        evaluate = panel[(panel.date > boundary) & (panel.date <= val_end) & (panel.label_end <= val_end)]
    return train, evaluate


def daily_metrics(frame):
    records = []
    for date, g in frame.groupby("date"):
        if len(g) < 9:
            continue
        g = g.sort_values(["prediction", "ticker"])
        ic = g.prediction.rank().corr(g.target.rank())
        spread = g.tail(3).target.mean() - g.head(3).target.mean()
        records.append({"date": date, "rank_ic": ic, "top3_minus_bottom3": spread})
    return pd.DataFrame(records)


def run(c, final=False):
    out = ROOT / "outputs" / ("holdout" if final else "validation")
    out.mkdir(parents=True, exist_ok=True)
    prices = pd.concat([read_series(t) for t in c["assets"]], axis=1).sort_index()
    # Union of actual ETF sessions; no equity filling or pre-inception backfilling.
    levels = pd.concat([read_series(f) for f in c["fred"]] +
                       [read_series(c["gold_proxy"]).rename("gold")], axis=1, sort=True).sort_index()
    # Preserve off-calendar observations before alignment; cap stale carries at two sessions.
    levels = levels.reindex(levels.index.union(prices.index)).sort_index().ffill(limit=2).reindex(prices.index)
    panel = build_panel(prices, levels, c)
    macro_cols = [f"{prefix}_{f}" for prefix in ["level", "change"] for f in FACTORS]
    beta_cols = [f"{prefix}_{f}" for prefix in ["beta", "interaction"] for f in FACTORS]
    all_cols = EQUITY + macro_cols + beta_cols
    valid = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=all_cols + ["target", "label_end"])
    # All comparisons use identical dates/assets; require at least nine assets per session.
    valid = valid[valid.groupby("date").ticker.transform("count") >= 9].copy()
    for ticker in c["assets"]:
        valid[f"asset_{ticker}"] = (valid.ticker == ticker).astype(float)
    ids = [f"asset_{t}" for t in c["assets"]]
    train, evaluate = split_panel(valid, c, final)
    if train.date.nunique() < 120 or evaluate.date.nunique() < 30:
        raise ValueError(f"Insufficient usable history: {train.date.nunique()} train / {evaluate.date.nunique()} evaluation dates")
    models = {"equity_only": EQUITY + ids,
              "equity_macro": EQUITY + macro_cols + ids,
              "equity_macro_betas": all_cols + ids}
    summaries = []
    for name, cols in models.items():
        scored = evaluate[["date", "ticker", "target"]].copy()
        scored["prediction"] = ridge_predict(train, evaluate, cols, c["model_ridge"])
        metrics = daily_metrics(scored)
        scored.to_csv(out / f"{name}_predictions.csv", index=False)
        metrics.to_csv(out / f"{name}_daily_metrics.csv", index=False)
        summaries.append({"model": name, "mean_rank_ic": metrics.rank_ic.mean(),
                          "mean_5_session_spread": metrics.top3_minus_bottom3.mean(),
                          "evaluation_dates": len(metrics)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    # Export development data only by default, leaving the holdout unscored and uninspected.
    development = valid if final else valid[valid.date <= pd.Timestamp(c["validation_end"])]
    development.to_csv(out / "panel.csv", index=False)
    development.groupby("ticker")[[f"beta_{f}" for f in FACTORS]].agg(["mean", "std", "min", "max"]).to_csv(out / "beta_diagnostics.csv")
    counts = {"train_dates": int(train.date.nunique()), "evaluation_dates": int(evaluate.date.nunique()),
              "train_start": str(train.date.min().date()), "train_end": str(train.date.max().date()),
              "evaluation_start": str(evaluate.date.min().date()), "evaluation_end": str(evaluate.date.max().date())}
    (out / "run.json").write_text(json.dumps({"config": c, "coverage": counts}, indent=2))
    report = "# Macro-sensitivity prototype — " + ("holdout" if final else "validation") + "\n\n"
    report += "Latest-vintage exploratory data. These results are not a point-in-time backtest or evidence of tradable alpha.\n\n"
    report += "```\n" + summary.to_string(index=False) + "\n```\n\n"
    report += "Coverage: " + json.dumps(counts) + "\n\n"
    report += "Credit history limits the common sample. Gold uses GLD adjusted returns; oil uses Brent spot. Macro data and betas have a five-session assumed lag, not verified historical release timestamps.\n\n"
    report += "Spreads are overlapping five-session gross forward-return diagnostics, not a daily portfolio return series. No costs, turnover, annualization, or significance claims are included.\n\n"
    report += "A shared linear macro term adds the same score to every asset on a date; it cannot directly alter within-date ranks. Betas and interactions can. A nonlinear macro baseline is required before claiming a graph advantage.\n\n"
    report += "Default execution evaluates validation only. The final period remains unscored until an explicit --holdout run.\n"
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(counts))
    print(f"Report: {out / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--holdout", action="store_true", help="Explicitly unlock final holdout evaluation")
    args = parser.parse_args()
    c = config()
    download(c) if args.command == "download" else run(c, args.holdout)


if __name__ == "__main__":
    main()
