"""
Validation suite for the Data Preparation phase.

Each check tests a specific property the research proposal requires -- not
generic unit tests, but domain checks: does the access layer actually block
look-ahead leakage, do regime labels agree with known market history, does
the entity crosswalk resolve correctly, etc. Run after any change to
dataprep.py to confirm the phase still does what the proposal asks for.

Usage: .venv/bin/python validate.py
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd

import dataprep as dp

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"raised {type(e).__name__}: {e}"
        RESULTS.append((name, passed, detail))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 1. Coverage: did the raw sources actually get fully processed?
# ---------------------------------------------------------------------------

@check("coverage: prices covers full FNSPID ticker universe")
def _():
    import pyarrow.parquet as pq
    from pathlib import Path
    paths = list(Path(dp.PRICES_DIR).glob("entity=*/data.parquet"))
    n_tickers = len(paths)
    n_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    ok = n_tickers >= 7000 and n_rows >= 20_000_000
    return ok, f"{n_tickers} tickers, {n_rows:,} rows"


@check("coverage: fundamentals covers all downloaded SEC quarters")
def _():
    from pathlib import Path
    n_quarters_raw = len(list(dp.RAW_SEC_DIR.glob("*.zip")))
    n_quarters_built = len(list(Path(dp.FUNDAMENTALS_DIR).glob("quarter=*/data.parquet")))
    ok = n_quarters_built == n_quarters_raw and n_quarters_built > 0
    return ok, f"{n_quarters_built}/{n_quarters_raw} quarters built"


@check("coverage: news dataset is non-trivial and spans expected years")
def _():
    import pyarrow.parquet as pq
    from pathlib import Path
    paths = list(Path(dp.NEWS_DIR).rglob("*.parquet"))
    n_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    years = sorted({int(p.parent.name.split("=", 1)[1]) for p in paths})
    ok = n_rows >= 10_000_000 and years and min(years) <= 2000 and max(years) >= 2023
    return ok, f"{n_rows:,} rows, years {min(years) if years else '?'}-{max(years) if years else '?'}"


# ---------------------------------------------------------------------------
# 2. PiT enforcement: does the access layer actually block future data?
# ---------------------------------------------------------------------------

@check("PiT enforcement: prices_as_of returns nothing at/after cutoff, and the filter is non-vacuous")
def _():
    store = dp.PointInTimeStore()
    cutoff = pd.Timestamp("2015-01-01", tz="UTC")
    filtered = store.prices_as_of(cutoff, entities=["AAPL"])
    unfiltered = dp.load_prices(entities=["AAPL"])
    leaked = (filtered["available_at"] >= cutoff).sum()
    actually_filtered = len(unfiltered) > len(filtered)  # proves cutoff wasn't a no-op
    return leaked == 0 and actually_filtered, (
        f"leaked={leaked}, kept {len(filtered)}/{len(unfiltered)} rows")


@check("PiT enforcement: news_as_of returns nothing at/after cutoff, and the filter is non-vacuous")
def _():
    store = dp.PointInTimeStore()
    cutoff = pd.Timestamp("2015-01-01", tz="UTC")
    filtered = store.news_as_of(cutoff, entities=["AAPL"])
    unfiltered = dp.load_news(entities=["AAPL"])
    leaked = (filtered["available_at"] >= cutoff).sum()
    actually_filtered = len(unfiltered) > len(filtered)
    return leaked == 0 and actually_filtered, (
        f"leaked={leaked}, kept {len(filtered)}/{len(unfiltered)} rows")


@check("PiT enforcement: fundamentals_as_of returns nothing at/after cutoff, and the filter is non-vacuous")
def _():
    store = dp.PointInTimeStore()
    cik = dp.ticker_to_cik("AAPL")
    cutoff = pd.Timestamp("2015-01-01", tz="UTC")
    filtered = store.fundamentals_as_of(cutoff, entities=[cik])
    unfiltered = dp.load_fundamentals(entities=[cik])
    leaked = (filtered["available_at"] >= cutoff).sum()
    actually_filtered = len(unfiltered) > len(filtered)
    return leaked == 0 and actually_filtered, (
        f"leaked={leaked}, kept {len(filtered)}/{len(unfiltered)} rows")


@check("PiT enforcement: available fundamentals never shrink as the cutoff moves forward")
def _():
    # As of a later date we should know at least as much as we did earlier
    # -- a monotonicity property that would break if leakage or a filtering
    # bug let a later query see less than an earlier one.
    store = dp.PointInTimeStore()
    cik = dp.ticker_to_cik("AAPL")
    counts = [len(store.fundamentals_as_of(pd.Timestamp(d, tz="UTC"), entities=[cik]))
              for d in ["2015-01-01", "2018-01-01", "2021-01-01", "2024-01-01"]]
    ok = all(b >= a for a, b in zip(counts, counts[1:]))
    return ok, f"counts over time: {counts}"


# ---------------------------------------------------------------------------
# 3. Entity resolution
# ---------------------------------------------------------------------------

@check("entity crosswalk: AAPL resolves to its known SEC CIK (320193)")
def _():
    cik = dp.ticker_to_cik("AAPL")
    return cik == 320193, f"resolved to {cik}"


@check("entity crosswalk: an unregistered ticker resolves to None, not a wrong guess")
def _():
    cik = dp.ticker_to_cik("THIS_TICKER_DOES_NOT_EXIST_XYZ")
    return cik is None, f"resolved to {cik}"


# ---------------------------------------------------------------------------
# 4. Regime labels vs. known market history
# ---------------------------------------------------------------------------

@check("regime labels: crisis periods (2008, 2018Q4, 2020) show majority-crisis/elevated days")
def _():
    spy = dp.load_prices(entities=["SPY"]).sort_values("date").set_index("date")
    returns = spy["adj_close"].pct_change()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regimes = dp.label_regimes(returns)
    windows = {
        "2008 GFC": ("2008-09-01", "2008-12-31"),
        "2018 Q4": ("2018-10-01", "2018-12-31"),
        "2020 COVID": ("2020-02-15", "2020-04-30"),
    }
    details = []
    ok = True
    for name, (s, e) in windows.items():
        seg = regimes.loc[s:e]
        crisis_frac = (seg["regime"] == "crisis").mean() if len(seg) else 0
        stressed_frac = crisis_frac + (seg["regime"] == "trending").mean() if len(seg) else 0
        window_ok = stressed_frac >= 0.5  # majority not "calm mean-reverting"
        ok &= window_ok
        details.append(f"{name}: crisis={crisis_frac:.0%} stressed={stressed_frac:.0%}")
    return ok, "; ".join(details)


@check("regime labels: calm bull periods (2013-14, 2017, 2023-24) stay mostly non-crisis")
def _():
    spy = dp.load_prices(entities=["SPY"]).sort_values("date").set_index("date")
    returns = spy["adj_close"].pct_change()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regimes = dp.label_regimes(returns)
    windows = {
        "2013-2014": ("2013-01-01", "2014-12-31"),
        "2017": ("2017-01-01", "2017-12-31"),
        "2023-2024": ("2023-01-01", "2024-12-31"),
    }
    details = []
    ok = True
    for name, (s, e) in windows.items():
        seg = regimes.loc[s:e]
        crisis_frac = (seg["regime"] == "crisis").mean() if len(seg) else 1
        window_ok = crisis_frac <= 0.10
        ok &= window_ok
        details.append(f"{name}: crisis={crisis_frac:.0%}")
    return ok, "; ".join(details)


# ---------------------------------------------------------------------------
# 5. Episode generation
# ---------------------------------------------------------------------------

@check("episodes: windows are non-overlapping and respect the purge gap")
def _():
    windows = dp.generate_episode_windows("2015-01-01", "2020-01-01",
                                           pd.Timedelta(days=90), pd.Timedelta(days=14))
    ok = all(windows[i + 1][0] >= windows[i][1] + pd.Timedelta(days=14) for i in range(len(windows) - 1))
    return ok, f"{len(windows)} windows generated"


@check("episodes: each non-empty window gets exactly one regime label, assigned before execution")
def _():
    spy = dp.load_prices(entities=["SPY"]).sort_values("date").set_index("date")
    returns = spy["adj_close"].pct_change()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regimes = dp.label_regimes(returns)
    windows = dp.generate_episode_windows("2018-01-01", "2021-01-01",
                                           pd.Timedelta(days=90), pd.Timedelta(days=14))
    ep = dp.assign_episode_regimes(windows, regimes)
    non_empty = ep[ep["n_days"] > 0]
    ok = non_empty["regime"].isin(["trending", "mean_reverting", "crisis"]).all()
    return ok, f"{len(non_empty)}/{len(ep)} windows labeled: {dict(non_empty['regime'].value_counts())}"


# ---------------------------------------------------------------------------
# 6. Derived features
# ---------------------------------------------------------------------------

@check("derived features: snapshot_as_of returns finite, plausible values for a known ticker")
def _():
    store = dp.PointInTimeStore()
    snap = store.snapshot_as_of("AAPL", pd.Timestamp("2020-06-01", tz="UTC"),
                                 price_lookback=90, news_lookback=pd.Timedelta(days=30))
    f = snap["derived_features"]
    checks = [
        f["realized_vol"] is not None and 0 < f["realized_vol"] < 0.2,
        f["leverage_ratio"] is None or 0 < f["leverage_ratio"] < 5,
        f["current_ratio"] is None or 0 < f["current_ratio"] < 20,
    ]
    return all(checks), str(f)


# ---------------------------------------------------------------------------
# 7. Model training-cutoff registry (parametric look-ahead guard)
# ---------------------------------------------------------------------------

@check("model cutoff registry: unregistered model raises, isn't silently treated as safe")
def _():
    try:
        dp.check_model_cutoff("__unregistered_test_model__", pd.Timestamp("2020-01-01", tz="UTC"))
        return False, "did not raise"
    except KeyError:
        return True, "raised KeyError as expected"


@check("model cutoff registry: correctly flags a cutoff that does NOT precede the episode")
def _():
    dp.MODEL_TRAINING_CUTOFFS["__test_model__"] = "2021-01-01"
    unsafe = dp.check_model_cutoff("__test_model__", pd.Timestamp("2020-01-01", tz="UTC"))
    safe = dp.check_model_cutoff("__test_model__", pd.Timestamp("2022-01-01", tz="UTC"))
    del dp.MODEL_TRAINING_CUTOFFS["__test_model__"]
    return (unsafe is False and safe is True), f"unsafe_case={unsafe}, safe_case={safe}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    for name, passed, detail in RESULTS:
        pass  # RESULTS is populated at import time via the @check decorators above

    width = max(len(name) for name, _, _ in RESULTS)
    n_pass = sum(1 for _, p, _ in RESULTS if p)
    for name, passed, detail in RESULTS:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name.ljust(width)}  {detail}")
    print()
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
