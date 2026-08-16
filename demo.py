"""
Demo: walks a mock trading agent through a real historical episode using
the Data Preparation phase's point-in-time access layer.

Purpose: prove the plumbing works end-to-end before wiring in a real agent
(self-built, or the TradingAgents repo) -- exactly the "insert mock data
first" step before merging everyone's components together.

Run: .venv/bin/python demo.py
"""

import logging
import warnings

import pandas as pd

import dataprep as dp

TICKER = "TSLA"

# dataprep's own INFO-level access logging and hmmlearn's per-restart
# convergence warnings are real signal during development, but at demo time
# they just bury the narration below in noise -- this demo shows the same
# information deliberately, via the access log and the printed tables
# instead, so silence both loggers rather than let them interleave with it.
logging.getLogger("dataprep").setLevel(logging.ERROR)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def section(n: int, title: str) -> None:
    print()
    print(f"[{n}] {title}")
    print("-" * 74)


def fmt_pct(x: float | None) -> str:
    return f"{x:+.1%}" if x is not None else "n/a"


def fmt_3f(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def mock_agent_decide(snapshot: dict) -> str:
    """Toy decision rule standing in for a real agent. The strategy itself
    is not the point -- what matters is that ANY agent (self-built or
    TradingAgents) can be dropped in here and would see exactly this shape
    of point-in-time-consistent data: recent prices, recent news, latest
    known fundamentals, and derived features, all cut off at `as_of` with
    zero risk of seeing the future."""
    feats = snapshot["derived_features"]
    vol = feats.get("realized_vol")
    ret_21d = feats.get("return_21d")
    if vol is None or ret_21d is None:
        return "HOLD (insufficient history yet)"
    if vol > 0.04:
        return "HOLD (volatility too high)"
    if ret_21d > 0.05:
        return "BUY (positive 21d momentum)"
    if ret_21d < -0.05:
        return "SELL (negative 21d momentum)"
    return "HOLD (no strong signal)"


def main() -> None:
    print("=" * 74)
    print("DATA PREPARATION PHASE -- DEMO")
    print("=" * 74)
    print(f"Ticker: {TICKER}")
    print("Point-in-time snapshot -> agent decision -> repeat. Swap")
    print("mock_agent_decide() for a real agent and nothing else here needs")
    print("to change -- that's the integration point for Phase 2.")

    store = dp.PointInTimeStore()

    # ------------------------------------------------------------------
    # 1. Temporal integrity: prove the cutoff is actually enforced
    # ------------------------------------------------------------------
    section(1, "TEMPORAL INTEGRITY -- cutoff enforced at the access layer")
    cutoff = pd.Timestamp("2020-01-01", tz="UTC")
    all_prices = dp.load_prices(entities=[TICKER])
    snap = store.snapshot_as_of(TICKER, cutoff, price_lookback=5)
    kept, total = len(snap["prices"]), len(all_prices)
    assert snap["prices"]["available_at"].max() < cutoff
    print(f"  query          {TICKER} prices as of {cutoff.date()}")
    print(f"  returned       {kept}/{total} rows "
          f"({total - kept} at/after cutoff blocked)")
    print(f"  latest row     {snap['prices']['date'].max().date()} (< cutoff, confirmed)")

    # ------------------------------------------------------------------
    # 2. Regime labels assigned to non-overlapping episode windows
    # ------------------------------------------------------------------
    section(2, "REGIME LABELS -- walk-forward HMM, non-overlapping episodes")
    print("  fitting rolling out-of-sample HMM on SPY returns (~30s)...")
    spy = dp.load_prices(entities=["SPY"]).sort_values("date").set_index("date")
    returns = spy["adj_close"].pct_change()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regimes = dp.label_regimes(returns)
    windows = dp.generate_episode_windows(
        "2020-01-01", "2020-07-01", pd.Timedelta(days=90), pd.Timedelta(days=14))
    episodes = dp.assign_episode_regimes(windows, regimes)
    print()
    print(episodes.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Mock agent walking a PiT decision loop through the COVID episode
    # ------------------------------------------------------------------
    section(3, "MOCK AGENT -- PiT decision loop through the COVID episode")
    dates = pd.date_range("2020-02-01", "2020-04-15", freq="7D", tz="UTC")
    header = f"{'date':<10} {'realized_vol':>12} {'return_21d':>10} {'news_n':>6}  decision"
    print()
    print(header)
    print("-" * len(header))
    # Printed as each snapshot is fetched, not batched until the loop ends --
    # each iteration does 3 dataset loads, so streaming rows here is what
    # tells you it's progressing rather than stuck.
    for as_of in dates:
        snap = store.snapshot_as_of(
            TICKER, as_of, price_lookback=90, news_lookback=pd.Timedelta(days=14))
        feats = snap["derived_features"]
        print(f"{as_of.date()!s:<10} {fmt_3f(feats.get('realized_vol')):>12} "
              f"{fmt_pct(feats.get('return_21d')):>10} {len(snap['news']):>6}  "
              f"{mock_agent_decide(snap)}")

    # ------------------------------------------------------------------
    # 4. Access log: every request this run made, blocked or not, is a
    #    durable record -- not just the warning lines scrolling by above.
    # ------------------------------------------------------------------
    section(4, "ACCESS LOG -- every PiT request this run made")
    log = pd.DataFrame(store.access_log)
    summary = log.groupby("dataset")[["rows_returned", "rows_blocked"]].sum()
    print(summary.to_string())
    print(f"\n  {len(log)} requests, {log['rows_blocked'].sum()} rows blocked total "
          "-- see PointInTimeStore.save_access_log() to persist this.")

    print()
    print("=" * 74)
    print("DONE")
    print("=" * 74)


if __name__ == "__main__":
    main()
