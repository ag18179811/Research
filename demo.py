"""
Demo: walks a mock trading agent through a real historical episode using
the Data Preparation phase's point-in-time access layer.

Purpose: prove the plumbing works end-to-end before wiring in a real agent
(self-built, or the TradingAgents repo) -- exactly the "insert mock data
first" step before merging everyone's components together.

Run: .venv/bin/python demo.py
"""

import warnings

import pandas as pd

import dataprep as dp

TICKER = "TSLA"


def print_header(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


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
    print_header("DATA PREPARATION PHASE -- DEMO")
    print(f"Ticker: {TICKER}")
    print("Shows the data-access layer driving an agent decision loop")
    print("end-to-end: point-in-time snapshot -> agent decision -> repeat.")
    print("Swap mock_agent_decide() for a real agent and nothing else here")
    print("needs to change -- that's the integration point for Phase 2.")

    store = dp.PointInTimeStore()

    # ------------------------------------------------------------------
    # 1. Temporal integrity: prove the cutoff is actually enforced
    # ------------------------------------------------------------------
    print_header("1. TEMPORAL INTEGRITY CHECK")
    cutoff = pd.Timestamp("2020-01-01", tz="UTC")
    snap = store.snapshot_as_of(TICKER, cutoff, price_lookback=5)
    latest_date = snap["prices"]["date"].max()
    print(f"Queried {TICKER} as of {cutoff.date()}")
    print(f"Latest price row returned is dated: {latest_date.date()}")
    assert snap["prices"]["available_at"].max() < cutoff
    print("Confirmed: nothing in the response is dated at/after the cutoff.")

    # ------------------------------------------------------------------
    # 2. Regime label assigned to an episode, before any agent runs
    # ------------------------------------------------------------------
    print_header("2. REGIME LABEL FOR AN EPISODE (fitting walk-forward HMM, ~30s)")
    spy = dp.load_prices(entities=["SPY"]).sort_values("date").set_index("date")
    returns = spy["adj_close"].pct_change()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regimes = dp.label_regimes(returns)
    windows = dp.generate_episode_windows(
        "2020-01-01", "2020-07-01", pd.Timedelta(days=90), pd.Timedelta(days=14))
    episodes = dp.assign_episode_regimes(windows, regimes)
    print(episodes.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Mock agent walking a PiT decision loop through the COVID episode
    # ------------------------------------------------------------------
    print_header("3. MOCK AGENT WALKING A PiT DECISION LOOP")
    dates = pd.date_range("2020-02-01", "2020-04-15", freq="7D", tz="UTC")
    for as_of in dates:
        snap = store.snapshot_as_of(
            TICKER, as_of, price_lookback=90, news_lookback=pd.Timedelta(days=14))
        decision = mock_agent_decide(snap)
        vol = snap["derived_features"].get("realized_vol")
        vol_str = f"{vol:.3f}" if vol is not None else "n/a "
        n_news = len(snap["news"])
        print(f"  {as_of.date()} | realized_vol={vol_str} | news available={n_news:3d} "
              f"| decision: {decision}")

    print_header("DONE")
    print("Every row above came from one call: store.snapshot_as_of(ticker, as_of).")
    print("That's the integration point -- a real agent replaces mock_agent_decide().")


if __name__ == "__main__":
    main()
