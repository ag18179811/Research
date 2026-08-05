
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np
import pandas as pd

from dataprep import (
    PointInTimeStore,
    generate_episode_windows,
    assign_episode_regimes,
    label_regimes,
)
from execution_replay import (
    ExecutionReplayEngine,
    ReplayConfig,
    MarketDataSnapshot,
    MarketRegime,
    Order,
)


REGIME_LABEL_TO_ENUM = {
    "trending": MarketRegime.TRENDING,
    "mean_reverting": MarketRegime.MEAN_REVERTING,
    "crisis": MarketRegime.CRISIS_HIGH_VOLATILITY,
}

BASE_SPREAD_BPS = 2.0        # baseline half-spread-driving constant, calm regime
VOL_SPREAD_SCALE = 40.0      # how much realized vol widens the spread
CRISIS_SPREAD_MULT = 3.0     # extra multiplier applied only in crisis regime
DEPTH_FRACTION_OF_ADV = 0.01  # top-of-book depth as a fraction of ADV


def estimate_quote(mid_price: float, realized_vol: Optional[float],
                    adv: Optional[float], regime: MarketRegime) -> dict:
    """Synthesize bid/ask/depth from a close price + realized vol + ADV.
    Returns a dict matching the relevant MarketDataSnapshot fields."""
    vol = realized_vol if realized_vol is not None else 0.0
    spread_bps = BASE_SPREAD_BPS + VOL_SPREAD_SCALE * vol
    if regime == MarketRegime.CRISIS_HIGH_VOLATILITY:
        spread_bps *= CRISIS_SPREAD_MULT
    half_spread = mid_price * (spread_bps / 10000.0) / 2.0

    adv_val = adv if adv and adv > 0 else 0.0
    depth = adv_val * DEPTH_FRACTION_OF_ADV

    return {
        "bid_price": mid_price - half_spread,
        "ask_price": mid_price + half_spread,
        "bid_depth": depth,
        "ask_depth": depth,
        "average_daily_volume": adv_val,
    }


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------

def build_execution_snapshot(day_row: pd.Series, ticker: str,
                              realized_vol: Optional[float],
                              adv: Optional[float],
                              regime_label: Optional[str]) -> MarketDataSnapshot:
    """Turn one row of Phase 1's PiT price data (for a single trading day)
    into the MarketDataSnapshot Phase 2's engine fills orders against."""
    mid_price = float(day_row["adj_close"])
    regime = REGIME_LABEL_TO_ENUM.get(regime_label, MarketRegime.MEAN_REVERTING)
    quote = estimate_quote(mid_price, realized_vol, adv, regime)

    return MarketDataSnapshot(
        timestamp=day_row["available_at"].isoformat(),
        symbol=ticker,
        bid_price=quote["bid_price"],
        ask_price=quote["ask_price"],
        bid_depth=quote["bid_depth"],
        ask_depth=quote["ask_depth"],
        daily_volatility=realized_vol if realized_vol is not None else 0.0,
        average_daily_volume=quote["average_daily_volume"],
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

AgentPolicy = Callable[[dict], list]

def _resolve_policy(agent) -> AgentPolicy:
    return agent.decide if hasattr(agent, "decide") else agent


def _reset_if_stateful(agent) -> None:
    if hasattr(agent, "reset"):
        agent.reset()

MARKET_OPEN_UTC_HOUR = 13  # ~9:30am ET, approximate, matches MARKET_CLOSE_UTC_HOUR's precision level in dataprep.py


def run_episode(
    store: PointInTimeStore,
    engine: ExecutionReplayEngine,
    ticker: str,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    regime_label: Optional[str],
    agent,
    price_lookback: int = 63,
    vol_window: int = 21,
) -> None:
    """Runs one non-overlapping episode window for a single ticker, driving
    the ExecutionReplayEngine off real Phase 1 data instead of mock
    snapshots. Call once per (episode, ticker) pair from your outer loop.

    `agent` is either a phase1_5_agent.TradingAgent instance or a plain
    callable(snapshot) -> list[Order]; see _resolve_policy/_reset_if_stateful
    above."""
    decide_fn = _resolve_policy(agent)
    _reset_if_stateful(agent)  # fresh state at the start of every episode

   
    full_history = store.prices_as_of(episode_end, entities=[ticker])
    window_rows = full_history[
        (full_history["date"] >= episode_start) & (full_history["date"] < episode_end)
    ].sort_values("date")

    if window_rows.empty:
        return

    for _, day_row in window_rows.iterrows():
        trade_date = day_row["date"]
        decision_cutoff = trade_date.normalize() + pd.Timedelta(hours=MARKET_OPEN_UTC_HOUR)

        # --- decision half: strictly PiT, cannot see trade_date's own close
        decision_snapshot = store.snapshot_as_of(
            ticker, decision_cutoff, price_lookback=price_lookback,
        )
        orders: list[Order] = decide_fn(decision_snapshot)

        # --- execution half: today's realized price, used only for fills
        prior_rows = full_history[full_history["date"] < trade_date].tail(vol_window + 1)
        realized_vol = None
        if len(prior_rows) >= 2:
            rets = prior_rows["adj_close"].pct_change().dropna()
            realized_vol = float(rets.std()) if len(rets) >= vol_window else None
        adv = float(prior_rows["volume"].mean()) if "volume" in prior_rows.columns and not prior_rows.empty else None

        exec_snapshot = build_execution_snapshot(day_row, ticker, realized_vol, adv, regime_label)

        for order in orders:
            engine.process_order(order, exec_snapshot)

        engine.record_daily_value(
            trade_date.date().isoformat(),
            {ticker: float(day_row["adj_close"])},
        )


def run_backtest(
    store: PointInTimeStore,
    ticker: str,
    start: str,
    end: str,
    window_len: pd.Timedelta,
    purge_gap: pd.Timedelta,
    agent,
    replay_config: Optional[ReplayConfig] = None,
) -> ExecutionReplayEngine:
   
    prices = store.prices_as_of(pd.Timestamp(end, tz="UTC"), entities=[ticker])
    if prices.empty:
        raise ValueError(f"no price history for {ticker}; run Phase 1's build-prices first")

    returns = prices.set_index("date")["adj_close"].pct_change()
    regimes = label_regimes(returns)

    windows = generate_episode_windows(start, end, window_len, purge_gap)
    episode_regimes = assign_episode_regimes(windows, regimes)

    engine = ExecutionReplayEngine(replay_config or ReplayConfig())
    for _, ep in episode_regimes.iterrows():
        run_episode(
            store, engine, ticker,
            ep["episode_start"], ep["episode_end"], ep["regime"],
            agent,
        )

    return engine

from phase1_5_agent import SmaCrossoverAgent  # the real Phase 1.5 strategy


def main():
    store = PointInTimeStore()
    agent = SmaCrossoverAgent(order_qty=10.0)  # Phase 1.5

    engine = run_backtest(               # Phase 1 data drives this,
        store,                           # Phase 1.5's agent.decide() picks
        ticker="AAPL",                   # the orders inside it, and
        start="2019-01-01",              # Phase 2's ExecutionReplayEngine
        end="2021-01-01",                # fills them -- this one call is
        window_len=pd.Timedelta(days=126),   # the whole three-phase flow.
        purge_gap=pd.Timedelta(days=5),
        agent=agent,
    )

    breakeven_bps = engine.calculate_breakeven_threshold()
    print(f"Fills logged: {len(engine.execution_log)}")
    print(f"Days marked to market: {len(engine.daily_portfolio_log)}")
    print(f"Execution-breakeven threshold: {breakeven_bps:.1f} bps")

    engine.export_logs_to_json("execution_log.json")
    store.save_access_log("access_log.parquet")

    access_log = pd.DataFrame(store.access_log)
    blocked = int(access_log["rows_blocked"].sum()) if not access_log.empty else 0
    print(f"Temporal integrity: {blocked} blocked access(es) across the run")


if __name__ == "__main__":
    main()
