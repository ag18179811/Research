"""
Execution harness: the glue between an agent's orders for one day and Phase
2's ExecutionReplayEngine.

This was previously inlined inside runner.py's run_episode loop. Pulled out
because it's genuinely Phase 2's concern (how do orders actually get
executed against a market snapshot), not Phase 3 orchestration (which
episode are we in, what regime is it, whose turn is it to decide). runner.py
now just calls execute_orders_for_day() once per trading day and moves on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from execution_replay import (
    ExecutionReplayEngine,
    MarketDataSnapshot,
    MarketRegime,
    Order,
)


# ---------------------------------------------------------------------------
# Synthetic microstructure
# ---------------------------------------------------------------------------
# Phase 1's raw data has no bid/ask/depth (FNSPID is daily OHLCV only), but
# Phase 2's engine requires a MarketDataSnapshot to execute against. This
# fabricates one from the single daily close price. It is a placeholder,
# not a validated microstructure model -- it exists so the Phase 2 engine
# has something to execute against. Treat any Executable Performance
# numbers produced this way as illustrative until real quote/depth data is
# wired in.


def synthesize_snapshot(
    ticker: str,
    execution_date: pd.Timestamp,
    todays_price_row: pd.Series,
    trailing_prices: pd.DataFrame,
    regime: MarketRegime,
    spread_bps: float = 5.0,
    depth_fraction: float = 0.02,
    vol_window: int = 21,
) -> MarketDataSnapshot:
    close = float(todays_price_row["adj_close"])
    half_spread = close * (spread_bps / 10000.0) / 2
    bid, ask = close - half_spread, close + half_spread

    vol_col = trailing_prices["adj_close"].pct_change().tail(vol_window)
    daily_vol = float(vol_col.std()) if vol_col.notna().sum() >= 2 else 0.02

    adv = float(trailing_prices["volume"].tail(vol_window).mean()) if "volume" in trailing_prices else 1e6
    if not np.isfinite(adv) or adv <= 0:
        adv = 1e6
    depth = adv * depth_fraction

    return MarketDataSnapshot(
        timestamp=execution_date.isoformat(),
        symbol=ticker,
        bid_price=bid,
        ask_price=ask,
        bid_depth=depth,
        ask_depth=depth,
        daily_volatility=daily_vol if np.isfinite(daily_vol) else 0.02,
        average_daily_volume=adv,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Order execution for one day
# ---------------------------------------------------------------------------


def execute_orders_for_day(
    engine: ExecutionReplayEngine,
    orders: list[Order],
    day: pd.Timestamp,
    all_prices: pd.DataFrame,
    tickers: list[str],
    regime: MarketRegime,
    order_counter_start: int = 0,
) -> tuple[dict[str, float], int]:
    """Takes one day's orders from the agent and actually runs them through
    Phase 2's engine, ticker by ticker: builds a synthetic market snapshot,
    submits any orders for that ticker, and records today's close price
    (for marking the portfolio to market afterward, in runner.py).

    Returns (todays_close, order_counter) -- today's close prices per
    ticker (used by the caller to mark-to-market) and the running order-id
    counter (passed back in so ids stay unique across the whole episode,
    not just within one day)."""
    todays_close: dict[str, float] = {}
    order_counter = order_counter_start

    for t in tickers:
        t_prices = all_prices[all_prices["entity"] == t].sort_values("date")
        todays_row = t_prices[t_prices["date"] == day]
        if todays_row.empty:
            continue  # no print for this ticker today (halt/holiday mismatch)
        todays_close[t] = float(todays_row.iloc[0]["adj_close"])

        ticker_orders = [o for o in orders if o.symbol == t]
        if not ticker_orders:
            continue

        trailing = t_prices[t_prices["date"] < day]
        snapshot = synthesize_snapshot(t, day, todays_row.iloc[0], trailing, regime)
        for order in ticker_orders:
            order_counter += 1
            if not order.order_id:
                order.order_id = f"{t}-{day.date()}-{order_counter}"
            engine.process_order(order, snapshot)

    return todays_close, order_counter
