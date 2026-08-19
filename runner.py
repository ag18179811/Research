"""
End-to-end Phase 3 runner.

This is the missing piece between Phase 1/2 and metrics.py: it actually
drives an agent through a sequence of episodes, day by day, and builds the
list[EpisodeRun] that build_joint_metric_report() expects.

Pipeline per episode:
  for each trading day t in the episode:
    1. decision snapshot = store.snapshot_as_of(midnight of day t)
       -> this only ever contains data with available_at < midnight(t),
          i.e. strictly before today -- the agent decides using yesterday's
          information, never today's own close.
    2. agent_fn(decision_snapshot, positions, cash) -> list[Order]
    3. each order is executed against day t's *own* market snapshot
       (synthesized bid/ask/depth around day t's close -- see the
       SYNTHETIC MICROSTRUCTURE note below) via Phase 2's
       ExecutionReplayEngine.
    4. portfolio value is marked to day t's close.
  at episode end: package execution_log + daily portfolio values + the PiT
  store's access_log into one EpisodeRun.

Deciding on T-1 data and executing at T's close is a standard daily-bar
backtest simplification (you can't act on a close price before it exists).
It is NOT the same as full intraday realism -- Phase 2's replay engine
supports maker/taker/impact mechanics, but this runner only feeds it one
snapshot per ticker per day, because the upstream data (FNSPID) is daily
OHLCV, not an intraday order book. If you get access to intraday quotes,
swap `execution_harness.synthesize_snapshot` for a real book snapshot and
this loop does not need to change.

SYNTHETIC MICROSTRUCTURE NOTE:
Phase 1's raw data has no bid/ask/depth (FNSPID is OHLCV only).
`execution_harness` fabricates bid/ask from the close price and a flat
assumed spread, and depth from a fraction of trailing average volume. That is a
placeholder, not a validated microstructure model -- it exists so the
Phase 2 engine has something to execute against. Treat any Executable
Performance numbers produced this way as illustrative until real
quote/depth data is wired in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from metrics import EpisodeRun

log = logging.getLogger("runner")

# These come from your Phase 1 / Phase 2 files. Import paths assume you've
# named them dataprep.py and execution_replay.py alongside this file --
# adjust if yours are named differently.
from dataprep import (  # noqa: E402
    PointInTimeStore,
    generate_episode_windows,
    assign_episode_regimes,
    label_regimes,
    load_prices,
)
from execution_replay import (  # noqa: E402
    ExecutionReplayEngine,
    ReplayConfig,
    Order,
    OrderSide,
    OrderType,
    MarketRegime,
)
from execution_harness import execute_orders_for_day  # noqa: E402

REGIME_LABEL_TO_ENUM = {
    "trending": MarketRegime.TRENDING,
    "mean_reverting": MarketRegime.MEAN_REVERTING,
    "crisis": MarketRegime.CRISIS_HIGH_VOLATILITY,
}


# ---------------------------------------------------------------------------
# Agent interface
# ---------------------------------------------------------------------------


class AgentFn(Protocol):
    """What your agent (single- or multi-agent system) must implement. Called
    once per trading day per episode, before that day's prices are known.
    `decision_snapshots` maps ticker -> the dict returned by
    PointInTimeStore.snapshot_as_of (prices/news/fundamentals/derived_features,
    all PiT-filtered to strictly before today). Return a list of Phase 2
    Order objects to submit for today; return [] to sit out the day."""

    def __call__(
        self,
        decision_snapshots: dict[str, dict],
        positions: dict[str, float],
        cash: float,
        as_of: pd.Timestamp,
    ) -> list[Order]: ...


# ---------------------------------------------------------------------------
# Single-episode driver
# ---------------------------------------------------------------------------


def run_episode(
    agent_fn: AgentFn,
    store: PointInTimeStore,
    tickers: list[str],
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    regime: str,
    replay_config: ReplayConfig | None = None,
) -> EpisodeRun:
    """Runs one agent through one episode window, day by day, and returns
    the EpisodeRun metrics.py needs. `store` should be a fresh
    PointInTimeStore (or at least have an access_log you're willing to
    attribute entirely to this episode) so the temporal-integrity report
    is scoped correctly per episode."""
    engine = ExecutionReplayEngine(replay_config or ReplayConfig())
    regime_enum = REGIME_LABEL_TO_ENUM.get(regime, MarketRegime.MEAN_REVERTING)

    all_prices = load_prices(tickers)
    if all_prices.empty:
        raise ValueError(f"no price data for tickers={tickers}")
    trading_days = sorted(
        d for d in all_prices["date"].unique()
        if episode_start <= pd.Timestamp(d) < episode_end
    )
    if not trading_days:
        log.warning("episode %s..%s has no trading days in price data", episode_start, episode_end)

    portfolio_values: dict[pd.Timestamp, float] = {}
    order_counter = 0

    for day in trading_days:
        day = pd.Timestamp(day)
        decision_as_of = day.normalize()  # midnight -- strictly before today's own close

        decision_snapshots = {t: store.snapshot_as_of(t, decision_as_of) for t in tickers}

        try:
            positions = dict(engine.positions)
            orders = agent_fn(decision_snapshots, positions, engine.cash_balance, decision_as_of)
        except Exception:
            log.exception("agent_fn raised on %s -- treating as no-trade for this day", day)
            orders = []

        todays_close, order_counter = execute_orders_for_day(
            engine, orders, day, all_prices, tickers, regime_enum, order_counter,
        )

        if todays_close:
            # record_daily_value both marks to market and appends to the
            # engine's daily_portfolio_log, so export_logs_to_json() emits a
            # populated "daily_portfolio_values" array instead of an empty
            # one. Reuse its return value rather than calling
            # compute_portfolio_value separately -- one mark per day, not two.
            portfolio_values[day] = engine.record_daily_value(
                day.date().isoformat(), todays_close
            ).portfolio_value

    pv_series = pd.Series(portfolio_values).sort_index()
    return EpisodeRun(
        episode_start=episode_start,
        episode_end=episode_end,
        regime=regime,
        execution_log=list(engine.execution_log),
        portfolio_values=pv_series,
        access_log=list(store.access_log),
    )


# ---------------------------------------------------------------------------
# Multi-episode driver
# ---------------------------------------------------------------------------


def run_all_episodes(
    agent_fn: AgentFn,
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    window_len: pd.Timedelta = pd.Timedelta(days=90),
    purge_gap: pd.Timedelta = pd.Timedelta(days=5),
    replay_config: ReplayConfig | None = None,
    store_factory: Callable[[], PointInTimeStore] = PointInTimeStore,
) -> list[EpisodeRun]:
    """Generates non-overlapping episode windows (Phase 1), labels each with
    its majority regime (Phase 1's walk-forward HMM), then runs the agent
    through every window (this module) and returns the list[EpisodeRun]
    ready for metrics.build_joint_metric_report(). A new PointInTimeStore
    per episode keeps each episode's temporal-integrity log scoped to that
    episode, matching what regime_robustness_table / temporal_integrity_report
    expect."""
    windows = generate_episode_windows(start, end, window_len, purge_gap)
    if not windows:
        raise ValueError("no episode windows generated -- check start/end/window_len/purge_gap")

    all_prices = load_prices(tickers)
    if all_prices.empty:
        raise ValueError(f"no price data for tickers={tickers}")

    # One regime label per ticker's return series would each disagree
    # slightly; use the first ticker as the regime reference series, which
    # is a simplification worth revisiting for a multi-asset benchmark
    # (e.g. label off a basket/index return instead of a single name).
    ref_prices = all_prices[all_prices["entity"] == tickers[0]].sort_values("date")
    ref_returns = ref_prices.set_index("date")["adj_close"].pct_change()
    regimes = label_regimes(ref_returns)
    episode_regimes = assign_episode_regimes(windows, regimes)

    runs = []
    for _, row in episode_regimes.iterrows():
        ep_start, ep_end, regime = row["episode_start"], row["episode_end"], row["regime"]
        if regime is None:
            log.warning("skipping episode %s..%s -- no regime label (insufficient HMM history)",
                        ep_start, ep_end)
            continue
        store = store_factory()
        run = run_episode(agent_fn, store, tickers, ep_start, ep_end, regime, replay_config)
        runs.append(run)
        log.info("episode %s..%s (%s): %d fills, %d trading days",
                  ep_start.date(), ep_end.date(), regime, len(run.execution_log), len(run.portfolio_values))
    return runs


# ---------------------------------------------------------------------------
# Baseline agents (control group, per the proposal's Experimental Setup)
# ---------------------------------------------------------------------------


def no_trade_agent(decision_snapshots, positions, cash, as_of) -> list[Order]:
    """Does nothing, ever. The floor: any real system should beat this
    net of costs, since trading against noise with nonzero fees loses to
    doing nothing."""
    return []


def buy_and_hold_agent_factory(tickers: list[str], notional_per_ticker: float = 10_000.0) -> AgentFn:
    """Buys once on the first day it's called for each ticker, then never
    trades again. This is the passive benchmark referenced throughout the
    proposal's metrics section (e.g. "if just buying the S&P 500 made 12%
    and your agent made 4%, your system is bad")."""
    bought: set[str] = set()

    def agent(decision_snapshots, positions, cash, as_of) -> list[Order]:
        orders = []
        for t in tickers:
            if t in bought:
                continue
            snap = decision_snapshots.get(t, {})
            px_feats = snap.get("derived_features", {})
            prices = snap.get("prices")
            if prices is None or prices.empty:
                continue
            last_close = float(prices.sort_values("date").iloc[-1]["adj_close"])
            if last_close <= 0:
                continue
            qty = notional_per_ticker / last_close
            orders.append(Order(order_id="", timestamp=as_of.isoformat(), symbol=t,
                                 side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=qty))
            bought.add(t)
        return orders

    return agent


def random_agent_factory(tickers: list[str], trade_prob: float = 0.1,
                          notional: float = 2_000.0, seed: int = 0) -> AgentFn:
    """Trades randomly with no information edge. A system that beats this
    baseline demonstrates something beyond noise; a system that doesn't is
    not adding value regardless of how sophisticated its coordination is."""
    rng = np.random.default_rng(seed)

    def agent(decision_snapshots, positions, cash, as_of) -> list[Order]:
        orders = []
        for t in tickers:
            if rng.random() > trade_prob:
                continue
            snap = decision_snapshots.get(t, {})
            prices = snap.get("prices")
            if prices is None or prices.empty:
                continue
            last_close = float(prices.sort_values("date").iloc[-1]["adj_close"])
            if last_close <= 0:
                continue
            side = OrderSide.BUY if rng.random() < 0.5 else OrderSide.SELL
            qty = notional / last_close
            orders.append(Order(order_id="", timestamp=as_of.isoformat(), symbol=t,
                                 side=side, order_type=OrderType.MARKET, quantity=qty))
        return orders

    return agent
