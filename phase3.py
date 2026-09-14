"""
Phase 3: episode runner, agent implementations, and the Joint Metric Report.

Consolidates what used to be four separate files (runner.py,
trading_agents_adapter.py, pit_fundamentals.py, metrics.py) into one, since
they form a single dependency chain end to end -- an EpisodeRun produced by
run_episode/run_all_episodes is exactly what build_joint_metric_report
consumes, and the TradingAgents adapter exists only to produce an AgentFn
those runners can drive. comparison.py (leaderboard + coordination breakeven
spread) stays separate; it's a consumer of this module's outputs, not part
of this dependency chain.

Sections, in pipeline order:

  1. EpisodeRun            -- the record type every other section produces
                               or consumes.
  2. Episode runner         -- drives an AgentFn through episodes day by day
                               (Phase 1 snapshots in, Phase 2 fills out).
  3. TradingAgents adapter  -- wires TauricResearch/TradingAgents into the
                               AgentFn interface, patching three temporal-
                               integrity gaps found by reading its source.
  4. Joint Metric Report    -- temporal integrity / regime robustness /
                               executable performance / factor attribution,
                               computed from EpisodeRun objects Phase 1 and
                               Phase 2 already produced (no re-simulation).

Deciding on T-1 data and executing at T's close (section 2) is a standard
daily-bar backtest simplification -- Phase 2's replay engine supports maker/
taker/impact mechanics, but this runner only feeds it one snapshot per
ticker per day, because the upstream data (FNSPID) is daily OHLCV, not an
intraday order book.

SYNTHETIC MICROSTRUCTURE NOTE: Phase 1's raw data has no bid/ask/depth.
`execution_harness` fabricates bid/ask from the close price and a flat
assumed spread, and depth from a fraction of trailing average volume. That
is a placeholder, not a validated microstructure model -- treat any
Executable Performance numbers as illustrative until real quote/depth data
is wired in.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import queue
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from azure_llm import _PROPAGATE_HARD_TIMEOUT_SECONDS, _request_with_hard_timeout
from dataprep import (
    PointInTimeStore,
    generate_episode_windows,
    assign_episode_regimes,
    label_regimes,
    load_prices,
    FACTOR_COLUMNS,
)
from execution_replay import (
    ExecutionReplayEngine,
    ReplayConfig,
    Order,
    OrderSide,
    OrderType,
    MarketRegime,
)
from execution_harness import execute_orders_for_day

log = logging.getLogger("phase3")

TRADING_DAYS_PER_YEAR = 252

REGIME_LABEL_TO_ENUM = {
    "trending": MarketRegime.TRENDING,
    "mean_reverting": MarketRegime.MEAN_REVERTING,
    "crisis": MarketRegime.CRISIS_HIGH_VOLATILITY,
}


# ---------------------------------------------------------------------------
# 1. EpisodeRun -- the record type every other section produces or consumes
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRun:
    """Everything the Joint Metric Report needs from one episode for one
    agent system. Deliberately holds raw Phase 1/2 outputs rather than
    pre-computed metrics, so every downstream number in this file is
    traceable back to an execution log row or an access-log row.

    One EpisodeRun per non-overlapping window from generate_episode_windows.
    A JointMetricReport is built from a list of these for a single agent
    system, so systems are compared report-to-report under identical
    episodes/regimes/costs."""

    episode_start: pd.Timestamp
    episode_end: pd.Timestamp
    regime: str  # "trending" | "mean_reverting" | "crisis"
    execution_log: list  # list[ExecutionFill] (Phase 2 dataclass instances,
    # or equivalent objects/dicts exposing the same fields)
    portfolio_values: pd.Series  # date-indexed, daily mark-to-market value
    access_log: list[dict]  # PointInTimeStore.access_log entries (Phase 1)

    def daily_returns(self) -> pd.Series:
        pv = self.portfolio_values.sort_index()
        return pv.pct_change().dropna()


def _fills_to_frame(execution_log) -> pd.DataFrame:
    if not execution_log:
        return pd.DataFrame(columns=["timestamp", "side", "executed_qty",
                                      "execution_price", "fee_paid", "regime"])
    rows = []
    for f in execution_log:
        # Accept either the Phase 2 ExecutionFill dataclass or a plain dict
        # (e.g. reloaded from execution_log.json) -- both expose the same
        # field names, so normalize once here rather than branching
        # throughout the rest of the module.
        rec = dataclasses.asdict(f) if dataclasses.is_dataclass(f) else dict(f)
        side = rec["side"]
        rec["side"] = side.value if hasattr(side, "value") else side
        regime = rec.get("regime")
        rec["regime"] = regime.value if hasattr(regime, "value") else regime
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Episode runner
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


class CadenceGatedAgent:
    """Wraps any AgentFn so it only actually decides every N trading days
    within an episode (N chosen per regime), holding on the days in
    between -- run_episode itself calls agent_fn unconditionally every
    trading day (see its loop above), so gating has to happen here, one
    layer up.

    Not TradingAgents-specific: used identically to reduce a single-agent
    baseline's decision frequency to match TradingAgents' reduced cadence
    for a valid, matched-cadence comparison (see run_tradingagents_episodes.py).

    The caller must call set_regime(regime) before each run_episode() call
    -- run_episode doesn't pass regime into agent_fn, so the orchestrator,
    which already knows the regime for the episode it's about to run, sets
    it explicitly rather than this class trying to infer it.

    Correctness note, found while designing this (not while debugging a
    live run): on a hold day, last_usage()/true_blended_cost_usd() must
    NOT just forward to the wrapped agent's own last_usage() -- that would
    return whatever the wrapped agent last reported for its most recent
    REAL decision, and CoordinationMeter would then double-count that same
    cost again on every hold day in between, since it calls
    agent.last_usage() unconditionally after every agent_fn(...) call. This
    class tracks whether its own most recent __call__ was a real decision
    or a hold and returns explicit zero usage on hold days."""

    def __init__(self, agent, regime_intervals: dict[str, int]):
        self.agent = agent
        self.regime_intervals = regime_intervals
        # Deliberately NOT getattr(agent, "n_agents", 1) -- confirmed live:
        # CoordinationMeter._record() has `if not calls and self.n_agents > 1:
        # calls = self.n_agents`, a fallback meant for "the agent didn't
        # expose real measurement, assume one call per seat." On a hold day
        # this class's last_usage() correctly reports {} (zero calls is the
        # TRUE, measured value, not a measurement failure), but
        # CoordinationMeter can't tell the difference between "empty because
        # unmeasured" and "empty because genuinely zero" -- it only checks
        # `not calls`. With n_agents inherited from the wrapped multi-agent
        # system (e.g. 2-6), every one of the many hold days in a reduced-
        # cadence run was silently getting an ASSUMED, non-zero cost
        # (n_agents calls at the generic assumed-tokens-per-call rate)
        # instead of the real zero -- caught in a real smoke test (62-day
        # episode, 55 hold days, reported cost inflated by exactly this).
        # n_agents=1 makes the fallback's `self.n_agents > 1` always False,
        # so a real zero is never overwritten; real decision days are
        # unaffected since they already report genuine non-empty usage.
        self.n_agents = 1
        self._day_index = 0
        self._regime: str | None = None
        self._last_call_was_real = False

    def set_regime(self, regime: str) -> None:
        """Call once at the start of each episode. Also resets the wrapped
        agent's own state (e.g. _LLMAgent.reset() clears trade history so
        it can't leak into the next episode's prompts) -- a new episode
        always means both a (possibly different) regime AND a fresh agent
        in this pipeline's own convention (run_episode's docstring: "the
        agent resets at every episode start"), so this does both in one
        call rather than making every caller remember two separate calls."""
        self._regime = regime
        self._day_index = 0
        if hasattr(self.agent, "reset"):
            self.agent.reset()

    def reset(self) -> None:
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        self._day_index = 0

    def __call__(self, decision_snapshots, positions, cash, as_of):
        interval = self.regime_intervals.get(self._regime, 1)
        is_decision_day = (self._day_index % interval) == 0
        self._day_index += 1
        self._last_call_was_real = is_decision_day
        if not is_decision_day:
            return []
        return self.agent(decision_snapshots, positions, cash, as_of)

    def last_usage(self) -> dict:
        if not self._last_call_was_real:
            return {}
        if hasattr(self.agent, "last_usage"):
            return self.agent.last_usage()
        return {}

    def true_blended_cost_usd(self) -> float:
        if not self._last_call_was_real:
            return 0.0
        if hasattr(self.agent, "true_blended_cost_usd"):
            return self.agent.true_blended_cost_usd()
        return 0.0


def run_episode(
    agent_fn: AgentFn,
    store: PointInTimeStore,
    tickers: list[str],
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    regime: str,
    replay_config: ReplayConfig | None = None,
) -> EpisodeRun:
    """Runs one agent through one episode window, day by day.

    Pipeline per trading day t:
      1. decision snapshot = store.snapshot_as_of(midnight of day t)
         -> this only ever contains data with available_at < midnight(t),
            i.e. strictly before today -- the agent decides using
            yesterday's information, never today's own close.
      2. agent_fn(decision_snapshot, positions, cash) -> list[Order]
      3. each order is executed against day t's *own* market snapshot
         (synthesized bid/ask/depth around day t's close) via Phase 2's
         ExecutionReplayEngine.
      4. portfolio value is marked to day t's close.

    `store` should be a fresh PointInTimeStore (or at least have an
    access_log you're willing to attribute entirely to this episode) so the
    temporal-integrity report is scoped correctly per episode."""
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

        # news_lookback/fundamentals_lookback_quarters bound snapshot_as_of's
        # news and fundamentals reads to the partitions they can actually use
        # (see dataprep.PointInTimeStore.snapshot_as_of) instead of scanning
        # the full news history / opening every quarterly fundamentals
        # archive once per ticker per day. 8 quarters (2 years) is generous
        # slack for "most recent as-filed fact per tag" -- companies file
        # quarterly, so the fact this feeds derived_features from is almost
        # always inside the last one or two archives, not near the edge of
        # this window.
        decision_snapshots = {
            t: store.snapshot_as_of(t, decision_as_of, news_lookback=pd.Timedelta(days=3),
                                     fundamentals_lookback_quarters=8)
            for t in tickers
        }

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
    through every window and returns the list[EpisodeRun] ready for
    build_joint_metric_report(). A new PointInTimeStore per episode keeps
    each episode's temporal-integrity log scoped to that episode, matching
    what regime_robustness_table / temporal_integrity_report expect."""
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
        # Duck-typed, not every AgentFn has state to clear (buy_and_hold's
        # closure doesn't) -- but one that does (e.g. _LLMAgent's trade
        # history, fed into its own prompt for narrative continuity) must
        # not carry state across a purge gap into an unrelated episode.
        if hasattr(agent_fn, "reset"):
            agent_fn.reset()
        store = store_factory()
        run = run_episode(agent_fn, store, tickers, ep_start, ep_end, regime, replay_config)
        runs.append(run)
        log.info("episode %s..%s (%s): %d fills, %d trading days",
                  ep_start.date(), ep_end.date(), regime, len(run.execution_log), len(run.portfolio_values))
    return runs


# ---------------------------------------------------------------------------
# 3. TradingAgents adapter
# ---------------------------------------------------------------------------
# Wires TauricResearch/TradingAgents into the AgentFn interface above,
# applying three fixes identified from actually reading the vendored source:
#
#   a. PiT-safe fundamentals -- TradingAgents' get_fundamentals accepts a
#      curr_date parameter all the way down to the vendor layer (see
#      tradingagents/agents/utils/fundamental_data_tools.py ->
#      route_to_vendor("get_fundamentals", ticker, curr_date)) -- but
#      neither the yfinance nor Alpha Vantage vendor implementation actually
#      uses it (both literally say "not used for X" in their own
#      docstrings). Every call returns today's live company snapshot
#      regardless of what historical date is being simulated -- a confirmed
#      look-ahead leak for any TradingAgents run inside this benchmark.
#      pit_safe_get_fundamentals below is a drop-in replacement, same call
#      signature (ticker, curr_date) -> str, backed by Phase 1's
#      PointInTimeStore instead of a live API.
#   b. get_insider_transactions -- excluded entirely, via a subclass that
#      rebuilds the "news" tool node without it. Its tool schema has no date
#      parameter at all, so there's nothing to patch; excluding is safer
#      than leaving a tool bound to the agent that structurally cannot
#      respect the temporal wall.
#   c. Cross-episode memory isolation -- TradingAgents persists a decision
#      log across runs (~/.tradingagents/memory/trading_memory.md) and
#      injects realized outcomes from prior runs into future prompts. That's
#      legitimate *if and only if* episodes for one system are always run
#      in true chronological order against one isolated memory file. This
#      module gives every system-under-test its own memory path and never
#      reorders episodes, but does NOT protect you if you re-run episodes
#      out of order yourself -- always call run_all_episodes with a single
#      forward pass over the date range, per system.
#
# NOT fixed here, flagged as still-open:
#   - Balance sheet / cashflow / income statement filter by fiscal
#     period-end date, not filing date -- a smaller, unconfirmed leak risk.
#   - News internals were spot-checked (get_news_yfinance genuinely filters
#     by pub_date) but not exhaustively audited across every vendor path the
#     way fundamentals was.
# ---------------------------------------------------------------------------


def pit_safe_get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """PiT-safe replacement for TradingAgents' get_fundamentals vendor call.
    Same signature and same kind of return value (a formatted text report,
    since this gets fed straight into an LLM prompt) as the yfinance/Alpha
    Vantage versions it replaces -- so no other TradingAgents code needs to
    change, only the vendor registration (register_pit_fundamentals_vendor
    below).

    Uses whatever fundamentals + derived ratios Phase 1's PointInTimeStore
    can produce as of curr_date. This will have different fields than the
    yfinance version (e.g. no PE ratio, since that needs live analyst
    estimates Phase 1 doesn't have) -- it reports what was genuinely
    knowable at curr_date, not a feature-parity clone of the live version.

    get_insider_transactions is NOT patched here -- its tool wrapper
    (tradingagents/agents/utils/news_data_tools.py) doesn't accept curr_date
    at all, so there's no date to thread through even if a vendor respected
    it, and insider-transaction data isn't part of Phase 1's PiT store to
    begin with. It's excluded from the toolset entirely instead -- see
    make_benchmark_safe_graph_class below."""
    if not curr_date:
        raise ValueError(
            "get_fundamentals requires curr_date -- refusing to guess a "
            "cutoff, since silently defaulting to 'now' is exactly the "
            "leak this function exists to prevent."
        )

    as_of = pd.Timestamp(curr_date, tz="UTC")
    store = PointInTimeStore()
    snap = store.snapshot_as_of(ticker, as_of)

    fundamentals = snap["fundamentals"]
    derived = snap["derived_features"]

    lines = [f"# Fundamentals for {ticker} as of {curr_date} (point-in-time)"]

    ratio_labels = {
        "leverage_ratio": "Leverage Ratio (Liabilities/Assets)",
        "current_ratio": "Current Ratio",
        "return_on_assets": "Return on Assets",
    }
    any_ratio = False
    for key, label in ratio_labels.items():
        value = derived.get(key)
        if value is not None:
            lines.append(f"{label}: {value:.4f}")
            any_ratio = True

    for key in ("return_5d", "return_21d", "return_63d", "realized_vol"):
        value = derived.get(key)
        if value is not None:
            lines.append(f"{key}: {value:.4f}")
            any_ratio = True

    if fundamentals is not None and not fundamentals.empty:
        latest = (
            fundamentals.sort_values("ddate")
            .drop_duplicates(subset=["tag"], keep="last")
            .sort_values("tag")
        )
        lines.append("")
        lines.append("# Most recent as-filed XBRL facts known as of this date:")
        for _, row in latest.iterrows():
            lines.append(f"{row['tag']}: {row['value']} (period end {row['ddate'].date()})")
        any_ratio = True

    if not any_ratio:
        return (
            f"NO_DATA_AVAILABLE: no point-in-time fundamentals known for "
            f"'{ticker}' as of {curr_date}. Do not estimate or fabricate "
            f"values -- report that fundamentals data is unavailable."
        )

    return "\n".join(lines)


def register_pit_fundamentals_vendor() -> None:
    """Call once, before constructing any TradingAgentsGraph. Adds our
    PointInTimeStore-backed implementation as a selectable vendor and
    nothing else -- doesn't touch yfinance/Alpha Vantage's registrations,
    so anything that explicitly asks for those vendors by name still gets
    the (broken) original, which is why get_pit_fundamentals_config()
    below also sets tool_vendors explicitly rather than relying on this
    registration alone."""
    from tradingagents.dataflows import interface as ta_interface

    ta_interface.VENDOR_METHODS["get_fundamentals"]["benchmark_pit"] = pit_safe_get_fundamentals


def register_wide_ohlcv_window() -> None:
    """Call once, before constructing any TradingAgentsGraph. Patches
    tradingagents.dataflows.stockstats_utils.load_ohlcv, whose price-history
    fetch/cache window is hardcoded to "5 years before wall-clock today"
    (today_date = pd.Timestamp.today(); start_date = today_date -
    DateOffset(years=5)) -- NOT relative to curr_date, the date actually
    being backtested. Confirmed live in a pre-run smoke test: every episode
    window starting more than ~5 years before the real calendar date this
    was run (15 of this benchmark's 18 balanced-regime episodes -- anything
    before ~2021) made get_verified_market_snapshot raise "No OHLCV data
    available" on every single call, silently caught by
    TradingAgentsAdapter's own exception handler as a Hold. Not a crash,
    but real money would have been spent on decisions with no real market
    data behind them for 83% of episodes -- caught before the paid run, not
    after.

    Fix: widen the raw fetch window to a fixed 2005-01-01..today range
    (Yahoo Finance returns this much history in one request without
    trouble) rather than trying to compute a per-curr_date window. This is
    safe: the existing look-ahead filter inside load_ohlcv
    (`data[data["Date"] <= curr_date_dt]`) is untouched and is the actual
    point-in-time enforcement here -- widening the raw fetch does not
    change what any call is allowed to see, only what's available to
    filter from. The rest of the original's logic (on-disk caching,
    yf_retry, the same-day-refresh check, the stale-frame assertion) is
    reused as-is via the module's own helpers, not reimplemented, so this
    changes only the window.

    load_ohlcv is imported by name (`from ... import load_ohlcv`) in three
    separate modules -- stockstats_utils itself, market_data_validator.py,
    and y_finance.py (confirmed by grepping the installed package) -- so
    the replacement has to be assigned in each of their namespaces
    individually; patching only the origin module leaves the other two
    still bound to the original function object."""
    from tradingagents.dataflows import market_data_validator as mdv
    from tradingagents.dataflows import stockstats_utils as ssu
    from tradingagents.dataflows import y_finance as yf_mod

    fixed_start = pd.Timestamp("2005-01-01")

    def _wide_load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
        canonical = ssu.normalize_symbol(symbol)
        safe_symbol = ssu.safe_ticker_component(canonical)

        config = ssu.get_config()
        curr_date_dt = pd.to_datetime(curr_date)

        today_date = pd.Timestamp.today()
        start_str = fixed_start.strftime("%Y-%m-%d")
        end_str = (today_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        os.makedirs(config["data_cache_dir"], exist_ok=True)
        data_file = os.path.join(
            config["data_cache_dir"],
            f"{safe_symbol}-YFin-data-{start_str}-{end_str}.csv",
        )

        data = None
        if os.path.exists(data_file):
            cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
            if (
                not cached.empty
                and "Close" in cached.columns
                and not ssu._needs_same_day_refresh(data_file, curr_date_dt, today_date)
            ):
                data = cached

        if data is None:
            downloaded = ssu.yf_retry(lambda: ssu.yf.download(
                canonical, start=start_str, end=end_str,
                multi_level_index=False, progress=False, auto_adjust=True,
            ))
            downloaded = ssu._ensure_date_column(downloaded.reset_index())
            if downloaded.empty or "Close" not in downloaded.columns:
                raise ssu.NoMarketDataError(symbol, canonical, "Yahoo Finance returned no rows")
            downloaded.to_csv(data_file, index=False, encoding="utf-8")
            data = downloaded

        data = ssu._clean_dataframe(data)
        data = data[data["Date"] <= curr_date_dt]
        ssu._assert_ohlcv_not_stale(data, curr_date, symbol, canonical)
        return data

    ssu.load_ohlcv = _wide_load_ohlcv
    mdv.load_ohlcv = _wide_load_ohlcv
    yf_mod.load_ohlcv = _wide_load_ohlcv


def get_pit_fundamentals_config(base_config: dict) -> dict:
    """Returns a copy of base_config with get_fundamentals routed to our
    PiT-safe vendor. Doesn't touch get_balance_sheet/get_cashflow/
    get_income_statement -- those are the "not fixed here" period-date gap
    noted in the section docstring above, left on their default vendor."""
    config = dict(base_config)
    tool_vendors = dict(config.get("tool_vendors", {}))
    tool_vendors["get_fundamentals"] = "benchmark_pit"
    config["tool_vendors"] = tool_vendors
    return config


def make_benchmark_safe_graph_class():
    """Returns a TradingAgentsGraph subclass with get_insider_transactions
    dropped from the "news" tool node. Built as a factory function (rather
    than a module-level class) so importing this module doesn't require
    tradingagents to be installed unless you actually call this."""
    from langgraph.prebuilt import ToolNode
    from tradingagents.agents.utils.agent_utils import (
        get_balance_sheet,
        get_cashflow,
        get_fundamentals,
        get_global_news,
        get_income_statement,
        get_indicators,
        get_macro_indicators,
        get_news,
        get_prediction_markets,
        get_stock_data,
        get_verified_market_snapshot,
    )
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    class BenchmarkSafeTradingAgentsGraph(TradingAgentsGraph):
        """Identical to TradingAgentsGraph except the "news" tool node
        excludes get_insider_transactions -- see section docstring above
        for why. Everything else is reproduced exactly as trading_graph.py
        builds it, so if their tool lists change upstream, this needs
        re-syncing (a known maintenance cost of excluding via
        reconstruction rather than a config flag; there isn't one)."""

        def _create_tool_nodes(self) -> dict[str, ToolNode]:
            return {
                "market": ToolNode([
                    get_stock_data,
                    get_indicators,
                    get_verified_market_snapshot,
                ]),
                "social": ToolNode([
                    get_news,
                ]),
                "news": ToolNode([
                    get_news,
                    get_global_news,
                    # get_insider_transactions intentionally omitted --
                    # no date parameter exists on its tool schema at all,
                    # so it cannot respect the episode's as_of cutoff.
                    get_macro_indicators,
                    get_prediction_markets,
                ]),
                "fundamentals": ToolNode([
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]),
            }

    return BenchmarkSafeTradingAgentsGraph


def isolated_memory_path(system_name: str, base_dir: str = "./ta_memory") -> Path:
    """A fresh memory file per system-under-test, so one system's run never
    contaminates another's, and so re-running a system from scratch (e.g.
    after a code change) starts clean rather than inheriting stale
    reflections from a previous, possibly-different version of itself."""
    path = Path(base_dir) / f"{system_name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_trading_agents_graph(system_name: str, base_config: dict, memory_base_dir: str = "./ta_memory"):
    """Constructs a fully patched TradingAgentsGraph: PiT fundamentals
    vendor registered, insider-transactions tool excluded, memory path
    isolated to this system, price-history fetch window widened to cover
    this benchmark's full 2010-2023 episode range (see
    register_wide_ohlcv_window -- without it, get_stock_data/
    get_indicators/get_verified_market_snapshot fail outright for any
    episode older than ~5 years before the real calendar date this is
    run, which is 15 of this benchmark's 18 episodes). Requires
    OpenAI/Anthropic/etc. credentials to already be configured per
    TradingAgents' normal setup -- this function only handles
    benchmark-integrity fixes, not general setup."""
    from tradingagents.default_config import DEFAULT_CONFIG

    register_pit_fundamentals_vendor()
    register_wide_ohlcv_window()
    # Merge over a fresh DEFAULT_CONFIG copy rather than trusting
    # base_config alone: TradingAgentsGraph.__init__ does `self.config =
    # config or DEFAULT_CONFIG` -- no per-key fallback -- so a caller
    # passing a partial dict (e.g. just llm_provider/deep_think_llm/
    # quick_think_llm, which is the natural thing to pass) previously hit
    # `self.config["data_cache_dir"]` with that key entirely absent: a
    # guaranteed KeyError on every construction, confirmed live.
    config = get_pit_fundamentals_config({**DEFAULT_CONFIG, **base_config})

    # Set memory_log_path directly on our own config dict rather than via
    # the TRADINGAGENTS_MEMORY_LOG_PATH env var: DEFAULT_CONFIG bakes that
    # env var's value in at the *first* time tradingagents.default_config
    # is imported anywhere in the process (a module-level dict, evaluated
    # once and cached) -- callers typically import it once already just to
    # build base_config, before ever reaching this function, so setting
    # the env var here is too late to change the already-materialized
    # DEFAULT_CONFIG. Writing the key onto our own dict has no such
    # ordering dependency and is what actually reaches self.config.
    config["memory_log_path"] = str(isolated_memory_path(system_name, memory_base_dir))

    GraphClass = make_benchmark_safe_graph_class()
    return GraphClass(config=config)


# TradingAgents' propagate() returns exactly one of five strings: "Buy",
# "Overweight", "Hold", "Underweight", "Sell" (tradingagents/agents/schemas.py
# PortfolioRating, extracted deterministically by SignalProcessor -- no
# free-text parsing needed). We treat each rating as a *target position*
# expressed as a fraction of notional_per_ticker, then submit an order for
# whatever delta gets the current position to that target. This is a
# rebalancing interpretation, not the only valid one -- e.g. you could
# instead treat every "Buy"/"Sell" as a fixed-size trade regardless of
# current position. Rebalancing was chosen because it's the only
# interpretation of a 5-tier *rating* (as opposed to a 3-tier *action*)
# that uses Overweight/Underweight for anything at all.
RATING_TO_TARGET_FRACTION = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": None,  # None = leave the current position unchanged
    "Underweight": 0.25,
    "Sell": 0.0,
}

MIN_REBALANCE_NOTIONAL = 1.0  # skip order submission below this $ delta (avoid dust-sized orders)

# Confirmed live, in the real full run before this fix existed: at
# n_workers=10 (one worker per ticker, fully parallel), roughly 7 of every
# 10 concurrent propagate() calls hit openai.RateLimitError (HTTP 429,
# "RateLimitReached") against the Azure Kimi-K2.6 deployment -- recurring
# every real decision-wave (roughly every 6-7 minutes), not just a
# startup burst. Before this fix, _rate_and_build_order's blanket
# `except Exception` treated a 429 identically to a genuine failure --
# one immediate give-up, logged, silently defaulted to Hold. That is
# exactly the failure mode already called out in this project's own
# azure_llm.py retry logic and in CoordinationMeter's design notes ("an
# unretried transient failure silently becomes a HOLD indistinguishable
# from a real decision, confounding infrastructure noise with trading
# behavior") -- it just hadn't been applied here yet. A rate-limited
# ticker on ~70% of real decisions is not a minor availability blip; it
# would have made most of a paid run's "decisions" rate-limit noise
# rather than TradingAgents' actual output.
#
# A first version of this retry (fixed exponential backoff, no jitter)
# was itself confirmed live to be insufficient on its own: with
# n_workers=10, every worker hits the limit at nearly the same instant
# (they all started concurrently), backs off on the identical schedule,
# and then all retry in near-lockstep -- reproducing the same burst, and
# the same 429, on every retry cycle. Observed live as a 15-minute stretch
# with almost no forward progress (CPU time barely moved while wall clock
# did) rather than a clean recover-and-continue. Random jitter on the
# delay is the standard fix for exactly this thundering-herd pattern --
# it desynchronizes concurrent retriers so they stop arriving at the
# limit together.
_RATE_LIMIT_MAX_RETRIES = 6
_RATE_LIMIT_BASE_DELAY_SECONDS = 20.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 180.0
_RATE_LIMIT_JITTER_FRACTION = 0.5  # actual delay = base * uniform(1 - frac, 1 + frac)


class TradingAgentsAdapter:
    """CoordinationMeter-compatible AgentFn: rates every ticker each day
    against a POOL of already-patched TradingAgentsGraph instances and
    translates each rating into a rebalancing Order. Also exposes
    last_usage(), so CoordinationMeter meters this on REAL measured token
    counts (its "way 1", most trustworthy) instead of falling back to an
    assumed per-call estimate or wall-clock-only.

    Takes a pool of graph instances -- build several via
    build_trading_agents_graph with the same system_name/config (so they
    share one isolated memory log) and pass them as a list -- rather than
    a single shared instance. propagate() mutates
    self.ticker/self.graph/self._checkpointer_ctx directly on the graph
    object (confirmed by reading trading_graph.py), so two tickers rated
    concurrently on ONE shared instance would silently contaminate each
    other's runs. With len(ta_graph_pool) instances, up to that many
    tickers rate concurrently instead of one at a time; the dominant
    per-ticker cost is real, sequential LLM latency inside propagate()
    (13-16 calls observed for a single ticker/day at this framework's own
    minimum debate-round setting), which this does not reduce in content,
    only in how many tickers pay that cost simultaneously. The memory
    log's own write (tradingagents/agents/utils/memory.py:store_decision)
    is an unlocked read-then-append, so concurrent writers can
    occasionally produce a duplicate "pending" entry for the same
    ticker+date -- accepted, since it only affects a future
    self-reflection prompt's context, not the rating returned for any
    ticker on any day.

    Usage measurement: each ticker's propagate() call runs inside a fresh
    langchain_core UsageMetadataCallbackHandler context
    (get_usage_metadata_callback() -- a context-var-based global hook, so
    it captures every LangChain chat-model call made anywhere inside
    LangGraph's execution, without TradingAgents needing to accept or
    forward a callbacks= parameter itself). Being context-var-based, it is
    correctly isolated per worker thread with no extra locking needed on
    our side. TradingAgents mixes two real models per decision --
    deep_think_llm (Sonnet, the expensive multi-round analysis/debate/risk
    calls) and quick_think_llm (Haiku, lighter calls) -- but
    CoordinationCostConfig prices a whole system at ONE $/token rate via
    cfg.model. last_usage() reports the real total input/output tokens
    summed across BOTH models; run_tradingagents_episodes.py sets
    cfg.model="claude-sonnet-5" (the more expensive of the two), so the
    dollar figure CoordinationMeter derives from this is a real measured
    token count priced at a deliberately conservative upper bound, not an
    exact blended figure -- true_blended_cost_usd() below computes the
    exact number (each model's real tokens at its own real rate) as a
    separately-reported supplement.

    Note this makes one full multi-agent LLM run per ticker per day --
    expect this to cost real API credits at scale even with concurrency.
    Consider testing on a small ticker list and a short date range
    first."""

    def __init__(self, ta_graph_pool: Sequence, tickers: list[str],
                 notional_per_ticker: float = 10_000.0):
        self._graph_pool: "queue.Queue" = queue.Queue()
        for g in ta_graph_pool:
            self._graph_pool.put(g)
        self.n_agents = len(ta_graph_pool)
        self.tickers = tickers
        self.notional_per_ticker = notional_per_ticker
        self._last_usage_by_model: dict[str, dict[str, int]] = {}
        # Running total across every __call__ since construction --
        # true_blended_cost_usd() only ever reflects the MOST RECENT call,
        # which run_tradingagents_episodes.py's first version wrongly
        # summed once per episode (after run_episode() had already made
        # many calls internally), silently reporting only the last
        # decision's cost as if it were the whole episode's. Found during
        # a pre-run code review, not in a live run.
        self._cumulative_usage_by_model: dict[str, dict[str, int]] = {}

    def _rate_and_build_order(self, ticker, as_of, decision_snapshots, positions):
        from langchain_core.callbacks import get_usage_metadata_callback

        trade_date = as_of.strftime("%Y-%m-%d")
        snap = decision_snapshots.get(ticker, {})
        prices = snap.get("prices")
        if prices is None or prices.empty:
            return None, {}
        last_close = float(prices.sort_values("date").iloc[-1]["adj_close"])
        if last_close <= 0:
            return None, {}

        graph = self._graph_pool.get()

        def _propagate_and_capture_usage():
            # get_usage_metadata_callback's registration is contextvars-based,
            # and a raw threading.Thread does NOT inherit the spawning
            # thread's contextvars.Context -- confirmed live: entering this
            # context manager around the OUTSIDE of
            # _request_with_hard_timeout (which runs graph.propagate on a
            # freshly spawned daemon thread, see azure_llm.py) captured
            # zero tokens despite a real, complete, 7+ minute propagate()
            # call underneath it. The context has to be entered INSIDE the
            # function that actually runs on that daemon thread, which is
            # what this closure exists for.
            with get_usage_metadata_callback() as cb:
                result = graph.propagate(ticker, trade_date)
            return result, dict(cb.usage_metadata)

        import openai

        delay = _RATE_LIMIT_BASE_DELAY_SECONDS
        try:
            for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    (_, rating), usage_by_model = _request_with_hard_timeout(
                        _propagate_and_capture_usage,
                        hard_timeout=_PROPAGATE_HARD_TIMEOUT_SECONDS,
                    )
                    break
                except openai.RateLimitError as exc:
                    if attempt == _RATE_LIMIT_MAX_RETRIES:
                        raise
                    jittered = delay * random.uniform(
                        1 - _RATE_LIMIT_JITTER_FRACTION, 1 + _RATE_LIMIT_JITTER_FRACTION
                    )
                    log.warning(
                        "%s on %s: rate limited (attempt %d/%d), retrying in %.0fs: %s",
                        ticker, trade_date, attempt + 1, _RATE_LIMIT_MAX_RETRIES, jittered, exc,
                    )
                    time.sleep(jittered)
                    delay = min(delay * 2, _RATE_LIMIT_MAX_DELAY_SECONDS)
        except Exception:
            # A genuine failure (not a rate limit -- those are retried
            # above) or a stalled propagate() call (see
            # _PROPAGATE_HARD_TIMEOUT_SECONDS -- TradingAgents' Yahoo
            # Finance tools have no reliable timeout of their own,
            # confirmed live) shouldn't crash an entire multi-year
            # backtest -- treat it as a no-trade day for this ticker
            # and move on, same as run_episode already does for a
            # crashing agent_fn at the whole-day level. Logged (not just
            # swallowed) -- this except block previously had no logging at
            # all, so a systematic failure (e.g. every real call failing
            # the same way) would produce zero fills and zero cost with no
            # visible cause, indistinguishable from "every day was a
            # legitimate Hold."
            log.exception("propagate() failed for %s on %s -- treating as Hold", ticker, trade_date)
            return None, {}
        finally:
            self._graph_pool.put(graph)

        target_fraction = RATING_TO_TARGET_FRACTION.get(rating)
        if target_fraction is None:
            return None, usage_by_model  # Hold, or an unrecognized rating string

        target_qty = (target_fraction * self.notional_per_ticker) / last_close
        current_qty = positions.get(ticker, 0.0)
        delta_qty = target_qty - current_qty

        if abs(delta_qty * last_close) < MIN_REBALANCE_NOTIONAL:
            return None, usage_by_model

        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        order = Order(
            order_id="", timestamp=as_of.isoformat(), symbol=ticker,
            side=side, order_type=OrderType.MARKET, quantity=abs(delta_qty),
        )
        return order, usage_by_model

    def __call__(self, decision_snapshots, positions, cash, as_of):
        orders = []
        combined: dict[str, dict[str, int]] = {}
        n_calls = 0
        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = [
                pool.submit(self._rate_and_build_order, ticker, as_of, decision_snapshots, positions)
                for ticker in self.tickers
            ]
            for future in as_completed(futures):
                order, usage_by_model = future.result()
                if order is not None:
                    orders.append(order)
                for raw_model_name, um in usage_by_model.items():
                    # response_metadata["model_name"] is inconsistent across
                    # both model families AND routes -- confirmed live twice:
                    # (1) langchain-anthropic: Haiku 4.5 comes back dated
                    # ("claude-haiku-4-5-20251001"), Sonnet 5 undated
                    # ("claude-sonnet-5"), same SDK version. (2) the Azure
                    # Foundry DeepSeek deployment routed through
                    # langchain_openai.ChatOpenAI echoes back the exact
                    # deployment name as configured in the portal
                    # ("DeepSeek-V4-Pro"), not the lowercase alias
                    # coordination_cost.MODEL_PRICES is keyed on
                    # ("deepseek-v4-pro") -- confirmed live: this silently
                    # zeroed true_blended_cost_usd() with no error, not an
                    # imprecise number but a wrong one (price=None ->
                    # excluded, not just uncredited). Strip a trailing
                    # -YYYYMMDD if present and lowercase; a no-op on names
                    # that already match.
                    model_name = re.sub(r"-\d{8}$", "", raw_model_name).lower()
                    n_calls += 1
                    agg = combined.setdefault(model_name, {"input_tokens": 0, "output_tokens": 0})
                    agg["input_tokens"] += int(um["input_tokens"])
                    agg["output_tokens"] += int(um["output_tokens"])
        self._last_usage_by_model = combined
        self._last_n_calls = n_calls
        for model_name, m in combined.items():
            cum = self._cumulative_usage_by_model.setdefault(model_name, {"input_tokens": 0, "output_tokens": 0})
            cum["input_tokens"] += m["input_tokens"]
            cum["output_tokens"] += m["output_tokens"]
        return orders

    def last_usage(self) -> dict:
        """CoordinationMeter's real-measurement path (see class docstring
        for the single-price-vs-mixed-model caveat)."""
        total_in = sum(m["input_tokens"] for m in self._last_usage_by_model.values())
        total_out = sum(m["output_tokens"] for m in self._last_usage_by_model.values())
        return {
            "llm_calls": self._last_n_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "messages": max(self.n_agents - 1, 0),
        }

    @staticmethod
    def _price_usage_by_model(usage_by_model: dict[str, dict[str, int]]) -> float:
        from coordination_cost import MODEL_PRICES

        total = 0.0
        for model_name, usage in usage_by_model.items():
            price = MODEL_PRICES.get(model_name)
            if price is None:
                log.warning("no MODEL_PRICES entry for %r -- excluded from cost total", model_name)
                continue
            total += (usage["input_tokens"] * price.input_per_mtok
                      + usage["output_tokens"] * price.output_per_mtok) / 1_000_000.0
        return total

    def true_blended_cost_usd(self) -> float:
        """Exact dollar cost for the most recent __call__ ONLY: each
        model's real measured tokens priced at that model's own real
        MODEL_PRICES rate, summed -- not an approximation, but also not a
        running total. Use cumulative_true_blended_cost_usd() for the
        total across an episode or a full run -- conflating the two is
        exactly the bug that motivated splitting them (see
        _cumulative_usage_by_model's comment in __init__)."""
        return self._price_usage_by_model(self._last_usage_by_model)

    def cumulative_true_blended_cost_usd(self) -> float:
        """Exact dollar cost across every __call__ since construction --
        the number to report for an episode or a full run. Supplements
        (does not replace) CoordinationMeter's own report, which prices
        the total at one rate (see class docstring)."""
        return self._price_usage_by_model(self._cumulative_usage_by_model)


# ---------------------------------------------------------------------------
# 4. Joint Metric Report
# ---------------------------------------------------------------------------
# Aggregates EpisodeRun objects into the four report categories described in
# the proposal's Joint Metric Collection Phase:
#
#   a. Temporal Integrity     -- from PointInTimeStore.access_log (Phase 1)
#   b. Regime Robustness      -- performance grouped by episode regime label
#                                (Phase 1's assign_episode_regimes)
#   c. Executable Performance -- net return / Sharpe / Sortino / max drawdown
#                                / breakeven threshold, computed from the
#                                cost-adjusted execution log (Phase 2)
#   d. Attribution            -- OLS alpha/beta vs Market/Size/Value factors,
#                                with block-bootstrap 95% CIs on Sharpe,
#                                Sortino, and alpha across episodes
#
# This section does not re-run the agent or re-simulate the market.
# Everything here is arithmetic over data sections 2/4 already produced -- an
# episode's execution log and access log are treated as fixed inputs, exactly
# the same "no need to rerun" principle Phase 2 uses for its breakeven sweep.
# ---------------------------------------------------------------------------


@dataclass
class TemporalIntegrityReport:
    total_queries: int
    total_rows_returned: int
    total_rows_blocked: int  # informational: see note below, NOT a failure count
    explicit_check_violations: int  # from PointInTimeStore.check() calls, if any were made
    checked_episodes: list[str]  # episode_start ISO strings where check() was actually exercised

    @property
    def clean(self) -> bool:
        # A blocked row is the filter *working*, not a leak -- see below.
        # The only genuine violation signal available from the access log is
        # an explicit check() call reporting violations > 0.
        return self.explicit_check_violations == 0


def temporal_integrity_report(episodes: Sequence[EpisodeRun]) -> TemporalIntegrityReport:
    """IMPORTANT: given how Phase 1's PointInTimeStore is implemented
    (prices_as_of/news_as_of/fundamentals_as_of load the dataset and then
    filter to `available_at < as_of`), every ordinary query blocks whatever
    rows are still in the future relative to that day. That is the guard
    working correctly, not an attempted leak -- treating total_rows_blocked
    as a failure count misreports completely normal operation as a
    violation. Expect total_rows_blocked > 0 on essentially every real run;
    it's kept here only as a diagnostic.

    The one signal in the access log that *does* indicate a genuine
    violation is an explicit PointInTimeStore.check() call -- used for
    auditing a candidate access request such as an inter-agent message or
    tool-output timestamp -- reporting rows_blocked > 0. Those log entries
    are distinguished from ordinary loader calls by dataset names your
    check() call sites choose (e.g. "agent_message", "tool_output"); this
    function treats any access_log entry whose dataset name is not one of
    the raw loaders as an explicit check and sums its violations. Kept in
    sync with dataprep.PointInTimeStore's own loader methods by name (not by
    import, to avoid a phase3<->dataprep cycle) -- prices/news/fundamentals
    plus the three added since: bars_as_of ("intraday_bars"), quotes_as_of
    ("quotes"), corporate_actions_as_of ("corporate_actions"). Not currently
    reachable from main.py's --agent llm path (phase3.run_episode's own
    store only ever calls snapshot_as_of), but Phase15Bridge.run_day and
    cbs.py both call quotes_as_of directly -- if either is ever wired into a
    JointMetricReport, an incomplete set here would misreport every ordinary
    quote-blocking event as a genuine temporal-integrity violation."""
    ORDINARY_LOADERS = {"prices", "news", "fundamentals",
                         "intraday_bars", "quotes", "corporate_actions"}
    total_q = total_ret = total_blk = total_check_violations = 0
    checked_episodes = []
    for ep in episodes:
        episode_had_check = False
        for entry in ep.access_log:
            total_q += 1
            total_ret += entry.get("rows_returned", 0)
            total_blk += entry.get("rows_blocked", 0)
            if entry.get("dataset") not in ORDINARY_LOADERS:
                episode_had_check = True
                total_check_violations += entry.get("rows_blocked", 0)
        if episode_had_check:
            checked_episodes.append(str(ep.episode_start))
    return TemporalIntegrityReport(
        total_queries=total_q,
        total_rows_returned=total_ret,
        total_rows_blocked=total_blk,
        explicit_check_violations=total_check_violations,
        checked_episodes=checked_episodes,
    )


def _excess_returns(daily_returns: pd.Series, rf: "pd.Series | float") -> pd.Series:
    """`daily_returns - rf`, aligned by date when `rf` is a Series.

    rf defaults to 0.0 everywhere it's threaded through below (SPY-proxy
    factor fallback has no rf column, and pre-fix callers that never pass rf
    at all get identical behavior to before this fix) -- passing 0.0 is a
    deliberate no-op, not an approximation. When `rf` IS a real Series
    (dataprep.load_factors()'s decimal-scaled Ken French daily rate), a date
    missing from it is filled with 0.0 rather than dropped, so this never
    changes the observation COUNT feeding Sharpe/Sortino/alpha -- only a
    genuinely absent rf quote for that day, which practice treats as "no
    adjustment" rather than as a reason to lose the return observation."""
    if isinstance(rf, (int, float)):
        return daily_returns - float(rf)
    return daily_returns - rf.reindex(daily_returns.index).fillna(0.0)


_DEGENERATE_RETURN_STD = 1e-9   # see _is_degenerate's docstring


def _is_degenerate(daily_returns: pd.Series) -> bool:
    """True when the RAW strategy produced no real variation -- a fully
    flat episode (e.g. the agent never traded, so portfolio value never
    changed and every return is exactly 0.0). Checked on the raw returns,
    NOT on rf-adjusted excess returns: confirmed live, a real
    deepseek-v4-pro episode (2019-05-26..2019-08-24, zero fills) had raw
    returns of exactly 0.0 every day, but rf (the real daily Ken French
    rate) varies slightly day to day -- so `excess = 0 - rf` is NOT exactly
    constant, giving excess.std(ddof=1) a tiny but nonzero value from rf's
    own wiggle alone, unrelated to anything the strategy did. Dividing by
    that produced a Sharpe of -9.68e15, printed and nearly shipped in a
    real report before this was caught. A strategy that did nothing should
    always read NaN, regardless of how much the risk-free rate itself
    happened to jitter that week -- checking degeneracy on the raw returns
    (which genuinely are exactly constant here) fixes this at the source
    rather than patching the symptom with a looser tolerance on the
    rf-adjusted series, which would just move the blowup threshold rather
    than remove its cause. A small tolerance (not exact ==0) guards the
    same case for near-flat (not bit-for-bit-identical) raw returns, e.g.
    floating-point noise from repeated arithmetic rather than genuine
    signal."""
    return daily_returns.empty or daily_returns.std(ddof=1) < _DEGENERATE_RETURN_STD


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR,
                  rf: "pd.Series | float" = 0.0) -> float:
    if _is_degenerate(daily_returns):
        return float("nan")
    excess = _excess_returns(daily_returns, rf)
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR,
                   rf: "pd.Series | float" = 0.0) -> float:
    """Downside deviation per the standard definition (Sortino & Price 1994,
    also what most quant libraries implement): target semi-deviation over
    ALL N observations, sqrt(mean(min(r_i - MAR, 0)^2)) with MAR = rf.
    The previous version computed std(ddof=1) of only the negative-return
    SUBSET, centered on that subset's own mean rather than on MAR/0 and
    divided by (n_downside - 1) rather than N -- a different, non-standard
    statistic that doesn't reduce to the textbook one by a constant factor,
    so a reviewer checking the definition against Sortino & Price would not
    be able to reproduce these numbers from the paper's stated formula."""
    if _is_degenerate(daily_returns):
        return float("nan")
    excess = _excess_returns(daily_returns, rf)
    downside_dev = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    if downside_dev < _DEGENERATE_RETURN_STD:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values: pd.Series) -> float:
    """Largest peak-to-trough percentage decline before a new high, as a
    positive fraction (0.25 == a 25% drawdown)."""
    pv = portfolio_values.sort_index()
    if pv.empty:
        return float("nan")
    running_max = pv.cummax()
    drawdown = (pv - running_max) / running_max
    return float(-drawdown.min())


def net_monetary_return(portfolio_values: pd.Series) -> float:
    pv = portfolio_values.sort_index()
    if len(pv) < 2:
        return float("nan")
    return float(pv.iloc[-1] - pv.iloc[0])


def execution_breakeven_bps(execution_log, step_bps: float = 1.0, max_bps: float = 100.0) -> float:
    """Same sweep as Phase 2's ExecutionReplayEngine.calculate_breakeven_threshold,
    reimplemented against a plain fills list/frame so this module doesn't need
    a live ExecutionReplayEngine instance -- only its logged output. If you
    still have the engine instance handy, prefer calling its method directly
    and pass the result in; this is here for offline/reloaded logs."""
    fills = _fills_to_frame(execution_log)
    if fills.empty:
        return 0.0
    direction = np.where(fills["side"] == "SELL", 1.0, -1.0)
    notional = fills["executed_qty"] * fills["execution_price"]

    current_bps = 0.0
    while current_bps <= max_bps:
        sim_fee = notional * (current_bps / 10000.0)
        net_pnl = float((direction * notional - sim_fee).sum())
        if net_pnl <= 0:
            return current_bps
        current_bps += step_bps
    return max_bps


@dataclass
class EpisodeMetrics:
    episode_start: pd.Timestamp
    regime: str
    net_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    breakeven_bps: float
    daily_returns: pd.Series = field(repr=False)


def episode_metrics(ep: EpisodeRun, rf: "pd.Series | float" = 0.0) -> EpisodeMetrics:
    rets = ep.daily_returns()
    return EpisodeMetrics(
        episode_start=ep.episode_start,
        regime=ep.regime,
        net_return=net_monetary_return(ep.portfolio_values),
        sharpe=sharpe_ratio(rets, rf=rf),
        sortino=sortino_ratio(rets, rf=rf),
        max_drawdown=max_drawdown(ep.portfolio_values),
        breakeven_bps=execution_breakeven_bps(ep.execution_log),
        daily_returns=rets,
    )


def executable_performance_table(episodes: Sequence[EpisodeRun], rf: "pd.Series | float" = 0.0) -> pd.DataFrame:
    """One row per episode -- the raw material for both the Executable
    Performance summary and the Regime Robustness groupby below.

    `rf` defaults to 0.0 (no adjustment -- identical to this function's
    behavior before risk-free-rate handling existed at all). Pass the real
    daily Ken French rf series (dataprep.load_factors()["rf"]) to report
    Sharpe/Sortino on excess returns, consistent with factor_attribution's
    alpha, which regresses against Mkt-RF (already excess) -- mixing a raw
    return Sharpe with an excess-return alpha in the same report would be an
    internally inconsistent pair of numbers."""
    rows = [dataclasses.asdict(episode_metrics(ep, rf=rf)) for ep in episodes]
    df = pd.DataFrame(rows).drop(columns=["daily_returns"])
    return df.sort_values("episode_start").reset_index(drop=True)


def regime_robustness_table(perf_table: pd.DataFrame) -> pd.DataFrame:
    """Groups the per-episode performance table by regime label so you can
    see, e.g., whether Sharpe collapses specifically in crisis windows --
    this is the check that catches a system that only "works" in calm
    markets, per the proposal's motivation for regime-stratified eval."""
    agg = perf_table.groupby("regime").agg(
        n_episodes=("episode_start", "count"),
        mean_net_return=("net_return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        std_sharpe=("sharpe", "std"),
        mean_sortino=("sortino", "mean"),
        mean_max_drawdown=("max_drawdown", "mean"),
        mean_breakeven_bps=("breakeven_bps", "mean"),
    )
    return agg.reset_index()


def _ols_alpha_beta(y: np.ndarray, factors: np.ndarray) -> tuple[float, np.ndarray]:
    """Plain OLS via lstsq (no statsmodels dependency): y = alpha + X @ beta.
    Returns (alpha, beta_vector). `factors` is (n_obs, n_factors); an
    intercept column is added here."""
    X = np.column_stack([np.ones(len(y)), factors])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), coef[1:]


def _available_factor_cols(factor_returns: pd.DataFrame) -> list[str]:
    """Which of dataprep.FACTOR_COLUMNS (mkt/smb/hml/rmw/cma/mom, in that
    canonical order) `factor_returns` actually has. "mkt" is the one
    non-negotiable column -- a market beta is the minimum any attribution
    needs to mean anything -- everything else is used if present, so this
    same code path handles the full 6-factor build, a user-supplied
    --factors FILE with fewer columns, and the SPY-only proxy (mkt alone)
    without three separate implementations."""
    cols = [c for c in FACTOR_COLUMNS if c in factor_returns.columns]
    if "mkt" not in cols:
        raise ValueError(
            "factor_returns must include at least a 'mkt' column -- "
            "market beta is the minimum needed for any attribution"
        )
    return cols


@dataclass
class AttributionResult:
    alpha_annualized: float
    betas: dict[str, float]  # factor name -> beta, only for factors actually regressed
    n_obs: int


def factor_attribution(
    daily_returns: pd.Series,
    factor_returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> AttributionResult:
    """Regresses the agent's EXCESS daily returns on daily style factor
    returns. `factor_returns` must be date-indexed with at least a "mkt"
    column (Market) -- "smb" (Size), "hml" (Value), "rmw" (Profitability),
    "cma" (Investment), and "mom" (Momentum) are each included in the
    regression if present, so this adapts to whatever factor set was
    actually sourced (see dataprep.build_factors for the full 6-factor
    build, or main.load_factor_returns for the --factors FILE / SPY-proxy
    fallbacks). You're responsible for aligning its index timezone/frequency
    to the portfolio value series before calling this.

    The standard Fama-French specification is R_portfolio - RF = alpha +
    beta*(Mkt-RF) + ... -- and dataprep.build_factors' "mkt" column IS
    already Mkt-RF (excess), by construction of the Ken French data itself.
    If `factor_returns` carries an "rf" column (the real Ken French build
    does; the SPY-only proxy in main.load_factor_returns does not), it is
    subtracted from `daily_returns` here before the regression, so the LHS
    and RHS are both in excess-return terms -- regressing a RAW agent return
    against an already-excess Mkt-RF is an inconsistent specification that
    biases alpha by roughly rf * (1 - net factor exposure), which is not
    negligible across a date range spanning the 2022-2023 rate-hike period.
    Falls back to rf=0.0 (unchanged prior behavior) when no "rf" column is
    present, e.g. the SPY-proxy path.

    Alpha is annualized (daily alpha * periods_per_year) and reported as
    factor-adjusted residual performance, per the proposal's framing --
    this is NOT a claim of "genuine skill," just the return unexplained by
    whatever style factors were included."""
    factor_cols = _available_factor_cols(factor_returns)
    rf = factor_returns["rf"] if "rf" in factor_returns.columns else 0.0
    excess_returns = _excess_returns(daily_returns, rf).rename("ret")
    aligned = pd.concat([excess_returns, factor_returns[factor_cols]],
                         axis=1, join="inner").dropna()
    if len(aligned) < 5:
        log.warning("factor_attribution: only %d aligned observations, results unreliable", len(aligned))
    y = aligned["ret"].to_numpy()
    X = aligned[factor_cols].to_numpy()
    alpha, beta = _ols_alpha_beta(y, X)
    return AttributionResult(
        alpha_annualized=alpha * periods_per_year,
        betas={name: float(b) for name, b in zip(factor_cols, beta)},
        n_obs=len(aligned),
    )


def _moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draws blocks of consecutive indices (with wraparound) until reaching
    length n. Block resampling (rather than iid resampling of single days)
    preserves the short-range autocorrelation in daily returns, which iid
    bootstrap would destroy and which matters for realistic Sharpe/Sortino
    CIs on financial return series."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])
    return idx[:n]


@dataclass
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int


def block_bootstrap_ci(
    daily_returns: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    metric: str = "sharpe",
    n_boot: int = 1000,
    block_size: int = 21,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """95% CI for `metric` in {"sharpe", "sortino", "alpha"} via moving
    block bootstrap over the (concatenated, across episodes) daily return
    series. Per the proposal: this guards against a single favorable
    window being reported as if it were a stable result. `factor_returns`
    is required (and resampled in lockstep with the returns, via shared
    block indices) when metric == "alpha", since alpha needs matched
    agent/factor observations on each resample.

    Every metric here is computed on EXCESS returns when `factor_returns`
    carries an "rf" column -- same rf treatment as sharpe_ratio/
    sortino_ratio/factor_attribution, kept consistent so a Sharpe CI and an
    alpha CI reported side by side in the same table are answering the same
    "excess of what" question rather than silently mixing raw and
    excess-return conventions. Falls back to rf=0.0 (no adjustment,
    unchanged prior behavior) when factor_returns is None or lacks "rf"
    (e.g. metric in {"sharpe","sortino"} called with no factor context at
    all, or the SPY-proxy fallback)."""
    rf = (factor_returns["rf"] if factor_returns is not None and "rf" in factor_returns.columns
          else 0.0)
    rets = _excess_returns(daily_returns, rf).dropna()
    n = len(rets)
    if n < block_size:
        raise ValueError(f"need at least {block_size} return observations, got {n}")

    if metric == "alpha":
        if factor_returns is None:
            raise ValueError("factor_returns is required for metric='alpha'")
        factor_cols = _available_factor_cols(factor_returns)
        aligned = pd.concat([rets.rename("ret"), factor_returns[factor_cols]],
                             axis=1, join="inner").dropna()
        rets_arr = aligned["ret"].to_numpy()
        fac_arr = aligned[factor_cols].to_numpy()
        n = len(aligned)

        def point(r, f):
            a, _ = _ols_alpha_beta(r, f)
            return a * TRADING_DAYS_PER_YEAR
    else:
        rets_arr = rets.to_numpy()
        fac_arr = None
        # rf already applied above (rets is excess); pass rf=0.0 here so
        # sharpe_ratio/sortino_ratio don't subtract it a second time.
        fn = {"sharpe": sharpe_ratio, "sortino": sortino_ratio}[metric]

        def point(r, f):
            return fn(pd.Series(r))

    estimate = point(rets_arr, fac_arr)

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = _moving_block_bootstrap_indices(n, block_size, rng)
        r_boot = rets_arr[idx]
        f_boot = fac_arr[idx] if fac_arr is not None else None
        boot_stats[b] = point(r_boot, f_boot)

    alpha_tail = (1 - confidence) / 2
    lo, hi = np.nanpercentile(boot_stats, [100 * alpha_tail, 100 * (1 - alpha_tail)])
    return BootstrapCI(point_estimate=float(estimate), ci_low=float(lo), ci_high=float(hi), n_boot=n_boot)


# Canonical factor-set sizes to ablate over, smallest to largest. Chosen to
# match the field's own standard progression -- CAPM (Sharpe/Lintner) ->
# Fama-French 3-factor (1993) -> Fama-French-Carhart 6-factor (1997/2015)
# -- not an arbitrary subset.
FACTOR_SPECS: dict[str, tuple[str, ...]] = {
    "capm_1factor": ("mkt",),
    "ff_3factor": ("mkt", "smb", "hml"),
    "ff_carhart_6factor": ("mkt", "smb", "hml", "rmw", "cma", "mom"),
}


def factor_attribution_ablation(
    daily_returns: pd.Series,
    factor_returns: pd.DataFrame,
    n_boot: int = 1000,
    block_size: int = 21,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Runs factor_attribution + block_bootstrap_ci(metric="alpha") at each
    of FACTOR_SPECS' factor-set sizes against the SAME underlying
    factor_returns, so alpha's sign/significance can be checked against
    whether it survives every specification or only shows up in one.

    Per spec, ALL of that spec's columns must be present in
    factor_returns or the spec is skipped entirely (e.g. a --factors FILE
    that only ever had 3 columns can't run the 6-factor spec) -- unlike
    factor_attribution's own graceful per-call degradation to whatever
    subset is available, here a partial match would silently compare a
    "6-factor" row that's actually only a 4-factor regression, defeating
    the point of the ablation.

    Explicitly NOT a second, independently-sourced dataset -- every spec
    here still comes from whatever single factor_returns was passed in.
    This answers "is this alpha robust to specification choice," not "is
    this alpha robust to which vendor's factor construction you trust";
    those are different, complementary questions."""
    rows = []
    for spec_name, wanted_cols in FACTOR_SPECS.items():
        if not all(c in factor_returns.columns for c in wanted_cols):
            continue
        # "rf" is carried along (not a regressor -- not in wanted_cols) so
        # factor_attribution/block_bootstrap_ci can find it and regress
        # excess-of-rf returns at every spec size, not just when the full,
        # unrestricted factor_returns happens to be passed elsewhere. Without
        # this, every row of the ablation table would silently fall back to
        # rf=0.0 regardless of the top-level rf fix, since restricted's
        # column set never included it.
        restrict_cols = list(wanted_cols) + (["rf"] if "rf" in factor_returns.columns else [])
        restricted = factor_returns[restrict_cols]
        attribution = factor_attribution(daily_returns, restricted, periods_per_year)
        alpha_ci = block_bootstrap_ci(daily_returns, factor_returns=restricted, metric="alpha",
                                       n_boot=n_boot, block_size=block_size, seed=seed)
        rows.append({
            "spec": spec_name,
            "n_factors": len(wanted_cols),
            "alpha_annualized": attribution.alpha_annualized,
            "alpha_ci_low": alpha_ci.ci_low,
            "alpha_ci_high": alpha_ci.ci_high,
            "n_obs": attribution.n_obs,
            **{f"beta_{name}": val for name, val in attribution.betas.items()},
        })
    return pd.DataFrame(rows)


@dataclass
class JointMetricReport:
    system_name: str
    temporal_integrity: TemporalIntegrityReport
    executable_performance: pd.DataFrame  # per-episode
    regime_robustness: pd.DataFrame  # per-regime rollup
    attribution: AttributionResult
    sharpe_ci: BootstrapCI
    sortino_ci: BootstrapCI
    alpha_ci: BootstrapCI
    attribution_ablation: pd.DataFrame  # alpha/betas at each FACTOR_SPECS size, see factor_attribution_ablation

    def summary(self) -> str:
        ti = self.temporal_integrity
        lines = [
            f"=== Joint Metric Report: {self.system_name} ===",
            "",
            f"-- Temporal Integrity: {'PASS' if ti.clean else 'FAIL'} --",
            f"  [informational] {ti.total_rows_blocked} rows filtered out as future-dated / "
            f"{ti.total_rows_returned} returned across {ti.total_queries} loader queries "
            f"-- this is the PiT filter operating normally, not a leak",
        ]
        if not ti.clean:
            lines.append(f"  FAIL reason: {ti.explicit_check_violations} explicit check() "
                          f"violation(s) in episode(s): {ti.checked_episodes}")
        lines += [
            "",
            "-- Regime Robustness --",
            self.regime_robustness.to_string(index=False),
            "",
            "-- Executable Performance (aggregate across episodes) --",
            f"mean Sharpe:  {self.sharpe_ci.point_estimate:.3f}  "
            f"95% CI [{self.sharpe_ci.ci_low:.3f}, {self.sharpe_ci.ci_high:.3f}]",
            f"mean Sortino: {self.sortino_ci.point_estimate:.3f}  "
            f"95% CI [{self.sortino_ci.ci_low:.3f}, {self.sortino_ci.ci_high:.3f}]",
            "",
            "-- Attribution --",
            f"alpha (annualized): {self.alpha_ci.point_estimate:.4f}  "
            f"95% CI [{self.alpha_ci.ci_low:.4f}, {self.alpha_ci.ci_high:.4f}]",
            "  " + "  ".join(f"beta_{name}={val:.3f}" for name, val in self.attribution.betas.items())
            + f"  (n_obs={self.attribution.n_obs})",
        ]
        if not self.attribution_ablation.empty:
            lines += [
                "",
                "-- Attribution ablation (same data, different factor-set sizes) --",
                "  is alpha robust to specification choice, or an artifact of picking one?",
                self.attribution_ablation.round(4).to_string(index=False),
            ]
        return "\n".join(lines)


def build_joint_metric_report(
    system_name: str,
    episodes: Sequence[EpisodeRun],
    factor_returns: pd.DataFrame,
    n_boot: int = 1000,
    block_size: int = 21,
    seed: int = 0,
) -> JointMetricReport:
    """Top-level entry point: run the full Joint Metric Collection phase for
    one agent system over its assigned episodes. `factor_returns` is shared
    across systems (same market conditions -> comparable attribution)."""
    ti = temporal_integrity_report(episodes)
    # rf: the real Ken French build carries a daily risk-free rate; the
    # SPY-proxy fallback (main.load_factor_returns, no data build available)
    # does not. Falls back to 0.0 (no adjustment) in that case, which is the
    # same behavior every one of these functions had before this fix existed
    # -- so a run without real factor data isn't newly broken, just not
    # excess-return-adjusted.
    rf = factor_returns["rf"] if "rf" in factor_returns.columns else 0.0
    perf = executable_performance_table(episodes, rf=rf)
    regime = regime_robustness_table(perf)

    # Concatenate daily returns across episodes for the aggregate attribution
    # and bootstrap. Episodes are non-overlapping with purge gaps (Phase 1),
    # so this concatenation doesn't stitch together autocorrelated periods
    # that were never actually adjacent in the agent's run.
    all_returns = pd.concat([ep.daily_returns() for ep in episodes]).sort_index()
    all_returns = all_returns[~all_returns.index.duplicated(keep="first")]

    attribution = factor_attribution(all_returns, factor_returns)
    # factor_returns passed to the sharpe/sortino CIs too (not just alpha's),
    # so all three CIs in one report answer the same "excess of what"
    # question -- previously these two calls omitted it entirely, meaning
    # the Sharpe/Sortino CI would silently use raw returns even once alpha
    # started using excess returns, an internally inconsistent report.
    sharpe_ci = block_bootstrap_ci(all_returns, factor_returns=factor_returns, metric="sharpe",
                                    n_boot=n_boot, block_size=block_size, seed=seed)
    sortino_ci = block_bootstrap_ci(all_returns, factor_returns=factor_returns, metric="sortino",
                                     n_boot=n_boot, block_size=block_size, seed=seed)
    alpha_ci = block_bootstrap_ci(all_returns, factor_returns=factor_returns, metric="alpha",
                                   n_boot=n_boot, block_size=block_size, seed=seed)
    ablation = factor_attribution_ablation(all_returns, factor_returns, n_boot=n_boot,
                                            block_size=block_size, seed=seed)

    return JointMetricReport(
        system_name=system_name,
        temporal_integrity=ti,
        executable_performance=perf,
        regime_robustness=regime,
        attribution=attribution,
        sharpe_ci=sharpe_ci,
        sortino_ci=sortino_ci,
        alpha_ci=alpha_ci,
        attribution_ablation=ablation,
    )
