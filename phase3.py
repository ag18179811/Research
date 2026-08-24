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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

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
    isolated to this system. Requires OpenAI/Anthropic/etc. credentials to
    already be configured per TradingAgents' normal setup -- this function
    only handles the three benchmark-integrity fixes, not general setup."""
    register_pit_fundamentals_vendor()
    config = get_pit_fundamentals_config(base_config)

    os.environ["TRADINGAGENTS_MEMORY_LOG_PATH"] = str(isolated_memory_path(system_name, memory_base_dir))

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


def build_trading_agents_adapter(ta_graph, tickers: list[str], notional_per_ticker: float = 10_000.0) -> AgentFn:
    """Returns an AgentFn (matching the interface in section 2) that calls
    the given (already-patched) TradingAgentsGraph once per ticker per day
    and translates its rating into a rebalancing Order.

    Note this makes one full multi-agent LLM run per ticker per day --
    expect this to be slow and to cost real API credits at scale. Consider
    testing on a small ticker list and a short date range first."""

    def agent(decision_snapshots, positions, cash, as_of):
        orders = []
        trade_date = as_of.strftime("%Y-%m-%d")

        for ticker in tickers:
            snap = decision_snapshots.get(ticker, {})
            prices = snap.get("prices")
            if prices is None or prices.empty:
                continue
            last_close = float(prices.sort_values("date").iloc[-1]["adj_close"])
            if last_close <= 0:
                continue

            try:
                _, rating = ta_graph.propagate(ticker, trade_date)
            except Exception:
                # A single failed LLM call (rate limit, provider hiccup,
                # malformed response) shouldn't crash an entire multi-year
                # backtest -- treat it as a no-trade day for this ticker
                # and move on, same as run_episode already does for a
                # crashing agent_fn at the whole-day level.
                continue

            target_fraction = RATING_TO_TARGET_FRACTION.get(rating)
            if target_fraction is None:
                continue  # Hold, or an unrecognized rating string

            target_qty = (target_fraction * notional_per_ticker) / last_close
            current_qty = positions.get(ticker, 0.0)
            delta_qty = target_qty - current_qty

            if abs(delta_qty * last_close) < MIN_REBALANCE_NOTIONAL:
                continue

            side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
            orders.append(Order(
                order_id="", timestamp=as_of.isoformat(), symbol=ticker,
                side=side, order_type=OrderType.MARKET, quantity=abs(delta_qty),
            ))

        return orders

    return agent


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
    the three raw loaders as an explicit check and sums its violations."""
    ORDINARY_LOADERS = {"prices", "news", "fundamentals"}
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


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if daily_returns.empty or daily_returns.std(ddof=1) == 0:
        return float("nan")
    return float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    downside = daily_returns[daily_returns < 0]
    if daily_returns.empty or downside.empty or downside.std(ddof=1) == 0:
        return float("nan")
    return float(daily_returns.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


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


def episode_metrics(ep: EpisodeRun) -> EpisodeMetrics:
    rets = ep.daily_returns()
    return EpisodeMetrics(
        episode_start=ep.episode_start,
        regime=ep.regime,
        net_return=net_monetary_return(ep.portfolio_values),
        sharpe=sharpe_ratio(rets),
        sortino=sortino_ratio(rets),
        max_drawdown=max_drawdown(ep.portfolio_values),
        breakeven_bps=execution_breakeven_bps(ep.execution_log),
        daily_returns=rets,
    )


def executable_performance_table(episodes: Sequence[EpisodeRun]) -> pd.DataFrame:
    """One row per episode -- the raw material for both the Executable
    Performance summary and the Regime Robustness groupby below."""
    rows = [dataclasses.asdict(episode_metrics(ep)) for ep in episodes]
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
    """Regresses the agent's daily returns on daily style factor returns.
    `factor_returns` must be date-indexed with at least a "mkt" column
    (Market) -- "smb" (Size), "hml" (Value), "rmw" (Profitability), "cma"
    (Investment), and "mom" (Momentum) are each included in the regression
    if present, so this adapts to whatever factor set was actually sourced
    (see dataprep.build_factors for the full 6-factor build, or
    main.load_factor_returns for the --factors FILE / SPY-proxy fallbacks).
    You're responsible for aligning its index timezone/frequency to the
    portfolio value series before calling this.

    Alpha is annualized (daily alpha * periods_per_year) and reported as
    factor-adjusted residual performance, per the proposal's framing --
    this is NOT a claim of "genuine skill," just the return unexplained by
    whatever style factors were included."""
    factor_cols = _available_factor_cols(factor_returns)
    aligned = pd.concat([daily_returns.rename("ret"), factor_returns[factor_cols]],
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
    agent/factor observations on each resample."""
    rets = daily_returns.dropna()
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
    perf = executable_performance_table(episodes)
    regime = regime_robustness_table(perf)

    # Concatenate daily returns across episodes for the aggregate attribution
    # and bootstrap. Episodes are non-overlapping with purge gaps (Phase 1),
    # so this concatenation doesn't stitch together autocorrelated periods
    # that were never actually adjacent in the agent's run.
    all_returns = pd.concat([ep.daily_returns() for ep in episodes]).sort_index()
    all_returns = all_returns[~all_returns.index.duplicated(keep="first")]

    attribution = factor_attribution(all_returns, factor_returns)
    sharpe_ci = block_bootstrap_ci(all_returns, metric="sharpe", n_boot=n_boot,
                                    block_size=block_size, seed=seed)
    sortino_ci = block_bootstrap_ci(all_returns, metric="sortino", n_boot=n_boot,
                                     block_size=block_size, seed=seed)
    alpha_ci = block_bootstrap_ci(all_returns, factor_returns=factor_returns, metric="alpha",
                                   n_boot=n_boot, block_size=block_size, seed=seed)

    return JointMetricReport(
        system_name=system_name,
        temporal_integrity=ti,
        executable_performance=perf,
        regime_robustness=regime,
        attribution=attribution,
        sharpe_ci=sharpe_ci,
        sortino_ci=sortino_ci,
        alpha_ci=alpha_ci,
    )
