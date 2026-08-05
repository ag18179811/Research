"""
TradingAgents adapter for the benchmark harness.

Wires TauricResearch/TradingAgents into runner.py's AgentFn interface,
applying the three fixes identified from actually reading the vendored
source:

  1. get_fundamentals -- replaced with a PiT-safe version backed by Phase
     1's PointInTimeStore (both stock vendors silently ignore curr_date;
     see pit_fundamentals.py for the full writeup).
  2. get_insider_transactions -- excluded entirely, via a subclass that
     rebuilds the "news" tool node without it. Its tool schema has no date
     parameter at all, so there's nothing to patch; excluding is safer
     than leaving a tool bound to the agent that structurally cannot
     respect the temporal wall.
  3. Cross-episode memory isolation -- TradingAgents persists a decision
     log across runs (~/.tradingagents/memory/trading_memory.md) and
     injects realized outcomes from prior runs into future prompts. That's
     legitimate *if and only if* episodes for one system are always run in
     true chronological order against one isolated memory file. This
     module gives every system-under-test its own memory path and never
     reorders episodes, but does NOT protect you if you re-run episodes
     out of order yourself -- always call run_all_episodes with a single
     forward pass over the date range, per system.

NOT fixed here, flagged as still-open:
  - Balance sheet / cashflow / income statement filter by fiscal
    period-end date, not filing date -- a smaller, unconfirmed leak risk.
  - News internals were spot-checked (get_news_yfinance genuinely filters
    by pub_date) but not exhaustively audited across every vendor path the
    way fundamentals was.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from execution_replay import Order, OrderSide, OrderType
from pit_fundamentals import get_fundamentals as get_pit_fundamentals

# ---------------------------------------------------------------------------
# 1. PiT-safe fundamentals vendor registration
# ---------------------------------------------------------------------------


def register_pit_fundamentals_vendor() -> None:
    """Call once, before constructing any TradingAgentsGraph. Adds our
    PointInTimeStore-backed implementation as a selectable vendor and
    nothing else -- doesn't touch yfinance/Alpha Vantage's registrations,
    so anything that explicitly asks for those vendors by name still gets
    the (broken) original, which is why get_pit_fundamentals_config()
    below also sets tool_vendors explicitly rather than relying on this
    registration alone."""
    from tradingagents.dataflows import interface as ta_interface

    ta_interface.VENDOR_METHODS["get_fundamentals"]["benchmark_pit"] = get_pit_fundamentals


def get_pit_fundamentals_config(base_config: dict) -> dict:
    """Returns a copy of base_config with get_fundamentals routed to our
    PiT-safe vendor. Doesn't touch get_balance_sheet/get_cashflow/
    get_income_statement -- those are the "not fixed here" period-date gap
    noted in the module docstring, left on their default vendor."""
    config = dict(base_config)
    tool_vendors = dict(config.get("tool_vendors", {}))
    tool_vendors["get_fundamentals"] = "benchmark_pit"
    config["tool_vendors"] = tool_vendors
    return config


# ---------------------------------------------------------------------------
# 2. get_insider_transactions exclusion
# ---------------------------------------------------------------------------


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
        excludes get_insider_transactions -- see module docstring for why.
        Everything else is reproduced exactly as trading_graph.py builds
        it, so if their tool lists change upstream, this needs re-syncing
        (a known maintenance cost of excluding via reconstruction rather
        than a config flag; there isn't one)."""

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


# ---------------------------------------------------------------------------
# 3. Per-system memory isolation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 4. Decision -> Order mapping
# ---------------------------------------------------------------------------
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


def build_trading_agents_adapter(ta_graph, tickers: list[str], notional_per_ticker: float = 10_000.0):
    """Returns an AgentFn (matching runner.py's interface) that calls the
    given (already-patched) TradingAgentsGraph once per ticker per day and
    translates its rating into a rebalancing Order.

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
                # and move on, same as runner.py already does for a
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
