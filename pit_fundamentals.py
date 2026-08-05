"""
PiT-safe fundamentals vendor for TradingAgents.

TradingAgents' get_fundamentals accepts a curr_date parameter all the way
down to the vendor layer (see tradingagents/agents/utils/fundamental_data_tools.py
-> route_to_vendor("get_fundamentals", ticker, curr_date)) -- but neither
the yfinance nor Alpha Vantage vendor implementation actually uses it
(both literally say "not used for X" in their own docstrings). Every call
returns today's live company snapshot regardless of what historical date
is being simulated -- a confirmed look-ahead leak for any TradingAgents
run inside this benchmark.

This module provides a drop-in replacement with the exact same call
signature (ticker, curr_date) -> str, backed by Phase 1's PointInTimeStore
instead of a live API, so it genuinely only ever reflects fundamentals
that were public knowledge as of curr_date.

Registration (do this once, before creating a TradingAgentsGraph):

    from tradingagents.dataflows import interface as ta_interface
    from pit_fundamentals import get_fundamentals as get_pit_fundamentals

    ta_interface.VENDOR_METHODS["get_fundamentals"]["benchmark_pit"] = get_pit_fundamentals

    config = DEFAULT_CONFIG.copy()
    config["tool_vendors"] = {"get_fundamentals": "benchmark_pit"}
    ta = TradingAgentsGraph(config=config)

get_insider_transactions is NOT patched here -- its tool wrapper
(tradingagents/agents/utils/news_data_tools.py) doesn't accept curr_date
at all, so there's no date to thread through even if a vendor respected
it, and insider-transaction data isn't part of Phase 1's PiT store to
begin with. Recommended fix there is to exclude that tool from the
agent's toolset for benchmark runs rather than patch it -- see the note
at the bottom of this file.
"""

from __future__ import annotations

import pandas as pd

from dataprep import PointInTimeStore


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """PiT-safe replacement for TradingAgents' get_fundamentals vendor call.
    Same signature and same kind of return value (a formatted text report,
    since this gets fed straight into an LLM prompt) as the yfinance/Alpha
    Vantage versions it replaces -- so no other TradingAgents code needs to
    change, only the vendor registration.

    Uses whatever fundamentals + derived ratios Phase 1's PointInTimeStore
    can produce as of curr_date. This will have different fields than the
    yfinance version (e.g. no PE ratio, since that needs live analyst
    estimates Phase 1 doesn't have) -- it reports what was genuinely
    knowable at curr_date, not a feature-parity clone of the live version.
    """
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


# ---------------------------------------------------------------------------
# get_insider_transactions -- deliberately not patched, see module docstring.
# Recommended: exclude it from the agent's toolset for benchmark runs, e.g.
# when building the graph's toolkit, drop "get_insider_transactions" from
# whatever tool list gets bound to the Fundamentals Analyst. This trades a
# small amount of the analyst's information (insider buy/sell activity) for
# a guarantee that nothing it sees is being pulled live -- the safer
# direction to be wrong in in a benchmark whose entire premise is that
# temporal integrity is non-negotiable.
# ---------------------------------------------------------------------------
