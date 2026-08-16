"""
Coordination Breakeven Spread (CBS), as defined in Nguyen & Pham,
"Toward Reliable Evaluation of LLM-Based Financial Multi-Agent Systems"
(arXiv:2603.27539), Section 6.2.

Exact definition from the paper: let delta_p(d) be the expected improvement
in entry/exit price from coordination depth d, and let s be the instrument's
bid-ask spread (round-trip cost = 2s). Then

    CBS(d) = delta_p(d) / 2

Coordination is worth its overhead only if s < CBS(d); when the instrument's
spread exceeds the CBS, the paper's conclusion is that coordination overhead
is not recovered and the system should revert to single-agent operation.

This file is deliberately split into two halves with very different levels
of confidence:

  - The spread side (s) is real. It's read directly from the quotes data
    already built into dataprep.py (Alpaca NBBO quotes), so every spread
    number this file produces is an actual observed market spread, not an
    estimate.
  - The delta_p(d) side is NOT computed here, and can't honestly be yet.
    Per the paper, delta_p(d) is "the difference in volume-weighted average
    execution price between a coordinated and single-agent baseline over
    matched trading windows" -- that requires real execution logs from both
    a coordinated multi-agent run and a single-agent baseline run on the
    same trades, which don't exist because no agent (Phase 1.5) and no
    replay engine (Phase 2) have been built and run yet. Every function
    below that needs delta_p(d) takes it as a required argument rather than
    inventing a plausible-looking number for it.
"""

from __future__ import annotations

import pandas as pd

import dataprep as dp


def spread_bps(bid_price: float, ask_price: float) -> float:
    """Bid-ask spread in basis points, relative to the midpoint. This is `s`
    from the paper, in the units the paper's worked examples (Fig. 2, the
    1-2bps large-cap / 10-50bps small-cap / 20-100bps crypto figures) use."""
    mid = (bid_price + ask_price) / 2
    return (ask_price - bid_price) / mid * 10_000


def real_spread_as_of(store: dp.PointInTimeStore, ticker: str, as_of: pd.Timestamp) -> dict:
    """Pulls the most recent real quote available as of `as_of` for `ticker`
    from the already-built PiT quotes dataset and returns its spread. Empty
    (all None) if no quote data exists for this ticker/date -- remember
    quotes only exist from ~2016 onward (ALPACA_HISTORY_START in
    dataprep.py); this will legitimately come back empty before that."""
    quotes = store.quotes_as_of(as_of, entities=[ticker])
    if quotes.empty:
        return {"ticker": ticker, "as_of": as_of, "bid_price": None,
                "ask_price": None, "spread_bps": None}
    latest = quotes.sort_values("timestamp").iloc[-1]
    return {
        "ticker": ticker,
        "as_of": as_of,
        "quote_timestamp": latest["timestamp"],
        "bid_price": float(latest["bid_price"]),
        "ask_price": float(latest["ask_price"]),
        "spread_bps": spread_bps(latest["bid_price"], latest["ask_price"]),
    }


def cbs_bps(delta_p_bps: float) -> float:
    """CBS(d) = delta_p(d) / 2, exactly as defined in the paper. `delta_p_bps`
    is the coordination-driven price improvement, in basis points -- this is
    NOT computed here; it must come from real matched-window execution data
    (see module docstring)."""
    return delta_p_bps / 2


def coordination_is_worth_it(delta_p_bps: float, bid_price: float, ask_price: float) -> dict:
    """The paper's decision rule, applied directly: coordination is only
    worth its overhead if the real spread s is less than CBS(d). Returns the
    inputs alongside the verdict so the reasoning is auditable, not just a
    bare boolean."""
    s = spread_bps(bid_price, ask_price)
    threshold = cbs_bps(delta_p_bps)
    return {
        "spread_bps": s,
        "cbs_threshold_bps": threshold,
        "coordination_worth_it": s < threshold,
    }
