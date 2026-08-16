"""
Phase 1.5: the step between Data Preparation (Phase 1) and Execution Replay
(Phase 2). Turns `PointInTimeStore.universe_snapshot_as_of()`'s output into
an actual prompt, and defines the `AgentDecision` shape the agent's response
is parsed into.

The agent operates at the whole-universe (portfolio) level, not one ticker
at a time -- deciding factor: the benchmark's own metrics (Sharpe, Sortino,
max drawdown, the Attribution regression) are computed on the combined
portfolio's return series, not per-ticker. Twenty independent single-ticker
calls per timestep can't produce a sensible portfolio-level decision -- e.g.
it could load up on five correlated tech names without any one of those
five decisions ever having visibility into the other four. build_prompt()
(single-ticker) is kept below anyway -- it's still a useful, correct
building block -- but build_universe_prompt() is what the agent actually
receives.

Scoped directly from the research proposal, not assumed: "Execution Replay
Phase" states that bid-ask spread and fill mechanics are applied by the
Replay Engine to the agent's decision afterward, not reasoned about by the
agent itself. So the agent decides a direction (buy/sell/hold) and a
rationale per ticker -- nothing about size or price. Converting that into
sized, priced Orders is a separate, not-yet-built step that belongs to the
Replay Engine side of the pipeline, using the same cash/positions state
universe_snapshot_as_of takes as input.

What this file does NOT do: actually call a real LLM. `decide()` and
`decide_universe()` below are explicit placeholders -- wiring in a real
model is a separate decision (which model, which API, which prompting
strategy) that hasn't been made yet, and this file shouldn't quietly invent
an answer to that. What it does provide, for real: concrete prompts built
from real Phase 1 data, and an honest token count of each.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
# Draft format, not a finalized spec. Every value in the prompt below is read
# directly out of the snapshot dictionary -- nothing here is invented data;
# only the wording around it is new.

MAX_NEWS_HEADLINES = 10  # a raw news table can have hundreds of rows; capped
MAX_PRICE_ROWS = 10      # same reasoning for the price table


def build_prompt(snapshot: dict) -> str:
    """snapshot must be the exact dict shape returned by
    PointInTimeStore.snapshot_as_of(): ticker, cik, as_of, prices, news,
    fundamentals, derived_features."""
    ticker = snapshot["ticker"]
    as_of: pd.Timestamp = snapshot["as_of"]
    prices: pd.DataFrame = snapshot["prices"]
    news: pd.DataFrame = snapshot["news"]
    feats: dict = snapshot["derived_features"]

    lines = []
    lines.append(f"You are a trading agent. Today's date is {as_of.date()}.")
    lines.append(
        f"Every fact below was already public before {as_of.date()} -- "
        "nothing after this date is available to you or exists in your "
        "training data for this decision."
    )
    lines.append(f"\nTicker: {ticker}")

    lines.append(f"\nRecent daily prices (most recent {MAX_PRICE_ROWS} trading days):")
    lines.append("date       open     high     low      close    volume")
    recent_prices = prices.sort_values("date").tail(MAX_PRICE_ROWS)
    for _, row in recent_prices.iterrows():
        lines.append(
            f"{row['date'].date()}  {row['open']:7.2f}  {row['high']:7.2f}  "
            f"{row['low']:7.2f}  {row['close']:7.2f}  {int(row['volume']):>10,}"
        )

    lines.append(f"\nRecent news headlines (most recent {MAX_NEWS_HEADLINES}):")
    if news.empty:
        lines.append("(none in the available window)")
    else:
        recent_news = news.sort_values("published_at").tail(MAX_NEWS_HEADLINES)
        for _, row in recent_news.iterrows():
            lines.append(f"- [{row['published_at'].date()}] {row['title']}")

    lines.append("\nCalculated indicators:")
    def fmt(v):
        return f"{v:.4f}" if v is not None else "not available (insufficient history)"
    lines.append(f"  5-day return:    {fmt(feats.get('return_5d'))}")
    lines.append(f"  21-day return:   {fmt(feats.get('return_21d'))}")
    lines.append(f"  63-day return:   {fmt(feats.get('return_63d'))}")
    lines.append(f"  21-day realized volatility (daily, not annualized): {fmt(feats.get('realized_vol'))}")
    lines.append(f"  Leverage ratio (Liabilities/Assets):        {fmt(feats.get('leverage_ratio'))}")
    lines.append(f"  Current ratio (AssetsCurrent/LiabilitiesCurrent): {fmt(feats.get('current_ratio'))}")
    lines.append(f"  Return on assets (NetIncomeLoss/Assets):    {fmt(feats.get('return_on_assets'))}")

    lines.append(
        "\nRespond with a single decision as JSON: "
        '{"side": "buy" | "sell" | "hold", "rationale": "<one sentence>"}. '
        "Do not specify quantity or price -- sizing and execution are "
        "handled separately."
    )
    return "\n".join(lines)


# Trimmed harder than the single-ticker version's MAX_PRICE_ROWS/
# MAX_NEWS_HEADLINES (10/10) -- bundling 20 tickers into one prompt
# multiplies whatever's kept per ticker by 20, so the per-ticker budget
# has to come down for the whole thing to stay a reasonable size.
UNIVERSE_MAX_PRICE_ROWS = 5
UNIVERSE_MAX_NEWS_HEADLINES = 3


def build_universe_prompt(universe_snapshot: dict) -> str:
    """universe_snapshot must be the exact dict shape returned by
    PointInTimeStore.universe_snapshot_as_of(): as_of, cash, positions,
    tickers (a {ticker: single-ticker snapshot dict} mapping)."""
    as_of: pd.Timestamp = universe_snapshot["as_of"]
    cash: float = universe_snapshot["cash"]
    positions: dict = universe_snapshot["positions"]
    tickers: dict = universe_snapshot["tickers"]

    lines = []
    lines.append(f"You are a trading agent managing a portfolio. Today's date is {as_of.date()}.")
    lines.append(
        f"Every fact below was already public before {as_of.date()} -- "
        "nothing after this date is available to you or exists in your "
        "training data for this decision."
    )
    lines.append(f"\nCash available: ${cash:,.2f}")
    lines.append("Current positions:")
    held = {t: q for t, q in positions.items() if q}
    if held:
        for t, qty in held.items():
            lines.append(f"  {t}: {qty:,.0f} shares")
    else:
        lines.append("  (none)")

    for ticker, snap in tickers.items():
        prices: pd.DataFrame = snap["prices"]
        news: pd.DataFrame = snap["news"]
        feats: dict = snap["derived_features"]

        lines.append(f"\n--- {ticker} ---")
        recent_prices = prices.sort_values("date").tail(UNIVERSE_MAX_PRICE_ROWS)
        if recent_prices.empty:
            lines.append("prices: (none available)")
        else:
            price_str = "; ".join(
                f"{row['date'].date()} close={row['close']:.2f}"
                for _, row in recent_prices.iterrows()
            )
            lines.append(f"prices: {price_str}")

        recent_news = news.sort_values("published_at").tail(UNIVERSE_MAX_NEWS_HEADLINES)
        if recent_news.empty:
            lines.append("news: (none)")
        else:
            for _, row in recent_news.iterrows():
                lines.append(f"news: [{row['published_at'].date()}] {row['title']}")

        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"
        lines.append(
            f"indicators: 5d={fmt(feats.get('return_5d'))} "
            f"21d={fmt(feats.get('return_21d'))} 63d={fmt(feats.get('return_63d'))} "
            f"vol21d={fmt(feats.get('realized_vol'))} "
            f"leverage={fmt(feats.get('leverage_ratio'))} "
            f"current_ratio={fmt(feats.get('current_ratio'))} "
            f"roa={fmt(feats.get('return_on_assets'))}"
        )

    lines.append(
        f"\nRespond with a JSON array of exactly {len(tickers)} decisions, one "
        "per ticker above, each: "
        '{"ticker": "<ticker>", "side": "buy" | "sell" | "hold", '
        '"rationale": "<one sentence>"}. '
        "Do not specify quantity or price -- sizing and execution are "
        "handled separately. Every ticker must appear exactly once."
    )
    return "\n".join(lines)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Real token count of `text`, via tiktoken. Caveat that matters: tiktoken
    implements OpenAI's tokenizers (cl100k_base = GPT-3.5/4-era models).
    Anthropic, Google, and other model families use their own tokenizers that
    aren't available as an installable library the same way -- this number is
    a reasonable approximation for planning purposes, not an exact count for
    whichever model actually ends up running this prompt."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# What the agent actually outputs -- and what it deliberately does NOT
# ---------------------------------------------------------------------------
# Per the research proposal (Methods, "The Task"): the agent is given only
# PiT records and asked to "output specific trading orders or make a
# no-trade decision" -- no cash balance, no position, no portfolio state is
# named anywhere as agent input. And per "Execution Replay Phase": "the
# agent system makes a trading decision, its actions are converted into
# orders and processed through a Replay Engine [which] reconstructs the
# historical execution environment by modeling... Bid-Ask Spreads... Fill
# Mechanics." Spread and fill handling are explicitly the Replay Engine's
# job, applied to the agent's decision afterward -- not something the agent
# itself reasons about or is shown.
#
# So the agent's output is a direction, not a sized, priced order. Turning
# an AgentDecision into an Order (real quantity, real fill price) is a
# separate step owned by the Replay Engine side of the pipeline, using
# portfolio state that Phase 2 tracks across the episode -- not built here,
# and deliberately not part of what the agent sees or decides.

@dataclass
class AgentDecision:
    """Exactly what the agent outputs -- nothing about size or price."""
    ticker: str
    as_of: pd.Timestamp
    side: str       # "buy" | "sell" | "hold"
    rationale: str


@dataclass
class Order:
    """What the Replay Engine (Phase 2) actually consumes -- built from an
    AgentDecision plus sizing/portfolio logic that lives downstream of the
    agent, not from the agent's own output directly."""
    ticker: str
    as_of: pd.Timestamp
    side: str
    quantity: float
    order_type: str
    limit_price: float | None
    rationale: str


def decide(snapshot: dict) -> AgentDecision:
    """Not implemented on purpose. This is where a real call to a real model
    goes -- build_prompt(snapshot) would be the input, and the model's JSON
    response (per the schema requested in the prompt) would get parsed into
    an AgentDecision here. Which model, which API, and how to handle a
    malformed or refused response are real decisions that haven't been made
    yet, so this raises rather than silently returning a fabricated
    decision."""
    raise NotImplementedError(
        "decide() needs a real model call wired in -- see build_prompt() for "
        "the exact input it would receive, and AgentDecision for the exact "
        "output shape. Converting that into a sized Order is a separate, "
        "not-yet-built step outside the agent's own decision."
    )


def decide_universe(universe_snapshot: dict) -> list[AgentDecision]:
    """Not implemented, same reasoning as decide() -- this is the real entry
    point once a model is wired in: build_universe_prompt(universe_snapshot)
    is the input, and the model's JSON array response gets parsed into one
    AgentDecision per ticker here."""
    raise NotImplementedError(
        "decide_universe() needs a real model call wired in -- see "
        "build_universe_prompt() for the exact input, and AgentDecision for "
        "each element of the expected output list."
    )
