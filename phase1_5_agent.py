from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase1_5_bridge")

# --- Phase 1 (rename to match your actual filename) -------------------------
import dataprep  # PointInTimeStore, load_prices, label_regimes, EVALUATION_TICKERS

# --- Phase 2 (rename to match your actual filename) --------------------------
from execution_replay import (
    ExecutionReplayEngine,
    MarketDataSnapshot,
    MarketRegime,
    Order,
    OrderSide,
    OrderType,
    ReplayConfig,
)

# --- Coordination cost ------------------------------------------------------
# One-way dependency: coordination_cost knows nothing about this module, so there
# is no cycle. The SDK-to-meter key mapping lives there, next to the price table
# that consumes it, rather than being restated at each call site.
from coordination_cost import usage_from_anthropic

# --- Claude ---------------------------------------------------------------
import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the current directory into os.environ


# =============================================================================
# Portfolio / trade-history bookkeeping
# =============================================================================

@dataclass
class PortfolioState:
    """Tracked across the whole episode, mutated in place by run_episode().
    Starts at $100k cash, no positions, per your spec."""

    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)

    def position_value(self, last_prices: dict[str, float]) -> float:
        return sum(qty * last_prices.get(sym, 0.0) for sym, qty in self.positions.items())

    def total_value(self, last_prices: dict[str, float]) -> float:
        return self.cash + self.position_value(last_prices)


@dataclass
class TradeRecord:
    """One day's outcome, HOLD or otherwise -- this is what gets fed back
    into the prompt on subsequent days so the LLM has continuity."""

    date: str
    action: str  # "BUY" | "SELL" | "HOLD"
    symbol: Optional[str]
    quantity: Optional[float]
    order_type: Optional[str]
    limit_price: Optional[float]
    reasoning: str

DECISION_SCHEMA_HINT = """
Respond with ONLY a JSON object (no markdown fences, no commentary) matching:
{
  "action": "BUY" | "SELL" | "HOLD",
  "symbol": string or null,       // required if action is BUY or SELL; must be one of the tickers listed above
  "quantity": number or null,     // shares, required if action is BUY or SELL
  "order_type": "MARKET" | "LIMIT" | "POST_ONLY",
  "limit_price": number or null,  // required if order_type is LIMIT or POST_ONLY
  "reasoning": string             // 1-3 sentences
}
You may place AT MOST ONE trade today. If nothing looks worth trading, return HOLD.
""".strip()


# The stable half of every request: role, rules, and output schema. Byte-identical
# across every decision in a run, which is the entire precondition for prompt
# caching -- caching is a PREFIX match, so everything that varies day to day (the
# date, the portfolio, the PiT payload) must come strictly AFTER this block. It
# used to sit at the tail of _format_prompt, behind the trading date, which left
# a cacheable prefix of exactly zero bytes.
#
# Whether this actually caches is a per-model question, not a property of the
# text: it is ~200 tokens, which clears Claude Opus 5's 512-token minimum but is
# far below Claude Haiku 4.5's 4096. Below the minimum there is no error and no
# discount -- cache_creation_input_tokens simply comes back 0.
# coordination_cost.cost_across_models() reports that per model rather than
# assuming it away.
SYSTEM_PROMPT = (
    "You are a JSON-only API acting as a trading agent. Respond with exactly one "
    "JSON object matching the requested schema and nothing else -- no markdown "
    "code fences, no commentary before or after.\n\n"
    "You may place at most ONE trade per day (or none). Use only the information "
    "in the user message -- it is already point-in-time filtered; nothing in it "
    "is from after the stated trading date.\n\n"
    + DECISION_SCHEMA_HINT
)


# =============================================================================
# The bridge
# =============================================================================

class Phase15Bridge:
    def __init__(
        self,
        store: "dataprep.PointInTimeStore",
        claude_api_key: str | None = None,
        tickers: list[str] | None = None,
        model_name: str = "claude-haiku-4-5",
        max_tokens: int = 1024,
        adv_window: int = 21,
        synthetic_spread_bps: float = 5.0,
        use_prompt_caching: bool = True,
    ):
        self.store = store
        self.tickers = tickers or dataprep.EVALUATION_TICKERS
        # Undated ID on purpose: this string is also the lookup key into
        # coordination_cost.MODEL_PRICES, which is keyed on undated IDs. The old
        # default ("claude-haiku-4-5-20251001") is a valid model but not a valid
        # price key, so metering it raised KeyError on the first cost lookup.
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.adv_window = adv_window
        self.synthetic_spread_bps = synthetic_spread_bps
        self.use_prompt_caching = use_prompt_caching
        # api_key=None lets the SDK resolve credentials itself (ANTHROPIC_API_KEY,
        # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile). Passing None is
        # not the same as having no credentials.
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        self._regime_series: pd.Series | None = None  # lazily built, cached
        # Set by _call_llm from response.usage; read by CoordinationMeter via
        # last_usage(). None until the first call, so an unused bridge reports
        # nothing rather than a fabricated zero.
        self._last_usage: dict | None = None
        self._prefix_tokens: int = 0   # measured lazily by measure_prefix_tokens()

    # -- market-wide regime, computed once from SPY, never shown to the LLM --
    def _ensure_regime_series(self) -> None:
        if self._regime_series is not None:
            return
        spy = dataprep.load_prices(["SPY"]).sort_values("date")
        returns = spy.set_index("date")["adj_close"].pct_change()
        labeled = dataprep.label_regimes(returns)
        self._regime_series = labeled["regime"]
        log.info("regime series built: %d labeled trading days", len(self._regime_series))

    def _regime_for(self, as_of: pd.Timestamp) -> MarketRegime:
        self._ensure_regime_series()
        prior = self._regime_series[self._regime_series.index < as_of]
        if prior.empty:
            log.warning("no regime label available before %s, defaulting to MEAN_REVERTING", as_of.date())
            return MarketRegime.MEAN_REVERTING
        label = prior.iloc[-1]
        return {
            "trending": MarketRegime.TRENDING,
            "mean_reverting": MarketRegime.MEAN_REVERTING,
            "crisis": MarketRegime.CRISIS_HIGH_VOLATILITY,
        }[label]

    # -- ADV from trailing daily volume ---------------------------------------
    def _adv(self, ticker: str, as_of: pd.Timestamp) -> float:
        prices = self.store.prices_as_of(as_of, entities=[ticker])
        if prices.empty or "volume" not in prices.columns:
            log.warning("no volume column for %s as of %s, using 1e6 fallback ADV", ticker, as_of.date())
            return 1_000_000.0
        trailing = prices.sort_values("date")["volume"].tail(self.adv_window)
        return float(trailing.mean()) if not trailing.empty else 1_000_000.0

    # -- bid/ask/depth: real Alpaca quotes if available, else synthetic ------
    def _quote_snapshot(self, ticker: str, as_of: pd.Timestamp) -> dict:
        try:
            quotes = self.store.quotes_as_of(as_of, entities=[ticker])
        except (FileNotFoundError, KeyError):
            # KeyError happens when quotes.py's build ran but never reached
            # this specific ticker (interrupted mid-run) -- load_quotes then
            # returns a fully empty DataFrame with no columns at all, and
            # the store's own available_at filter KeyErrors on it. Treat
            # both cases the same way: no real quotes for this ticker, fall
            # through to the synthetic estimate below.
            quotes = pd.DataFrame()

        if not quotes.empty:
            latest = quotes.sort_values("timestamp").iloc[-1]
            return {
                "bid_price": float(latest["bid_price"]),
                "ask_price": float(latest["ask_price"]),
                "bid_depth": float(latest["bid_size"]),
                "ask_depth": float(latest["ask_size"]),
                "source": "alpaca_quote",
            }

        # Fallback: pre-2016 episodes, or a gap in pulled quote data. Do NOT
        # trust this for a real backtest -- it's a placeholder so the day
        # doesn't just fail, built off last close with an assumed spread and
        # an ADV-derived depth guess.
        prices = self.store.prices_as_of(as_of, entities=[ticker])
        if prices.empty:
            raise ValueError(f"no price data at all for {ticker} as of {as_of}")
        last_close = float(prices.sort_values("date")["adj_close"].iloc[-1])
        half_spread = last_close * (self.synthetic_spread_bps / 10000.0) / 2
        adv = self._adv(ticker, as_of)
        log.warning("no real quotes for %s as of %s, using synthetic spread fallback", ticker, as_of.date())
        return {
            "bid_price": last_close - half_spread,
            "ask_price": last_close + half_spread,
            "bid_depth": adv * 0.01,
            "ask_depth": adv * 0.01,
            "source": "synthetic_fallback",
        }

    # -- universe snapshot for one day ----------------------------------------
    def build_daily_snapshot(self, as_of: pd.Timestamp, portfolio: PortfolioState) -> dict:
        return self.store.universe_snapshot_as_of(
            as_of, cash=portfolio.cash, positions=portfolio.positions, tickers=self.tickers,
        )

    # -- prompt construction ---------------------------------------------------
    def _format_prompt(
        self, snap: dict, portfolio: PortfolioState, history: list[TradeRecord], as_of: pd.Timestamp
    ) -> str:
        lines = [
            f"Trading date: {as_of.date().isoformat()}",
            f"Cash: ${portfolio.cash:,.2f}",
            f"Positions: {portfolio.positions or 'none'}",
            "",
            "Recent trade history (most recent last, may be empty on day 1):",
        ]
        if not history:
            lines.append("  (none yet)")
        else:
            for h in history[-10:]:
                lines.append(f"  {h.date}: {h.action} {h.symbol or ''} {h.quantity or ''} -- {h.reasoning}")

        lines += ["", "Universe snapshot (point-in-time, no future data):"]
        for ticker, tsnap in snap["tickers"].items():
            feats = tsnap["derived_features"]
            price_rows = tsnap["prices"]
            last_price = (
                float(price_rows.sort_values("date")["adj_close"].iloc[-1]) if not price_rows.empty else None
            )
            lines.append(
                f"  {ticker}: last_close={last_price}, "
                f"ret_5d={feats.get('return_5d')}, ret_21d={feats.get('return_21d')}, "
                f"ret_63d={feats.get('return_63d')}, realized_vol={feats.get('realized_vol')}, "
                f"leverage_ratio={feats.get('leverage_ratio')}, current_ratio={feats.get('current_ratio')}, "
                f"return_on_assets={feats.get('return_on_assets')}, recent_news_count={len(tsnap['news'])}"
            )

        # The role, the one-trade-per-day rule, and DECISION_SCHEMA_HINT used to be
        # appended here. They now live in SYSTEM_PROMPT, ahead of this text, so the
        # request has a stable cacheable prefix instead of leading with the trading
        # date. Nothing stable belongs after this point.
        return "\n".join(lines)

    # -- system block, with or without a cache breakpoint --------------------
    def _system_blocks(self) -> list[dict]:
        """SYSTEM_PROMPT as a content-block list, carrying a cache breakpoint when
        caching is on. Sending it as blocks rather than a bare string is what makes
        `cache_control` expressible at all."""
        block: dict = {"type": "text", "text": SYSTEM_PROMPT}
        if self.use_prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def measure_prefix_tokens(self) -> int:
        """Real token length of the cacheable prefix, via the Anthropic count_tokens
        endpoint -- not tiktoken, which implements OpenAI's tokenizers and would give
        a number for the wrong model family.

        This drives the `prefix_caches` column in
        coordination_cost.CoordinationMeter.cost_across_models(), which is the only
        thing that will tell you a prefix silently fell below a model's minimum and
        earned no discount at all. Measured once and cached; 0 if the call fails,
        which reads downstream as "unknown" rather than as a false "yes it cached"."""
        if self._prefix_tokens:
            return self._prefix_tokens
        try:
            counted = self.client.messages.count_tokens(
                model=self.model_name,
                system=self._system_blocks(),
                messages=[{"role": "user", "content": "x"}],
            )
            self._prefix_tokens = int(counted.input_tokens)
        except Exception as exc:                     # network, auth, bad model id
            log.warning("count_tokens failed (%s); prefix size unknown", exc)
            self._prefix_tokens = 0
        return self._prefix_tokens

    # -- usage, in the shape CoordinationMeter expects -----------------------
    def last_usage(self) -> dict | None:
        """Token counts from the most recent call, keyed the way
        coordination_cost.DecisionRecord names them. CoordinationMeter looks for
        exactly this method (duck-typed) and falls back to assumed figures without
        it. None before the first call."""
        return self._last_usage

    # -- call Claude -------------------------------------------------------
    def _call_llm(self, prompt: str) -> dict:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=self._system_blocks(),
            messages=[{"role": "user", "content": prompt}],
        )
        # Record usage BEFORE parsing. A malformed-JSON response still cost real
        # money, and losing the count because json.loads raised is exactly how a
        # committee ends up looking cheaper than it was.
        self._last_usage = usage_from_anthropic(response)
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Belt-and-suspenders: strip stray code fences and retry once
            # before giving up. More important here than it was with Gemini,
            # since Claude has no request-level JSON-mode enforcement -- the
            # system prompt above is the only thing steering it, and it can
            # still occasionally wrap output in ```json fences anyway.
            cleaned = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
            return json.loads(cleaned)

    # -- validate + clip the LLM's decision against real constraints ---------
    def _validate_decision(self, decision: dict, portfolio: PortfolioState, snap: dict) -> dict:
        action = decision.get("action", "HOLD")
        if action not in ("BUY", "SELL", "HOLD"):
            log.warning("invalid action %r from LLM, treating as HOLD", action)
            return {
                "action": "HOLD", "symbol": None, "quantity": None,
                "order_type": None, "limit_price": None,
                "reasoning": decision.get("reasoning", "invalid action from LLM, defaulted to HOLD"),
            }
        if action == "HOLD":
            return decision

        symbol = decision.get("symbol")
        if symbol not in self.tickers:
            log.warning("LLM chose symbol %r outside the tradeable universe, treating as HOLD", symbol)
            decision["action"] = "HOLD"
            return decision

        quantity = decision.get("quantity")
        if not quantity or quantity <= 0:
            log.warning("LLM gave invalid quantity %r, treating as HOLD", quantity)
            decision["action"] = "HOLD"
            return decision

        last_price = float(snap["tickers"][symbol]["prices"].sort_values("date")["adj_close"].iloc[-1])

        if action == "SELL":
            held = portfolio.positions.get(symbol, 0.0)
            if held <= 0:
                log.warning("LLM tried to SELL %s with no position held, treating as HOLD", symbol)
                decision["action"] = "HOLD"
                return decision
            decision["quantity"] = min(quantity, held)

        if action == "BUY":
            max_affordable = portfolio.cash / last_price if last_price > 0 else 0.0
            if quantity > max_affordable:
                log.info("clipping BUY %s from %.4f to affordable %.4f shares", symbol, quantity, max_affordable)
                decision["quantity"] = max(0.0, max_affordable)
            if decision["quantity"] <= 0:
                decision["action"] = "HOLD"

        return decision

    # -- decision -> Phase 2 Order --------------------------------------------
    def _decision_to_order(self, decision: dict, as_of: pd.Timestamp, order_seq: int) -> Optional[Order]:
        if decision["action"] == "HOLD":
            return None
        order_type_str = decision.get("order_type") or "MARKET"
        try:
            order_type = OrderType[order_type_str]
        except KeyError:
            log.warning("unknown order_type %r, defaulting to MARKET", order_type_str)
            order_type = OrderType.MARKET
        return Order(
            order_id=f"ORD_{as_of.date().isoformat()}_{order_seq:04d}",
            timestamp=as_of.isoformat(),
            symbol=decision["symbol"],
            side=OrderSide[decision["action"]],
            order_type=order_type,
            quantity=float(decision["quantity"]),
            limit_price=float(decision["limit_price"]) if decision.get("limit_price") else None,
        )

    # -- decision -> Phase 2 MarketDataSnapshot -------------------------------
    def _decision_to_market_snapshot(self, symbol: str, as_of: pd.Timestamp) -> MarketDataSnapshot:
        quote = self._quote_snapshot(symbol, as_of)
        tsnap = self.store.snapshot_as_of(symbol, as_of)
        realized_vol = tsnap["derived_features"].get("realized_vol") or 0.0
        return MarketDataSnapshot(
            timestamp=as_of.isoformat(),
            symbol=symbol,
            bid_price=quote["bid_price"],
            ask_price=quote["ask_price"],
            bid_depth=quote["bid_depth"],
            ask_depth=quote["ask_depth"],
            daily_volatility=realized_vol,
            average_daily_volume=self._adv(symbol, as_of),
            regime=self._regime_for(as_of),
        )

    # -- one trading day, start to finish -------------------------------------
    def run_day(
        self,
        as_of: pd.Timestamp,
        portfolio: PortfolioState,
        history: list[TradeRecord],
        order_seq: int = 0,
    ) -> tuple[Optional[Order], Optional[MarketDataSnapshot], TradeRecord]:
        snap = self.build_daily_snapshot(as_of, portfolio)
        prompt = self._format_prompt(snap, portfolio, history, as_of)
        raw_decision = self._call_llm(prompt)
        decision = self._validate_decision(raw_decision, portfolio, snap)

        record = TradeRecord(
            date=as_of.date().isoformat(),
            action=decision["action"],
            symbol=decision.get("symbol"),
            quantity=decision.get("quantity"),
            order_type=decision.get("order_type"),
            limit_price=decision.get("limit_price"),
            reasoning=decision.get("reasoning", ""),
        )

        order = self._decision_to_order(decision, as_of, order_seq)
        market_snapshot = self._decision_to_market_snapshot(decision["symbol"], as_of) if order else None
        return order, market_snapshot, record


# =============================================================================
# Episode loop
# =============================================================================

def trading_calendar(start: pd.Timestamp, n_days: int) -> list[pd.Timestamp]:
    """Real trading days (from SPY's own price index) starting at `start`,
    up to `n_days` of them. Using calendar days would wrongly count weekends
    and market holidays as tradeable steps."""
    spy = dataprep.load_prices(["SPY"]).sort_values("date")
    dates = spy[spy["date"] >= start]["date"]
    return list(dates.iloc[:n_days])


def _mark_to_market_prices(store: "dataprep.PointInTimeStore", tickers: list[str],
                            as_of: pd.Timestamp) -> dict[str, float]:
    """Every held ticker's actual close on the day being marked -- deliberately
    NOT the decision snapshot's prices. The decision snapshot is (correctly)
    filtered to strictly before today, since the agent must decide before
    today's close exists; reusing it to mark today's portfolio value would
    price today's positions off yesterday's close, silently shifting the
    entire daily-return series by one trading day (this is what execute-on-
    T-close means in the standard T-1-decide/T-execute backtest design
    execution_harness.py already follows -- this function makes run_episode's
    own loop follow it too). Pass `as_of = day + 1 trading-day's worth of
    time (a full calendar day is enough, since PiT filtering is `< as_of`
    and the next real trading day's own close is always further out) so the
    PiT cutoff admits today's close but nothing from tomorrow onward.

    A separate, prices-only fetch rather than reusing/rebuilding the full
    per-ticker snapshot (news + fundamentals + derived features) that
    build_daily_snapshot assembles -- mark-to-market only ever needs
    adj_close, and re-fetching the rest for all tickers every day would
    roughly double this loop's non-LLM cost for nothing this function reads."""
    out = {}
    for t in tickers:
        prices = store.prices_as_of(as_of, entities=[t])
        if not prices.empty:
            out[t] = float(prices.sort_values("date")["adj_close"].iloc[-1])
    return out


def run_episode(
    bridge: Phase15Bridge,
    start_date: str,
    n_days: int = 90,
    engine: ExecutionReplayEngine | None = None,
) -> tuple[ExecutionReplayEngine, list[TradeRecord]]:
    """Walks `n_days` trading days forward from `start_date`, calling the LLM
    once per day (max 1 trade/day), running any resulting order through
    Phase 2's real ExecutionReplayEngine, and marking the account to market
    every day (trade or not). Cash/positions fed into tomorrow's prompt come
    straight from the engine's own post-fill state -- not an approximation --
    so the LLM sees real fills, slippage, and fees, not idealized ones.

    Pass your own `engine` (e.g. with a non-default ReplayConfig) if you
    want different fee/impact assumptions than the $100k-cash default.
    """
    if engine is None:
        engine = ExecutionReplayEngine(ReplayConfig(initial_cash=100_000.0))

    start = pd.Timestamp(start_date, tz="UTC")
    days = trading_calendar(start, n_days)
    if not days:
        raise ValueError(f"no trading days found on/after {start_date} -- check SPY price coverage")

    history: list[TradeRecord] = []

    for i, day in enumerate(days):
        # Portfolio state handed to the LLM's snapshot/prompt is read
        # straight from the engine -- authoritative, not tracked separately.
        portfolio = PortfolioState(cash=engine.cash_balance, positions=dict(engine.positions))

        snap = bridge.build_daily_snapshot(day, portfolio)
        prompt = bridge._format_prompt(snap, portfolio, history, day)
        raw_decision = bridge._call_llm(prompt)
        decision = bridge._validate_decision(raw_decision, portfolio, snap)

        record = TradeRecord(
            date=day.date().isoformat(),
            action=decision["action"],
            symbol=decision.get("symbol"),
            quantity=decision.get("quantity"),
            order_type=decision.get("order_type"),
            limit_price=decision.get("limit_price"),
            reasoning=decision.get("reasoning", ""),
        )

        # Execution and mark-to-market both need TODAY's own close -- unlike
        # `day` (the decision cutoff, correctly strictly-before-today so the
        # agent never sees its own close before deciding), `execution_cutoff`
        # is one calendar day later so PiT filtering (`available_at < as_of`)
        # admits today's close. Standard T-1-decide/T-execute backtest
        # design; see _mark_to_market_prices for the full rationale.
        execution_cutoff = day + pd.Timedelta(days=1)
        # Structural invariant, not a data-content check (store.check() is
        # for that) -- asserts the *relationship* between the two cutoffs
        # used in this loop is still correct. A future edit that
        # accidentally reused `day` for execution again (the exact bug
        # already found once here) would silently lag every fill/mark by a
        # trading day with no error; this turns that into an immediate,
        # loud failure instead of a subtly-wrong number someone has to
        # notice by hand.
        assert execution_cutoff > day, (
            f"execution_cutoff ({execution_cutoff}) must be strictly after the decision "
            f"cutoff `day` ({day}) -- execution/mark-to-market would otherwise use data "
            f"from at-or-before the decision cutoff instead of today's own close"
        )

        order = bridge._decision_to_order(decision, day, order_seq=i)
        if order is not None:
            market_snapshot = bridge._decision_to_market_snapshot(decision["symbol"], execution_cutoff)
            fill = engine.process_order(order, market_snapshot)
            log.info(
                "%s: %s %s -- requested %.2f, filled %.2f @ $%.4f (fee $%.4f, maker=%s) -- %s",
                record.date, record.action, record.symbol,
                fill.requested_qty, fill.executed_qty, fill.execution_price,
                fill.fee_paid, fill.is_maker, record.reasoning,
            )
        else:
            log.info("%s: HOLD -- %s", record.date, record.reasoning)

        # Mark to market every day, trade or not, so the daily value series
        # has no gaps -- this is what Phase 3's Sharpe/Sortino/max-drawdown
        # calculations read. Uses today's own close (execution_cutoff), not
        # the decision snapshot `snap` -- see _mark_to_market_prices.
        engine.record_daily_value(
            record.date, _mark_to_market_prices(bridge.store, bridge.tickers, execution_cutoff)
        )
        history.append(record)

    return engine, history


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="e.g. 2022-06-09")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("set CLAUDE_API_KEY in your environment before running this")

    store = dataprep.PointInTimeStore()
    bridge = Phase15Bridge(store, claude_api_key=api_key, tickers=args.tickers, model_name=args.model)
    engine, history = run_episode(bridge, args.start_date, n_days=args.days)

    n_trades = sum(1 for h in history if h.action != "HOLD")
    print(f"\nFinal cash: ${engine.cash_balance:,.2f}")
    print(f"Final positions: {engine.positions}")
    print(f"Trades placed: {n_trades} / {len(history)} days")

    engine.export_logs_to_json("execution_log.json")


if __name__ == "__main__":
    main()