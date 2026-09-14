

from __future__ import annotations

import argparse
import logging
import os
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

import dataprep
from azure_llm import _PROPAGATE_HARD_TIMEOUT_SECONDS, _request_with_hard_timeout
from execution_replay import ExecutionReplayEngine, Order, OrderSide, OrderType, ReplayConfig

# Reuse everything that isn't decision-making, unchanged, from the Claude
# bridge -- no reason to duplicate PortfolioState, TradeRecord, the trading
# calendar, or the mark-to-market logic, all of that is decision-maker-agnostic.
from phase1_5_agent import (
    PortfolioState,
    TradeRecord,
    Phase15Bridge,
    trading_calendar,
    _mark_to_market_prices,
)

# phase3.py already builds a PiT-safe TradingAgentsGraph (fundamentals vendor
# patched, insider-transactions tool excluded, isolated memory path) -- reuse
# that instead of re-solving the same integrity problems here.
from phase3 import build_trading_agents_graph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase1_5_tradingagents")

# TradingAgents' propagate() returns exactly one of these five strings
# (tradingagents/agents/schemas.py PortfolioRating). Same interpretation
# phase3.py's own adapter uses: a target position, not a sized order.
RATING_PRIORITY = ["Buy", "Sell", "Overweight", "Underweight"]  # Hold excluded: never actionable
RATING_TO_TARGET_FRACTION = {"Buy": 1.0, "Overweight": 0.5, "Underweight": 0.25, "Sell": 0.0}


class TradingAgentsBridge:
    """Same role as Phase15Bridge, but decisions come from a
    TradingAgentsGraph's propagate() call per ticker instead of one Claude
    call across the whole universe.

    Design choice, stated plainly: TradingAgents natively wants to rate
    every ticker every day (see phase3.build_trading_agents_adapter for
    that native, multi-order-per-day mode, used by phase3.run_all_episodes).
    To keep THIS class a fair, apples-to-apples comparison against the
    single-agent Claude bridge's "at most one trade per day" rule, it calls
    propagate() for every ticker, then picks AT MOST ONE trade for the day:
    the highest-priority non-Hold rating found, in RATING_PRIORITY order,
    with ticker list order as the tiebreak. Every other ticker's rating
    that day is discarded. If you want TradingAgents' native
    everyone-gets-rebalanced-daily behavior instead, use
    phase3.run_all_episodes with build_trading_agents_adapter directly."""

    def __init__(
        self,
        store: "dataprep.PointInTimeStore",
        tickers: list[str] | None = None,
        notional_per_ticker: float = 10_000.0,
        system_name: str = "tradingagents_baseline",
        base_config: dict | None = None,
        memory_base_dir: str = "./ta_memory",
        n_workers: int = 6,
    ):
        self.store = store
        self.tickers = tickers or dataprep.EVALUATION_TICKERS
        self.notional_per_ticker = notional_per_ticker
        self.n_workers = max(1, min(n_workers, len(self.tickers)))

        from tradingagents.default_config import DEFAULT_CONFIG

        config = dict(base_config or DEFAULT_CONFIG.copy())
        # Route TradingAgents through Claude, same model family as the
        # single-agent baseline, so a comparison isolates coordination
        # itself rather than also varying the underlying LLM.
        config.setdefault("llm_provider", "anthropic")
        config.setdefault("deep_think_llm", "claude-sonnet-5")
        config.setdefault("quick_think_llm", "claude-haiku-4-5")

        # One TradingAgentsGraph instance per worker, not one shared
        # instance rated ticker-by-ticker. propagate() mutates
        # self.ticker / self.graph / self._checkpointer_ctx directly on
        # the graph object (confirmed by reading trading_graph.py) -- two
        # tickers rated concurrently on ONE shared instance would silently
        # contaminate each other's runs (e.g. ticker B's analysis
        # executing under self.ticker == "A" if A's call is still
        # in-flight when B's overwrites it). Every instance here still
        # points at the same isolated memory log path (same
        # system_name/memory_base_dir, by design -- see
        # phase3.isolated_memory_path), so cross-ticker memory context
        # still works the same as the unparallelized version. That log's
        # own write (tradingagents/agents/utils/memory.py:store_decision)
        # is an unlocked read-then-append, so concurrent writers can
        # occasionally produce a duplicate "pending" entry for the same
        # ticker+date -- confirmed by reading it, and accepted: it only
        # degrades the quality of a *future* self-reflection prompt, it
        # does not change the rating returned for any ticker on any day,
        # which is what this benchmark actually measures.
        self._graph_pool: "queue.Queue" = queue.Queue()
        for _ in range(self.n_workers):
            self._graph_pool.put(build_trading_agents_graph(system_name, config, memory_base_dir))

        # Reused purely for its market-data machinery (quotes/ADV/regime),
        # not for any LLM call -- avoids re-implementing _quote_snapshot,
        # _adv, _regime_for, and _decision_to_market_snapshot a second time.
        # claude_api_key=None lets the SDK resolve credentials lazily; since
        # this helper's LLM-calling methods are never invoked, an unset key
        # here is harmless.
        self._market_helper = Phase15Bridge(store, tickers=self.tickers)

    def build_daily_snapshot(self, as_of: pd.Timestamp, portfolio: PortfolioState) -> dict:
        return self.store.universe_snapshot_as_of(
            as_of, cash=portfolio.cash, positions=portfolio.positions, tickers=self.tickers,
        )

    def _rate_one_ticker(self, ticker: str, trade_date: str) -> tuple[str, str]:
        """Borrows one graph instance from the pool for the duration of this
        ticker's propagate() call, returns it when done (success, failure,
        or timeout alike) so the next queued ticker can use it."""
        graph = self._graph_pool.get()
        try:
            _, rating = _request_with_hard_timeout(
                graph.propagate, ticker, trade_date,
                hard_timeout=_PROPAGATE_HARD_TIMEOUT_SECONDS,
            )
        except Exception:
            log.exception(
                "TradingAgents propagate() failed for %s on %s, treating as Hold",
                ticker, trade_date,
            )
            rating = "Hold"
        finally:
            self._graph_pool.put(graph)
        return ticker, rating

    def _rate_all_tickers(self, as_of: pd.Timestamp) -> dict[str, str]:
        trade_date = as_of.strftime("%Y-%m-%d")
        ratings: dict[str, str] = {}
        # self.n_workers independent graph instances (see __init__) means
        # up to self.n_workers tickers rate concurrently instead of one at
        # a time -- the dominant cost per ticker is real, sequential LLM
        # latency inside propagate() (13-16 calls observed for a single
        # ticker/day at the framework's own minimum debate-round setting,
        # not something this benchmark's code controls), so wall-clock
        # time for a full day's universe scales down roughly by
        # n_workers, not by luck.
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = [pool.submit(self._rate_one_ticker, ticker, trade_date) for ticker in self.tickers]
            for future in as_completed(futures):
                ticker, rating = future.result()
                ratings[ticker] = rating
        return ratings

    def _pick_days_trade(self, ratings: dict[str, str], portfolio: PortfolioState, snap: dict) -> dict:
        for wanted_rating in RATING_PRIORITY:
            for ticker in self.tickers:
                if ratings.get(ticker) != wanted_rating:
                    continue
                prices = snap["tickers"][ticker]["prices"]
                if prices.empty:
                    continue
                last_close = float(prices.sort_values("date")["adj_close"].iloc[-1])
                if last_close <= 0:
                    continue

                target_fraction = RATING_TO_TARGET_FRACTION[wanted_rating]
                target_qty = (target_fraction * self.notional_per_ticker) / last_close
                current_qty = portfolio.positions.get(ticker, 0.0)
                delta_qty = target_qty - current_qty
                if abs(delta_qty) < 1e-9:
                    continue  # already at target -- nothing to actually submit

                side = "BUY" if delta_qty > 0 else "SELL"
                return {
                    "action": side, "symbol": ticker, "quantity": abs(delta_qty),
                    "order_type": "MARKET", "limit_price": None,
                    "reasoning": f"TradingAgents rated {ticker} {wanted_rating}",
                }
        reason = "no actionable rating across the universe today" if ratings else "no ratings produced"
        return {"action": "HOLD", "symbol": None, "quantity": None,
                "order_type": None, "limit_price": None, "reasoning": reason}

    def _validate_decision(self, decision: dict, portfolio: PortfolioState, snap: dict) -> dict:
        # Same real-world constraints as Phase15Bridge._validate_decision:
        # can't sell what you don't hold, can't buy more than cash allows.
        if decision["action"] == "HOLD":
            return decision
        symbol, quantity = decision["symbol"], decision["quantity"]
        last_price = float(snap["tickers"][symbol]["prices"].sort_values("date")["adj_close"].iloc[-1])

        if decision["action"] == "SELL":
            held = portfolio.positions.get(symbol, 0.0)
            if held <= 0:
                decision["action"] = "HOLD"
                return decision
            decision["quantity"] = min(quantity, held)

        if decision["action"] == "BUY":
            max_affordable = portfolio.cash / last_price if last_price > 0 else 0.0
            decision["quantity"] = min(quantity, max_affordable)
            if decision["quantity"] <= 0:
                decision["action"] = "HOLD"

        return decision

    def _decision_to_order(self, decision: dict, as_of: pd.Timestamp, order_seq: int) -> Order | None:
        if decision["action"] == "HOLD":
            return None
        return Order(
            order_id=f"ORD_{as_of.date().isoformat()}_{order_seq:04d}",
            timestamp=as_of.isoformat(),
            symbol=decision["symbol"],
            side=OrderSide[decision["action"]],
            order_type=OrderType.MARKET,
            quantity=float(decision["quantity"]),
            limit_price=None,
        )

    def _decision_to_market_snapshot(self, symbol: str, as_of: pd.Timestamp):
        return self._market_helper._decision_to_market_snapshot(symbol, as_of)


def run_episode(
    bridge: TradingAgentsBridge,
    start_date: str,
    n_days: int = 90,
    engine: ExecutionReplayEngine | None = None,
) -> tuple[ExecutionReplayEngine, list[TradeRecord]]:
    """Identical structure to phase1_5_agent.run_episode -- same trading
    calendar, same T-1-decide/T-execute cutoff discipline, same daily
    mark-to-market -- with TradingAgents' per-ticker rating + single-trade
    selection standing in for the single Claude call."""
    if engine is None:
        engine = ExecutionReplayEngine(ReplayConfig(initial_cash=100_000.0))

    start = pd.Timestamp(start_date, tz="UTC")
    days = trading_calendar(start, n_days)
    if not days:
        raise ValueError(f"no trading days found on/after {start_date} -- check SPY price coverage")

    history: list[TradeRecord] = []

    for i, day in enumerate(days):
        portfolio = PortfolioState(cash=engine.cash_balance, positions=dict(engine.positions))

        snap = bridge.build_daily_snapshot(day, portfolio)
        ratings = bridge._rate_all_tickers(day)
        raw_decision = bridge._pick_days_trade(ratings, portfolio, snap)
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

        # Same T-1-decide/T-execute discipline as phase1_5_agent.py: `day`
        # is the decision cutoff (strictly before today's own close);
        # execution/mark-to-market need today's own close, one calendar
        # day later, so PiT filtering (`available_at < as_of`) admits it.
        execution_cutoff = day + pd.Timedelta(days=1)
        assert execution_cutoff > day, (
            f"execution_cutoff ({execution_cutoff}) must be strictly after the decision "
            f"cutoff `day` ({day})"
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

        engine.record_daily_value(
            record.date, _mark_to_market_prices(bridge.store, bridge.tickers, execution_cutoff)
        )
        history.append(record)

    return engine, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="e.g. 2022-06-09")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--notional-per-ticker", type=float, default=10_000.0)
    parser.add_argument("--deep-think-llm", default="claude-sonnet-5")
    parser.add_argument("--quick-think-llm", default="claude-haiku-4-5")
    parser.add_argument("--system-name", default="tradingagents_baseline")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "set ANTHROPIC_API_KEY in your environment before running this -- "
            "TradingAgents reads this exact variable name, separate from "
            "whatever CLAUDE_API_KEY convention phase1_5_agent.py uses."
        )

    store = dataprep.PointInTimeStore()
    bridge = TradingAgentsBridge(
        store,
        tickers=args.tickers,
        notional_per_ticker=args.notional_per_ticker,
        system_name=args.system_name,
        base_config={
            "llm_provider": "anthropic",
            "deep_think_llm": args.deep_think_llm,
            "quick_think_llm": args.quick_think_llm,
        },
    )
    engine, history = run_episode(bridge, args.start_date, n_days=args.days)

    n_trades = sum(1 for h in history if h.action != "HOLD")
    print(f"\nFinal cash: ${engine.cash_balance:,.2f}")
    print(f"Final positions: {engine.positions}")
    print(f"Trades placed: {n_trades} / {len(history)} days")

    engine.export_logs_to_json("execution_log_tradingagents.json")


if __name__ == "__main__":
    main()