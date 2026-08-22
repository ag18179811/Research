"""

WHAT THIS MODULE ADDS
---------------------
A. A meter that wraps ANY agent and records, per decision: LLM calls, tokens,
   inter-agent messages, and wall-clock latency. No agent code changes.
B. Two cost channels, charged separately so you can ablate them:
     TOKENS  -> a direct dollar expense against the portfolio.
     LATENCY -> a decision-staleness penalty, because a committee that
                deliberates for 90 seconds is acting on a stale signal.
C. `coordination_adjusted_cbs`: CBS where the coordinated system also pays its
   own coordination bill. This is the number the proposal actually promises.
D. `coordination_ablation`: same signals, same episodes, same seeds, N analysts
   vs 1 -- the controlled experiment that isolates coordination from strategy.

USAGE
-----
    from agent_payload import resolve_counter
    from coordination_cost import CoordinationMeter, CoordinationCostConfig
    from metered_committee import MeteredCommittee

    counter = resolve_counter("claude-opus-5")       # count_tokens if creds exist
    agent   = MeteredCommittee(tickers, counter, n_analysts=8)
    metered = CoordinationMeter(agent, CoordinationCostConfig(model="claude-opus-5"))
    runs    = run_all_episodes(metered, ...)
    print(metered.report())

`run_cbs.py` runs the whole thing end to end.

THE METER IS ONLY AS GOOD AS THE AGENT UNDER IT
-----------------------------------------------
CoordinationMeter cannot invent token counts. An agent that exposes neither
`last_usage()` nor `n_agents` -- which is what `build_trading_agents_adapter`
returns today -- meters latency only and reports a token bill of $0.00.
report() labels that case "ASSUMED -- do not publish", but coordination_adjusted_cbs
will still hand back a CBS computed against a free committee. Wire real counts
in (agent_payload.py measures them off the actual PiT payload) before believing
any number this module prints.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

TRADING_SECONDS_PER_DAY = 23_400.0      # 6.5 hours of continuous session


# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Model price registry
# ---------------------------------------------------------------------------
# Prices are USD per MILLION tokens, the unit vendors publish in.
#
# SOURCES AND DATES -- update these together, never one without the other:
#   Anthropic : https://platform.claude.com/docs/en/about-claude/pricing
#               verified 2026-08-14. Rates below are the FIRST-PARTY API rates.
#               Bedrock/Vertex are partner-operated and priced separately; do
#               not reuse these figures for those.
#
# Three corrections against the previous version of this table, all of which
# moved the CBS materially:
#
#   1. Cache WRITES were not modelled at all -- only reads. A cache write costs
#      MORE than an uncached token (1.25x base at 5m TTL, 2x at 1h), so a
#      committee that re-writes its prefix every session was being under-billed.
#   2. `token_inflation` is gone. It existed to inflate an ASSUMED token count
#      for the newer tokenizer. Token counts are now measured (agent_payload.py)
#      against the deployed model, so the correction is already baked in and
#      applying it again double-counts.
#   3. `cache_min_tokens` added. A prefix shorter than the model's minimum
#      silently does not cache -- no error, just cache_creation_input_tokens=0.
#      The minimum is NOT monotonic across generations (512 on Opus 5, but 4096
#      on Haiku 4.5), so a design that caches on one model silently pays full
#      price on another.
#
# Cost is a MODEL-SELECTION question, not a constant. Report a curve across
# this table, not one number.


@dataclass(frozen=True)
class ModelPrice:
    name: str
    input_per_mtok: float
    output_per_mtok: float
    source: str
    verified: str
    cache_min_tokens: int = 1024      # prefixes below this silently do not cache
    cache_read_mult: float = 0.10     # cache hit  ~= 0.1x base input
    cache_write_mult: float = 1.25    # cache write = 1.25x base (5m TTL); 2.0 at 1h

    @property
    def cached_input_per_mtok(self) -> float:
        return self.input_per_mtok * self.cache_read_mult

    @property
    def cache_write_per_mtok(self) -> float:
        return self.input_per_mtok * self.cache_write_mult

    def cost(self, input_tokens: int, output_tokens: int,
             cached_input_tokens: int = 0, cache_write_tokens: int = 0,
             batch: bool = False) -> float:
        """Dollar cost of one usage record.

        The three input channels are disjoint -- a token is billed exactly
        once, as uncached, as a cache read, or as a cache write. Double-
        counting cached tokens inside `input_tokens` is the classic way to
        overstate a committee's bill by ~3x.
        """
        mult = 0.5 if batch else 1.0          # batch API = 50% off both directions
        return mult * (
            input_tokens / 1e6 * self.input_per_mtok
            + cached_input_tokens / 1e6 * self.cached_input_per_mtok
            + cache_write_tokens / 1e6 * self.cache_write_per_mtok
            + output_tokens / 1e6 * self.output_per_mtok
        )


_A = "anthropic docs (first-party API)"

MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-haiku-4-5":  ModelPrice("Claude Haiku 4.5",  1.0,  5.0, _A, "2026-08-14", cache_min_tokens=4096),
    "claude-sonnet-5":   ModelPrice("Claude Sonnet 5",   3.0, 15.0, _A, "2026-08-14", cache_min_tokens=1024),
    "claude-sonnet-5-intro": ModelPrice("Claude Sonnet 5 (intro, to 2026-08-31)",
                                                         2.0, 10.0, _A, "2026-08-14", cache_min_tokens=1024),
    "claude-opus-4-8":   ModelPrice("Claude Opus 4.8",   5.0, 25.0, _A, "2026-08-14", cache_min_tokens=1024),
    "claude-opus-5":     ModelPrice("Claude Opus 5",     5.0, 25.0, _A, "2026-08-14", cache_min_tokens=512),
    "claude-fable-5":    ModelPrice("Claude Fable 5",   10.0, 50.0, _A, "2026-08-14", cache_min_tokens=512),
}


@dataclass
class CoordinationCostConfig:
    """Every assumption in one place. Report these values in the paper.

    model
        A key into MODEL_PRICES. This is now an explicit, required decision --
        there is no default price, because there is no such thing as "the"
        price of an LLM call and a 45x spread should never be hidden in a
        default argument.

    use_prompt_caching / cache_ttl_1h
        A committee re-sends the same system prefix to every seat, which is the
        case prompt caching exists for: reads bill at ~0.1x base input. But
        caching is not free and not automatic --
          * a WRITE costs 1.25x base (5m TTL) or 2.0x (`cache_ttl_1h`), so the
            first seat pays MORE than it would uncached;
          * a prefix shorter than `ModelPrice.cache_min_tokens` silently does
            not cache at all -- no error, just no discount.
        There is deliberately no assumed hit rate here. The agent reports the
        real read/write/uncached split (see metered_committee.py); an agent
        that reports nothing gets no caching credit, because an unmeasured
        discount is just a thumb on the scale.

    use_batch_api
        50% off both directions, 24-hour turnaround. Applies to an offline
        backtest sweep; does NOT apply to anything claiming to be live trading.
        Say which one you are reporting.

    latency_alpha_halflife_seconds
        THE important dial, and the one that needs justifying. Latency does
        not cost money by itself -- a random walk moves against you exactly as
        often as it moves for you, so pure timing noise adds variance, not a
        systematic loss. What latency actually costs you is DECAY: a signal
        worth X now is worth X * 2^(-delay/halflife) by the time you act.
        Daily-bar momentum decays over days, so 6 hours (21600s) is a
        conservative default. An intraday signal might be 60s. SWEEP THIS.

    charge_timing_noise
        If True, also inject symmetric price noise of size
        sigma * sqrt(delay / trading_day). Mean-zero, so it widens confidence
        intervals rather than shifting the mean -- which is itself worth
        reporting, since a slow system is a noisier system.
    """
    model: str = "claude-opus-5"
    use_prompt_caching: bool = True
    cache_ttl_1h: bool = False        # True -> writes cost 2.0x base instead of 1.25x
    use_batch_api: bool = False
    charge_tokens_to_portfolio: bool = True

    latency_alpha_halflife_seconds: float = 21_600.0
    charge_latency: bool = True
    charge_timing_noise: bool = False

    # Fallbacks used ONLY when an agent reports no real counts via last_usage().
    # With agent_payload.py wired in, nothing should reach these -- they exist
    # so an unmetered agent fails loudly in report() rather than silently
    # costing $0.00.
    assumed_tokens_per_llm_call_in: int = 4_000
    assumed_tokens_per_llm_call_out: int = 800
    assumed_seconds_per_llm_call: float = 8.0

    @property
    def price(self) -> ModelPrice:
        if self.model not in MODEL_PRICES:
            raise KeyError(
                f"{self.model!r} is not in MODEL_PRICES. Add it with its source "
                f"and verification date rather than guessing -- known keys: "
                f"{sorted(MODEL_PRICES)}"
            )
        return MODEL_PRICES[self.model]

    def effective_price(self) -> ModelPrice:
        """Price with the configured cache TTL applied to the write rate."""
        p = self.price
        if self.cache_ttl_1h:
            return replace(p, cache_write_mult=2.0)
        return p


# ===========================================================================
# A. The meter
# ===========================================================================

@dataclass
class DecisionRecord:
    as_of: pd.Timestamp
    llm_calls: int = 0
    input_tokens: int = 0            # UNCACHED input only
    cached_input_tokens: int = 0     # cache READS, ~0.1x base
    cache_write_tokens: int = 0      # cache WRITES, 1.25x base (2.0x at 1h TTL)
    output_tokens: int = 0
    messages: int = 0
    latency_seconds: float = 0.0
    n_orders: int = 0
    measured: bool = False           # True if the agent reported real counts
    failed: bool = False             # agent raised; latency still charged

    def token_cost_usd(self, cfg: CoordinationCostConfig) -> float:
        return cfg.effective_price().cost(
            self.input_tokens, self.output_tokens,
            self.cached_input_tokens, self.cache_write_tokens,
            batch=cfg.use_batch_api)

    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.cache_write_tokens


class CoordinationMeter:
    """Wraps an AgentFn (runner.py signature) and meters every decision.

    Works three ways, in descending order of trustworthiness:

    1. The agent exposes `last_usage()` returning a dict with any of
       {llm_calls, input_tokens, output_tokens, messages}. Real measurement --
       use this. For TradingAgents, read the LangChain callback token counts
       and expose them here.
    2. The agent exposes `n_agents` (int). Falls back to the assumed
       per-call figures in the config, multiplied by that count.
    3. Neither. Only wall-clock latency is metered, which is still real.

    Latency is always measured for real, with time.perf_counter().
    """

    def __init__(self, agent, cfg: Optional[CoordinationCostConfig] = None,
                 name: str = "", n_agents: Optional[int] = None):
        self.agent = agent
        self.cfg = cfg or CoordinationCostConfig()
        self.name = name or getattr(agent, "name", agent.__class__.__name__)
        self.n_agents = n_agents if n_agents is not None else getattr(agent, "n_agents", 1)
        self.records: list[DecisionRecord] = []

    # --- AgentFn interface passthrough -------------------------------------
    def reset(self) -> None:
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        # records deliberately NOT cleared: cost accumulates across episodes,
        # which is what you want for a per-system cost report. Call
        # reset_meter() if you need per-episode scoping.

    def reset_meter(self) -> None:
        self.records.clear()

    def __call__(self, decision_snapshots, positions, cash, as_of):
        t0 = time.perf_counter()
        failed = False
        try:
            orders = self.agent(decision_snapshots, positions, cash, as_of)
        except Exception:
            # A committee that burns 90 seconds and then throws still consumed
            # 90 seconds and every token it sent. Letting the exception skip
            # the record made failures free, which under-charges exactly the
            # slowest, most expensive decisions. Record, then re-raise so
            # runner.run_episode still treats the day as no-trade.
            elapsed = time.perf_counter() - t0
            self._record(as_of, {}, elapsed, 0, failed=True)
            raise
        elapsed = time.perf_counter() - t0

        usage = {}
        if hasattr(self.agent, "last_usage"):
            try:
                usage = self.agent.last_usage() or {}
            except Exception:
                usage = {}

        self._record(as_of, usage, elapsed, len(orders or []), failed=failed)
        return orders or []

    def _record(self, as_of, usage, elapsed, n_orders, failed=False):
        cfg = self.cfg
        calls = int(usage.get("llm_calls", 0))
        measured = bool(usage)
        if not calls and self.n_agents > 1:
            calls = self.n_agents              # assume one call per agent per decision

        if measured and ("input_tokens" in usage or "cached_input_tokens" in usage):
            # Real counts, already in the deployed model's tokenizer.
            uncached = int(usage.get("input_tokens", 0))
            cached = int(usage.get("cached_input_tokens", 0))
            written = int(usage.get("cache_write_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
        else:
            # Assumed counts -- no caching credit is given here on purpose.
            # The old code assumed an 80% hit rate against an unmeasured
            # prefix, which is the single most flattering assumption available
            # and produced a near-free committee. If you have not measured the
            # split, you have not earned the discount.
            uncached = int(calls * cfg.assumed_tokens_per_llm_call_in)
            out_tok = int(calls * cfg.assumed_tokens_per_llm_call_out)
            cached = written = 0

        self.records.append(DecisionRecord(
            as_of=as_of,
            llm_calls=calls,
            input_tokens=uncached,
            cached_input_tokens=cached,
            cache_write_tokens=written,
            output_tokens=out_tok,
            messages=int(usage.get("messages", max(self.n_agents - 1, 0))),
            # An agent that models its own latency (no real LLM behind it)
            # reports it explicitly; otherwise fall back to measured wall-clock
            # floored at the assumed per-call cost.
            latency_seconds=float(usage["latency_seconds"]) if "latency_seconds" in usage
            else (max(elapsed, calls * cfg.assumed_seconds_per_llm_call) if calls else elapsed),
            n_orders=n_orders,
            measured=measured,
            failed=failed,
        ))

    # --- reporting ----------------------------------------------------------
    def total_token_cost_usd(self) -> float:
        return sum(r.token_cost_usd(self.cfg) for r in self.records)

    def mean_latency_seconds(self) -> float:
        return float(np.mean([r.latency_seconds for r in self.records])) if self.records else 0.0

    def frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.__dict__ for r in self.records])

    def report(self) -> str:
        if not self.records:
            return f"{self.name}: no decisions metered."
        f = self.frame()
        p = self.cfg.effective_price()
        provenance = (getattr(self, "token_provenance", None)
                      or ("MEASURED from the agent" if f["measured"].any()
                          else "ASSUMED -- not measured, do not publish"))
        n_fail = int(f["failed"].sum())
        return "\n".join([
            f"=== Coordination cost: {self.name} ===",
            f"  model                : {p.name}  (${p.input_per_mtok}/${p.output_per_mtok} "
            f"per MTok, {p.source}, verified {p.verified})",
            f"  cache rates          : read ${p.cached_input_per_mtok:.2f} / "
            f"write ${p.cache_write_per_mtok:.2f} per MTok "
            f"(min cacheable prefix {p.cache_min_tokens:,} tok)",
            f"  billing              : caching {'on' if self.cfg.use_prompt_caching else 'off'}, "
            f"batch {'on' if self.cfg.use_batch_api else 'off'}, "
            f"TTL {'1h' if self.cfg.cache_ttl_1h else '5m'}",
            f"  token counts         : {provenance}",
            f"  decisions            : {len(f):,}" + (f"  ({n_fail} failed)" if n_fail else ""),
            f"  LLM calls            : {int(f['llm_calls'].sum()):,} "
            f"({f['llm_calls'].mean():.1f} per decision)",
            f"  input tokens         : {int(f['input_tokens'].sum()):,} uncached / "
            f"{int(f['cached_input_tokens'].sum()):,} cache-read / "
            f"{int(f['cache_write_tokens'].sum()):,} cache-write",
            f"  output tokens        : {int(f['output_tokens'].sum()):,}",
            f"  inter-agent messages : {int(f['messages'].sum()):,}",
            f"  token cost           : ${self.total_token_cost_usd():,.2f}",
            f"  mean decision latency: {self.mean_latency_seconds():.2f} s",
            f"  signal decay at that latency: "
            f"{alpha_decay_factor(self.mean_latency_seconds(), self.cfg):.1%} of edge retained",
        ])

    # --- the cost curve: what this run costs on every model in the ladder ---
    def cost_across_models(self, models=None, batch: bool = False) -> pd.DataFrame:
        """Same token volume, priced on every model. Report this instead of one
        number: coordination cost is a model-selection decision, and hiding a
        45x spread behind a single figure is exactly the kind of buried
        assumption this benchmark exists to surface."""
        if not self.records:
            return pd.DataFrame()
        f = self.frame()
        unc, cac, wrt, out = (int(f["input_tokens"].sum()),
                              int(f["cached_input_tokens"].sum()),
                              int(f["cache_write_tokens"].sum()),
                              int(f["output_tokens"].sum()))
        # Token volume was measured against ONE model's tokenizer. Repricing it
        # on another model is a price swap, not a re-tokenization -- models in
        # different tokenizer families would emit different counts for the same
        # text. Flagged rather than silently corrected, because the correction
        # requires re-measuring, not a multiplier.
        rows = []
        for key in (models or MODEL_PRICES):
            mp = MODEL_PRICES[key]
            rows.append({
                "model": mp.name,
                "usd_per_mtok": f"${mp.input_per_mtok:g}/${mp.output_per_mtok:g}",
                "cost_usd": round(mp.cost(unc, out, cac, wrt, batch=batch), 2),
                "prefix_caches": (
                    "?" if not self._prefix_tokens
                    else ("yes" if self._prefix_tokens >= mp.cache_min_tokens
                          else f"NO ({self._prefix_tokens}<{mp.cache_min_tokens})")),
                "source": mp.source,
            })
        return (pd.DataFrame(rows).sort_values("cost_usd").reset_index(drop=True))

    # Set by the metered agent so cost_across_models can tell you whether the
    # cacheable prefix actually clears each model's minimum.
    _prefix_tokens: int = 0


# ===========================================================================
# B. Charging the two cost channels
# ===========================================================================

def alpha_decay_factor(latency_seconds: float, cfg: CoordinationCostConfig) -> float:
    """Fraction of a signal's edge still available after `latency_seconds`.

    Exponential decay with the configured half-life. 1.0 = instant decision,
    0.5 = you waited exactly one half-life and half your edge is gone.
    """
    if latency_seconds <= 0 or cfg.latency_alpha_halflife_seconds <= 0:
        return 1.0
    return float(2.0 ** (-latency_seconds / cfg.latency_alpha_halflife_seconds))


def timing_noise_bps(latency_seconds: float, daily_vol: float) -> float:
    """Std-dev of the price move that happens WHILE the committee deliberates.

    Random walk: sigma over a fraction of a session scales with sqrt(time).
    Mean zero -- this widens dispersion, it does not create a systematic loss.
    """
    if latency_seconds <= 0 or daily_vol <= 0:
        return 0.0
    return float(daily_vol * math.sqrt(latency_seconds / TRADING_SECONDS_PER_DAY) * 1e4)


def apply_coordination_costs(portfolio_values: pd.Series,
                             meter: CoordinationMeter,
                             cfg: Optional[CoordinationCostConfig] = None) -> pd.Series:
    """Return a coordination-adjusted equity curve.

    Two adjustments, both applied to the SAME logged run -- no re-simulation,
    same arithmetic-on-the-log principle the execution breakeven sweep uses:

      TOKENS  subtract the running token bill from portfolio value.
      LATENCY scale the strategy's excess return (over its own starting value)
              by the alpha-decay factor. A system whose signal has fully
              decayed by the time it acts keeps none of its edge -- it ends
              flat, not negative, because a stale signal is uninformative
              rather than actively wrong.

    LIMITATION to state in the paper: this scales realized return by a decay
    factor rather than re-running the agent with delayed inputs. Re-running is
    strictly better and is the obvious follow-up; this is the cheap version
    that lets you report the magnitude today.
    """
    cfg = cfg or meter.cfg
    pv = pd.Series(portfolio_values).sort_index().astype(float)
    if len(pv) < 2:
        return pv
    start = float(pv.iloc[0])
    adjusted = pv.copy()

    if cfg.charge_latency and meter.records:
        keep = alpha_decay_factor(meter.mean_latency_seconds(), cfg)
        adjusted = start + (adjusted - start) * keep

    if cfg.charge_tokens_to_portfolio and meter.records:
        costs = pd.Series([r.token_cost_usd(cfg) for r in meter.records],
                          index=pd.to_datetime([r.as_of for r in meter.records]).normalize())
        by_day = costs.groupby(level=0).sum().sort_index()

        # Charge every decision whose timestamp falls at or before each mark,
        # rather than reindexing onto the mark dates. The old reindex silently
        # DROPPED any decision made on a day with no portfolio value -- which
        # run_episode produces whenever no ticker printed a close (halts,
        # holiday mismatches). Those decisions still cost money.
        pv_days = pv.index.normalize()
        window = by_day[(by_day.index >= pv_days[0]) & (by_day.index <= pv_days[-1])]
        cum = window.cumsum()
        # asof: cumulative spend as of each mark, carrying forward between marks
        running = pd.Series(
            cum.reindex(cum.index.union(pv_days)).ffill().reindex(pv_days).fillna(0.0).to_numpy(),
            index=pv.index)

        # Decisions outside the mark window are still real spend; charge them
        # at the end rather than losing them.
        stray = float(by_day.sum() - window.sum())
        adjusted = adjusted - running
        if stray:
            adjusted.iloc[-1] -= stray

    return adjusted


# ===========================================================================
# C. CBS that actually charges for coordination
# ===========================================================================

@dataclass
class CoordinationCBS:
    system: str
    baseline: str
    advantage_trading_costs_only: float     # what comparison.py measures today
    advantage_with_coordination: float      # after tokens + latency
    cbs_bps_trading_only: Optional[float]
    cbs_bps_with_coordination: Optional[float]
    token_cost_usd: float
    mean_latency_seconds: float
    edge_retained: float

    def verdict(self) -> str:
        a, b = self.cbs_bps_trading_only, self.cbs_bps_with_coordination
        if self.advantage_with_coordination <= 0:
            return ("coordination NEVER pays once its own cost is charged "
                    f"(advantage {self.advantage_trading_costs_only:+,.0f} -> "
                    f"{self.advantage_with_coordination:+,.0f})")
        if a is not None and b is not None:
            if a > 0:
                return (f"CBS falls from {a:.0f} to {b:.0f} bps once coordination "
                        f"is billed -- a {100*(1-b/a):.0f}% reduction in headroom")
            # a == 0 means the edge did not survive even zero added trading
            # cost. The old code reported "coordination pays" here, which is
            # the opposite of what a zero CBS means.
            return ("no CBS headroom even before coordination is billed "
                    "(edge vanishes at 0 bps added cost)")
        if a is None and b is None:
            return "coordination survives the full tested cost grid"
        if b is None:
            return f"CBS >{self.cbs_bps_trading_only or 0:.0f} bps and survives the full grid once billed"
        return (f"edge survives the full grid on trading costs alone, but only "
                f"{b:.0f} bps once coordination is billed")


def coordination_adjusted_cbs(
    system_name: str, baseline_name: str,
    system_episodes: Sequence, baseline_episodes: Sequence,
    system_meter: CoordinationMeter,
    baseline_meter: Optional[CoordinationMeter] = None,
    step_bps: float = 1.0, max_bps: float = 200.0,
    cfg: Optional[CoordinationCostConfig] = None,
) -> CoordinationCBS:
    """CBS where the coordinated system pays its own coordination bill.

    Same sweep as comparison.coordination_breakeven_spread, but each side's
    per-episode PnL is first adjusted by apply_coordination_costs(). Pass a
    meter for the baseline too if it also makes LLM calls; omit it for a
    deterministic baseline, which then correctly pays nothing.
    """
    from comparison import _episode_adjusted_pnl          # your existing function
    cfg = cfg or system_meter.cfg

    def coordination_bill(episodes, meter) -> float:
        """Total coordination bill in dollars. Independent of the fee sweep, so
        it is computed ONCE rather than rebuilt at every bps step -- the old
        version re-ran a full pandas rebuild per episode per bps, 200x more
        work than needed for an identical answer."""
        if meter is None or not meter.records:
            return 0.0
        total = 0.0
        for ep in episodes:
            raw = pd.Series(ep.portfolio_values).sort_index()
            if len(raw) < 2:
                continue
            adj = apply_coordination_costs(ep.portfolio_values, meter, cfg)
            total += float(raw.iloc[-1] - adj.iloc[-1])
        return total

    sys_bill = coordination_bill(system_episodes, system_meter)
    base_bill = coordination_bill(baseline_episodes, baseline_meter)

    def side_pnl(episodes, bill, bps):
        return sum(_episode_adjusted_pnl(ep, bps) for ep in episodes) - bill

    adv_trading = side_pnl(system_episodes, 0.0, 0.0) - side_pnl(baseline_episodes, 0.0, 0.0)
    adv_full = side_pnl(system_episodes, sys_bill, 0.0) - side_pnl(baseline_episodes, base_bill, 0.0)

    def sweep(bs, bb):
        bps = 0.0
        if side_pnl(system_episodes, bs, 0.0) - side_pnl(baseline_episodes, bb, 0.0) <= 0:
            return 0.0
        while bps <= max_bps:
            if side_pnl(system_episodes, bs, bps) - side_pnl(baseline_episodes, bb, bps) <= 0:
                return bps
            bps += step_bps
        return None

    return CoordinationCBS(
        system=system_name, baseline=baseline_name,
        advantage_trading_costs_only=adv_trading,
        advantage_with_coordination=adv_full,
        cbs_bps_trading_only=sweep(0.0, 0.0),
        cbs_bps_with_coordination=sweep(sys_bill, base_bill),
        token_cost_usd=system_meter.total_token_cost_usd(),
        mean_latency_seconds=system_meter.mean_latency_seconds(),
        edge_retained=alpha_decay_factor(system_meter.mean_latency_seconds(), cfg),
    )


# ===========================================================================
# D. The controlled coordination ablation
# ===========================================================================

def coordination_ablation_table(results: dict[str, tuple]) -> pd.DataFrame:
    """results maps system_name -> (episodes, meter). One row per system.

    Designed for the paper's key comparison: run the SAME signals with 1, 3,
    and 8 analysts over the SAME episodes and seeds, so the only thing that
    varies is the amount of coordination. Anything else and you are comparing
    strategies, not coordination.
    """
    from phase3 import sharpe_ratio
    rows = []
    for name, (episodes, meter) in results.items():
        rets = pd.concat([ep.daily_returns() for ep in episodes]).sort_index()
        gross = float(sum(float(ep.portfolio_values.iloc[-1] - ep.portfolio_values.iloc[0])
                          for ep in episodes if len(ep.portfolio_values) > 1))
        adj = float(sum(float(apply_coordination_costs(ep.portfolio_values, meter).iloc[-1]
                              - ep.portfolio_values.iloc[0])
                        for ep in episodes if len(ep.portfolio_values) > 1)) if meter else gross
        rows.append({
            "system": name,
            "n_agents": getattr(meter, "n_agents", 1) if meter else 1,
            "sharpe": sharpe_ratio(rets),
            "pnl_before_coordination_cost": gross,
            "pnl_after_coordination_cost": adj,
            "coordination_bill": gross - adj,
            "token_cost_usd": meter.total_token_cost_usd() if meter else 0.0,
            "mean_latency_s": meter.mean_latency_seconds() if meter else 0.0,
            "pnl_per_dollar_of_tokens": (adj / meter.total_token_cost_usd()
                                         if meter and meter.total_token_cost_usd() > 0
                                         else float("nan")),
        })
    return pd.DataFrame(rows).set_index("system")
