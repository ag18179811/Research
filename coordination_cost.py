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

    from coordination_cost import CoordinationMeter, CoordinationCostConfig
    metered = CoordinationMeter(my_agent, CoordinationCostConfig())
    runs = run_all_episodes(metered, ...)
    print(metered.report())
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_SECONDS_PER_DAY = 23_400.0      # 6.5 hours of continuous session


# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Model price registry
# ---------------------------------------------------------------------------
# Prices are USD per MILLION tokens, the unit both vendors publish in.
#
# SOURCES AND DATES -- update these together, never one without the other:
#   Anthropic : https://platform.claude.com/docs/en/about-claude/pricing
#               verified 2026-08-05 against the official docs page
#   OpenAI    : https://platform.openai.com/docs/pricing
#               figures below are from third-party aggregators and are NOT
#               yet verified against the official page. Verify before citing.
#
# Cost is a MODEL-SELECTION question, not a constant: the same 29.6M-token
# committee run costs $10.87 on GPT-5.6 Luna and $494.08 on Claude Fable 5,
# a 45x spread. Report a curve across this table, not one number.


@dataclass(frozen=True)
class ModelPrice:
    name: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float      # cache hit / read rate
    source: str
    verified: str
    token_inflation: float = 1.0      # see note below

    def cost(self, input_tokens: int, output_tokens: int,
             cached_input_tokens: int = 0, batch: bool = False) -> float:
        """Dollar cost of one usage record. Cached tokens are billed at the
        cache-hit rate and are NOT double-counted in `input_tokens`."""
        mult = 0.5 if batch else 1.0          # both vendors: batch = 50% off
        return mult * (
            input_tokens / 1e6 * self.input_per_mtok
            + cached_input_tokens / 1e6 * self.cached_input_per_mtok
            + output_tokens / 1e6 * self.output_per_mtok
        )


_A = "anthropic docs"
_O = "openai (3rd-party, UNVERIFIED)"

# `token_inflation`: Claude 4.7 and later use a newer tokenizer that emits
# roughly 30% more tokens for the same text than 4.6 and earlier. It is applied
# only to ASSUMED token counts -- when an agent reports measured counts via
# last_usage(), those are already in the right tokenizer and no factor applies.
MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-haiku-4.5":  ModelPrice("Claude Haiku 4.5",   1.0,  5.0, 0.10, _A, "2026-08-05"),
    "claude-sonnet-5":   ModelPrice("Claude Sonnet 5",    3.0, 15.0, 0.30, _A, "2026-08-05", 1.3),
    "claude-sonnet-5-intro": ModelPrice("Claude Sonnet 5 (intro, to 2026-08-31)",
                                                          2.0, 10.0, 0.20, _A, "2026-08-05", 1.3),
    "claude-opus-5":     ModelPrice("Claude Opus 5",      5.0, 25.0, 0.50, _A, "2026-08-05", 1.3),
    "claude-fable-5":    ModelPrice("Claude Fable 5",    10.0, 50.0, 1.00, _A, "2026-08-05", 1.3),
    "gpt-5.6-luna":      ModelPrice("GPT-5.6 Luna",       0.20, 1.20, 0.02, _O, "2026-08-05"),
    "gpt-5.6-terra":     ModelPrice("GPT-5.6 Terra",      2.0, 12.0, 0.20, _O, "2026-08-05"),
    "gpt-5.6-sol":       ModelPrice("GPT-5.6 Sol",        5.0, 30.0, 0.50, _O, "2026-08-05"),
    "gpt-5.4":           ModelPrice("GPT-5.4",            2.50, 15.0, 0.25, _O, "2026-08-05"),
}


@dataclass
class CoordinationCostConfig:
    """Every assumption in one place. Report these values in the paper.

    model
        A key into MODEL_PRICES. This is now an explicit, required decision --
        there is no default price, because there is no such thing as "the"
        price of an LLM call and a 45x spread should never be hidden in a
        default argument.

    use_prompt_caching / assumed_cache_hit_rate
        A committee re-sends the same large system prompt on every call, which
        is exactly the case prompt caching exists for. Cache hits bill at ~10%
        of base input. Ignoring this overstates a realistic deployment's
        coordination bill by roughly 3x. `assumed_cache_hit_rate` is only used
        when the agent doesn't report real cached-token counts.

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
    model: str = "claude-sonnet-5"
    use_prompt_caching: bool = True
    assumed_cache_hit_rate: float = 0.80
    use_batch_api: bool = False
    charge_tokens_to_portfolio: bool = True

    latency_alpha_halflife_seconds: float = 21_600.0
    charge_latency: bool = True
    charge_timing_noise: bool = False

    # Fallbacks used ONLY when an agent reports no real counts via last_usage().
    # These are ASSUMPTIONS -- measure and replace before publishing.
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


# ===========================================================================
# A. The meter
# ===========================================================================

@dataclass
class DecisionRecord:
    as_of: pd.Timestamp
    llm_calls: int = 0
    input_tokens: int = 0            # UNCACHED input only
    cached_input_tokens: int = 0     # billed at the cache-hit rate, ~10% of base
    output_tokens: int = 0
    messages: int = 0
    latency_seconds: float = 0.0
    n_orders: int = 0
    measured: bool = False           # True if the agent reported real counts

    def token_cost_usd(self, cfg: CoordinationCostConfig) -> float:
        return cfg.price.cost(self.input_tokens, self.output_tokens,
                              self.cached_input_tokens, batch=cfg.use_batch_api)

    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens


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
        orders = self.agent(decision_snapshots, positions, cash, as_of)
        elapsed = time.perf_counter() - t0

        usage = {}
        if hasattr(self.agent, "last_usage"):
            try:
                usage = self.agent.last_usage() or {}
            except Exception:
                usage = {}

        cfg = self.cfg
        calls = int(usage.get("llm_calls", 0))
        measured = bool(usage)
        if not calls and self.n_agents > 1:
            calls = self.n_agents              # assume one call per agent per decision

        if "input_tokens" in usage or "cached_input_tokens" in usage:
            # Real counts. Already in the deployed model's tokenizer, so no
            # inflation factor -- applying one here would double-count.
            uncached = int(usage.get("input_tokens", 0))
            cached = int(usage.get("cached_input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
        else:
            # Assumed counts. Scale by the model's tokenizer factor, then split
            # into cached/uncached at the assumed hit rate.
            infl = cfg.price.token_inflation
            total_in = int(calls * cfg.assumed_tokens_per_llm_call_in * infl)
            out_tok = int(calls * cfg.assumed_tokens_per_llm_call_out * infl)
            hit = cfg.assumed_cache_hit_rate if cfg.use_prompt_caching else 0.0
            cached = int(total_in * hit)
            uncached = total_in - cached

        rec = DecisionRecord(
            as_of=as_of,
            llm_calls=calls,
            input_tokens=uncached,
            cached_input_tokens=cached,
            output_tokens=out_tok,
            messages=int(usage.get("messages", max(self.n_agents - 1, 0))),
            latency_seconds=max(elapsed, calls * cfg.assumed_seconds_per_llm_call)
            if calls else elapsed,
            n_orders=len(orders or []),
            measured=measured,
        )
        self.records.append(rec)
        return orders or []

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
        p = self.cfg.price
        provenance = ("MEASURED from the agent" if f["measured"].any()
                      else "ASSUMED -- not measured, do not publish")
        return "\n".join([
            f"=== Coordination cost: {self.name} ===",
            f"  model                : {p.name}  (${p.input_per_mtok}/${p.output_per_mtok} "
            f"per MTok, {p.source}, verified {p.verified})",
            f"  billing              : caching {'on' if self.cfg.use_prompt_caching else 'off'}, "
            f"batch {'on' if self.cfg.use_batch_api else 'off'}",
            f"  token counts         : {provenance}",
            f"  decisions            : {len(f):,}",
            f"  LLM calls            : {int(f['llm_calls'].sum()):,} "
            f"({f['llm_calls'].mean():.1f} per decision)",
            f"  tokens               : {int(f['input_tokens'].sum()):,} uncached in / "
            f"{int(f['cached_input_tokens'].sum()):,} cached in / "
            f"{int(f['output_tokens'].sum()):,} out",
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
        unc, cac, out = (int(f["input_tokens"].sum()),
                         int(f["cached_input_tokens"].sum()),
                         int(f["output_tokens"].sum()))
        rows = []
        for key in (models or MODEL_PRICES):
            mp = MODEL_PRICES[key]
            rows.append({
                "model": mp.name,
                "usd_per_mtok": f"${mp.input_per_mtok:g}/${mp.output_per_mtok:g}",
                "cost_usd": round(mp.cost(unc, out, cac, batch=batch), 2),
                "source": mp.source,
            })
        return (pd.DataFrame(rows).sort_values("cost_usd").reset_index(drop=True))


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
        by_day = costs.groupby(level=0).sum()
        running = by_day.reindex(pv.index.normalize(), fill_value=0.0).cumsum()
        running.index = pv.index
        adjusted = adjusted - running

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
            return (f"CBS falls from {a:.0f} to {b:.0f} bps once coordination "
                    f"is billed -- a {100*(1-b/a):.0f}% reduction in headroom"
                    if a > 0 else "coordination pays")
        return "coordination survives the full tested cost grid"


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

    def side_pnl(episodes, meter, bps):
        total = 0.0
        for ep in episodes:
            base = _episode_adjusted_pnl(ep, bps)
            if meter is None or not meter.records:
                total += base
                continue
            adj = apply_coordination_costs(ep.portfolio_values, meter, cfg)
            raw = pd.Series(ep.portfolio_values).sort_index()
            # difference between adjusted and raw final value = the coordination bill
            bill = float(raw.iloc[-1] - adj.iloc[-1]) if len(raw) > 1 else 0.0
            total += base - bill
        return total

    adv_trading = (side_pnl(system_episodes, None, 0.0)
                   - side_pnl(baseline_episodes, None, 0.0))
    adv_full = (side_pnl(system_episodes, system_meter, 0.0)
                - side_pnl(baseline_episodes, baseline_meter, 0.0))

    def sweep(meter_sys, meter_base):
        bps = 0.0
        if (side_pnl(system_episodes, meter_sys, 0.0)
                - side_pnl(baseline_episodes, meter_base, 0.0)) <= 0:
            return 0.0
        while bps <= max_bps:
            if (side_pnl(system_episodes, meter_sys, bps)
                    - side_pnl(baseline_episodes, meter_base, bps)) <= 0:
                return bps
            bps += step_bps
        return None

    return CoordinationCBS(
        system=system_name, baseline=baseline_name,
        advantage_trading_costs_only=adv_trading,
        advantage_with_coordination=adv_full,
        cbs_bps_trading_only=sweep(None, None),
        cbs_bps_with_coordination=sweep(system_meter, baseline_meter),
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
    from metrics import sharpe_ratio
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
