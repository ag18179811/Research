"""
Runs a single-agent baseline through the SAME 18-episode design, SAME
ticker subset, and SAME per-regime decision cadence as a reduced-cadence
TradingAgents run (see run_tradingagents_episodes.py), via
phase3.CadenceGatedAgent wrapping main._build_llm_agent. This is the
matched control a coordination-value comparison against a cadence-reduced
TradingAgents run actually needs -- without it, TradingAgents (few
decisions, holding in between) would be compared against a baseline that
decided every single day, which answers "does coordination help" and
"does deciding more often help" at once instead of isolating coordination.

Cheap by construction: the single-agent baseline makes one LLM call per
real decision (not ~16 like TradingAgents), so this costs a small fraction
of the TradingAgents run it's meant to match.

Usage:
    python run_matched_control.py --model kimi-k2.6 \\
        --tickers AAPL MSFT GOOG AMZN NVDA TSLA JPM V JNJ WMT \\
        --crisis-interval 10 --other-interval 63 \\
        --system kimi_matched_control --max-spend 5
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import dataprep
import main as main_mod
import phase3
from execution_replay import ReplayConfig
from run_regime_episodes import _balanced_regime_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                         help="a key in coordination_cost.MODEL_PRICES")
    parser.add_argument("--tickers", nargs="*", required=True,
                         help="must match the TradingAgents run's ticker list exactly")
    parser.add_argument("--system", required=True,
                         help="output filename stem -- writes runs/<system>.json")
    parser.add_argument("--crisis-interval", type=int, default=1)
    parser.add_argument("--other-interval", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--max-spend", type=float, default=None)
    args = parser.parse_args()

    # Chronological order, not _balanced_regime_episodes()'s regime-grouped
    # order -- matches the fix in run_tradingagents_episodes.py (see that
    # file for the full explanation: TradingAgents' cross-episode
    # self-reflection memory is only valid in true chronological order, and
    # a regime-grouped run was confirmed to leak future entries into past
    # decisions). _LLMAgent has no equivalent memory-reflection feature, so
    # this system was never actually exposed to that leak -- sorting here
    # anyway so both systems in the comparison ran their episodes in the
    # identical order, removing any doubt rather than relying on "it
    # shouldn't matter for this one."
    episodes = sorted(_balanced_regime_episodes(), key=lambda e: pd.Timestamp(e[1]))
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    regime_intervals = {
        "crisis": max(1, args.crisis_interval),
        "trending": max(1, args.other_interval),
        "mean_reverting": max(1, args.other_interval),
    }

    from coordination_cost import CoordinationCostConfig, CoordinationMeter, estimate_cost

    agent = main_mod._build_llm_agent(args.tickers, model=args.model, use_prompt_caching=not args.no_cache)
    gated = phase3.CadenceGatedAgent(agent, regime_intervals)

    cost_cfg = CoordinationCostConfig(model=args.model, use_prompt_caching=not args.no_cache)
    meter = CoordinationMeter(gated, cost_cfg, name=args.system)
    meter._prefix_tokens = agent.prefix_tokens()

    n_decisions = 0
    for regime, start, end in episodes:
        n_days = max(int((pd.Timestamp(end) - pd.Timestamp(start)).days * (252.0 / 365.0)), 1)
        interval = regime_intervals.get(regime, 1)
        n_decisions += -(-n_days // interval)  # ceil, one call per decision (whole-universe prompt)

    est = estimate_cost(cost_cfg, n_decisions, cost_cfg.assumed_tokens_per_llm_call_in,
                         prefix_tokens=meter._prefix_tokens)
    print(f"\npreflight cost estimate ({len(episodes)} episode(s), {len(args.tickers)} tickers, "
          f"crisis every {regime_intervals['crisis']}d / other every {regime_intervals['trending']}d)")
    print(est.summary(cost_cfg))
    if args.estimate_only:
        print("\n--estimate-only: stopping before any API call.")
        return
    if args.max_spend is None:
        raise SystemExit(
            f"\nrefusing to start an unbounded paid run.\n"
            f"Pass --max-spend USD to authorise it (projected list price above is "
            f"${est.list_usd:,.2f}), or --estimate-only to just price it."
        )
    if est.list_usd > args.max_spend:
        raise SystemExit(
            f"\nprojected list cost ${est.list_usd:,.2f} exceeds --max-spend "
            f"${args.max_spend:,.2f}."
        )
    print(f"  authorised: ${est.list_usd:,.2f} <= --max-spend ${args.max_spend:,.2f}\n")

    config = ReplayConfig()
    runs = []
    for regime, start, end in episodes:
        ep_start = pd.Timestamp(start, tz="UTC")
        ep_end = pd.Timestamp(end, tz="UTC")
        gated.set_regime(regime)
        store = dataprep.PointInTimeStore()
        run = phase3.run_episode(meter, store, args.tickers, ep_start, ep_end, regime, config)
        runs.append(run)
        print(f"{ep_start.date()}..{ep_end.date()} ({regime}): "
              f"{len(run.execution_log)} fills, {len(run.portfolio_values)} trading days", flush=True)

    main_mod.RUNS_DIR.mkdir(exist_ok=True)
    out = main_mod.RUNS_DIR / f"{args.system}.json"
    payload = {
        "system": args.system,
        "impact_gamma": config.impact_gamma,
        "episodes": [main_mod._episode_to_dict(r) for r in runs],
        "coordination": meter.to_dict(),
        "regime_intervals": regime_intervals,
        "tickers": args.tickers,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n{len(runs)} episode(s) -> {out}")
    print(meter.report())
    print(f"\nNow run: python main.py report --systems {args.system}")


if __name__ == "__main__":
    main()
