"""
Runs the TradingAgents multi-agent system through the SAME 18-episode
balanced regime design (6 crisis + 6 trending + 6 mean_reverting) already
used for the single-agent baselines, via run_regime_episodes.py's own
_balanced_regime_episodes() -- imported directly, not re-hardcoded, so
there is exactly one source of truth for these 18 windows. Matching the
episode set is not optional: comparison.coordination_breakeven_spread
requires both systems in a comparison to have run the identical episode
set before it will compute anything.

Supports a per-regime decision cadence (--crisis-interval /
--other-interval, trading days between real decisions -- 1 means every
day) via phase3.CadenceGatedAgent, because TradingAgents' real per-decision
cost (measured, not assumed -- ~16 real LLM calls per ticker per day) makes
full daily cadence at full ticker/episode scope cost thousands of dollars.
Holding position between decisions keeps the trading ENVIRONMENT identical
to the baselines (same tickers, same episodes, same execution/fee model) --
only decision frequency is reduced, and it's reduced per-regime so the
regime that most needs reactive testing (crisis) gets more of the budget
than regimes where a single informed entry is a smaller compromise
(trending, mean-reverting). See CadenceGatedAgent's docstring for exactly
why this needs its own class rather than being inlined here -- there is a
real double-counting bug it exists to avoid.

Routes through Kimi-K2.6 on Azure AI Foundry by default (config below),
via TradingAgents' generic "openai"-compatible provider with a custom
backend_url pointed at the Foundry resource's unified /openai/v1 route --
confirmed working live (real BUY orders, real usage captured, correct
cost computed) against two separate Azure accounts before this was wired
in as the default.

Usage:
    # price it first -- no API calls, no cost
    python run_tradingagents_episodes.py --estimate-only

    # a cheap correctness check before spending real money
    python run_tradingagents_episodes.py --tickers AAPL MSFT --max-episodes 1 \\
        --crisis-interval 10 --other-interval 63 --estimate-only

    # the real run
    python run_tradingagents_episodes.py --system tradingagents_kimi \\
        --tickers AAPL MSFT GOOG AMZN NVDA TSLA JPM V JNJ WMT \\
        --crisis-interval 10 --other-interval 63 --max-spend 190

Then:
    python main.py report --systems tradingagents_kimi
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv

import dataprep
import main as main_mod
import phase3
from execution_replay import ReplayConfig
from run_regime_episodes import _balanced_regime_episodes

load_dotenv()

# Measured empirically this session (multiple real, complete propagate()
# calls across Claude, DeepSeek, and Kimi backbones, all at
# max_debate_rounds=1 / max_risk_discuss_rounds=1 -- TradingAgents' own
# floor, unchanged by this script): consistently in the 1-16+ real LLM
# call range per ticker's decision, varying by backbone. Used only to size
# the preflight --max-spend guard; the real run is priced from actually
# measured usage via TradingAgentsAdapter.cumulative_true_blended_cost_usd(),
# not this constant.
MEASURED_CALLS_PER_TICKER_DECISION = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=None,
                         help="default: dataprep.EVALUATION_TICKERS (full 20)")
    parser.add_argument("--system", default="tradingagents_kimi",
                         help="output filename stem -- writes runs/<system>.json")
    parser.add_argument("--notional-per-ticker", type=float, default=10_000.0)
    parser.add_argument("--model", default="kimi-k2.6",
                         help="a key in coordination_cost.MODEL_PRICES, used for both "
                              "deep_think_llm and quick_think_llm and for cost estimation")
    parser.add_argument("--azure-deployment", default="Kimi-K2.6",
                         help="exact deployment name as configured in the Foundry portal")
    parser.add_argument("--n-workers", type=int, default=6,
                         help="concurrent TradingAgentsGraph instances (one real "
                              "multi-agent run per ticker per decision; see "
                              "phase3.TradingAgentsAdapter for why each worker "
                              "needs its own graph instance, not a shared one)")
    parser.add_argument("--memory-base-dir", default="./ta_memory")
    parser.add_argument("--crisis-interval", type=int, default=1,
                         help="trading days between real decisions in crisis episodes "
                              "(1 = every day)")
    parser.add_argument("--other-interval", type=int, default=1,
                         help="trading days between real decisions in trending/"
                              "mean_reverting episodes (1 = every day)")
    parser.add_argument("--max-episodes", type=int, default=None,
                         help="override: run only the first N of the 18 episodes "
                              "(for a cheap correctness check -- NOT for the "
                              "reported results, which need all 18 to match the "
                              "baselines and to satisfy comparison.py's matched-"
                              "episode-set requirement)")
    parser.add_argument("--estimate-only", action="store_true",
                         help="print the projected cost and exit without calling the API")
    parser.add_argument("--max-spend", type=float, default=None,
                         help="USD ceiling on the projected cost. Required for a paid "
                              "run; the run refuses to start without it.")
    parser.add_argument("--endpoint-env", default="AZURE_FOUNDRY_ENDPOINT",
                         help="env var name holding the Azure Foundry endpoint URL to "
                              "use -- lets a second Azure account/deployment (its own "
                              "separate rate-limit quota) be selected without editing "
                              "this file, e.g. --endpoint-env AZURE_KIMI2_ENDPOINT")
    parser.add_argument("--api-key-env", default="AZURE_FOUNDRY_API_KEY",
                         help="env var name holding the matching API key")
    args = parser.parse_args()

    tickers = args.tickers or dataprep.EVALUATION_TICKERS
    episodes = _balanced_regime_episodes()
    # Chronological order, not the regime-grouped order _balanced_regime_episodes()
    # returns (all 6 crisis, then all 6 trending, then all 6 mean_reverting).
    # Confirmed live, from reading the actual ta_memory/*.md file a completed
    # run produced: TradingAgents' cross-episode self-reflection memory
    # (tradingagents/agents/utils/memory.py:get_past_context) injects PAST
    # log entries into EVERY future prompt for the same system, and that log
    # is written in EXECUTION order. Regime-grouped order meant a decision
    # dated 2010-01-04 (the first trending episode, executed 7th) could read
    # -- and did read, confirmed by inspecting the file -- memory entries
    # dated as late as 2022-12-23 from crisis episodes executed 1st-6th. The
    # data STORE's own point-in-time filter (dataprep.PointInTimeStore) was
    # never violated; this is a second, separate leak channel the standard
    # Temporal Integrity check does not audit at all. build_trading_agents_graph's
    # own docstring already warned this feature is "legitimate if and only if
    # episodes... are always run in true chronological order" -- the fix is
    # to actually do that, not to disable the feature TradingAgents is meant
    # to have.
    episodes = sorted(episodes, key=lambda e: pd.Timestamp(e[1]))
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    regime_intervals = {
        "crisis": max(1, args.crisis_interval),
        "trending": max(1, args.other_interval),
        "mean_reverting": max(1, args.other_interval),
    }

    from coordination_cost import MODEL_PRICES

    if args.model not in MODEL_PRICES:
        raise SystemExit(
            f"{args.model!r} has no entry in coordination_cost.MODEL_PRICES.\n"
            f"Known: {', '.join(sorted(MODEL_PRICES))}"
        )
    price = MODEL_PRICES[args.model]

    n_decisions = 0
    for regime, start, end in episodes:
        n_days = max(int((pd.Timestamp(end) - pd.Timestamp(start)).days * (252.0 / 365.0)), 1)
        interval = regime_intervals.get(regime, 1)
        n_real_decisions = -(-n_days // interval)  # ceil
        n_decisions += n_real_decisions * len(tickers)

    n_calls = n_decisions * MEASURED_CALLS_PER_TICKER_DECISION
    # Rough estimate only -- see run_tradingagents_episodes' real, measured
    # per-decision costs in conversation for this exact model; this preflight
    # guard uses a generic assumed-tokens-per-call figure, not the real split.
    assumed_tokens_in, assumed_tokens_out = 154466 / MEASURED_CALLS_PER_TICKER_DECISION, 36978 / MEASURED_CALLS_PER_TICKER_DECISION
    est_cost = n_calls * (assumed_tokens_in * price.input_per_mtok + assumed_tokens_out * price.output_per_mtok) / 1_000_000.0

    print(f"\npreflight cost estimate ({len(episodes)} episode(s), {len(tickers)} tickers, "
          f"crisis every {regime_intervals['crisis']}d / other every {regime_intervals['trending']}d)")
    print(f"  ~{n_decisions:,} real ticker-decisions, ~{n_calls:,} underlying LLM calls")
    print(f"  model: {price.name} (${price.input_per_mtok}/${price.output_per_mtok} per MTok)")
    print(f"  projected: ${est_cost:,.2f}  (rough -- generic per-call token assumption, "
          f"not this model's real measured split; the real run is priced from actual usage)")
    if args.estimate_only:
        print("\n--estimate-only: stopping before any API call.")
        return
    if args.max_spend is None:
        raise SystemExit(
            f"\nrefusing to start an unbounded paid run.\n"
            f"Pass --max-spend USD to authorise it (rough projected cost above is "
            f"${est_cost:,.2f}), or --estimate-only to just price it."
        )
    if est_cost > args.max_spend * 1.5:
        # Generous slack over the rough estimate -- this guard exists to catch
        # a wildly wrong CLI arg (e.g. forgetting --crisis-interval), not to
        # second-guess a well-reasoned budget with an admittedly rough number.
        raise SystemExit(
            f"\nrough projected cost ${est_cost:,.2f} exceeds 1.5x --max-spend "
            f"${args.max_spend:,.2f}. If this is intentional, raise --max-spend."
        )
    print(f"  authorised: proceeding under --max-spend ${args.max_spend:,.2f}\n")

    endpoint = os.environ.get(args.endpoint_env)
    api_key = os.environ.get(args.api_key_env)
    if not endpoint or not api_key:
        raise SystemExit(f"{args.endpoint_env} / {args.api_key_env} not set in .env")
    # Two URL shapes confirmed live this session, from two different Azure
    # accounts: (1) a project-scoped Foundry endpoint
    # (".../api/projects/<project>") -- strip that suffix, append /models
    # for the unified inference route; (2) an already-complete chat-
    # completions URL (".../openai/v1/chat/completions") -- strip that
    # suffix directly to get the base_url ChatOpenAI wants. Detected by
    # which pattern is actually present rather than assumed, since guessing
    # wrong here silently sends every request to a 404.
    if "/api/projects/" in endpoint:
        backend_url = f"{endpoint.split('/api/projects/')[0]}/models"
    elif endpoint.endswith("/chat/completions"):
        backend_url = endpoint.rsplit("/chat/completions", 1)[0]
    else:
        raise SystemExit(f"unrecognized endpoint URL shape: {endpoint!r}")
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"  routing via {args.endpoint_env} -> backend_url={backend_url}")

    n_workers = max(1, min(args.n_workers, len(tickers)))
    ta_config = {
        "llm_provider": "openai",
        "deep_think_llm": args.azure_deployment,
        "quick_think_llm": args.azure_deployment,
        "backend_url": backend_url,
    }
    graph_pool = [
        phase3.build_trading_agents_graph(args.system, ta_config, args.memory_base_dir)
        for _ in range(n_workers)
    ]
    ta_adapter = phase3.TradingAgentsAdapter(graph_pool, tickers, args.notional_per_ticker)
    gated = phase3.CadenceGatedAgent(ta_adapter, regime_intervals)

    from coordination_cost import CoordinationCostConfig, CoordinationMeter
    cost_cfg = CoordinationCostConfig(model=args.model, use_prompt_caching=False)
    meter = CoordinationMeter(gated, cost_cfg, name=args.system)

    config = ReplayConfig()
    runs = []
    for regime, start, end in episodes:
        ep_start = pd.Timestamp(start, tz="UTC")
        ep_end = pd.Timestamp(end, tz="UTC")
        gated.set_regime(regime)
        store = dataprep.PointInTimeStore()
        run = phase3.run_episode(meter, store, tickers, ep_start, ep_end, regime, config)
        runs.append(run)
        print(f"{ep_start.date()}..{ep_end.date()} ({regime}): "
              f"{len(run.execution_log)} fills, {len(run.portfolio_values)} trading days "
              f"(running true cost: ${ta_adapter.cumulative_true_blended_cost_usd():,.2f})",
              flush=True)

    true_cost_total = ta_adapter.cumulative_true_blended_cost_usd()

    main_mod.RUNS_DIR.mkdir(exist_ok=True)
    out = main_mod.RUNS_DIR / f"{args.system}.json"
    payload = {
        "system": args.system,
        "impact_gamma": config.impact_gamma,
        "episodes": [main_mod._episode_to_dict(r) for r in runs],
        "coordination": meter.to_dict(),
        "true_blended_cost_usd": true_cost_total,
        "regime_intervals": regime_intervals,
        "tickers": tickers,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n{len(runs)} episode(s) -> {out}")
    print(meter.report())
    print(f"\nTrue cost (real tokens at real rate, exact): ${true_cost_total:,.2f}")
    print(f"\nNow run: python main.py report --systems {args.system}")


if __name__ == "__main__":
    main()
