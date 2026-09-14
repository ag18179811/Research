"""
Runs one agent through a hand-picked set of specific episode windows --
by default, real VIX-derived crisis periods -- instead of the full
evaluation-range sweep main.py backtest generates automatically.

Exists because label_regimes' walk-forward HMM makes "crisis" structurally
rare at 90-day-window granularity (majority-vote over a long window smooths
out shorter stress periods): across the full 2010-2024 range it produces
exactly 2 crisis episodes out of 49. dataprep.find_vix_crisis_windows()
supplies a second, independent, more commonly-used market-stress signal
(CBOE's own VIX) that yields more explicitly-defined crisis periods to
specifically test against -- these are supplementary episodes, run
alongside (not instead of) the standard HMM-labeled sweep.

Also exists because main.py's CLI only takes a single contiguous
--start/--end range: generate_episode_windows() auto-generates every
non-overlapping window inside it, so there's no way to ask for several
specific, disjoint windows through the CLI alone. This script builds the
same agent main.py backtest --agent llm would (via main._build_llm_agent,
so CLAUDE_API_KEY wiring and model/cache config are identical), drives it
through each window one at a time via phase3.run_episode, and writes a
normal runs/<system>.json main.py report can load unmodified.

Usage:
    python run_regime_episodes.py                          # VIX crisis windows, full ticker universe
    python run_regime_episodes.py --system llm_crisis       # write to a separate file to merge later
    python run_regime_episodes.py --preset three_regimes    # the original 1-per-regime spot-check
    python run_regime_episodes.py --preset balanced_regimes # 6 crisis + 6 trending + 6 mean_reverting

Then, to see these episodes on their own:
    python main.py report --systems llm_crisis

Or merge with an existing runs/llm.json from a normal main.py backtest run
(same "episodes" list format) before reporting, so Regime Robustness sees
both the standard sweep's episodes and these crisis windows together.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import dataprep
import phase3
import main as main_mod
from execution_replay import ReplayConfig

# The original 3-window spot-check (one per HMM regime label), kept as an
# alternate preset -- see --preset three_regimes.
THREE_REGIME_EPISODES: list[tuple[str, str, str]] = [
    ("trending", "2013-03-16", "2013-04-25"),        # 2013 bull run
    ("mean_reverting", "2011-03-27", "2011-05-06"),  # choppy pre-debt-ceiling-crisis 2011
    ("crisis", "2020-02-08", "2020-03-19"),           # COVID crash
]

# 6 trending + 6 mean_reverting, real walk-forward-HMM-labeled windows (the
# SAME dataprep.label_regimes/assign_episode_regimes method the VIX-crisis
# preset's own regime labeling relies on elsewhere in the pipeline -- not
# hand-picked), spread evenly across the 18 trending / 29 mean_reverting
# windows genuinely present in the full 2010-2024 EVALUATION_TICKERS[0]
# reference series -- AFTER first filtering out any candidate window that
# overlaps one of the 6 real VIX-crisis windows below. That filter step is
# not optional: an earlier version of this list picked windows by spread
# alone and one (trending, 2022-03-31..2022-06-29) overlapped the crisis
# window 2022-01-25..2022-04-25 by 25 real trading days -- those days would
# have been independently re-decided by the agent under two different
# regime labels (agent.reset() runs between every episode in the loop
# below, so overlap isn't just mislabeling, it's the same calendar days
# being fed through the decision loop twice, once as "crisis" and once as
# "trending"), and every aggregate statistic across episodes would have
# silently double-counted those days' returns. Caught by a direct pairwise
# overlap check across all 18 final dates, not assumed safe just because
# generate_episode_windows() enforces non-overlap for windows it generates
# itself -- these two hardcoded lists are concatenated in
# _balanced_regime_episodes with no such automatic check between them.
#
# Hardcoded rather than computed at import time so every model in a run is
# guaranteed the exact same dates (comparison.py's CBS guard requires
# matched episode sets across compared systems) and so this doesn't
# silently drift if label_regimes' HMM refit (8 random EM restarts) ever
# produces a slightly different labeling on a later dataprep rebuild.
#
# Why 6-per-regime specifically: no established field standard for this
# exists (checked directly -- even CLQT, arXiv:2606.29771, explicitly defers
# regime-stratified analysis to future work). Grounded instead in giving
# each regime ~370 real trading-day observations, well past the point where
# the 21-day bootstrap-CI floor would dominate the estimate (per arXiv
# 2605.16895's finding that short-window Sharpe CIs routinely exceed the
# point estimate itself).
TRENDING_MEAN_REVERTING_EPISODES: list[tuple[str, str, str]] = [
    ("trending", "2010-01-01", "2010-04-01"),
    ("trending", "2013-02-18", "2013-05-19"),
    ("trending", "2013-09-14", "2013-12-13"),
    ("trending", "2015-05-31", "2015-08-29"),
    ("trending", "2020-07-15", "2020-10-13"),
    ("trending", "2021-05-23", "2021-08-21"),
    ("mean_reverting", "2011-02-21", "2011-05-22"),
    ("mean_reverting", "2014-04-10", "2014-07-09"),
    ("mean_reverting", "2016-07-20", "2016-10-18"),
    ("mean_reverting", "2017-09-09", "2017-12-08"),
    ("mean_reverting", "2019-05-26", "2019-08-24"),
    ("mean_reverting", "2023-09-02", "2023-12-01"),
]


def _vix_crisis_episodes() -> list[tuple[str, str, str]]:
    """Real crisis windows from dataprep.find_vix_crisis_windows() (CBOE VIX
    close > 30, sustained >= 5 trading days), not a hardcoded snapshot --
    re-running this always reflects whatever data/processed/vix currently
    holds. Requires `python -c "import dataprep; dataprep.build_vix()"` (or
    `python dataprep.py build-vix`) to have been run first."""
    windows = dataprep.find_vix_crisis_windows()
    return [("crisis", row.episode_start.date().isoformat(), row.episode_end.date().isoformat())
            for row in windows.itertuples()]


def _balanced_regime_episodes() -> list[tuple[str, str, str]]:
    """6 crisis (real VIX windows, live) + 6 trending + 6 mean_reverting
    (real HMM windows, hardcoded above) = 18 episodes, genuinely balanced
    across all three regime labels -- not the old 7/1/1 split produced by
    combining the VIX scan with the one-example-per-regime spot-check."""
    return _vix_crisis_episodes() + TRENDING_MEAN_REVERTING_EPISODES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=None,
                         help="default: dataprep.EVALUATION_TICKERS (matches a normal backtest run)")
    parser.add_argument("--model", default=main_mod.DEFAULT_MODEL)
    parser.add_argument("--system", default="llm",
                         help="output filename stem -- writes runs/<system>.json")
    parser.add_argument("--preset",
                         choices=["vix_crisis", "three_regimes", "balanced_regimes",
                                  "trending_mean_reverting_only"],
                         default="vix_crisis")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--estimate-only", action="store_true",
                         help="print the projected cost and exit without calling the API")
    parser.add_argument("--max-spend", type=float, default=None,
                         help="USD ceiling on the projected list cost. Required for a "
                              "paid run; the run refuses to start without it.")
    args = parser.parse_args()

    tickers = args.tickers or dataprep.EVALUATION_TICKERS
    episodes = {
        "vix_crisis": _vix_crisis_episodes,
        "three_regimes": lambda: THREE_REGIME_EPISODES,
        "balanced_regimes": _balanced_regime_episodes,
        # For a model that already has a clean --preset vix_crisis run on file:
        # run just this (12 episodes, not 18), then merge_regime_runs.py it with
        # the existing crisis file instead of re-paying for those 6 crisis
        # windows a second time.
        "trending_mean_reverting_only": lambda: TRENDING_MEAN_REVERTING_EPISODES,
    }[args.preset]()

    agent = main_mod._build_llm_agent(tickers, model=args.model, use_prompt_caching=not args.no_cache)
    config = ReplayConfig()

    # This script previously built the agent directly (main_mod._build_llm_agent)
    # with no CoordinationMeter wrapper at all -- every run produced this way
    # (which is every crisis/regimes/balanced_regimes run this whole session)
    # wrote a payload with no "coordination" key. main.py's cmd_report only
    # populates `meters` when that key is present, so _report_coordination_cost
    # silently printed nothing for any of these models -- the paper's
    # coordination-cost/CBS-with-billing metric had no real data behind it for
    # every model run through this script, discovered via a live code audit,
    # not by design. Mirrors cmd_backtest's own metering block in main.py
    # exactly (CoordinationCostConfig -> CoordinationMeter -> agent_fn = meter
    # -> payload["coordination"] = meter.to_dict()) so a report over a mix of
    # main.py backtest and run_regime_episodes.py runs prices them consistently.
    from coordination_cost import CoordinationCostConfig, CoordinationMeter, estimate_cost

    cost_cfg = CoordinationCostConfig(model=args.model, use_prompt_caching=not args.no_cache)
    meter = CoordinationMeter(agent, cost_cfg, name=args.system)
    meter._prefix_tokens = agent.prefix_tokens()

    # Lightweight preflight, not main.py's full _preflight (that one derives
    # n_decisions from generate_episode_windows(start,end,window,purge), which
    # doesn't apply here -- episodes is an explicit, already-fixed list). Same
    # ~252/365 trading-day approximation main.py uses for its own guard.
    n_decisions = sum(
        max(int((pd.Timestamp(end) - pd.Timestamp(start)).days * (252.0 / 365.0)), 1)
        for _, start, end in episodes
    )
    est = estimate_cost(cost_cfg, n_decisions, cost_cfg.assumed_tokens_per_llm_call_in,
                         prefix_tokens=meter._prefix_tokens)
    print(f"\npreflight cost estimate ({len(episodes)} episode(s), ~{n_decisions:,} decisions)")
    print(est.summary(cost_cfg))
    print("  input tokens per call are ASSUMED, not measured -- treat this as a "
          "budget guard, not a forecast.")
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

    runs = []
    for regime, start, end in episodes:
        ep_start = pd.Timestamp(start, tz="UTC")
        ep_end = pd.Timestamp(end, tz="UTC")
        if hasattr(meter, "reset"):
            meter.reset()
        store = dataprep.PointInTimeStore()
        run = phase3.run_episode(meter, store, tickers, ep_start, ep_end, regime, config)
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
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n{len(runs)} episode(s) -> {out}")
    print(meter.report())
    print(f"\nNow run: python main.py report --systems {args.system}")


if __name__ == "__main__":
    main()
