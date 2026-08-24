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
    python run_regime_episodes.py                       # VIX crisis windows, full ticker universe
    python run_regime_episodes.py --system llm_crisis    # write to a separate file to merge later
    python run_regime_episodes.py --preset three_regimes # the original trending/mean_reverting/crisis spot-check

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


def _vix_crisis_episodes() -> list[tuple[str, str, str]]:
    """Real crisis windows from dataprep.find_vix_crisis_windows() (CBOE VIX
    close > 30, sustained >= 5 trading days), not a hardcoded snapshot --
    re-running this always reflects whatever data/processed/vix currently
    holds. Requires `python -c "import dataprep; dataprep.build_vix()"` (or
    `python dataprep.py build-vix`) to have been run first."""
    windows = dataprep.find_vix_crisis_windows()
    return [("crisis", row.episode_start.date().isoformat(), row.episode_end.date().isoformat())
            for row in windows.itertuples()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=None,
                         help="default: dataprep.EVALUATION_TICKERS (matches a normal backtest run)")
    parser.add_argument("--model", default=main_mod.DEFAULT_MODEL)
    parser.add_argument("--system", default="llm",
                         help="output filename stem -- writes runs/<system>.json")
    parser.add_argument("--preset", choices=["vix_crisis", "three_regimes"], default="vix_crisis")
    args = parser.parse_args()

    tickers = args.tickers or dataprep.EVALUATION_TICKERS
    episodes = _vix_crisis_episodes() if args.preset == "vix_crisis" else THREE_REGIME_EPISODES

    agent = main_mod._build_llm_agent(tickers, model=args.model)
    config = ReplayConfig()

    runs = []
    for regime, start, end in episodes:
        ep_start = pd.Timestamp(start, tz="UTC")
        ep_end = pd.Timestamp(end, tz="UTC")
        if hasattr(agent, "reset"):
            agent.reset()
        store = dataprep.PointInTimeStore()
        run = phase3.run_episode(agent, store, tickers, ep_start, ep_end, regime, config)
        runs.append(run)
        print(f"{ep_start.date()}..{ep_end.date()} ({regime}): "
              f"{len(run.execution_log)} fills, {len(run.portfolio_values)} trading days", flush=True)

    main_mod.RUNS_DIR.mkdir(exist_ok=True)
    out = main_mod.RUNS_DIR / f"{args.system}.json"
    payload = {
        "system": args.system,
        "impact_gamma": config.impact_gamma,
        "episodes": [main_mod._episode_to_dict(r) for r in runs],
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n{len(runs)} episode(s) -> {out}")
    print(f"Now run: python main.py report --systems {args.system}")


if __name__ == "__main__":
    main()
