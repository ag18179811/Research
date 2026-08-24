"""
Runs one agent through a hand-picked set of specific, pre-labeled episode
windows -- one (or more) per regime -- instead of the full evaluation-range
sweep main.py backtest generates automatically.

Exists because main.py's CLI only takes a single contiguous --start/--end
range: generate_episode_windows() auto-generates every non-overlapping
window inside it, so there's no way to ask for three specific, disjoint,
regime-diverse windows through the CLI alone. This script builds the same
agent main.py backtest --agent llm would (via main._build_llm_agent, so
CLAUDE_API_KEY wiring and model/cache config are identical), drives it
through EPISODES below one at a time via phase3.run_episode, and writes a
normal runs/<system>.json main.py report can load unmodified.

Usage:
    python run_regime_episodes.py [--tickers AAPL MSFT] [--system llm]

Then:
    python main.py report --systems llm

Edit EPISODES below to pick different windows. To find your own
regime-labeled candidates:
    python3 -c "
    import dataprep, pandas as pd
    windows = dataprep.generate_episode_windows(START, END, WINDOW_LEN, PURGE_GAP)
    prices = dataprep.load_prices(['AAPL']).sort_values('date')
    returns = prices.set_index('date')['adj_close'].pct_change()
    regimes = dataprep.label_regimes(returns)
    print(dataprep.assign_episode_regimes(windows, regimes))
    "
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import dataprep
import phase3
import main as main_mod
from execution_replay import ReplayConfig

# One window per regime, verified via dataprep.assign_episode_regimes against
# the walk-forward HMM labels (see module docstring for how to find more).
EPISODES: list[tuple[str, str, str]] = [
    ("trending", "2013-03-16", "2013-04-25"),        # 2013 bull run
    ("mean_reverting", "2011-03-27", "2011-05-06"),  # choppy pre-debt-ceiling-crisis 2011
    ("crisis", "2020-02-08", "2020-03-19"),           # COVID crash
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT"])
    parser.add_argument("--model", default=main_mod.DEFAULT_MODEL)
    parser.add_argument("--system", default="llm",
                         help="output filename stem -- writes runs/<system>.json")
    args = parser.parse_args()

    agent = main_mod._build_llm_agent(args.tickers, model=args.model)
    config = ReplayConfig()

    runs = []
    for regime, start, end in EPISODES:
        ep_start = pd.Timestamp(start, tz="UTC")
        ep_end = pd.Timestamp(end, tz="UTC")
        if hasattr(agent, "reset"):
            agent.reset()
        store = dataprep.PointInTimeStore()
        run = phase3.run_episode(agent, store, args.tickers, ep_start, ep_end, regime, config)
        runs.append(run)
        print(f"{ep_start.date()}..{ep_end.date()} ({regime}): "
              f"{len(run.execution_log)} fills, {len(run.portfolio_values)} trading days")

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
