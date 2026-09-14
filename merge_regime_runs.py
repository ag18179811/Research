"""Combines one model's separate crisis-preset and three_regimes-preset run
files into a single file with all 9 episodes, so main.py report sees one
model with genuine trending/mean_reverting/crisis diversity -- rather than
two separate "systems" (crisis-only and regimes-only) that would fragment
one model's results across two leaderboard rows and give Regime Robustness
nothing to differentiate within either half on its own.

Usage:
    python merge_regime_runs.py --models claude-opus-5 gpt-5.6-sol deepseek-v4-pro kimi-k2.6

For each model, reads runs/<model>-crisis.json and runs/<model>-regimes.json
(both must already exist -- run_regime_episodes.py --preset vix_crisis and
--preset three_regimes respectively), concatenates their "episodes" lists,
and writes runs/<model>-final.json. Then:

    python main.py report --systems claude-opus-5-final gpt-5.6-sol-final deepseek-v4-pro-final kimi-k2.6-final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--suffix", default="final",
                         help="output filename stem suffix -- writes runs/<model>-<suffix>.json")
    args = parser.parse_args()

    for model in args.models:
        crisis_path = RUNS_DIR / f"{model}-crisis.json"
        regimes_path = RUNS_DIR / f"{model}-regimes.json"
        missing = [p for p in (crisis_path, regimes_path) if not p.exists()]
        if missing:
            print(f"SKIP {model}: missing {[p.name for p in missing]}")
            continue

        with open(crisis_path) as f:
            crisis = json.load(f)
        with open(regimes_path) as f:
            regimes = json.load(f)

        out_name = f"{model}-{args.suffix}"
        combined = dict(crisis)
        combined["system"] = out_name
        combined["episodes"] = crisis["episodes"] + regimes["episodes"]

        out_path = RUNS_DIR / f"{out_name}.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)

        n_crisis, n_regimes = len(crisis["episodes"]), len(regimes["episodes"])
        print(f"{model}: {n_crisis} crisis + {n_regimes} regime-diversity "
              f"= {n_crisis + n_regimes} episode(s) -> {out_path}")


if __name__ == "__main__":
    main()
