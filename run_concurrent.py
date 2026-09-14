"""Runs several LLM backbones through the pipeline at the same time, each as
its own independent OS process.

Why a separate process per model, not a thread pool inside one process: each
model drives its own Phase15Bridge -> ExecutionReplayEngine -> episode loop,
which is sequential-per-model (day N's prompt embeds day N-1's trade history,
so within one model's run nothing can parallelize -- see main.py's --batch
refusal for the same reason). Across models there is no such dependency, and
a real subprocess means one model's slow API (Grok-4.6 measured at ~20s per
decision, driven by hidden reasoning tokens -- see coordination cost output)
or an outright crash cannot block or corrupt another model's run. Each run
writes to its own runs/<model>.json (via main.py backtest's --system flag),
so the outputs never collide.

This is infrastructure, not a new metric or agent: every subprocess is
exactly the `python main.py backtest ...` invocation you would type by hand,
just N of them launched together and watched to completion.

Usage:
    python run_concurrent.py --models claude-haiku-4-5 gpt-5.6-sol grok-4.6 deepseek-v4-pro \\
        --tickers AAPL MSFT GOOG --start 2022-01-03 --end 2022-03-01 \\
        --window-days 21 --purge-days 5 --max-spend 5.0

Then, once everything finishes:
    python main.py report --systems claude-haiku-4-5 gpt-5.6-sol grok-4.6 deepseek-v4-pro

(or pass --report to this script to chain into that automatically).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import dataprep

PROJECT_DIR = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_DIR / "runs" / "logs"

DEFAULT_MODELS = ["claude-haiku-4-5", "gpt-5.6-sol", "grok-4.6", "deepseek-v4-pro"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                         help=f"MODEL_PRICES keys to run concurrently (default: {DEFAULT_MODELS})")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--window-days", type=int, required=True)
    parser.add_argument("--purge-days", type=int, default=dataprep.EPISODE_PURGE_GAP.days,
                         help="days between consecutive episode windows, to prevent state/memory "
                             "leaking across the boundary. Previously defaulted to 0 here, silently "
                             "diverging from `main.py backtest`'s own default -- an omitted flag "
                             "gave a materially weaker leakage buffer than the equivalent plain "
                             "backtest invocation would, found by a code audit, not by a user "
                             "noticing thinner purge gaps in their results.")
    parser.add_argument("--max-spend", type=float, required=True,
                         help="USD ceiling applied to EACH model independently, not pooled "
                              "-- N models each get up to this much, so real spend can reach "
                              "N x this figure.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--impact-gamma", type=float, default=0.1)
    parser.add_argument("--report", action="store_true",
                         help="after every subprocess exits, chain into "
                              "`main.py report --systems <models>` automatically")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, Path] = {}
    t0 = time.time()

    for model in args.models:
        log_path = LOGS_DIR / f"{model}.log"
        logs[model] = log_path
        cmd = [
            sys.executable, "main.py", "backtest",
            "--agent", "llm",
            "--model", model,
            "--system", model,          # runs/<model>.json, never collides across models
            "--start", args.start,
            "--end", args.end,
            "--window-days", str(args.window_days),
            "--purge-days", str(args.purge_days),
            "--impact-gamma", str(args.impact_gamma),
            "--max-spend", str(args.max_spend),
        ]
        if args.tickers:
            cmd += ["--tickers", *args.tickers]
        if args.no_cache:
            cmd.append("--no-cache")

        print(f"launching {model} -> {log_path}")
        with open(log_path, "w") as logf:
            procs[model] = subprocess.Popen(
                cmd, cwd=PROJECT_DIR, stdout=logf, stderr=subprocess.STDOUT,
            )

    print(f"\n{len(procs)} model(s) running concurrently, PIDs: "
          + ", ".join(f"{m}={p.pid}" for m, p in procs.items()))
    print("tail -f runs/logs/<model>.log to watch one live.\n")

    results: dict[str, int] = {}
    remaining = dict(procs)
    while remaining:
        for model, proc in list(remaining.items()):
            rc = proc.poll()
            if rc is not None:
                elapsed = time.time() - t0
                status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                print(f"[{elapsed:7.1f}s] {model}: {status} -- see {logs[model]}")
                results[model] = rc
                del remaining[model]
        if remaining:
            time.sleep(5)

    print(f"\nall {len(results)} model(s) finished in {time.time() - t0:.1f}s")
    ok = [m for m, rc in results.items() if rc == 0]
    failed = [m for m, rc in results.items() if rc != 0]
    print(f"  succeeded: {ok}")
    if failed:
        print(f"  FAILED   : {failed}  (check runs/logs/<model>.log for each)")

    if args.report and ok:
        print(f"\nrunning: python main.py report --systems {' '.join(ok)}\n")
        subprocess.run([sys.executable, "main.py", "report", "--systems", *ok], cwd=PROJECT_DIR)
    elif ok:
        print(f"\nnext: python main.py report --systems {' '.join(ok)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
