"""
Single entrypoint for the research pipeline.

This is a dispatcher, not a new layer: every subcommand delegates to code
that already exists in this repo. Nothing here computes a metric, fills an
order, or reads a raw dataset on its own.

    Phase 1    dataprep.py          point-in-time data + regime labels
    Phase 1.5  phase1_5_agent.py    LLM decision loop (opt-in, see `backtest`)
    Phase 2    execution_replay.py  fills, fees, market impact
               execution_harness.py orders -> engine, one day at a time
    Phase 3    phase3.py            episode runner, agents, joint metric report
               comparison.py        leaderboard + coordination breakeven spread

Typical run, from nothing to a leaderboard:

    python main.py data build --limit 5     # build a small slice of Phase 1
    python main.py validate                 # check the built data
    python main.py all                      # baselines + report

`backtest` and `report` are deliberately separate commands: a full backtest
is expensive, and the block bootstrap in `report` (n_boot=1000 by default)
gets re-run far more often than the episodes themselves change.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import dataprep

log = logging.getLogger("main")

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs"

# The passive-benchmark control group from phase3.py. `llm` is handled
# separately (see _build_agent) because it is not currently runnable.
BASELINE_AGENTS = ("buy_and_hold",)
AGENT_CHOICES = BASELINE_AGENTS + ("llm",)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _build_agent(name: str, tickers: list[str], seed: int = 0):
    """Returns a phase3.AgentFn for `name`. Baselines come straight from
    phase3.py; the Phase 1.5 LLM agent is not wired up (see below)."""
    import phase3  # local import: pulls in numpy/pandas, not needed for `data`

    if name == "buy_and_hold":
        return phase3.buy_and_hold_agent_factory(tickers)

    if name == "llm":
        # Two independent blockers, reported together so you find out about
        # both at once rather than fixing one and hitting the other.
        problems = []
        try:
            from google import genai  # noqa: F401
        except ImportError:
            problems.append(
                "  - google-genai is not installed (pip install google-genai), "
                "so phase1_5_agent.py cannot even be imported."
            )
        problems.append(
            "  - phase1_5_agent.Phase15Bridge does not implement runner.AgentFn. "
            "It exposes run_day(...)/run_episode(...) and drives its own engine, "
            "whereas run_all_episodes() expects "
            "agent_fn(decision_snapshots, positions, cash, as_of) -> list[Order]. "
            "An adapter has to be written before this path can run."
        )
        raise SystemExit(
            "--agent llm is not runnable yet:\n"
            + "\n".join(problems)
            + "\n\nNote also that a code review flagged 3 HIGH issues in "
            "phase1_5_agent.py, including a daily mark-to-market series that is "
            "off by one trading day -- which feeds Sharpe/Sortino/max-drawdown. "
            "Fix those before trusting any number this agent produces.\n"
            "The baselines (%s) are unaffected and runnable now."
            % ", ".join(BASELINE_AGENTS)
        )

    raise ValueError(f"unknown agent {name!r}")


# ---------------------------------------------------------------------------
# Factor returns
# ---------------------------------------------------------------------------


def load_factor_returns(path: str | None) -> pd.DataFrame:
    """Returns the date-indexed ["mkt","smb","hml"] frame that
    phase3.factor_attribution() regresses against. Three-tier priority:

      --factors FILE   an explicit override -- any 3-factor daily CSV
                       (Ken French format or equivalent), highest priority
                       so you can substitute a vetted/custom series.

      (omitted, and    the real thing: Ken French's own daily 3-factor
       data/processed/  series, downloaded and parsed by
       factors/ exists) dataprep.build_factors() (`python main.py data
                       build` runs this) -- see dataprep.load_factors.

      (omitted, and    a SPY-only proxy: mkt = SPY daily returns,
       factors were     smb = hml = 0. beta_market stays meaningful; beta_size
       never built)     and beta_value are regressed against a column of
                       zeros and are therefore NOT identified -- an
                       artifact, not a measurement. Alpha absorbs all
                       size/value exposure. Warns loudly precisely so it
                       never gets quietly reported as a genuine three-factor
                       attribution -- this is the last-resort path, not the
                       default."""
    if path:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in ("date", "dates", "unnamed: 0") if c in df.columns), None)
        if date_col is None:
            raise ValueError(f"{path}: no date column found (looked for date/dates)")
        df["date"] = pd.to_datetime(df[date_col], utc=True)

        # Ken French ships the market factor as "Mkt-RF"; accept both spellings.
        if "mkt" not in df.columns and "mkt-rf" in df.columns:
            df = df.rename(columns={"mkt-rf": "mkt"})

        missing = [c for c in ("mkt", "smb", "hml") if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}")

        out = df.set_index("date")[["mkt", "smb", "hml"]].astype(float).sort_index()

        # Ken French publishes these in percent (0.53 == 0.53%), while the
        # portfolio returns they get regressed against are decimals. Detect
        # the scale rather than silently mixing units by a factor of 100.
        if out["mkt"].abs().mean() > 0.05:
            log.warning(
                "%s looks percent-scaled (mean |mkt| = %.3f); dividing by 100 to "
                "match decimal portfolio returns", path, out["mkt"].abs().mean(),
            )
            out = out / 100.0
        return out

    # --- real Ken French factors, built locally by `data build` -----------
    try:
        return dataprep.load_factors()
    except FileNotFoundError:
        pass

    # --- proxy path (last resort) ------------------------------------------
    spy = dataprep.load_prices(entities=["SPY"])
    if spy.empty:
        raise SystemExit(
            "no SPY price data, so even the proxy market factor cannot be built. "
            "Run `python main.py data build` first, or pass --factors FILE."
        )
    spy = spy.sort_values("date")
    mkt = spy.set_index("date")["adj_close"].pct_change().dropna()

    log.warning(
        "\n" + "=" * 72 + "\n"
        "USING PROXY FACTORS -- mkt = SPY returns, smb = hml = 0.\n"
        "beta_size and beta_value are regressed against zero columns and are\n"
        "NOT identified. Do not report them as a Fama-French 3-factor\n"
        "attribution. Alpha here absorbs all size/value exposure.\n"
        "Pass --factors FILE with real daily factors to fix this.\n"
        + "=" * 72
    )

    return pd.DataFrame({"mkt": mkt, "smb": 0.0, "hml": 0.0})


# ---------------------------------------------------------------------------
# Run persistence
# ---------------------------------------------------------------------------
# JSON rather than pickle, deliberately. metrics.py was built for it --
# _fills_to_frame() says it accepts "the Phase 2 ExecutionFill dataclass or a
# plain dict (e.g. reloaded from execution_log.json)", and
# temporal_integrity_report() only ever .get()s str/int keys off access_log
# entries. So the round trip below loses nothing metrics needs, while keeping
# run files inspectable and avoiding pickle's arbitrary-code-execution risk on
# load (these files get shared between machines and checked into analyses).


def _episode_to_dict(ep) -> dict:
    fills = []
    for f in ep.execution_log:
        rec = dataclasses.asdict(f) if dataclasses.is_dataclass(f) else dict(f)
        # side/regime are enums; store their .value so JSON stays readable
        # and reloads into exactly what _fills_to_frame() normalizes to.
        for k in ("side", "regime"):
            if hasattr(rec.get(k), "value"):
                rec[k] = rec[k].value
        fills.append(rec)

    pv = ep.portfolio_values
    return {
        "episode_start": ep.episode_start.isoformat(),
        "episode_end": ep.episode_end.isoformat(),
        "regime": ep.regime,
        "execution_log": fills,
        "portfolio_values": {str(k): float(v) for k, v in pv.items()},
        "access_log": [
            {k: (str(v) if k == "as_of" else v) for k, v in entry.items()}
            for entry in ep.access_log
        ],
    }


def _episode_from_dict(d):
    from phase3 import EpisodeRun

    pv = pd.Series(
        {pd.Timestamp(k): v for k, v in d["portfolio_values"].items()},
        dtype="float64",
    ).sort_index()
    return EpisodeRun(
        episode_start=pd.Timestamp(d["episode_start"]),
        episode_end=pd.Timestamp(d["episode_end"]),
        regime=d["regime"],
        execution_log=d["execution_log"],  # plain dicts, supported by metrics
        portfolio_values=pv,
        access_log=d["access_log"],
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_data(args) -> int:
    """Phase 1 build. Delegates to dataprep's own build_* functions -- the
    same work `python dataprep.py build-all` does, minus news by default."""
    dataprep.build_prices(limit_tickers=args.limit)
    dataprep.build_fundamentals()
    dataprep.build_entity_map()

    try:
        dataprep.build_factors()
    except Exception:
        # Unlike prices/fundamentals/entity-map, factors has a built-in
        # fallback (load_factor_returns' SPY-only proxy) -- a flaky fetch
        # from Dartmouth's server shouldn't abort the rest of the Phase 1
        # build over a dataset only the `report` command's Attribution
        # phase needs.
        log.exception("failed to fetch Ken French factors; `report` will fall back to the SPY proxy")

    if args.with_news:
        # RAW_NEWS_CSV (~/nasdaq_exteral_data.csv) is not present in every
        # checkout, so this is opt-in rather than part of the default build.
        if not dataprep.RAW_NEWS_CSV.exists():
            log.error("--with-news given but %s does not exist; skipping news",
                      dataprep.RAW_NEWS_CSV)
        else:
            dataprep.build_news()
            dataprep.build_news_entity_backfill()
    else:
        log.info("skipping news (pass --with-news to build it)")

    print(f"\nPhase 1 build complete -> {dataprep.PROCESSED_DIR}")
    return 0


def cmd_validate(args) -> int:
    import validate
    return validate.main()


def cmd_demo(args) -> int:
    import demo
    demo.main()
    return 0


def cmd_backtest(args) -> int:
    import phase3
    from execution_replay import ReplayConfig

    tickers = args.tickers or dataprep.EVALUATION_TICKERS
    agent_fn = _build_agent(args.agent, tickers, seed=args.seed)
    config = ReplayConfig(impact_gamma=args.impact_gamma)

    print(f"backtest: agent={args.agent} tickers={len(tickers)} "
          f"{args.start} -> {args.end}")

    runs = phase3.run_all_episodes(
        agent_fn,
        tickers,
        start=args.start,
        end=args.end,
        window_len=pd.Timedelta(days=args.window_days),
        purge_gap=pd.Timedelta(days=args.purge_days),
        replay_config=config,
    )

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{args.agent}.json"
    payload = {
        "system": args.agent,
        "impact_gamma": args.impact_gamma,
        "episodes": [_episode_to_dict(r) for r in runs],
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    fills = sum(len(r.execution_log) for r in runs)
    print(f"  {len(runs)} episode(s), {fills} fill(s) -> {out}")
    return 0


def cmd_report(args) -> int:
    import phase3
    import comparison

    systems = args.systems or [p.stem for p in sorted(RUNS_DIR.glob("*.json"))]
    if not systems:
        raise SystemExit(
            f"no runs found in {RUNS_DIR}. Run `python main.py backtest ...` first."
        )

    episodes_by_system: dict[str, list] = {}
    impact_enabled: dict[str, bool] = {}
    for name in systems:
        path = RUNS_DIR / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"no run file for {name!r} ({path})")
        with open(path) as f:
            payload = json.load(f)
        episodes_by_system[name] = [_episode_from_dict(d) for d in payload["episodes"]]
        impact_enabled[name] = payload.get("impact_gamma", 0.1) > 0

    factor_returns = load_factor_returns(args.factors)

    reports = {}
    for name, eps in episodes_by_system.items():
        if not eps:
            log.warning("%s has no episodes; skipping", name)
            continue
        reports[name] = phase3.build_joint_metric_report(
            name, eps, factor_returns, n_boot=args.n_boot, seed=args.seed,
        )
        print()
        print(reports[name].summary())

    baseline = args.baseline
    if baseline not in reports:
        # CBS is measured against a passive benchmark; fall back to whatever
        # is actually present rather than failing the whole report.
        fallback = "buy_and_hold" if "buy_and_hold" in reports else next(iter(reports), None)
        if fallback is None:
            log.warning("no baseline available; skipping leaderboard")
            return 0
        log.warning("baseline %r not among runs; using %r", baseline, fallback)
        baseline = fallback

    leaderboard = comparison.build_leaderboard(
        reports, episodes_by_system, baseline_name=baseline,
        market_impact_enabled=impact_enabled,
    )
    print()
    print(comparison.leaderboard_summary(leaderboard))
    return 0


def cmd_all(args) -> int:
    for agent in BASELINE_AGENTS:
        args.agent = agent
        rc = cmd_backtest(args)
        if rc != 0:
            return rc
    return cmd_report(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_backtest_args(p) -> None:
    p.add_argument("--tickers", nargs="*", default=None,
                   help="default: dataprep.EVALUATION_TICKERS (20 names)")
    p.add_argument("--start", default=str(dataprep.EPISODE_RANGE_START.date()))
    p.add_argument("--end", default=str(dataprep.EPISODE_RANGE_END.date()))
    p.add_argument("--window-days", type=int, default=dataprep.EPISODE_WINDOW_LEN.days)
    p.add_argument("--purge-days", type=int, default=dataprep.EPISODE_PURGE_GAP.days)
    p.add_argument("--impact-gamma", type=float, default=0.1,
                   help="Almgren-Chriss impact scale; 0 disables market impact")
    p.add_argument("--seed", type=int, default=0)


def _add_report_args(p) -> None:
    p.add_argument("--systems", nargs="*", default=None,
                   help="default: every *.pkl in runs/")
    p.add_argument("--factors", default=None,
                   help="CSV of daily mkt/smb/hml; omitted = real Ken French factors "
                        "if `data build` fetched them, else SPY proxy (warns)")
    p.add_argument("--baseline", default="buy_and_hold",
                   help="system that CBS is measured against")
    p.add_argument("--n-boot", type=int, default=1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="Phase 1: build point-in-time datasets")
    p.add_argument("action", choices=["build"])
    p.add_argument("--limit", type=int, default=None,
                   help="only process first N tickers (use a small N to smoke-test)")
    p.add_argument("--with-news", action="store_true",
                   help="also build news (needs ~/nasdaq_exteral_data.csv)")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("validate", help="run validate.py's data checks")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("demo", help="narrated point-in-time walkthrough")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("backtest", help="drive one agent through all episodes")
    p.add_argument("--agent", choices=AGENT_CHOICES, required=True)
    _add_backtest_args(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="joint metric report + leaderboard")
    _add_report_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("all", help="backtest every baseline, then report")
    _add_backtest_args(p)
    _add_report_args(p)
    p.set_defaults(func=cmd_all)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
