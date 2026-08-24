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
import os
import sys
from pathlib import Path

import pandas as pd

import dataprep

log = logging.getLogger("main")

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs"

# No standing passive-baseline agent exists any more (buy_and_hold was
# removed) -- `llm` is currently the only runnable agent. Kept as a tuple,
# not a bare string constant, so cmd_all's loop and any future non-LLM
# baseline slot in cleanly.
BASELINE_AGENTS = ()
AGENT_CHOICES = BASELINE_AGENTS + ("llm",)

# Cheapest rung of the current ladder, and the one this pipeline was already
# pointed at. Undated on purpose: this string doubles as the lookup key into
# coordination_cost.MODEL_PRICES, which is keyed on undated IDs. Cost is a
# model-selection question -- `report` prints the whole curve, not just this row.
DEFAULT_MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _build_agent(name: str, tickers: list[str], seed: int = 0,
                 model: str = DEFAULT_MODEL, use_prompt_caching: bool = True):
    """Returns a phase3.AgentFn for `name`. `llm` (the only agent currently
    wired up) is the Anthropic-backed Phase 1.5 agent -- see _LLMAgent."""
    if name == "llm":
        return _build_llm_agent(tickers, model=model,
                                use_prompt_caching=use_prompt_caching)

    raise ValueError(f"unknown agent {name!r}")


class _LLMAgent:
    """Adapts phase1_5_agent.Phase15Bridge to phase3.AgentFn.

    This is the adapter whose absence used to make `--agent llm` unrunnable. It
    reuses the bridge's prompt construction, LLM call, decision validation, and
    order conversion, and supplies the one thing phase3.run_episode needs and
    the bridge did not: the (decision_snapshots, positions, cash, as_of) ->
    list[Order] shape.

    TWO DELIBERATE DIFFERENCES FROM Phase15Bridge.run_episode:

    1. It does NOT drive the bridge's own ExecutionReplayEngine and does NOT call
       _decision_to_market_snapshot. phase3.run_episode owns execution and
       portfolio marking. That is what sidesteps the off-by-one daily
       mark-to-market series flagged in the bridge's own episode loop -- the one
       that feeds Sharpe/Sortino/max-drawdown. The bug is not fixed here; this
       path simply never runs the code containing it. Anything calling
       Phase15Bridge.run_episode directly is still affected.

    2. The prompt is built from the `decision_snapshots` phase3.run_episode hands
       over, not from a second read of the bridge's own store.
       universe_snapshot_as_of returns {as_of, cash, positions, tickers}, and
       `decision_snapshots` is exactly that `tickers` mapping, so the bundle is
       reassembled rather than re-fetched. Re-reading would duplicate every
       access in the temporal integrity log and scope it to the wrong store --
       run_all_episodes builds a fresh PointInTimeStore per episode precisely
       to keep that log per-episode.

    Portfolio state is copied from the runner every day rather than tracked
    independently: the engine is authoritative on cash and positions, and a
    second running total would silently diverge from it.
    """

    def __init__(self, tickers: list[str], model: str, api_key: str,
                 use_prompt_caching: bool = True):
        import dataprep as _dataprep
        import phase1_5_agent

        self.tickers = tickers
        self.model = model
        # The store is only needed for the bridge's regime series (built from SPY
        # and never shown to the LLM); snapshots come from phase3.run_episode.
        self.bridge = phase1_5_agent.Phase15Bridge(
            store=_dataprep.PointInTimeStore(),
            claude_api_key=api_key,
            tickers=tickers,
            model_name=model,
            use_prompt_caching=use_prompt_caching,
        )
        self._PortfolioState = phase1_5_agent.PortfolioState
        self._TradeRecord = phase1_5_agent.TradeRecord
        self.portfolio = self._PortfolioState()
        self.history: list = []
        self._order_seq = 0
        self.n_agents = 1          # single seat; CoordinationMeter reads this

    # -- CoordinationMeter hooks --------------------------------------------
    def last_usage(self) -> dict | None:
        return self.bridge.last_usage()

    def prefix_tokens(self) -> int:
        return self.bridge.measure_prefix_tokens()

    def reset(self) -> None:
        """Called by phase3.run_all_episodes between episodes. Trade history
        must not leak across an episode boundary -- it goes into the prompt,
        and carrying it over would show the agent decisions from a window it
        is not supposed to have seen."""
        self.portfolio = self._PortfolioState()
        self.history = []
        self._order_seq = 0

    # -- phase3.AgentFn -------------------------------------------------------
    def __call__(self, decision_snapshots, positions, cash, as_of):
        self.portfolio.cash = float(cash)
        self.portfolio.positions = dict(positions)

        snap = {
            "as_of": as_of,
            "cash": float(cash),
            "positions": dict(positions),
            "tickers": decision_snapshots,
        }

        prompt = self.bridge._format_prompt(snap, self.portfolio, self.history, as_of)
        raw = self.bridge._call_llm(prompt)
        decision = self.bridge._validate_decision(raw, self.portfolio, snap)

        self.history.append(self._TradeRecord(
            date=as_of.date().isoformat(),
            action=decision["action"],
            symbol=decision.get("symbol"),
            quantity=decision.get("quantity"),
            order_type=decision.get("order_type"),
            limit_price=decision.get("limit_price"),
            reasoning=decision.get("reasoning", ""),
        ))

        order = self.bridge._decision_to_order(decision, as_of, self._order_seq)
        if order is None:
            return []
        self._order_seq += 1
        return [order]


def _build_llm_agent(tickers: list[str], model: str,
                     use_prompt_caching: bool = True) -> _LLMAgent:
    """Constructs the Anthropic-backed agent, failing early and specifically."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise SystemExit(
            "--agent llm needs the Anthropic SDK: pip install anthropic\n"
            "(it is in requirements.txt)"
        )

    # Explicit, not left to the SDK's own ANTHROPIC_API_KEY auto-detection --
    # this repo's .env has always used CLAUDE_API_KEY (phase1_5_agent.py's
    # own standalone `main()` reads the same variable), and letting the SDK
    # silently look for a differently-named var meant this path failed deep
    # inside anthropic._base_client's header validation, mid-episode, with
    # an error that never mentions CLAUDE_API_KEY at all.
    #
    # load_dotenv() here, not just relying on phase1_5_agent's own
    # module-level call, because that module hasn't been imported yet at
    # this point (only _LLMAgent.__init__ imports it) -- without this,
    # os.environ.get below runs before .env has ever been read into the
    # environment and always misses even when the key is genuinely set.
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise SystemExit("--agent llm needs CLAUDE_API_KEY set in your environment (.env is fine)")

    from coordination_cost import MODEL_PRICES
    if model not in MODEL_PRICES:
        raise SystemExit(
            f"{model!r} has no entry in coordination_cost.MODEL_PRICES, so the run "
            f"could not be priced. Add it with its source and verification date "
            f"rather than guessing.\nKnown: {', '.join(sorted(MODEL_PRICES))}"
        )

    return _LLMAgent(tickers, model=model, api_key=api_key, use_prompt_caching=use_prompt_caching)


# ---------------------------------------------------------------------------
# Factor returns
# ---------------------------------------------------------------------------


def load_factor_returns(path: str | None) -> pd.DataFrame:
    """Returns the date-indexed factor-returns frame that
    phase3.factor_attribution() regresses against -- at minimum a "mkt"
    column, plus whichever of dataprep.FACTOR_COLUMNS (smb/hml/rmw/cma/mom)
    are available. Three-tier priority:

      --factors FILE   an explicit override -- any daily factor CSV (Ken
                       French format or equivalent) with at least a mkt
                       column, highest priority so you can substitute a
                       vetted/custom series. Fewer columns than the full
                       6-factor build just means fewer betas get identified
                       -- factor_attribution adapts to whatever is present.

      (omitted, and    the real thing: Ken French's own daily
       data/processed/  Fama-French-Carhart 6-factor series (mkt/smb/hml/
       factors/ exists) rmw/cma/mom), downloaded and parsed by
                       dataprep.build_factors() (`python main.py data
                       build` runs this) -- see dataprep.load_factors.

      (omitted, and    a SPY-only proxy: mkt = SPY daily returns, nothing
       factors were     else. Only beta_mkt is identified; alpha absorbs
       never built)     all size/value/profitability/investment/momentum
                       exposure. Warns loudly precisely so it never gets
                       quietly reported as a full factor-model attribution
                       -- this is the last-resort path, not the default."""
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

        if "mkt" not in df.columns:
            raise ValueError(f"{path}: missing required column 'mkt' (or 'Mkt-RF')")

        present = [c for c in dataprep.FACTOR_COLUMNS if c in df.columns]
        out = df.set_index("date")[present].astype(float).sort_index()

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
        "USING PROXY FACTORS -- mkt = SPY returns only, no other factors.\n"
        "Only beta_mkt is identified; alpha absorbs all size/value/\n"
        "profitability/investment/momentum exposure. Do not report this as\n"
        "a full factor-model attribution.\n"
        "Pass --factors FILE, or run `python main.py data build`, to fix this.\n"
        + "=" * 72
    )

    return pd.DataFrame({"mkt": mkt})


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


# NYSE trades ~252 days a year against a 365-day calendar. Used only to turn an
# episode's calendar length into a decision count for the estimate.
_TRADING_DAY_FRACTION = 252.0 / 365.0


def _preflight(agent, args, cfg) -> int:
    """Project the bill and refuse to start if it exceeds --max-spend.

    A full sweep is ~3,000 calls, so an accidental `backtest --agent llm` on the
    default date range is a real amount of money. Returns 0 to continue, non-zero
    to stop. Costs one count_tokens call, which is not billed as inference.
    """
    from coordination_cost import estimate_cost

    # Exact episode count from the same generator run_all_episodes uses, rather
    # than a second derivation that could drift from it.
    windows = dataprep.generate_episode_windows(
        args.start, args.end,
        pd.Timedelta(days=args.window_days),
        pd.Timedelta(days=args.purge_days),
    )
    n_episodes = len(windows)
    if not n_episodes:
        raise SystemExit(f"no episode windows between {args.start} and {args.end}")
    # One decision per trading day per episode; the trading-day count inside a
    # calendar window is approximated, which is why this is a guard not a quote.
    per_episode = max(int(args.window_days * _TRADING_DAY_FRACTION), 1)
    n_decisions = n_episodes * per_episode

    prefix = agent.prefix_tokens()
    # Input per call is dominated by the per-ticker feature block. Measured
    # length would be better, but it needs a built store and a real snapshot;
    # the configured assumption is the honest stand-in and is labelled as such.
    per_call_in = cfg.assumed_tokens_per_llm_call_in
    est = estimate_cost(cfg, n_decisions, per_call_in, prefix_tokens=prefix)

    print("\npreflight cost estimate")
    print(f"  ({n_episodes} episode(s) x ~{per_episode} trading days "
          f"= ~{n_decisions:,} decisions)")
    print(est.summary(cfg))
    print("  input tokens per call are ASSUMED, not measured -- treat this as a "
          "budget guard, not a forecast.")

    if args.estimate_only:
        print("\n--estimate-only: stopping before any API call.")
        return 1

    if args.max_spend is None:
        raise SystemExit(
            "\nrefusing to start an unbounded paid run.\n"
            "Pass --max-spend USD to authorise it (the projected list price "
            f"above is ${est.list_usd:,.2f}), or --estimate-only to just price it."
        )
    if est.list_usd > args.max_spend:
        raise SystemExit(
            f"\nprojected list cost ${est.list_usd:,.2f} exceeds --max-spend "
            f"${args.max_spend:,.2f}. Raise the cap, pick a cheaper model, or "
            f"shorten the date range."
        )
    print(f"  authorised: ${est.list_usd:,.2f} <= --max-spend ${args.max_spend:,.2f}")
    return 0


def cmd_backtest(args) -> int:
    import phase3
    from execution_replay import ReplayConfig
    from coordination_cost import CoordinationCostConfig, CoordinationMeter

    tickers = args.tickers or dataprep.EVALUATION_TICKERS
    model = getattr(args, "model", DEFAULT_MODEL)
    use_cache = not getattr(args, "no_cache", False)

    # Argument validation before anything expensive is constructed, so this fires
    # without the SDK installed and without touching the network.
    if getattr(args, "batch", False):
        raise SystemExit(
            "--batch cannot be used with this agent.\n"
            "The Batch API's 50% discount is real, but each day's prompt embeds "
            "the trade history through the previous day, so decisions within an "
            "episode are strictly sequential and cannot be submitted as one "
            "batch. Applying the discount anyway would under-report the bill by "
            "half.\n"
            "A stateless decision variant (no history in the prompt) would be "
            "batchable; that is a separate change."
        )

    agent_fn = _build_agent(args.agent, tickers, seed=args.seed,
                            model=model, use_prompt_caching=use_cache)
    config = ReplayConfig(impact_gamma=args.impact_gamma)

    # Only the LLM path costs anything, so only it gets metered. Wrapping a
    # baseline would produce a meter full of $0.00 records and invite someone to
    # read that as a measured result.
    meter = None
    if args.agent == "llm":
        cost_cfg = CoordinationCostConfig(model=model, use_prompt_caching=use_cache)
        rc = _preflight(agent_fn, args, cost_cfg)
        if rc:
            return 0
        meter = CoordinationMeter(agent_fn, cost_cfg, name=args.agent)
        meter._prefix_tokens = agent_fn.prefix_tokens()
        agent_fn = meter

    print(f"backtest: agent={args.agent} tickers={len(tickers)} "
          f"{args.start} -> {args.end}"
          + (f" model={model}" if args.agent == "llm" else ""))

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
    # Sibling key, not a change to the existing schema: run files written before
    # metering existed still load, and cmd_report treats a missing "coordination"
    # block as "this system paid nothing", which is correct for the baselines.
    if meter is not None:
        payload["coordination"] = meter.to_dict()
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    fills = sum(len(r.execution_log) for r in runs)
    print(f"  {len(runs)} episode(s), {fills} fill(s) -> {out}")
    if meter is not None:
        print()
        print(meter.report())
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
    meters: dict = {}          # system -> CoordinationMeter, for systems that paid
    for name in systems:
        path = RUNS_DIR / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"no run file for {name!r} ({path})")
        with open(path) as f:
            payload = json.load(f)
        episodes_by_system[name] = [_episode_from_dict(d) for d in payload["episodes"]]
        impact_enabled[name] = payload.get("impact_gamma", 0.1) > 0
        # Absent for the deterministic baselines and for any run file written
        # before metering existed. Absent means "paid nothing", which is exactly
        # right for an agent that makes no API calls.
        if "coordination" in payload:
            from coordination_cost import CoordinationMeter
            meters[name] = CoordinationMeter.from_dict(payload["coordination"])

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
        # CBS is measured against a reference system; fall back to whatever
        # is actually present rather than failing the whole report. There is
        # no standing passive baseline agent any more (buy_and_hold was
        # removed), so --baseline defaults to None and this fallback is the
        # normal path, not an edge case.
        fallback = next(iter(reports), None)
        if fallback is None:
            log.warning("no baseline available; skipping leaderboard")
            return 0
        if baseline is not None:
            log.warning("baseline %r not among runs; using %r", baseline, fallback)
        baseline = fallback

    leaderboard = comparison.build_leaderboard(
        reports, episodes_by_system, baseline_name=baseline,
        market_impact_enabled=impact_enabled,
    )
    print()
    print(comparison.leaderboard_summary(leaderboard))

    _report_coordination_cost(meters, episodes_by_system, baseline)
    return 0


def _report_coordination_cost(meters: dict, episodes_by_system: dict,
                              baseline: str) -> None:
    """The CBS the proposal actually promises: the coordinated system charged for
    its own coordination, not just for trading frictions.

    The leaderboard above measures CBS on trading costs alone. A system that only
    wins because its LLM bill was never counted has not won."""
    if not meters:
        return
    from coordination_cost import coordination_adjusted_cbs

    print()
    print("=" * 72)
    print("COORDINATION COST")
    print("=" * 72)

    for name, meter in meters.items():
        print()
        print(meter.report())

        curve = meter.cost_across_models()
        if not curve.empty:
            print()
            print("  same token volume, priced across the ladder "
                  "(cost is a model-selection decision, not a constant):")
            print("    " + curve.to_string(index=False).replace("\n", "\n    "))
            print("    NB: repricing swaps the RATE, not the tokenization. Models "
                  "in a different tokenizer family would emit different counts "
                  "for the same text; re-measure rather than trusting the swap.")

        if name == baseline or baseline not in episodes_by_system:
            continue
        cbs = coordination_adjusted_cbs(
            name, baseline,
            episodes_by_system[name], episodes_by_system[baseline],
            system_meter=meter,
            # The baseline is deterministic unless it too has a meter on file,
            # in which case it pays its own bill rather than being treated as free.
            baseline_meter=meters.get(baseline),
        )
        print()
        print(f"  CBS vs {baseline}:")
        print(f"    trading costs only     : "
              f"{'>grid' if cbs.cbs_bps_trading_only is None else f'{cbs.cbs_bps_trading_only:.0f} bps'}")
        print(f"    with coordination bill : "
              f"{'>grid' if cbs.cbs_bps_with_coordination is None else f'{cbs.cbs_bps_with_coordination:.0f} bps'}")
        print(f"    edge retained at {cbs.mean_latency_seconds:.1f}s latency: "
              f"{cbs.edge_retained:.1%}")
        print(f"    verdict: {cbs.verdict()}")


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

    # --- cost controls (only bite on --agent llm) --------------------------
    from coordination_cost import MODEL_PRICES
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_PRICES),
                   help=f"model to run and price against (default: {DEFAULT_MODEL}). "
                        "Constrained to MODEL_PRICES keys so an unpriceable run "
                        "fails at parse time rather than mid-sweep.")
    p.add_argument("--no-cache", action="store_true",
                   help="disable prompt caching. Caching only pays when the stable "
                        "prefix clears the model's minimum; the preflight says "
                        "whether it does.")
    p.add_argument("--batch", action="store_true",
                   help="request the Batch API's 50%% discount. Currently refused "
                        "for the sequential LLM agent -- see the error for why.")
    p.add_argument("--estimate-only", action="store_true",
                   help="print the projected cost and exit without calling the API")
    p.add_argument("--max-spend", type=float, default=None,
                   help="USD ceiling on the projected list cost. Required for a "
                        "paid run; the run refuses to start without it.")


def _add_report_args(p) -> None:
    p.add_argument("--systems", nargs="*", default=None,
                   help="default: every *.pkl in runs/")
    p.add_argument("--factors", default=None,
                   help="CSV of daily mkt/smb/hml; omitted = real Ken French factors "
                        "if `data build` fetched them, else SPY proxy (warns)")
    p.add_argument("--baseline", default=None,
                   help="system that CBS is measured against; default: whichever "
                        "run is loaded first (no standing passive baseline agent "
                        "exists any more)")
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
