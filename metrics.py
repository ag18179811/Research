"""
Joint Metric Collection layer for the multi-agent financial benchmark.

Aggregates the outputs of the two upstream phases into the four report
categories described in the proposal's Joint Metric Collection Phase:

  1. Temporal Integrity   -- from PointInTimeStore.access_log (Phase 1)
  2. Regime Robustness    -- performance grouped by episode regime label
                              (Phase 1's assign_episode_regimes)
  3. Executable Performance -- net return / Sharpe / Sortino / max drawdown /
                              breakeven threshold, computed from the
                              cost-adjusted execution log (Phase 2)
  4. Attribution          -- OLS alpha/beta vs Market/Size/Value factors,
                              with block-bootstrap 95% CIs on Sharpe,
                              Sortino, and alpha across episodes

This module does not re-run the agent or re-simulate the market. Everything
here is arithmetic over data Phase 1 and Phase 2 already produced -- an
episode's execution log and access log are treated as fixed inputs, exactly
the same "no need to rerun" principle Phase 2 uses for its breakeven sweep.

Expected inputs (you assemble these once per (agent_system, episode) run):

  EpisodeRun(
      episode_start, episode_end, regime,
      execution_log      -> list[ExecutionFill] from Phase 2's
                             ExecutionReplayEngine.execution_log
      portfolio_values    -> pd.Series of daily mark-to-market portfolio
                             value, indexed by date, covering the episode
                             (use ExecutionReplayEngine.compute_portfolio_value
                             once per trading day of the episode)
      access_log          -> the accumulated list[dict] from
                             PointInTimeStore.access_log for every store
                             query issued while running this episode
  )

One EpisodeRun per non-overlapping window from Phase 1's
generate_episode_windows. A JointMetricReport is built from a list of these
for a single agent system, so systems are compared report-to-report under
identical episodes/regimes/costs.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("metrics")

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRun:
    """Everything Phase 3 needs from one episode for one agent system.
    Deliberately holds raw Phase 1/2 outputs rather than pre-computed
    metrics, so every downstream number in this file is traceable back to
    an execution log row or an access-log row."""

    episode_start: pd.Timestamp
    episode_end: pd.Timestamp
    regime: str  # "trending" | "mean_reverting" | "crisis", from Phase 1
    execution_log: list  # list[ExecutionFill] (Phase 2 dataclass instances,
    # or equivalent objects/dicts exposing the same fields)
    portfolio_values: pd.Series  # date-indexed, daily mark-to-market value
    access_log: list[dict]  # PointInTimeStore.access_log entries (Phase 1)

    def daily_returns(self) -> pd.Series:
        pv = self.portfolio_values.sort_index()
        return pv.pct_change().dropna()


def _fills_to_frame(execution_log) -> pd.DataFrame:
    if not execution_log:
        return pd.DataFrame(columns=["timestamp", "side", "executed_qty",
                                      "execution_price", "fee_paid", "regime"])
    rows = []
    for f in execution_log:
        # Accept either the Phase 2 ExecutionFill dataclass or a plain dict
        # (e.g. reloaded from execution_log.json) -- both expose the same
        # field names, so normalize once here rather than branching
        # throughout the rest of the module.
        rec = dataclasses.asdict(f) if dataclasses.is_dataclass(f) else dict(f)
        side = rec["side"]
        rec["side"] = side.value if hasattr(side, "value") else side
        regime = rec.get("regime")
        rec["regime"] = regime.value if hasattr(regime, "value") else regime
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Temporal Integrity
# ---------------------------------------------------------------------------


@dataclass
class TemporalIntegrityReport:
    total_queries: int
    total_rows_returned: int
    total_rows_blocked: int  # informational: see note below, NOT a failure count
    explicit_check_violations: int  # from PointInTimeStore.check() calls, if any were made
    checked_episodes: list[str]  # episode_start ISO strings where check() was actually exercised

    @property
    def clean(self) -> bool:
        # A blocked row is the filter *working*, not a leak -- see below.
        # The only genuine violation signal available from the access log is
        # an explicit check() call reporting violations > 0.
        return self.explicit_check_violations == 0


def temporal_integrity_report(episodes: Sequence[EpisodeRun]) -> TemporalIntegrityReport:
    """IMPORTANT: given how Phase 1's PointInTimeStore is implemented
    (prices_as_of/news_as_of/fundamentals_as_of load the *entire* dataset
    and then filter to `available_at < as_of`), every ordinary query
    blocks whatever rows are still in the future relative to that day.
    That is the guard working correctly, not an attempted leak -- treating
    total_rows_blocked as a failure count (an earlier version of this
    function did) misreports completely normal operation as a violation.
    Expect total_rows_blocked > 0 on essentially every real run; it's kept
    here only as a diagnostic (e.g. sanity-checking it trends toward zero
    as as_of approaches the end of the dataset).

    The one signal in the access log that *does* indicate a genuine
    violation is an explicit PointInTimeStore.check() call -- used for
    auditing a candidate access request such as an inter-agent message or
    tool-output timestamp -- reporting rows_blocked > 0. Those log entries
    are distinguished from ordinary loader calls by dataset names your
    check() call sites choose (e.g. "agent_message", "tool_output"); this
    function treats any access_log entry whose dataset name is not one of
    the three raw loaders as an explicit check and sums its violations."""
    ORDINARY_LOADERS = {"prices", "news", "fundamentals"}
    total_q = total_ret = total_blk = total_check_violations = 0
    checked_episodes = []
    for ep in episodes:
        episode_had_check = False
        for entry in ep.access_log:
            total_q += 1
            total_ret += entry.get("rows_returned", 0)
            total_blk += entry.get("rows_blocked", 0)
            if entry.get("dataset") not in ORDINARY_LOADERS:
                episode_had_check = True
                total_check_violations += entry.get("rows_blocked", 0)
        if episode_had_check:
            checked_episodes.append(str(ep.episode_start))
    return TemporalIntegrityReport(
        total_queries=total_q,
        total_rows_returned=total_ret,
        total_rows_blocked=total_blk,
        explicit_check_violations=total_check_violations,
        checked_episodes=checked_episodes,
    )


# ---------------------------------------------------------------------------
# 2 & 3. Executable Performance (per-episode) and Regime Robustness (rollup)
# ---------------------------------------------------------------------------


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if daily_returns.empty or daily_returns.std(ddof=1) == 0:
        return float("nan")
    return float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    downside = daily_returns[daily_returns < 0]
    if daily_returns.empty or downside.empty or downside.std(ddof=1) == 0:
        return float("nan")
    return float(daily_returns.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values: pd.Series) -> float:
    """Largest peak-to-trough percentage decline before a new high, as a
    positive fraction (0.25 == a 25% drawdown)."""
    pv = portfolio_values.sort_index()
    if pv.empty:
        return float("nan")
    running_max = pv.cummax()
    drawdown = (pv - running_max) / running_max
    return float(-drawdown.min())


def net_monetary_return(portfolio_values: pd.Series) -> float:
    pv = portfolio_values.sort_index()
    if len(pv) < 2:
        return float("nan")
    return float(pv.iloc[-1] - pv.iloc[0])


def execution_breakeven_bps(execution_log, step_bps: float = 1.0, max_bps: float = 100.0) -> float:
    """Same sweep as Phase 2's ExecutionReplayEngine.calculate_breakeven_threshold,
    reimplemented against a plain fills list/frame so this module doesn't need
    a live ExecutionReplayEngine instance -- only its logged output. If you
    still have the engine instance handy, prefer calling its method directly
    and pass the result in; this is here for offline/reloaded logs."""
    fills = _fills_to_frame(execution_log)
    if fills.empty:
        return 0.0
    direction = np.where(fills["side"] == "SELL", 1.0, -1.0)
    notional = fills["executed_qty"] * fills["execution_price"]

    current_bps = 0.0
    while current_bps <= max_bps:
        sim_fee = notional * (current_bps / 10000.0)
        net_pnl = float((direction * notional - sim_fee).sum())
        if net_pnl <= 0:
            return current_bps
        current_bps += step_bps
    return max_bps


@dataclass
class EpisodeMetrics:
    episode_start: pd.Timestamp
    regime: str
    net_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    breakeven_bps: float
    daily_returns: pd.Series = field(repr=False)


def episode_metrics(ep: EpisodeRun) -> EpisodeMetrics:
    rets = ep.daily_returns()
    return EpisodeMetrics(
        episode_start=ep.episode_start,
        regime=ep.regime,
        net_return=net_monetary_return(ep.portfolio_values),
        sharpe=sharpe_ratio(rets),
        sortino=sortino_ratio(rets),
        max_drawdown=max_drawdown(ep.portfolio_values),
        breakeven_bps=execution_breakeven_bps(ep.execution_log),
        daily_returns=rets,
    )


def executable_performance_table(episodes: Sequence[EpisodeRun]) -> pd.DataFrame:
    """One row per episode -- the raw material for both the Executable
    Performance summary and the Regime Robustness groupby below."""
    rows = [dataclasses.asdict(episode_metrics(ep)) for ep in episodes]
    df = pd.DataFrame(rows).drop(columns=["daily_returns"])
    return df.sort_values("episode_start").reset_index(drop=True)


def regime_robustness_table(perf_table: pd.DataFrame) -> pd.DataFrame:
    """Groups the per-episode performance table by regime label so you can
    see, e.g., whether Sharpe collapses specifically in crisis windows --
    this is the check that catches a system that only "works" in calm
    markets, per the proposal's motivation for regime-stratified eval."""
    agg = perf_table.groupby("regime").agg(
        n_episodes=("episode_start", "count"),
        mean_net_return=("net_return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        std_sharpe=("sharpe", "std"),
        mean_sortino=("sortino", "mean"),
        mean_max_drawdown=("max_drawdown", "mean"),
        mean_breakeven_bps=("breakeven_bps", "mean"),
    )
    return agg.reset_index()


# ---------------------------------------------------------------------------
# 4. Attribution: factor regression + block bootstrap
# ---------------------------------------------------------------------------


def _ols_alpha_beta(y: np.ndarray, factors: np.ndarray) -> tuple[float, np.ndarray]:
    """Plain OLS via lstsq (no statsmodels dependency): y = alpha + X @ beta.
    Returns (alpha, beta_vector). `factors` is (n_obs, n_factors); an
    intercept column is added here."""
    X = np.column_stack([np.ones(len(y)), factors])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), coef[1:]


@dataclass
class AttributionResult:
    alpha_annualized: float
    beta_market: float
    beta_size: float
    beta_value: float
    n_obs: int


def factor_attribution(
    daily_returns: pd.Series,
    factor_returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> AttributionResult:
    """Regresses the agent's daily returns on daily Market/Size/Value factor
    returns. `factor_returns` must be date-indexed with columns
    ["mkt", "smb", "hml"] (e.g. Fama-French daily factors) covering at least
    the same dates as `daily_returns`; you're responsible for sourcing that
    series (Ken French data library or equivalent) and aligning its index
    timezone/frequency to the portfolio value series before calling this.

    Alpha is annualized (daily alpha * periods_per_year) and reported as
    factor-adjusted residual performance, per the proposal's framing --
    this is NOT a claim of "genuine skill," just the return unexplained by
    market/size/value exposure."""
    aligned = pd.concat([daily_returns.rename("ret"), factor_returns[["mkt", "smb", "hml"]]],
                         axis=1, join="inner").dropna()
    if len(aligned) < 5:
        log.warning("factor_attribution: only %d aligned observations, results unreliable", len(aligned))
    y = aligned["ret"].to_numpy()
    X = aligned[["mkt", "smb", "hml"]].to_numpy()
    alpha, beta = _ols_alpha_beta(y, X)
    return AttributionResult(
        alpha_annualized=alpha * periods_per_year,
        beta_market=float(beta[0]),
        beta_size=float(beta[1]),
        beta_value=float(beta[2]),
        n_obs=len(aligned),
    )


def _moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draws blocks of consecutive indices (with wraparound) until reaching
    length n. Block resampling (rather than iid resampling of single days)
    preserves the short-range autocorrelation in daily returns, which iid
    bootstrap would destroy and which matters for realistic Sharpe/Sortino
    CIs on financial return series."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])
    return idx[:n]


@dataclass
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int


def block_bootstrap_ci(
    daily_returns: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    metric: str = "sharpe",
    n_boot: int = 1000,
    block_size: int = 21,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """95% CI for `metric` in {"sharpe", "sortino", "alpha"} via moving
    block bootstrap over the (concatenated, across episodes) daily return
    series. Per the proposal: this guards against a single favorable
    window being reported as if it were a stable result. `factor_returns`
    is required (and resampled in lockstep with the returns, via shared
    block indices) when metric == "alpha", since alpha needs matched
    agent/factor observations on each resample."""
    rets = daily_returns.dropna()
    n = len(rets)
    if n < block_size:
        raise ValueError(f"need at least {block_size} return observations, got {n}")

    if metric == "alpha":
        if factor_returns is None:
            raise ValueError("factor_returns is required for metric='alpha'")
        aligned = pd.concat([rets.rename("ret"), factor_returns[["mkt", "smb", "hml"]]],
                             axis=1, join="inner").dropna()
        rets_arr = aligned["ret"].to_numpy()
        fac_arr = aligned[["mkt", "smb", "hml"]].to_numpy()
        n = len(aligned)

        def point(r, f):
            a, _ = _ols_alpha_beta(r, f)
            return a * TRADING_DAYS_PER_YEAR
    else:
        rets_arr = rets.to_numpy()
        fac_arr = None
        fn = {"sharpe": sharpe_ratio, "sortino": sortino_ratio}[metric]

        def point(r, f):
            return fn(pd.Series(r))

    estimate = point(rets_arr, fac_arr)

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = _moving_block_bootstrap_indices(n, block_size, rng)
        r_boot = rets_arr[idx]
        f_boot = fac_arr[idx] if fac_arr is not None else None
        boot_stats[b] = point(r_boot, f_boot)

    alpha_tail = (1 - confidence) / 2
    lo, hi = np.nanpercentile(boot_stats, [100 * alpha_tail, 100 * (1 - alpha_tail)])
    return BootstrapCI(point_estimate=float(estimate), ci_low=float(lo), ci_high=float(hi), n_boot=n_boot)


# ---------------------------------------------------------------------------
# Joint report
# ---------------------------------------------------------------------------


@dataclass
class JointMetricReport:
    system_name: str
    temporal_integrity: TemporalIntegrityReport
    executable_performance: pd.DataFrame  # per-episode
    regime_robustness: pd.DataFrame  # per-regime rollup
    attribution: AttributionResult
    sharpe_ci: BootstrapCI
    sortino_ci: BootstrapCI
    alpha_ci: BootstrapCI

    def summary(self) -> str:
        ti = self.temporal_integrity
        lines = [
            f"=== Joint Metric Report: {self.system_name} ===",
            "",
            "-- Temporal Integrity --",
            f"clean: {ti.clean}  (explicit check() violations: {ti.explicit_check_violations})",
            f"  [informational] {ti.total_rows_blocked} rows filtered out as future-dated / "
            f"{ti.total_rows_returned} returned across {ti.total_queries} loader queries "
            f"-- this is the PiT filter operating normally, not a leak",
        ]
        if not ti.clean:
            lines.append(f"  episodes with explicit check() violations: {ti.checked_episodes}")
        lines += [
            "",
            "-- Regime Robustness --",
            self.regime_robustness.to_string(index=False),
            "",
            "-- Executable Performance (aggregate across episodes) --",
            f"mean Sharpe:  {self.sharpe_ci.point_estimate:.3f}  "
            f"95% CI [{self.sharpe_ci.ci_low:.3f}, {self.sharpe_ci.ci_high:.3f}]",
            f"mean Sortino: {self.sortino_ci.point_estimate:.3f}  "
            f"95% CI [{self.sortino_ci.ci_low:.3f}, {self.sortino_ci.ci_high:.3f}]",
            "",
            "-- Attribution --",
            f"alpha (annualized): {self.alpha_ci.point_estimate:.4f}  "
            f"95% CI [{self.alpha_ci.ci_low:.4f}, {self.alpha_ci.ci_high:.4f}]",
            f"beta_market={self.attribution.beta_market:.3f}  "
            f"beta_size={self.attribution.beta_size:.3f}  "
            f"beta_value={self.attribution.beta_value:.3f}  "
            f"(n_obs={self.attribution.n_obs})",
        ]
        return "\n".join(lines)


def build_joint_metric_report(
    system_name: str,
    episodes: Sequence[EpisodeRun],
    factor_returns: pd.DataFrame,
    n_boot: int = 1000,
    block_size: int = 21,
    seed: int = 0,
) -> JointMetricReport:
    """Top-level entry point: run the full Joint Metric Collection phase for
    one agent system over its assigned episodes. `factor_returns` is shared
    across systems (same market conditions -> comparable attribution)."""
    ti = temporal_integrity_report(episodes)
    perf = executable_performance_table(episodes)
    regime = regime_robustness_table(perf)

    # Concatenate daily returns across episodes for the aggregate attribution
    # and bootstrap. Episodes are non-overlapping with purge gaps (Phase 1),
    # so this concatenation doesn't stitch together autocorrelated periods
    # that were never actually adjacent in the agent's run.
    all_returns = pd.concat([ep.daily_returns() for ep in episodes]).sort_index()
    all_returns = all_returns[~all_returns.index.duplicated(keep="first")]

    attribution = factor_attribution(all_returns, factor_returns)
    sharpe_ci = block_bootstrap_ci(all_returns, metric="sharpe", n_boot=n_boot,
                                    block_size=block_size, seed=seed)
    sortino_ci = block_bootstrap_ci(all_returns, metric="sortino", n_boot=n_boot,
                                     block_size=block_size, seed=seed)
    alpha_ci = block_bootstrap_ci(all_returns, factor_returns=factor_returns, metric="alpha",
                                   n_boot=n_boot, block_size=block_size, seed=seed)

    return JointMetricReport(
        system_name=system_name,
        temporal_integrity=ti,
        executable_performance=perf,
        regime_robustness=regime,
        attribution=attribution,
        sharpe_ci=sharpe_ci,
        sortino_ci=sortino_ci,
        alpha_ci=alpha_ci,
    )
