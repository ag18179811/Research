"""
Multi-system comparison table.

This is the code counterpart to the proposal's comparison grid (the table
contrasting Look-Ahead-Bench / TraderBench / PredictionMarketBench / etc.
against "This benchmark"). That table compares *benchmarks* in the paper;
this module produces the analogous thing for *agent systems* run through
this benchmark -- a leaderboard where each row is one system's
JointMetricReport, scored against the same six criteria, plus the
Coordination Breakeven Spread (CBS) the proposal names as the key
falsifiable-hypothesis metric: does a coordinated system's edge over a
simpler baseline survive realistic costs, or does it evaporate once fees
are high enough?

Requires the raw EpisodeRun lists (not just the JointMetricReport) per
system, because CBS needs each episode's fill-level execution log, which
metrics.py aggregates away into per-episode scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from metrics import EpisodeRun, JointMetricReport, _fills_to_frame, net_monetary_return

# ---------------------------------------------------------------------------
# Coordination Breakeven Spread (CBS)
# ---------------------------------------------------------------------------


def _episode_adjusted_pnl(ep: EpisodeRun, bps: float) -> float:
    """Counterfactual net return for one episode if every trade had paid
    `bps` instead of whatever fee was actually charged.

    Unlike Phase 2's calculate_breakeven_threshold (which sums signed
    notional across fills and implicitly assumes trades close out in
    round trips), this starts from the *portfolio value* net return --
    which already correctly includes unrealized mark-to-market gains on
    any position still open at episode end -- and adjusts only the fee
    component: add back whatever fees were actually paid, then subtract
    fees at the counterfactual rate. A buy-and-hold system that never
    sells still shows its real gain here; the fills-only version silently
    zeroed it out because it has no closing sell to net against."""
    actual_net_return = net_monetary_return(ep.portfolio_values)
    if pd.isna(actual_net_return):
        actual_net_return = 0.0
    fills = _fills_to_frame(ep.execution_log)
    if fills.empty:
        return actual_net_return
    total_fees_paid = float(fills["fee_paid"].sum())
    total_notional = float((fills["executed_qty"] * fills["execution_price"]).sum())
    counterfactual_fee = total_notional * (bps / 10000.0)
    return actual_net_return + total_fees_paid - counterfactual_fee


def _system_adjusted_pnl(episodes: Sequence[EpisodeRun], bps: float) -> float:
    return sum(_episode_adjusted_pnl(ep, bps) for ep in episodes)


@dataclass
class CBSResult:
    system_name: str
    baseline_name: str
    advantage_at_zero_bps: float  # coordinated_pnl - baseline_pnl, no added cost
    cbs_bps: float | None  # bps at which the advantage hits zero; None = never
    # crosses within the swept range (advantage holds at every cost level
    # tested, i.e. a robust edge); 0.0 = no advantage even before costs.
    max_bps_tested: float


def coordination_breakeven_spread(
    system_name: str,
    baseline_name: str,
    system_episodes: Sequence[EpisodeRun],
    baseline_episodes: Sequence[EpisodeRun],
    step_bps: float = 1.0,
    max_bps: float = 200.0,
) -> CBSResult:
    """Sweeps an additional cost level from 0 to max_bps and finds the point
    where `system`'s net PnL drops to or below `baseline`'s net PnL. This
    operationalizes the proposal's framing of coordination value as a
    falsifiable, cost-sensitive hypothesis rather than an assumed
    advantage: a high CBS means the coordinated system's edge survives
    even sharply higher trading costs; a low or zero CBS means the edge is
    fragile or nonexistent once realistic frictions are applied.

    Both episode sets must have been run over the *same* episode windows
    (same dates, same regime labels) -- this is what makes the comparison
    valid rather than an artifact of different market conditions, per the
    proposal's Motivation section. This function doesn't verify that for
    you; pass in episodes generated from a single shared
    run_all_episodes(..., start=..., end=...) call reused across systems."""
    advantage_0 = _system_adjusted_pnl(system_episodes, 0.0) - _system_adjusted_pnl(baseline_episodes, 0.0)
    if advantage_0 <= 0:
        return CBSResult(system_name, baseline_name, advantage_0, 0.0, max_bps)

    bps = 0.0
    while bps <= max_bps:
        adv = _system_adjusted_pnl(system_episodes, bps) - _system_adjusted_pnl(baseline_episodes, bps)
        if adv <= 0:
            return CBSResult(system_name, baseline_name, advantage_0, bps, max_bps)
        bps += step_bps
    return CBSResult(system_name, baseline_name, advantage_0, None, max_bps)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def build_leaderboard(
    reports: dict[str, JointMetricReport],
    episodes_by_system: dict[str, Sequence[EpisodeRun]],
    baseline_name: str,
    market_impact_enabled: dict[str, bool] | None = None,
    cbs_step_bps: float = 1.0,
    cbs_max_bps: float = 200.0,
) -> pd.DataFrame:
    """One row per system. Columns mirror the proposal's benchmark-comparison
    grid, but scored per *system* rather than per *benchmark paper*:

      pit_integrity        from JointMetricReport.temporal_integrity.clean
      realistic_execution  True for every system here by construction --
                            every system was run through Phase 2's
                            ExecutionReplayEngine, unlike a raw-return
                            backtest. Kept as an explicit column (rather
                            than assumed) so a future system evaluated
                            through a different/bypassed path shows False
                            instead of silently inheriting True.
      market_impact         whether that system's ReplayConfig had
                            impact_gamma > 0 -- pass market_impact_enabled
                            if any system used impact_gamma == 0 (i.e. the
                            square-root impact term was effectively off)
      multi_seed_ci          True for every system here, since
                            build_joint_metric_report always runs the
                            block bootstrap
      regime_stratified      number of distinct regimes actually observed
                            in that system's episodes (>= 2 means the
                            report can distinguish regime-dependent
                            performance, not just report a single pooled
                            number)
      cbs_bps                Coordination Breakeven Spread vs baseline_name,
                            in bps; None ("n/a") for the baseline's own row,
                            since a system's CBS relative to itself is
                            undefined
    """
    market_impact_enabled = market_impact_enabled or {}
    rows = []
    for name, report in reports.items():
        n_regimes = report.regime_robustness["regime"].nunique()
        if name == baseline_name:
            cbs_bps = None
            cbs_note = "baseline"
        else:
            cbs = coordination_breakeven_spread(
                name, baseline_name,
                episodes_by_system[name], episodes_by_system[baseline_name],
                step_bps=cbs_step_bps, max_bps=cbs_max_bps,
            )
            if cbs.cbs_bps is None:
                cbs_bps, cbs_note = cbs_max_bps, f">{cbs_max_bps:.0f} (edge survives max tested cost)"
            elif cbs.cbs_bps == 0.0 and cbs.advantage_at_zero_bps <= 0:
                cbs_bps, cbs_note = 0.0, "no advantage even at zero added cost"
            else:
                cbs_bps, cbs_note = cbs.cbs_bps, f"edge vanishes by {cbs.cbs_bps:.0f} bps added cost"

        rows.append({
            "system": name,
            "pit_integrity": report.temporal_integrity.clean,
            "realistic_execution": True,
            "market_impact": market_impact_enabled.get(name, True),
            "multi_seed_ci": True,
            "regime_stratified": n_regimes >= 2,
            "n_regimes_observed": n_regimes,
            "sharpe": report.sharpe_ci.point_estimate,
            "sharpe_ci_low": report.sharpe_ci.ci_low,
            "sharpe_ci_high": report.sharpe_ci.ci_high,
            "cbs_bps": cbs_bps,
            "cbs_note": cbs_note,
        })

    df = pd.DataFrame(rows).set_index("system")
    return df


def leaderboard_summary(df: pd.DataFrame) -> str:
    lines = ["=== Agent System Leaderboard ===", ""]
    display_cols = ["pit_integrity", "realistic_execution", "market_impact",
                     "multi_seed_ci", "regime_stratified", "sharpe", "cbs_bps"]
    lines.append(df[display_cols].to_string())
    lines.append("")
    for name, row in df.iterrows():
        lines.append(f"{name}: {row['cbs_note']}")
    return "\n".join(lines)
