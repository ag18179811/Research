# Related Works — audited against actual pipeline execution

Every paper below is named in one of your two Consensus threads — nothing outside those two files
is included here. For each, the question answered is not "does code exist that could compute this"
but "does a real run of this pipeline actually produce this, right now" — checked directly against
the codebase and against what every real report this session has actually printed.

---

## 1. Nguyen & Pham, "Toward Reliable Evaluation of LLM-Based Financial Multi-Agent Systems"
(arXiv:2603.27539)

**Concept:** A taxonomy for financial multi-agent systems; five named failure modes (look-ahead
bias, survivorship bias, overfitting, transaction-cost neglect, regime-shift blindness); Coordination
Breakeven Spread (CBS) as the metric for whether coordination adds value net of costs, defined as
`CBS(d) = delta_p(d) / 2` where `delta_p(d)` is the execution-price improvement from coordination.

**Actually demonstrated in our pipeline: partially, and CBS specifically — no.**
- The five-failure-mode taxonomy: genuinely addressed. Temporal integrity, regime robustness, and
  cost-adjusted execution are all real, running mechanisms (see #2 and #3 below).
- **CBS itself: never computed with real data, anywhere, in any run.** I checked this directly rather
  than assume:
  - `cbs.py` implements the paper's exact formula but is **never imported or called by anything** —
    confirmed by grep across the whole repo. It's dead code.
  - `comparison.py`/`coordination_cost.py` implement a *different*, self-built metric that's also
    called "CBS" (a breakeven-cost-sweep: at what added bps does system X's advantage over
    system Y disappear) — conceptually adjacent to the paper's CBS, but not the same definition
    (no `delta_p(d)`, no execution-price-improvement comparison).
  - Even that self-built version has **never produced a real number**: it requires a second system
    to compare against, and `runs/` has only ever contained one file, `llm.json`. Every report this
    session has printed `cbs_note = "baseline"` for the single system, because there was nothing
    to compare it to.
  - The root cause: TradingAgents (the intended multi-agent comparator — see #6) isn't installed
    (`ModuleNotFoundError: No module named 'tradingagents'`) and isn't wired into `main.py`'s CLI
    (`AGENT_CHOICES` only has `"llm"`). Coordination value, in any form, cannot be measured until
    a second, genuinely multi-agent system is actually run through the same pipeline.

---

## 2. Benhenda, "Look-Ahead-Bench" (arXiv:2601.13770)

**Concept:** A standardized benchmark measuring point-in-time look-ahead bias in realistic trading
workflows (not Q&A leakage probes), via alpha decay between in-sample and out-of-sample
performance.

**Actually demonstrated: yes, and this is the most solidly real part of the whole pipeline.**
- `PointInTimeStore._enforce_and_log` filters every single data query to `available_at < as_of` —
  this ran on every real run this session (thousands of queries in the full sweep alone).
- `PointInTimeStore.check()` and `snapshot_as_of`'s self-audit go further than a standalone check —
  I verified this directly with adversarial tests (fed it deliberately bad timestamps, confirmed it
  correctly reported violations rather than trivially passing).
- Every real report has printed a genuine `Temporal Integrity: PASS`, backed by real audit-log
  entries, not a default/assumed value.
- **Not extended yet**: the multi-agent channel leakage (shared memory, inter-agent messages)
  Look-Ahead-Bench's single-agent check doesn't cover — `shared_memory.py` has this exact
  machinery built (PiT-filtered `read_inbox`/`read_tool_outputs`/`check_provenance`) but is never
  imported anywhere, because no multi-agent system has been run yet to generate messages worth
  auditing.

---

## 3. Arora et al., "PredictionMarketBench" (arXiv:2602.00133)

**Concept:** Deterministic replay of historical order books/trades/settlement with maker/taker
semantics and fee modeling for reproducible execution-realistic backtesting.

**Actually demonstrated: yes, with real numbers from real runs.**
- Every fill in every run this session has a real execution price, real slippage, real fee, and a real
  maker/taker flag (`is_maker`) — e.g. the AAPL buy fill from the Claude Haiku demo: 610 shares @
  $163.71, fee $49.93, slippage 0.20%.
- `phase3.py`'s `execution_breakeven_bps` re-scores one fixed execution log under a sweep of cost
  assumptions without re-running anything — the same "log once, re-score many times" principle.
- **Not built**: PMB's own execution replay is one system at a time, same as ours currently is —
  the three-way same-episode comparison (single-agent / uncoordinated multi-agent / coordinated)
  that would isolate a coordination effect through this same replay engine doesn't exist yet, for the
  same reason CBS doesn't (#1): there's only ever been one runnable system.

---

## 4. Yuan et al., "TraderBench" (arXiv:2603.00285)

**Concept:** Evaluates agents under adversarial market conditions using realized Sharpe/return/
drawdown rather than LLM-judge scores; flags that single-run evaluation misses variance.

**Actually demonstrated: partially, and one part is a deliberate divergence, not a gap.**
- Realized-metric evaluation: yes — every number in every report (Sharpe, Sortino, drawdown,
  breakeven) comes from the actual execution log, never a model's self-assessment.
- Multi-seed/variance reporting: yes — `block_bootstrap_ci` has produced real 95% CIs in every
  report (e.g. Sharpe 4.72 [95% CI 0.06, 7.90] from the 3-episode regime test).
- **Adversarial stress conditions specifically: not built, and not planned to be** — our regime
  robustness uses only real historical episodes (both the HMM-labeled sweep and the VIX-derived
  crisis windows), never synthetic/manufactured adversarial perturbations. This is a real, stated
  divergence from TraderBench's approach, not an oversight.

---

## 5. Zhu et al., "From Knowing to Doing" (KTD-Fin, arXiv:2605.28359)

**Concept:** A masking protocol (anonymizing tickers/dates/calendar cues) plus Barra-style
attribution decomposing returns into market, style, and stock-selection alpha — finding that LLM
agents' returns under leakage-controlled evaluation are mostly passive exposure, not persistent
selection skill.

**Actually demonstrated: the attribution regression, yes with real numbers. The masking protocol,
no.**
- `phase3.py`'s `factor_attribution` regresses real agent daily returns against real market-factor
  data via OLS, producing real alpha + beta values in every run (e.g. the demo run: alpha +7.7%
  annualized, beta_mkt=0.81) — this is the same market/style-exposure-vs-residual decomposition
  KTD-Fin's attribution framework describes.
- `factor_attribution_ablation` checks whether alpha survives across different factor-set sizes
  instead of trusting one specification — matching the spirit of KTD-Fin's finding that raw returns
  can be explained away by exposure rather than skill.
- **KTD-Fin's actual named contribution — the masking protocol** (anonymizing identifiers to test
  whether the LLM is reasoning or reciting memorized market narratives) **is not implemented.**
  Nothing in this pipeline currently masks ticker/date identity before it reaches the agent.

---

## 6. Xiao et al., "TradingAgents" (arXiv:2412.20138)

**Concept:** Multi-agent LLM trading architecture — specialized analyst agents, a Bull/Bear debate,
a trader, a risk-management debate team — cited in your threads as an external multi-agent
comparator baseline.

**Actually demonstrated: no — confirmed not runnable in this environment right now.**
- `phase3.py` has real, source-verified integration code: `build_trading_agents_adapter`,
  `pit_safe_get_fundamentals`, and `make_benchmark_safe_graph_class` patch three specific leaks
  found by reading TradingAgents' actual source (a fundamentals vendor that ignores `curr_date`,
  an unpatchable insider-transactions tool, cross-run memory persistence).
- But `import tradingagents` fails — the package isn't installed — and `main.py`'s CLI has no branch
  for it (`AGENT_CHOICES` is `("llm",)` only). This is the single biggest reason coordination value
  (#1) has never been measurable: our only comparator system doesn't run yet.

---

## 7. The Hidden Markov Model regime-detection thread

Your "Financial Market Regime Evaluation" file has its own sub-thread specifically asking where the
HMM idea came from. It names **Wang 2020** as "the closest template" (HMM regime detection
driving downstream regime-conditioned trading logic) and **Oelschläger 2020/2024** for the caution
that "simple HMMs can confuse short-term fluctuations with longer-trend shifts" versus hierarchical
variants. I'm not treating this as a new research task — just reporting what your own file already
says, since it's the direct answer to "why HMM" that your own thread asked and answered.

**Actually demonstrated: yes, with real regime labels from every real run.**
- `dataprep.py`'s `label_regimes` produces real trending/mean_reverting/crisis labels for every
  episode, walk-forward (refit every 63 days on only prior data, never full-sample).
- Every regime robustness table this session has shown real, differentiated numbers by regime —
  e.g. Sharpe 4.68 (trending) vs. Sharpe -2.60 (crisis) from the 2020 COVID episode.
- Matches your thread's own critique honestly: we use a plain 3-state HMM, not the hierarchical
  variant Oelschläger's work argues for — worth stating as a scope decision in the paper rather than
  leaving unaddressed, since your own file already flagged this exact question.

---

# The one-paragraph version, if you need it fast

Of the six papers/threads named in your Consensus files, **temporal integrity (Look-Ahead-Bench),
execution realism (PredictionMarketBench), regime labeling (the HMM thread), and attribution
regression (KTD-Fin's framing)** are all genuinely running and producing real numbers in this
pipeline right now. **Coordination value — Nguyen & Pham's CBS specifically — has never been
computed with real data in any run this session**, because `cbs.py` is dead code and the only system
ever actually run is the single-agent Claude Haiku baseline; TradingAgents, the intended multi-agent
comparator, is neither installed nor wired into the CLI. TraderBench's realized-metric and variance-
reporting principles are followed; its adversarial-perturbation approach is a deliberate divergence,
not a gap. KTD-Fin's masking protocol specifically is not implemented.

If coordination value is central to the paper's claims, that's the actual highest-priority gap between
what your research threads describe and what this pipeline currently proves.
