# 3. Method

Our benchmark runs in four phases. Phase 1 prepares point-in-time data and regime-labeled episodes. Phase 1.5 turns each day's snapshot into a validated trading decision. Phase 2 replays each decision as a real fill against historical market data. Phase 3 scores the accumulated fills on four properties, with confidence intervals. The same pipeline runs unmodified whether the system under test is a single agent or a coordinating multi-agent team — nothing downstream of Phase 1.5 depends on which.

## 3.1 Phase 1 — Point-in-Time Data Preparation

**Point-in-time enforcement.** Every historical record — prices, fundamentals, news, intraday bars, quotes, corporate actions — carries an `available_at` timestamp: the moment that value actually became knowable, not the date it describes. A query at decision time $t$ returns only rows with $\text{available\_at} < t$. A request that would return a later row is blocked at the loader and logged, not silently dropped; this log is Phase 3's Temporal Integrity signal (§3.4A).

**Self-audit.** A second, independent pass re-checks the same condition over every timestamp actually returned in a snapshot — on every snapshot request, not a sample — and counts any row at or after $t$ as a violation. This catches a regression in the loader's own comparison logic (for example, a $<$ that silently became $\leq$) rather than trusting the loader's output at face value.

**Shared-memory provenance.** The same $<$ rule governs inter-agent communication. Every write to shared memory — a posted fact, an agent-to-agent message, a cached tool output — is stamped with the timestamp at which it became available; a read at time $t$ returns only entries stamped before $t$, so a teammate cannot receive a fact before it existed for the agent that produced it. A separate check covers a different failure mode specific to derived facts: when a written entry is computed from other point-in-time data, it also carries the availability timestamp of that underlying data, and the check flags any entry stamped at or before the data it was built from — a fact claiming to exist no later than its own inputs did.

**Regime labeling** combines two independent signals:

- *Walk-forward HMM.* Three states (trending / mean-reverting / crisis) over (return, realized volatility), refit every 63 trading days with 8 random EM restarts, using only data strictly before each refit point.
- *VIX threshold.* CBOE VIX close $> 30$ sustained $\geq 5$ trading days, clusters merged across gaps $\leq 45$ days. The HMM alone is crisis-sparse at this granularity (2 crisis episodes across the full 2010–2024 range); VIX detection recovers 6.

**Episode construction.** Episodes are non-overlapping windows separated by a purge gap so no state crosses a boundary; the agent resets at every episode start. The reported design is 18 episodes, 6 per regime, spanning 2010–2023 across 20 tickers — the trending and mean-reverting windows are drawn from the same systematic HMM process used for crisis detection, spread across different years, and pairwise non-overlapping against every other window. This replaces an earlier 7:1:1 split (the systematic VIX-detected crisis set plus a small hand-picked trending/mean-reverting sample), which gave the crisis regime disproportionate statistical power. Each regime totals roughly 370 trading-day observations, past the point where the bootstrap CI (§3.4) is dominated by sampling noise. A Fama-French-Carhart six-factor daily series and the risk-free rate (Ken French Data Library) are also prepared here, for Phase 3's attribution stage.

## 3.2 Phase 1.5 — Agent Decision Loop

**Daily observation.** Each trading day, the agent receives, per ticker: prior close, 5/21/63-day returns, realized volatility, available fundamentals (leverage, current ratio, ROA), and a *count* — never the text — of recent news items, removing a prompt-injection surface while preserving the count as signal. It also receives its own cash, positions, and up to 10 days of its own trade history. The regime label is withheld: showing "you are in a CRISIS" would test label-reading, not whether an edge survives an unlabeled crash. Realized volatility, the HMM's own input feature, remains visible.

**Decision schema and validation.** The agent returns one JSON object per day: action $\in\{\text{BUY}, \text{SELL}, \text{HOLD}\}$, symbol, quantity, order type $\in\{\text{MARKET}, \text{LIMIT}, \text{POST\_ONLY}\}$, limit price where applicable, and a short rationale; at most one trade per day. Before becoming an order, the decision is validated: symbol must be in the universe; SELL quantity is clipped to the position actually held, read fresh from the execution engine's state each day; BUY quantity is clipped by cash affordability at the pre-decision price, a first-pass estimate that Phase 2 supersedes once the real fill price is known. A ticker missing price data on a given day falls back to HOLD rather than aborting the episode.

**Backend integrations.** The same loop runs unmodified across three backend integrations — the native Anthropic Messages API, Azure AI Foundry's unified OpenAI-compatible route serving multiple model families through one interface, and Anthropic's Messages API hosted on Azure via Marketplace billing — so the cross-backbone coordination-cost comparison is not fit to any one provider's API shape. Every call retries with capped exponential backoff on transient failure (empty or malformed response, connection error, rate limit, server error) but never on a client error that would fail identically on every retry; an unretried transient failure would otherwise be indistinguishable from a genuine HOLD decision, conflating infrastructure noise with trading behavior. Real token usage and wall-clock latency are measured per call and feed Phase 3's coordination-cost accounting directly.

## 3.3 Phase 2 — Execution Replay

**Fill pricing.** Fills are priced against real historical bid/ask/depth (a flagged synthetic fallback where quote coverage gaps exist); a BUY crosses the ask, a SELL crosses the bid. POST_ONLY orders and resting LIMIT orders fill at their own price at the maker rate; MARKET orders and marketable LIMIT orders cross the spread at the taker rate.

**Market impact.** For a taker fill of size $q$, execution price is

$$p_{\text{exec}} = p_{\text{touch}} \cdot (1 \pm \gamma \cdot \sigma \cdot \sqrt{q / \text{ADV}})$$

the Almgren-Chriss square-root impact law ($\gamma = 0.1$; $\sigma$ = realized volatility; ADV = average daily volume; sign matches trade direction), applied only to taker fills — a resting order provides liquidity rather than walking the book, so impact applies only to the order types that actually consume it.

**Fill-time constraints.** Three constraints are enforced at the point of fill, using information unavailable beforehand: fill quantity is capped at available depth; a BUY cannot spend more than cash on hand once the real post-impact, post-fee price is known; a LIMIT order, resting or marketable, never executes worse than its own stated limit. Fees are charged on the post-slippage notional ($q \cdot p_{\text{exec}} \cdot \text{fee}_{\text{bps}}/10^4$; maker $=-1$bps rebate, taker $=+5$bps). The portfolio is marked to that day's own close every trading day, traded or not, rather than to the prior decision-time snapshot, avoiding a one-day lag in the return series.

**Execution log.** Every fill — timestamp, symbol, side, quantity, execution price, fee, maker/taker flag, regime — is written to one execution log. Because the log already holds real fills, Phase 3 re-prices it under any counterfactual fee schedule without re-simulating, which is what makes the breakeven sweep in §3.4E cheap enough to run at fine granularity.

## 3.4 Phase 3 — Joint Metric Scoring

All four properties are computed from one pipeline run over one set of episodes, not four separate passes that could disagree.

**(A) Temporal Integrity.** PASS iff Phase 1's aggregated access log contains zero violations across every episode.

**(B) Regime Robustness.** Per-episode Sharpe, Sortino, net return, max drawdown, and breakeven fee, grouped by the pre-assigned regime label and reported as mean $\pm$ std within each of the three regimes, so a system whose edge only exists in calm markets shows a visible crisis-vs-trending gap rather than having it averaged away.

**(C) Executable Performance.** With $r_t$ the day-$t$ simulated return and $r_f$ the matching Ken French daily rate,

$$\text{Sharpe} = \frac{\overline{r_t - r_f}}{\text{std}(r_t - r_f)}\sqrt{252}, \qquad \text{Sortino} = \frac{\overline{r_t - r_f}}{\sqrt{\frac{1}{N}\sum_t \min(r_t - r_f,\, 0)^2}}\sqrt{252}$$

Sortino follows the Sortino & Price (1994) definition — downside semi-deviation against the target over all $N$ observations, not the standard deviation of a filtered negative-return subset. Both are computed on simulated returns from actual fills, not an idealized signal-following series.

**(D) Attribution.** Excess daily returns are regressed on the Fama-French-Carhart factors,

$$r_t - r_f = \alpha + \beta_{\text{mkt}}\cdot\text{Mkt}_t + \beta_{\text{smb}}\cdot\text{SMB}_t + \beta_{\text{hml}}\cdot\text{HML}_t + \beta_{\text{rmw}}\cdot\text{RMW}_t + \beta_{\text{cma}}\cdot\text{CMA}_t + \beta_{\text{mom}}\cdot\text{MOM}_t + \varepsilon_t$$

reported at three nested specifications — CAPM, 3-factor, full 6-factor — so a reader can see whether $\alpha$ survives which factors are included. A positive $\alpha$ indicates returns unexplained by these passive exposures, not proof of skill.

**Confidence intervals.** 95% CIs for Sharpe, Sortino, and $\alpha$ via a circular moving-block bootstrap: blocks of $b=21$ consecutive trading days drawn with wraparound over $B=1000$ resamples, preserving the short-range autocorrelation an i.i.d. resample would destroy.

**(E) Coordination value.** Holding history, costs, and regime labels fixed, we sweep an added fee $c$ from 0 upward and report the smallest $c$ at which a comparison system's cumulative advantage over a baseline reaches zero, re-scoring Phase 2's logged fills rather than re-simulating; an edge surviving a much larger $c$ before vanishing is real, one vanishing near $c=0$ likely was not. The mechanism is baseline-agnostic: any two systems that ran an identical episode set can be compared, so in the results reported here the baseline is one of the evaluated single-agent backbones. A companion accounting layer charges the comparison system for its own measured coordination overhead — token cost plus a latency-driven signal-decay penalty — so a system cannot be reported as winning on overhead it was never billed for.

| Component | Parameter | Value |
|---|---|---|
| HMM regime labeling | refit interval / restarts | 63 trading days / 8 |
| VIX crisis detection | threshold / min length / merge gap | close $>30$ / $\geq5$ days / $\leq45$ days |
| Episode design | count / per-regime / purge gap | 18 / 6 / 14 days |
| Universe | tickers | 20 |
| Execution | maker / taker fee | $-1$bps / $+5$bps |
| Execution | impact scale $\gamma$ | 0.1 |
| Starting capital | per episode | \$100,000 |
| Bootstrap CI | resamples / block size / confidence | 1000 / 21 days / 95% |
