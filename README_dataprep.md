# Data Preparation Phase (+ Phase 1.5 agent interface, + CBS metric)

Implements Phase 1 (Data Preparation) of the research proposal: turns raw market data into
point-in-time (PiT) records, enforces that nothing "future" can leak into a query, and
produces the regime-labeled historical episodes the rest of the benchmark evaluates against.
Also includes Phase 1.5 (turning Phase 1's output into an agent-facing prompt) and a standalone
implementation of the Coordination Breakeven Spread (CBS) metric from Nguyen & Pham 2026.

## High-level overview

Four raw sources get normalized into their own point-in-time datasets, each row carrying an
`available_at` timestamp -- the moment that row would have actually been knowable:

| Source | What it is | Keyed by |
|---|---|---|
| FNSPID prices | Daily OHLCV bars, ~7,700 tickers, 1980-2023 | ticker |
| FNSPID news | ~12.7M articles (deduplicated), 1999-2024 | ticker (where tagged) |
| SEC EDGAR | As-filed, never-revised quarterly financial statements, 2009-2026 | CIK |
| Alpaca Market Data | Intraday bars, NBBO quotes, corporate actions -- ~2016+ only | ticker |

A single access layer, `PointInTimeStore`, sits in front of all four. Every read is filtered
to `available_at < as_of` and logged -- an agent (or anything else querying this) physically
cannot see data from after its own decision point. That's the core guarantee this whole phase
exists to provide.

On top of that:
- **Entity resolution** joins SEC's CIK-keyed fundamentals to the ticker-keyed prices/news via
  a crosswalk built from SEC's public ticker mapping.
- **Derived features** (rolling returns, realized volatility, a few fundamentals ratios) are
  computed on top of already-PiT-filtered data, so they inherit the same guarantee.
- **Regime labeling** fits a walk-forward Hidden Markov Model (Trending / Mean-Reverting /
  Crisis) on market-wide (SPY) returns + realized volatility, refit periodically using only
  data before each point -- validated against 2008, 2018Q4, and 2020 by checking it actually
  flags those as high-stress periods. Regime labels are deliberately never exposed to the
  agent (see `assign_episode_regimes`'s docstring for why) -- they're a Phase 3 bookkeeping
  input only.
- **Episode generation** produces non-overlapping historical windows (with purge gaps between
  them) and assigns each one a regime label *before* any agent runs on it.
- **Official evaluation scope is now decided, not ad hoc**: `EVALUATION_TICKERS` (20 verified
  large-cap equities), `BENCHMARK_TICKERS` (SPY/GLD/FXI), and `EPISODE_WINDOW_LEN` /
  `EPISODE_PURGE_GAP` / `EPISODE_RANGE_START` / `EPISODE_RANGE_END` (90-day episodes, 14-day
  purge gaps, 2010-01-01 to 2024-01-01) are real constants in `dataprep.py` now, each with the
  verification/reasoning behind it in a comment above the constant -- see that file directly
  rather than this doc for the full reasoning, since it's long.

## What the code roughly does

Everything lives in `dataprep.py`. The pieces, in the order data flows through them:

1. `build_prices()` / `build_fundamentals()` / `build_news()` -- parse the raw FNSPID/SEC
   archives into partitioned Parquet, adding the `available_at` column to each.
2. `build_entity_map()` -- fetches SEC's CIK↔ticker crosswalk.
3. `build_news_entity_backfill()` -- FNSPID's ticker tagging on news turned out to be sparse
   (e.g. AAPL was only tagged in 3 of its 25 years of coverage); this recovers a large chunk
   of that by matching company names against untagged article titles. Also deduplicates by
   URL -- a real, verified duplication issue in the raw data (18.3M duplicate rows found and
   removed) that isn't obvious unless you specifically check for it.
4. `build_intraday_bars()` / `build_quotes()` / `build_corporate_actions()` -- pull minute
   bars, real bid/ask quotes, and splits/dividends from Alpaca's Market Data API, for the
   Execution Replay phase. Needs `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` in a `.env` file
   (never commit this). Coverage is empirically bounded to ~2016 onward -- verified directly,
   not assumed; querying earlier dates returns empty, not an error.
5. `PointInTimeStore` -- the query layer.
   - `snapshot_as_of(ticker, as_of)` -- one ticker, PiT-consistent bundle (prices, news,
     fundamentals, derived features).
   - `universe_snapshot_as_of(as_of, cash, positions, tickers=None)` -- the same thing for
     every ticker in `EVALUATION_TICKERS` at once, plus portfolio state (cash/positions are
     required arguments -- this class has no way to know a running portfolio's state on its
     own, that's owned by whichever phase is actually running the episode loop).
   - `bars_as_of` / `quotes_as_of` / `corporate_actions_as_of` -- same enforcement pattern,
     for the Alpaca datasets.
6. `label_regimes()` / `generate_episode_windows()` / `assign_episode_regimes()` -- build the
   regime-labeled episode windows evaluation will run against.

**`PointInTimeStore.snapshot_as_of(ticker, as_of)` and `universe_snapshot_as_of(...)` are the
two methods anything downstream (a mock loop, a real agent, Phase 2's replay engine) actually
needs to call.** Everything else in this file exists to make those calls correct.

## Phase 1.5 -- `agent_interface.py`

Turns `universe_snapshot_as_of()`'s output into an actual prompt (`build_universe_prompt`),
and defines `AgentDecision` -- the shape an agent's response parses into (ticker, side,
rationale; deliberately no quantity or price, since sizing/execution are Phase 2's job per the
proposal's own framing). `decide_universe()` is an explicit placeholder -- no LLM is called
anywhere in this file; wiring one in is a separate, not-yet-made decision. `count_tokens()`
gives a real (not estimated) token count via `tiktoken`.

**Known open interface question, flagged directly rather than silently resolved**: Phase 2's
`Order` (see `execution_replay.py`) requires a real `quantity`, computed by the agent function
itself in every example agent currently in this repo (`no_trade_agent`,
`buy_and_hold_agent_factory`, `random_agent_factory`, the TradingAgents adapter). That doesn't
match `AgentDecision`'s current no-quantity design -- worth resolving as a team before wiring
a real agent through this file.

## CBS metric -- `cbs.py`

A standalone implementation of the Coordination Breakeven Spread from Nguyen & Pham 2026
(arXiv:2603.27539), Section 6.2: `CBS(d) = delta_p(d) / 2`. The spread side (`s`) is computed
from real Alpaca quote data via `real_spread_as_of()`. The price-improvement side (`delta_p`)
is deliberately NOT computed here -- per the paper, it requires real matched-window execution
data comparing a coordinated run against a single-agent baseline, which doesn't exist until
Phase 1.5 has a real agent and Phase 2 has run it.

**Note this is a different implementation than `comparison.py`'s `coordination_breakeven_spread`
and `coordination_cost.py`'s `coordination_adjusted_cbs`** -- those operationalize "CBS" as a
fee-sweep against a baseline system's PnL rather than the paper's literal price-improvement
formula. All three measure a related but distinct thing and will report different numbers --
worth the team deciding which one is the reported metric before publishing anything.

## Setup

Requires Python 3.11+. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need Alpaca-backed data (`build-intraday-bars`, `build-quotes`,
`build-corporate-actions`), create a `.env` file (gitignored, never commit it) with:

```
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
```

## Getting the raw data

Not included in the repo (too large for git). Download these three:

```bash
# FNSPID prices (~560MB)
curl -L -o full_history.zip https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip

# FNSPID news (~23GB -- the big one, budget real time for this download)
curl -L -o nasdaq_exteral_data.csv https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv

# SEC EDGAR financial statement data sets (~5.6GB across 69 quarterly zips)
# See https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
# for the current list of quarterly archive URLs; download each into one folder.
```

By default `dataprep.py` looks for these at `~/full_history.zip`, `~/nasdaq_exteral_data.csv`,
and `~/financial-statement-data-sets/*.zip` -- see the `RAW_*` constants near the top of the
file if you'd rather point it elsewhere.

## Building the processed dataset

```bash
python dataprep.py build-all                    # prices + fundamentals + news + entity map
python dataprep.py build-news-entity-backfill    # recovers sparse news ticker tags, dedupes
```

`build-all` takes on the order of 15-20 minutes depending on disk speed (most of it is the
23GB news CSV). Output lands in `data/processed/` (~6.6GB): `prices/`, `fundamentals/`,
`news/`, `entity_map.parquet`.

Alpaca-backed datasets are opt-in, per-ticker, per-date-range (unlike the archive-based builds
above, there's no "download everything" version -- pulling detailed intraday/quote data for
every ticker over years of history would take a very long time):

```bash
python dataprep.py build-intraday-bars --tickers AAPL TSLA --start 2020-01-01 --end 2020-02-01
python dataprep.py build-quotes --tickers AAPL --start 2020-01-01 --end 2020-01-02
python dataprep.py build-corporate-actions --tickers AAPL TSLA --start 2015-01-01 --end 2024-01-01
```

## Running validation

```bash
python validate.py
```

Runs a suite of domain-specific checks (not generic unit tests) against the built dataset --
does the access layer actually block look-ahead leakage, do regime labels agree with known
market history, does the entity crosswalk resolve correctly, etc. Run after any change to
`dataprep.py` to confirm the phase still does what the proposal asks for.

## Running the demo

```bash
python demo.py
```

Walks a toy "mock agent" through TSLA's early-2020 COVID episode, calling
`store.snapshot_as_of()` once a week and making a trivial volatility-based decision, across 4
sections: temporal integrity proof, regime labels for the episode, the mock decision loop, and
a summary of the full access log (every PiT request the run made, blocked or not). Takes under
two minutes end-to-end (needs the processed dataset built first, per above).

**The integration point for Phase 2:** `mock_agent_decide()` in `demo.py` is a stand-in. A
real agent -- self-built or the TradingAgents repo -- consumes exactly the same
`snapshot_as_of()` / `universe_snapshot_as_of()` output; nothing about the data layer needs to
change to swap it in.
