# Data Preparation Phase

Implements Phase 1 (Data Preparation) of the research proposal: turns raw market data into
point-in-time (PiT) records, enforces that nothing "future" can leak into a query, and
produces the regime-labeled historical episodes the rest of the benchmark evaluates against.

## High-level overview

Three raw sources get normalized into their own point-in-time datasets, each row carrying an
`available_at` timestamp -- the moment that row would have actually been knowable:

| Source | What it is | Keyed by |
|---|---|---|
| FNSPID prices | Daily OHLCV bars, ~7,700 tickers, 1980-2023 | ticker |
| FNSPID news | ~15.5M articles, 1999-2024 | ticker (where tagged) |
| SEC EDGAR | As-filed, never-revised quarterly financial statements, 2009-2026 | CIK |

A single access layer, `PointInTimeStore`, sits in front of all three. Every read is filtered
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
  flags those as high-stress periods.
- **Episode generation** produces non-overlapping historical windows (with purge gaps between
  them) and assigns each one a regime label *before* any agent runs on it.

## What the code roughly does

Everything lives in `dataprep.py`. The pieces, in the order data flows through them:

1. `build_prices()` / `build_fundamentals()` / `build_news()` -- parse the raw FNSPID/SEC
   archives into partitioned Parquet, adding the `available_at` column to each.
2. `build_entity_map()` -- fetches SEC's CIK↔ticker crosswalk.
3. `build_news_entity_backfill()` -- FNSPID's ticker tagging on news turned out to be sparse
   (e.g. AAPL was only tagged in 3 of its 25 years of coverage); this recovers a large chunk
   of that by matching company names against untagged article titles.
4. `PointInTimeStore` -- the query layer. `snapshot_as_of(ticker, as_of)` is the one method
   meant to be called from outside this file: it returns a single PiT-consistent bundle
   (recent prices, recent news, latest known fundamentals, derived features) for one ticker as
   of one timestamp.
5. `label_regimes()` / `generate_episode_windows()` / `assign_episode_regimes()` -- build the
   regime-labeled episode windows evaluation will run against.

**The one function anything downstream (a mock loop, a real agent, Phase 2's replay engine)
actually needs to call is `PointInTimeStore.snapshot_as_of(ticker, as_of)`.** Everything else
in this file exists to make that one call correct.

## Setup

Requires Python 3.11+. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy hmmlearn pyarrow flashtext
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
python dataprep.py build-news-entity-backfill    # recovers sparse news ticker tags
```

`build-all` takes on the order of 15-20 minutes depending on disk speed (most of it is the
23GB news CSV). Output lands in `data/processed/` (~6.6GB): `prices/`, `fundamentals/`,
`news/`, `entity_map.parquet`.

## Running the demo

```bash
python demo.py
```

Walks a toy "mock agent" through TSLA's early-2020 COVID episode, calling
`store.snapshot_as_of()` once a week and making a trivial volatility-based decision. It's
three things in sequence:

1. **Proves the temporal wall works** -- queries as of a cutoff, confirms nothing returned is
   dated at or after it.
2. **Shows a regime-labeled episode** -- fits the walk-forward HMM (~30s) and prints the
   regime assigned to a Jan-Mar 2020 window.
3. **Runs the mock decision loop** -- ten weekly snapshots through the crash, printing realized
   volatility and the toy agent's decision at each step. You can watch the agent go cautious
   (`HOLD, volatility too high`) exactly as volatility spikes through late February and March.

Takes under two minutes end-to-end (needs the processed dataset built first, per above).

**The integration point for Phase 2:** `mock_agent_decide()` in `demo.py` is a stand-in. A
real agent -- self-built or the TradingAgents repo -- consumes exactly the same
`snapshot_as_of()` output; nothing about the data layer needs to change to swap it in.
