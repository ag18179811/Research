"""
Data preparation layer for the multi-agent financial benchmark.

Turns raw FNSPID (prices + news) and SEC EDGAR (fundamentals) archives into
point-in-time (PiT) normalized Parquet datasets, then exposes a query layer
(`PointInTimeStore`) that only ever returns data that would have actually
been available as of a given timestamp. This is the enforcement point
described in the proposal's Data Preparation phase: every record carries an
`available_at` timestamp, and any query for a cutoff is filtered to
`available_at < as_of` with the attempt logged either way.

Also provides the two other Data Preparation phase pieces:
  - a rolling, out-of-sample HMM regime labeler (Trending / Mean-Reverting /
    Crisis), refit walk-forward so a label for date T never sees data at or
    after T.
  - a non-overlapping episode window generator with purge gaps, for
    rolling-window evaluation.

Plus three datasets pulled live from Alpaca's Market Data API for the
Execution Replay phase (order sequencing, fill mechanics, bid/ask spreads --
none of which the daily FNSPID bars below can support): intraday bars,
NBBO quotes, and corporate actions. Coverage is empirically bounded to
~2016 onward; see ALPACA_HISTORY_START.

Raw sources (already downloaded):
  ~/full_history.zip                    FNSPID prices, one CSV per ticker
  ~/nasdaq_exteral_data.csv             FNSPID news, ~9GB, one row per article
  ~/financial-statement-data-sets/*.zip SEC EDGAR quarterly financial
                                         statement data sets (sub/num/tag/pre)

Raw sources (fetched live, need APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env):
  Alpaca Market Data API                intraday bars, NBBO quotes, corporate
                                         actions -- see build_intraday_bars,
                                         build_quotes, build_corporate_actions
"""

from __future__ import annotations

import argparse
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dataprep")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HOME = Path.home()
PROJECT_DIR = Path(__file__).resolve().parent

RAW_PRICE_ZIP = HOME / "full_history.zip"
RAW_NEWS_CSV = HOME / "nasdaq_exteral_data.csv"
RAW_SEC_DIR = HOME / "financial-statement-data-sets"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PRICES_DIR = PROCESSED_DIR / "prices"
NEWS_DIR = PROCESSED_DIR / "news"
FUNDAMENTALS_DIR = PROCESSED_DIR / "fundamentals"

# Approximate US equity market close, used as the availability timestamp for
# a trading day's OHLCV bar. Not DST-adjusted; fine for daily-bar research,
# not for anything that needs the exact close second.
MARKET_CLOSE_UTC_HOUR = 20  # ~4pm ET


# ---------------------------------------------------------------------------
# Official evaluation scope: which stocks, which episodes
# ---------------------------------------------------------------------------
# Neither of these was decided anywhere before -- every prior script in this
# repo (demo.py, validate.py) just picked its own convenient ticker/date
# range for its own testing purpose. These are the actual decisions,
# verified against the built data rather than assumed:

# 20 large-cap, highly liquid US equities. Picked over the full 7,693-ticker
# FNSPID universe for three concrete reasons: (1) household-recognizable
# names a reader can sanity-check results against, (2) high liquidity, which
# matters directly for the CBS metric (cbs.py) -- coordination is only
# economically viable on narrow-spread instruments, and narrow spreads
# correlate with liquidity; (3) guaranteed Alpaca intraday/quote coverage
# for the post-2016 portion of the episode range below.
#
# Known limitation, stated plainly rather than hidden: this is a fixed list,
# not a point-in-time index membership snapshot. Nguyen & Pham 2026
# (arXiv:2603.27539) name exactly this ("point-in-time universe") as one of
# their five minimum evaluation standards -- a proper fix means sourcing
# historical index-constituent data (e.g. the Sharadar bundle named in the
# proposal), which this pipeline does not have. Using a fixed list of
# still-listed large-caps is survivorship-biased in the standard way (it
# excludes companies that were large-cap and later failed/were acquired/were
# delisted). Flagged as a real gap, not silently accepted.
#
# Every ticker below was verified two ways, not just "exists": (1) price
# history actually spans the full EPISODE_RANGE_START-EPISODE_RANGE_END
# window below with no early cutoff, and (2) ticker_to_cik resolves, so
# fundamentals aren't silently empty. This caught real problems on the
# first pass -- GOOGL, FB, PG, HD, MA, and BAC all looked fine by ticker
# existence alone but their raw FNSPID price history actually stops dead in
# 2020 (verified against the raw zip directly, not a processing bug), and FB
# never resolves a CIK at all. Swapped for the closest verified-complete
# large-cap peer in each case: GOOGL->GOOG (same company, other share
# class, full history), FB->INTC, PG->CL, HD->LOW, MA->AXP, BAC->WFC.
EVALUATION_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "INTC", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "CL", "XOM", "ABBV", "LOW", "AXP", "DIS", "WFC", "KO", "PFE",
]

# SPY and GLD are the SPDR S&P 500 / SPDR Gold Shares named directly in the
# proposal. MCHI (the proposal's named China ETF) has the same problem as
# the swapped equities above -- its FNSPID price history stops at
# 2020-07-02, verified directly -- so FXI (iShares China Large-Cap) is used
# instead as the closest verified-complete China-exposure ETF. Both FXI and
# MCHI fail ticker_to_cik (cik=None) -- unlike the operating companies
# above, that's expected and not a gap: these are ETF trusts, not treated
# as filers this crosswalk resolves, and benchmarks are only ever used for
# passive-return comparison, never for the fundamentals-derived ratios that
# CIK resolution would feed.
BENCHMARK_TICKERS = ["SPY", "GLD", "FXI"]

# Episode window length and purge gap: 90 days / 14 days. Not a new choice --
# this is what demo.py and both validate.py episode checks already used
# consistently, just never declared as an official constant until now.
EPISODE_WINDOW_LEN = pd.Timedelta(days=90)
EPISODE_PURGE_GAP = pd.Timedelta(days=14)

# Episode range: 2010-01-01 to 2024-01-01. Bounded on both ends by verified
# data coverage, not picked arbitrarily:
#   start -- SEC EDGAR's earliest quarterly archive is 2009q1 (verified:
#            ls ~/financial-statement-data-sets/*.zip). Starting exactly at
#            2009q1 would mean the earliest episodes see only one quarter of
#            fundamentals history; 2010-01-01 gives a full year of buffer so
#            snapshot_as_of's fundamentals ratios aren't running on
#            single-quarter data for the first several episodes.
#   end   -- verified news coverage extends through 2024 (validate.py's own
#            coverage check requires max(years) >= 2023; the actual built
#            data reaches into 2024).
# Consequence, stated directly: this excludes the 2008 GFC. Fundamentals
# data doesn't exist in this pipeline before 2009q1, so an episode
# overlapping 2008 would hand the agent a snapshot with real prices/news but
# an empty fundamentals table and three None-valued ratios out of
# derived_features -- a degraded, inconsistent bundle compared to every
# other episode. The regime *labeling* itself is unaffected (label_regimes
# runs on SPY back to 1993 and already correctly flags 2008 as crisis, shown
# earlier), but the multi-source snapshot an agent would actually receive
# for that period is incomplete, so 2008 is excluded from the official
# episode set until fundamentals coverage before 2009 is sourced separately.
EPISODE_RANGE_START = pd.Timestamp("2010-01-01", tz="UTC")
EPISODE_RANGE_END = pd.Timestamp("2024-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# Prices: FNSPID full_history.zip -> partitioned Parquet
# ---------------------------------------------------------------------------

def build_prices(limit_tickers: int | None = None) -> None:
    """Parse full_history.zip (one CSV per ticker) into data/processed/prices,
    partitioned by ticker. Adds `available_at` = trading date at market close."""
    if not RAW_PRICE_ZIP.exists():
        raise FileNotFoundError(f"missing {RAW_PRICE_ZIP}")

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_PRICE_ZIP) as zf:
        members = [
            m for m in zf.namelist()
            if m.startswith("full_history/") and m.endswith(".csv") and "__MACOSX" not in m
        ]
        if limit_tickers:
            members = members[:limit_tickers]
        log.info("prices: %d ticker files to process", len(members))

        n_ok, n_bad = 0, 0
        for i, member in enumerate(members, 1):
            # Uppercased: the raw zip's filenames are inconsistently cased
            # (mostly upper, some lower, e.g. "nvda.csv"), but every other
            # ticker lookup in this codebase assumes uppercase (ticker_to_cik
            # does .upper(), the entity map is uppercase). Left as-is, a
            # lookup for "NVDA" would silently return zero rows instead of
            # an error -- normalizing here is what prevents that.
            ticker = Path(member).stem.upper()
            try:
                with zf.open(member) as f:
                    df = pd.read_csv(f)
            except Exception as exc:  # malformed member, skip and keep going
                log.warning("prices: failed to read %s: %s", member, exc)
                n_bad += 1
                continue

            if df.empty or "date" not in df.columns:
                n_bad += 1
                continue

            df = df.rename(columns={"adj close": "adj_close"})
            df["entity"] = ticker
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["available_at"] = df["date"] + pd.Timedelta(hours=MARKET_CLOSE_UTC_HOUR)
            df = df.sort_values("date").reset_index(drop=True)

            out_dir = PRICES_DIR / f"entity={ticker}"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_dir / "data.parquet", index=False)
            n_ok += 1

            if i % 1000 == 0:
                log.info("prices: %d/%d done", i, len(members))

    log.info("prices: done. %d tickers written, %d skipped", n_ok, n_bad)


def load_prices(entities: list[str] | None = None) -> pd.DataFrame:
    if not PRICES_DIR.exists():
        raise FileNotFoundError(f"no processed prices at {PRICES_DIR}, run build-prices first")
    if entities:
        paths = [PRICES_DIR / f"entity={e}" / "data.parquet" for e in entities]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(PRICES_DIR.glob("entity=*/data.parquet"))
    if not paths:
        return pd.DataFrame()
    # Read files individually rather than pointing pyarrow at the whole
    # directory: the `entity=<ticker>` folder names look like hive
    # partitioning, which clashes with the `entity` column already stored
    # inside each file (dictionary-encoded partition column vs plain string
    # column -> schema merge error).
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# ---------------------------------------------------------------------------
# News: FNSPID nasdaq_exteral_data.csv (~9GB) -> partitioned Parquet, streamed
# ---------------------------------------------------------------------------

NEWS_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author"]
# Article body + summaries are large text blobs; kept out of the default
# processed columns to keep the store fast to query. Use --with-text to
# include the full article body if a downstream step needs it.


def build_news(chunksize: int = 200_000, limit_rows: int | None = None, with_text: bool = False) -> None:
    """Stream nasdaq_exteral_data.csv into data/processed/news, partitioned by
    year. `available_at` = article publish timestamp (already UTC in the
    source), which is exact -- news is available the instant it's published."""
    if not RAW_NEWS_CSV.exists():
        raise FileNotFoundError(f"missing {RAW_NEWS_CSV}")

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    usecols = NEWS_COLUMNS + (["Article"] if with_text else [])
    # Publisher/Author are almost entirely empty; letting pandas infer their
    # dtype per internal chunk flips between float64 (all-NaN) and object
    # across chunk boundaries, which trips a bug in pandas' low_memory
    # dtype-mismatch warning path (IndexError in _concatenate_chunks).
    # Forcing str + low_memory=False avoids the mixed-inference path entirely.
    dtype = {c: str for c in usecols if c != "Date"}

    rows_seen = 0
    chunk_i = 0
    reader = pd.read_csv(RAW_NEWS_CSV, usecols=usecols, dtype=dtype,
                          low_memory=False, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk.dropna(subset=["Date"])
        chunk["published_at"] = pd.to_datetime(chunk["Date"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["published_at"])
        # A handful of rows carry unrelated, pre-1999 dates (scraper
        # contamination unrelated to FNSPID's documented 1999-2023 coverage,
        # e.g. digitized 1914 newspaper headlines) -- drop them.
        chunk = chunk[chunk["published_at"] >= pd.Timestamp("1999-01-01", tz="UTC")]
        chunk = chunk.rename(columns={
            "Article_title": "title",
            "Stock_symbol": "entity",
            "Url": "url",
            "Publisher": "publisher",
            "Author": "author",
        })
        chunk["available_at"] = chunk["published_at"]
        chunk["year"] = chunk["published_at"].dt.year
        keep = ["entity", "published_at", "available_at", "title", "url", "publisher", "author", "year"]
        if with_text:
            keep.append("Article")
        chunk = chunk[keep]

        chunk.to_parquet(NEWS_DIR, partition_cols=["year"], index=False,
                          basename_template=f"part-{chunk_i}-{{i}}.parquet")

        rows_seen += len(chunk)
        chunk_i += 1
        if chunk_i % 10 == 0:
            log.info("news: %d rows processed", rows_seen)
        if limit_rows and rows_seen >= limit_rows:
            break

    log.info("news: done, %d rows written", rows_seen)


# ---------------------------------------------------------------------------
# News entity backfill: recover missing ticker tags from article titles
# ---------------------------------------------------------------------------
# FNSPID's Stock_symbol tagging turned out to be sparse and inconsistent
# across years -- verified empirically: AAPL/MSFT/TSLA/GOOGL each only have
# tagged articles in a handful of disconnected calendar years, despite
# clearly relevant untagged content (spot-checked real Apple coverage from
# 2005) sitting in the file for the "missing" years. This recovers some of
# that gap by matching company names against titles for rows FNSPID left
# untagged. It's an additive, lower-confidence signal, not a replacement
# for the original tag -- see entity_source below.

# Pure legal-entity-type suffixes -- never carry brand meaning, always
# safe to strip, and can legitimately stack (e.g. "Amazon Com Inc" needs
# both "Inc" and the legacy ".com"-style "Com" stripped to reach "Amazon").
_NAME_SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LTD", "LIMITED", "LLC", "LLP", "PLC", "COM", "SA", "NV", "AG",
}

# Deliberately NOT stripped, unlike the above: "Group"/"Holdings"/"Trust"
# sometimes carry real distinguishing meaning (found via testing --
# "National Bank Holdings Corp" must keep "Holdings" or it collapses to
# the dangerously generic "National Bank"), so they're left in place
# rather than risk under-specifying the name. Costs a little recall for
# names that really are shortened in news prose; that's the safer
# direction to be wrong in.

# Single common English words that also happen to be real company names.
# Matching on these alone produces too many false positives to be useful
# (e.g. "Target Corporation" -> "target", which shows up in headlines like
# "analysts raise price target" constantly, unrelated to the retailer).
# Not exhaustive -- a known, accepted precision limitation of name matching.
_GENERIC_NAME_STOPLIST = {
    "target", "gap", "square", "match", "align", "enable", "chart",
    "carnival", "generac", "west", "best", "cash", "block", "vail",
    "core", "verify", "figs", "olo", "on", "so", "up", "us",
    "news", "city", "post", "team", "sea", "bill", "nice", "hello",
    "joint", "noble", "lion", "good", "geo", "hub", "icon", "spar",
    # found via empirical spot-check: some companies' actual legal names
    # are themselves generic words (e.g. "Strategy Inc", "Bancorp, Inc.",
    # "Awareness Group, Inc.") -- these mismatch constantly against
    # unrelated ordinary usage of the same word.
    "strategy", "bancorp", "awareness",
    # MSCI Inc. is a real, distinctively-named company, but "MSCI" is used
    # far more often in this news corpus as an index-family reference
    # (MSCI World, MSCI EM, ...) unrelated to the company itself.
    "msci",
    # more of the same pattern, found on a second larger sample: a real
    # company's short legal name that collides with ordinary word usage
    # ("Eastern Co", "Employers Holdings", "Coffee Holding Co",
    # "Citizens, Inc."), or with a different, unrelated company that
    # happens to share its first word ("Eaton Corp" vs. the unrelated
    # asset manager "Eaton Vance").
    "eastern", "employers", "coffee", "citizens", "eaton",
    # third sample: ordinary-word homographs -- "pool" (verb, as in "pool
    # resources"), "visa" (travel document, not Visa Inc.), "crane"
    # (construction equipment / bird, not Crane Co).
    "pool", "visa", "crane",
    # fourth sample: "Dana" (a first name, "Dana Perino"; not Dana Inc),
    # "Nova" (matched inside "Terra Nova," an oil field, not Nova Measuring
    # Instruments), "Toro" (matched inside "Concha y Toro," a winery, not
    # The Toro Company), "Flex" (a utility "Flex Alert" program, not Flex
    # Ltd), "Reliance" (ordinary usage "reliance on X," not Reliance Inc).
    "dana", "nova", "toro", "flex", "reliance",
}


def _normalize_company_name(name: str) -> str:
    """Strip punctuation and trailing legal-entity-type suffixes so a
    company name matches how it's actually written in news prose. Tuned to
    SEC's naming conventions, not a general-purpose normalizer.

    Loops fully through _NAME_SUFFIXES (safe: they're pure legal
    boilerplate that can legitimately stack, e.g. "Amazon Com Inc" needs
    both "Inc" and "Com" stripped to reach "Amazon") but never touches
    Group/Holdings/Trust, which are excluded from that set entirely --
    found via testing that stripping those is what breaks names like
    "National Bank Holdings Corp" down to the dangerously generic
    "National Bank"."""
    name = re.sub(r"[.,]", "", name.upper())
    tokens = name.split()
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return " ".join(tokens).strip()


def build_entity_name_matcher():
    """flashtext KeywordProcessor mapping normalized company name -> ticker.
    Restricted to tickers we have price data for, and to names specific
    enough that a whole-word title match is a reasonable signal:
    multi-word names are always kept (an exact multi-word phrase match is
    unlikely to be coincidental); single-word names are kept only if
    they're long enough and not in the generic-word stoplist above."""
    from flashtext import KeywordProcessor

    em = load_entity_map()
    price_tickers = {p.name.split("=", 1)[1] for p in Path(PRICES_DIR).glob("entity=*")}
    em = em[em["ticker"].isin(price_tickers)].copy()
    em["norm"] = em["name"].map(_normalize_company_name)
    em = em[em["norm"] != ""]

    # A normalized name shared by multiple different tickers doesn't
    # uniquely identify any one of them. Verified this is a real, non-rare
    # case: dozens of bank-issued ETN/ETF tickers are registered under just
    # the parent bank's name (e.g. GLDI, USOI, and SLVO are three different
    # tickers all registered as "CREDIT SUISSE AG"; several more under
    # "DEUTSCHE BANK AKTIENGESELLSCHAFT" and "UBS AG"). Matching on the
    # bank's name would misattribute every general mention of that bank to
    # one arbitrary ticker among several equally-plausible candidates.
    ticker_counts = em.groupby("norm")["ticker"].nunique()
    ambiguous_names = set(ticker_counts[ticker_counts > 1].index)

    kp = KeywordProcessor(case_sensitive=False)
    added = set()
    for _, row in em.drop_duplicates(subset="norm").iterrows():
        norm = row["norm"]
        if norm in ambiguous_names:
            continue
        is_multiword = " " in norm
        if not is_multiword and (len(norm) < 4 or norm.lower() in _GENERIC_NAME_STOPLIST):
            continue
        kp.add_keyword(norm, row["ticker"])
        added.add(norm)

    # Common brand name != SEC legal entity name, so the legal name alone
    # misses obvious cases. Small, manually verified, low-risk (these words
    # are unambiguous in news text) -- not a general alias system.
    for alias, ticker in {"GOOGLE": "GOOGL", "FACEBOOK": "META"}.items():
        if ticker in price_tickers and alias not in added:
            kp.add_keyword(alias, ticker)
            added.add(alias)
    return kp


def build_news_entity_backfill() -> None:
    """Adds two columns to the existing news dataset, processed one year
    partition at a time:
      entity_source   "tagged" (FNSPID's original tag), "title_match"
                      (recovered here), or null (still unmatched)
      entity          unchanged for already-tagged rows; filled in with the
                       title-matched ticker where the original tag was
                       missing and exactly one company matched (a title
                       mentioning multiple companies is left unmatched
                       rather than guessed)

    Inherently lower-recall (titles only, not full article body -- we never
    stored article body to keep the store queryable-sized) and imperfect
    precision (name matching can mismatch). Treat entity_source ==
    "title_match" rows as a lower-confidence signal than "tagged" ones."""
    if not NEWS_DIR.exists():
        raise FileNotFoundError(f"no processed news at {NEWS_DIR}, run build-news first")

    kp = build_entity_name_matcher()
    year_dirs = sorted(Path(NEWS_DIR).glob("year=*"))
    log.info("entity backfill: matching against %d company names across %d year partitions",
             len(kp.get_all_keywords()), len(year_dirs))

    def match_one(title):
        if not isinstance(title, str):
            return None
        found = set(kp.extract_keywords(title))
        return found.pop() if len(found) == 1 else None

    total_tagged, total_matched, total_rows, total_deduped = 0, 0, 0, 0
    for ydir in year_dirs:
        files = list(ydir.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

        # A year directory can end up holding both an already-backfilled
        # file and a fresh, not-yet-backfilled set of files in it -- e.g.
        # build_news got re-run after backfill had already completed once,
        # without a matching backfill re-run afterward, since build_news
        # doesn't clear its output directory first. Concatenating both
        # copies would silently double every duplicated article. Dedupe on
        # url (the most reliable unique identifier for one article -- more
        # reliable than title, which is a plain string) before anything
        # else, and drop any stale entity_source column so it gets
        # recomputed fresh for every row below, not left half-populated
        # from whichever copy happened to have it already.
        before = len(df)
        df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
        total_deduped += before - len(df)
        if "entity_source" in df.columns:
            df = df.drop(columns=["entity_source"])

        tagged_mask = df["entity"].notna()
        df["entity_source"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df.loc[tagged_mask, "entity_source"] = "tagged"

        untagged_idx = df.index[~tagged_mask]
        matches = df.loc[untagged_idx, "title"].map(match_one)
        matched_idx = matches.index[matches.notna()]
        df.loc[matched_idx, "entity"] = matches.loc[matched_idx]
        df.loc[matched_idx, "entity_source"] = "title_match"

        tmp_path = ydir / "part-backfilled.tmp.parquet"
        df.to_parquet(tmp_path, index=False)
        for f in files:
            f.unlink()
        tmp_path.rename(ydir / "part-backfilled.parquet")

        total_tagged += int(tagged_mask.sum())
        total_matched += len(matched_idx)
        total_rows += len(df)
        log.info("entity backfill: %s -> %d duplicate rows dropped, %d tagged, %d recovered via title match, %d still untagged",
                 ydir.name, before - len(df), int(tagged_mask.sum()), len(matched_idx),
                 len(df) - int(tagged_mask.sum()) - len(matched_idx))

    log.info("entity backfill: done. %d duplicate rows dropped total. %d/%d rows tagged (%d original + %d recovered)",
              total_deduped,
              total_tagged + total_matched, total_rows, total_tagged, total_matched)


def load_news(entities: list[str] | None = None, years: list[int] | None = None) -> pd.DataFrame:
    if not NEWS_DIR.exists():
        raise FileNotFoundError(f"no processed news at {NEWS_DIR}, run build-news first")
    # Push entity filtering down to the parquet read (row-group pruning)
    # instead of materializing all 15.5M rows and filtering in pandas --
    # matters at 16GB RAM, where the full table is expensive to hold.
    filters = []
    if years:
        filters.append(("year", "in", years))
    if entities:
        filters.append(("entity", "in", entities))
    return pd.read_parquet(NEWS_DIR, filters=filters or None)


# ---------------------------------------------------------------------------
# Fundamentals: SEC EDGAR quarterly zips -> partitioned Parquet
# ---------------------------------------------------------------------------

SUB_COLS = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "accepted"]
NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "value"]


def build_fundamentals(quarters: list[str] | None = None) -> None:
    """Parse each SEC EDGAR quarterly zip (sub.txt + num.txt) into
    data/processed/fundamentals, partitioned by quarter. `available_at` =
    `accepted` timestamp (falls back to `filed` date at midnight if
    `accepted` is missing) -- the moment the filing actually became public,
    which is the whole point of using as-filed SEC data instead of a
    vendor's point-in-time-adjusted feed."""
    if not RAW_SEC_DIR.exists():
        raise FileNotFoundError(f"missing {RAW_SEC_DIR}")

    zips = sorted(RAW_SEC_DIR.glob("*.zip"))
    if quarters:
        wanted = set(quarters)
        zips = [z for z in zips if z.stem in wanted]
    log.info("fundamentals: %d quarterly archives to process", len(zips))

    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)

    for zpath in zips:
        quarter = zpath.stem  # e.g. "2023q4"
        out_path = FUNDAMENTALS_DIR / f"quarter={quarter}"
        if (out_path / "data.parquet").exists():
            log.info("fundamentals: %s already built, skipping", quarter)
            continue

        with zipfile.ZipFile(zpath) as zf:
            with zf.open("sub.txt") as f:
                sub = pd.read_csv(f, sep="\t", usecols=SUB_COLS, dtype={"cik": "int64"},
                                   low_memory=False)
            with zf.open("num.txt") as f:
                num = pd.read_csv(f, sep="\t", usecols=NUM_COLS, low_memory=False)

        df = num.merge(sub, on="adsh", how="inner")

        df["filed"] = pd.to_datetime(df["filed"], format="%Y%m%d", utc=True, errors="coerce")
        accepted = pd.to_datetime(df["accepted"], utc=True, errors="coerce")
        df["accepted"] = accepted
        df["available_at"] = accepted.fillna(df["filed"])
        df["ddate"] = pd.to_datetime(df["ddate"], format="%Y%m%d", utc=True, errors="coerce")
        df["period"] = pd.to_datetime(df["period"], format="%Y%m%d", utc=True, errors="coerce")
        df = df.rename(columns={"cik": "entity"})
        df = df.dropna(subset=["available_at"])

        out_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path / "data.parquet", index=False)
        log.info("fundamentals: %s -> %d facts", quarter, len(df))

    log.info("fundamentals: done")


def load_fundamentals(entities: list[int] | None = None, quarters: list[str] | None = None) -> pd.DataFrame:
    if not FUNDAMENTALS_DIR.exists():
        raise FileNotFoundError(f"no processed fundamentals at {FUNDAMENTALS_DIR}, run build-fundamentals first")
    paths = sorted(FUNDAMENTALS_DIR.glob("quarter=*/data.parquet"))
    if quarters:
        wanted = set(quarters)
        paths = [p for p in paths if p.parent.name.split("=", 1)[1] in wanted]
    if not paths:
        return pd.DataFrame()
    # Read files individually rather than as one pyarrow dataset: the 69
    # quarterly files were written independently and don't have identical
    # inferred schemas (a column that's all-null in one quarter infers as
    # `null` type there but `string` elsewhere), which breaks pyarrow's
    # cross-file schema unification when reading the directory as a whole.
    # Per-file reads still push the entity filter down to row-group pruning.
    filters = [("entity", "in", entities)] if entities else None
    return pd.concat([pd.read_parquet(p, filters=filters) for p in paths], ignore_index=True)


# ---------------------------------------------------------------------------
# Intraday bars, quotes, corporate actions: Alpaca Market Data API
# ---------------------------------------------------------------------------
# FNSPID's daily bars (build_prices, above) can't support the Execution
# Replay phase: order sequencing and fill mechanics need sub-daily price
# movement, and neither FNSPID nor SEC EDGAR carry bid/ask quotes at all.
# Alpaca's Market Data API fills both gaps -- but only from ~2016 onward.
# Verified empirically against this account's SIP (full consolidated tape,
# not just IEX) feed: 1-minute bars and quotes for AAPL are empty on
# 2015-11-02, present by 2016-01-05. Episodes before that boundary --
# including the 2008 GFC window the regime engine already validates
# against -- can't get intraday-fidelity replay from this source and fall
# back to the daily-bar approximation.
#
# These three datasets are deliberately kept separate from build_prices'
# PRICES_DIR: they serve the Execution Replay phase (fills, spreads, fees),
# not the agent's own decision-time snapshot, which is what snapshot_as_of
# bundles for the daily-granularity PiT view.

ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_HISTORY_START = pd.Timestamp("2016-01-01", tz="UTC")  # empirical floor, see above

INTRADAY_DIR = PROCESSED_DIR / "intraday_bars"
QUOTES_DIR = PROCESSED_DIR / "quotes"
CORP_ACTIONS_DIR = PROCESSED_DIR / "corporate_actions"

_ALPACA_TIMEFRAME_MINUTES = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60}
CORP_ACTION_TYPES = "forward_split,reverse_split,cash_dividend"


def _alpaca_headers() -> dict:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set -- put them in .env")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca_get_all_pages(path: str, params: dict) -> list[dict]:
    """GET one Alpaca Market Data endpoint, following `next_page_token`
    until exhausted -- bars/quotes paginate at up to 10k records/page, and a
    multi-year pull needs many pages, so this is the one place pagination
    logic lives rather than repeating it per dataset."""
    import time

    import requests

    headers = _alpaca_headers()
    pages = []
    page_token = None
    while True:
        req_params = dict(params)
        if page_token:
            req_params["page_token"] = page_token
        resp = requests.get(f"{ALPACA_DATA_URL}{path}", headers=headers, params=req_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pages.append(data)
        page_token = data.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.05)  # stay well under the per-minute rate limit
    return pages


def build_intraday_bars(tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp,
                         timeframe: str = "1Min", feed: str = "sip") -> None:
    """Pull historical OHLCV bars from Alpaca for `tickers` into
    data/processed/intraday_bars, partitioned by ticker. `available_at` =
    bar start time + one bar interval -- a bar isn't knowable until the
    interval it covers has actually elapsed, the same reasoning build_prices
    uses for its market-close offset on daily bars, just at bar
    granularity instead of a full trading day.

    Coverage is empirically bounded at ALPACA_HISTORY_START (~2016);
    requesting an earlier start just returns no rows for that stretch, it
    doesn't raise -- callers evaluating pre-2016 episodes need to know that
    silently-empty is expected, not a bug."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    bar_len = pd.Timedelta(minutes=_ALPACA_TIMEFRAME_MINUTES[timeframe])

    for ticker in tickers:
        pages = _alpaca_get_all_pages(
            f"/v2/stocks/{ticker}/bars",
            {"timeframe": timeframe, "start": start.isoformat(), "end": end.isoformat(),
             "feed": feed, "limit": 10000},
        )
        bars = [b for page in pages for b in (page.get("bars") or [])]
        if not bars:
            log.warning("intraday bars: no data for %s in requested range", ticker)
            continue
        df = pd.DataFrame(bars).rename(columns={
            "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "n": "trade_count", "vw": "vwap",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        df["available_at"] = df["timestamp"] + bar_len
        df["entity"] = ticker

        out_dir = INTRADAY_DIR / f"entity={ticker}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "data.parquet", index=False)
        log.info("intraday bars: %s -> %d bars", ticker, len(df))


def load_intraday_bars(entities: list[str] | None = None) -> pd.DataFrame:
    if not INTRADAY_DIR.exists():
        raise FileNotFoundError(f"no processed intraday bars at {INTRADAY_DIR}, run build-intraday-bars first")
    if entities:
        paths = [INTRADAY_DIR / f"entity={e}" / "data.parquet" for e in entities]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(INTRADAY_DIR.glob("entity=*/data.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_quotes(tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp,
                  feed: str = "sip") -> None:
    """Pull historical NBBO quotes (bid/ask) from Alpaca for `tickers` into
    data/processed/quotes. `available_at` = the quote's own timestamp --
    unlike a bar, a quote is knowable the instant it prints, no interval
    offset needed (same reasoning as news' published_at).

    This is the one dataset in the whole pipeline that carries an actual
    bid/ask spread -- neither FNSPID nor SEC EDGAR have one at all. Tick
    quote volume is large for liquid names (tens of thousands of quotes per
    ticker per day); pulling a wide ticker universe over years of history
    at full tick resolution is a real storage/time cost, not just an API
    call count -- worth resampling to a coarser snapshot (e.g. quote-at-each-
    bar-close) before this feeds the replay engine, rather than keeping raw
    ticks by default."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    QUOTES_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        pages = _alpaca_get_all_pages(
            f"/v2/stocks/{ticker}/quotes",
            {"start": start.isoformat(), "end": end.isoformat(), "feed": feed, "limit": 10000},
        )
        quotes = [q for page in pages for q in (page.get("quotes") or [])]
        if not quotes:
            log.warning("quotes: no data for %s in requested range", ticker)
            continue
        df = pd.DataFrame(quotes).rename(columns={
            "t": "timestamp", "bp": "bid_price", "bs": "bid_size",
            "ap": "ask_price", "as": "ask_size", "bx": "bid_exchange", "ax": "ask_exchange",
        })
        # Quote timestamps come back at inconsistent sub-second precision --
        # some carry nanosecond fractions, some don't -- which breaks
        # pandas' single-format auto-detection; ISO8601 mode handles both.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        df["available_at"] = df["timestamp"]
        df["entity"] = ticker
        df["spread"] = df["ask_price"] - df["bid_price"]
        df = df[["entity", "timestamp", "available_at", "bid_price", "bid_size",
                 "ask_price", "ask_size", "spread", "bid_exchange", "ask_exchange"]]

        out_dir = QUOTES_DIR / f"entity={ticker}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "data.parquet", index=False)
        log.info("quotes: %s -> %d quotes", ticker, len(df))


def load_quotes(entities: list[str] | None = None) -> pd.DataFrame:
    if not QUOTES_DIR.exists():
        raise FileNotFoundError(f"no processed quotes at {QUOTES_DIR}, run build-quotes first")
    if entities:
        paths = [QUOTES_DIR / f"entity={e}" / "data.parquet" for e in entities]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(QUOTES_DIR.glob("entity=*/data.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_corporate_actions(tickers: list[str], start: str | pd.Timestamp,
                             end: str | pd.Timestamp) -> None:
    """Pull splits and cash dividends from Alpaca's corporate-actions API
    into data/processed/corporate_actions. `available_at` = ex_date at
    midnight UTC -- the API doesn't expose a declaration/announcement date,
    only ex/record/payable/process dates, so ex_date is the latest-known-
    safe proxy: the real announcement happened at or before ex_date, so
    using ex_date as available_at can only be overly conservative (treating
    something as not-yet-knowable when it technically was, slightly
    earlier), never leak something too early. Same safe-direction tradeoff
    build_fundamentals makes falling back from `accepted` to `filed`.

    Verified against AAPL's real 2020-08-31 4:1 split and its quarterly
    dividends for 2020 -- both come back with correct real-world dates and
    ratios, not FNSPID's retroactively-adjusted price series."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    CORP_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pages = _alpaca_get_all_pages(
        "/v1beta1/corporate-actions",
        {"symbols": ",".join(tickers), "start": start.date().isoformat(),
         "end": end.date().isoformat(), "types": CORP_ACTION_TYPES, "limit": 1000},
    )
    rows = []
    for page in pages:
        actions = page.get("corporate_actions", {})
        for div in actions.get("cash_dividends") or []:
            rows.append({"entity": div["symbol"], "action_type": "cash_dividend",
                         "ex_date": div["ex_date"], "rate": div["rate"], "split_ratio": None})
        for split in (actions.get("forward_splits") or []) + (actions.get("reverse_splits") or []):
            rows.append({"entity": split["symbol"], "action_type": "split",
                         "ex_date": split["ex_date"], "rate": None,
                         "split_ratio": split["new_rate"] / split["old_rate"]})
    if not rows:
        log.warning("corporate actions: no data for requested tickers/range")
        return

    df = pd.DataFrame(rows)
    df["ex_date"] = pd.to_datetime(df["ex_date"], utc=True)
    df["available_at"] = df["ex_date"]
    df.to_parquet(CORP_ACTIONS_DIR / "data.parquet", index=False)
    log.info("corporate actions: %d actions written (%d dividends, %d splits)", len(df),
              int((df["action_type"] == "cash_dividend").sum()), int((df["action_type"] == "split").sum()))


def load_corporate_actions(entities: list[str] | None = None) -> pd.DataFrame:
    path = CORP_ACTIONS_DIR / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no processed corporate actions at {CORP_ACTIONS_DIR}, run build-corporate-actions first")
    df = pd.read_parquet(path)
    if entities:
        df = df[df["entity"].isin(entities)]
    return df


# ---------------------------------------------------------------------------
# Entity resolution: SEC CIK <-> ticker crosswalk
# ---------------------------------------------------------------------------
# Prices/news are keyed by ticker; SEC fundamentals are keyed by CIK (SEC
# has no ticker column at all). Without this crosswalk the three datasets
# can't be joined for the same company.

ENTITY_MAP_PATH = PROCESSED_DIR / "entity_map.parquet"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def build_entity_map() -> None:
    import json
    import subprocess

    # urllib hits a broken local CA bundle on this machine's python.org
    # install (SSLCertVerificationError); curl already works fine here and
    # is a hard dependency of this whole pipeline anyway, so just shell out
    # rather than add a certifi dependency for one JSON fetch.
    result = subprocess.run(
        ["curl", "-s", "-A", "research-agent test@example.com", SEC_TICKERS_URL],
        capture_output=True, check=True,
    )
    data = json.loads(result.stdout)

    rows = [{"ticker": v["ticker"], "cik": int(v["cik_str"]), "name": v["title"]}
            for v in data.values()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ENTITY_MAP_PATH, index=False)
    log.info("entity map: %d ticker<->CIK mappings written", len(df))


def load_entity_map() -> pd.DataFrame:
    if not ENTITY_MAP_PATH.exists():
        raise FileNotFoundError(f"no entity map at {ENTITY_MAP_PATH}, run build-entity-map first")
    return pd.read_parquet(ENTITY_MAP_PATH)


def ticker_to_cik(ticker: str) -> int | None:
    match = load_entity_map()
    match = match[match["ticker"] == ticker.upper()]
    return int(match.iloc[0]["cik"]) if not match.empty else None


# ---------------------------------------------------------------------------
# Derived financial features
# ---------------------------------------------------------------------------
# The proposal names four PiT input types: prices, fundamentals, news, and
# "derived financial features" -- computed signals, not raw records. These
# functions take already PiT-filtered data (whatever a caller got back from
# prices_as_of/fundamentals_as_of) and compute on top of it, so they inherit
# the same temporal-integrity guarantee as their inputs rather than needing
# a separate enforcement path.

PRICE_FEATURE_WINDOWS = (5, 21, 63)  # trading days: ~1wk, ~1mo, ~1qtr

# (feature_name, numerator us-gaap tag, denominator us-gaap tag). Kept small
# and to a handful of tags reported near-universally by 10-K/10-Q filers.
# XBRL tag usage isn't perfectly standardized -- smaller filers sometimes
# use custom extension tags instead of the standard us-gaap ones -- so a
# missing ratio means the standard tag wasn't found as-of this cutoff, not
# that the pipeline failed.
FUNDAMENTAL_RATIO_TAGS = {
    "leverage_ratio": ("Liabilities", "Assets"),
    "current_ratio": ("AssetsCurrent", "LiabilitiesCurrent"),
    "return_on_assets": ("NetIncomeLoss", "Assets"),
}


def compute_price_features(prices: pd.DataFrame, windows: tuple[int, ...] = PRICE_FEATURE_WINDOWS,
                            vol_window: int = 21) -> dict:
    """Trailing return over each window in `windows`, plus trailing realized
    volatility -- both computed only from rows already in `prices`, so a
    PiT-filtered input keeps this PiT-safe with no extra checks needed."""
    if prices.empty:
        return {f"return_{w}d": None for w in windows} | {"realized_vol": None}
    closes = prices.sort_values("date")["adj_close"]
    out = {}
    for w in windows:
        out[f"return_{w}d"] = float(closes.iloc[-1] / closes.iloc[-1 - w] - 1) if len(closes) > w else None
    daily_ret = closes.pct_change()
    out["realized_vol"] = float(daily_ret.tail(vol_window).std()) if daily_ret.notna().sum() >= vol_window else None
    return out


def compute_fundamental_ratios(fundamentals: pd.DataFrame,
                                ratio_tags: dict = FUNDAMENTAL_RATIO_TAGS) -> dict:
    """Best-effort ratios from the most recently reported value of each
    required us-gaap tag in `fundamentals` (already PiT-filtered/deduped by
    the caller). Returns None for a ratio whose tags aren't present rather
    than raising, since incomplete XBRL tag coverage is expected, not
    exceptional."""
    if fundamentals.empty:
        return {name: None for name in ratio_tags}
    latest = (fundamentals.sort_values("ddate")
              .drop_duplicates(subset=["tag"], keep="last")
              .set_index("tag")["value"])
    out = {}
    for name, (num_tag, den_tag) in ratio_tags.items():
        if num_tag in latest.index and den_tag in latest.index and latest[den_tag] != 0:
            out[name] = float(latest[num_tag] / latest[den_tag])
        else:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Point-in-time access layer
# ---------------------------------------------------------------------------

class TemporalIntegrityError(RuntimeError):
    """Raised when a query would leak data from at or after its cutoff."""


@dataclass
class PointInTimeStore:
    """Single access point for prices/news/fundamentals that enforces
    `available_at < as_of` on every read and logs the access, per the
    proposal's requirement that the temporal wall live in the data-access
    layer itself rather than be checked after the fact."""

    access_log: list[dict] = field(default_factory=list)

    def _enforce_and_log(self, dataset: str, as_of: pd.Timestamp, df: pd.DataFrame) -> pd.DataFrame:
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        before = len(df)
        safe = df if "available_at" not in df.columns else df[df["available_at"] < as_of]
        blocked = before - len(safe)
        self.access_log.append({
            "dataset": dataset,
            "as_of": as_of,
            "rows_returned": len(safe),
            "rows_blocked": blocked,
        })
        if blocked:
            log.warning(
                "temporal integrity: blocked %d/%d %s rows at/after as_of=%s",
                blocked, before, dataset, as_of,
            )
        return safe

    def prices_as_of(self, as_of: pd.Timestamp, entities: list[str] | None = None) -> pd.DataFrame:
        df = load_prices(entities)
        return self._enforce_and_log("prices", as_of, df)

    def news_as_of(self, as_of: pd.Timestamp, entities: list[str] | None = None,
                    years: list[int] | None = None) -> pd.DataFrame:
        df = load_news(entities, years)
        return self._enforce_and_log("news", as_of, df)

    def fundamentals_as_of(self, as_of: pd.Timestamp, entities: list[int] | None = None,
                            quarters: list[str] | None = None) -> pd.DataFrame:
        df = load_fundamentals(entities, quarters)
        return self._enforce_and_log("fundamentals", as_of, df)

    def bars_as_of(self, as_of: pd.Timestamp, entities: list[str] | None = None) -> pd.DataFrame:
        """Intraday bars (Alpaca) for the Execution Replay phase -- same
        enforcement as prices_as_of, just at bar granularity. Empty for any
        `as_of` before ALPACA_HISTORY_START regardless of cutoff, since the
        underlying data doesn't exist that far back (see build_intraday_bars)."""
        df = load_intraday_bars(entities)
        return self._enforce_and_log("intraday_bars", as_of, df)

    def quotes_as_of(self, as_of: pd.Timestamp, entities: list[str] | None = None) -> pd.DataFrame:
        """NBBO bid/ask quotes (Alpaca) for spread modeling in the Execution
        Replay phase. Same enforcement and same ~2016 coverage floor as
        bars_as_of."""
        df = load_quotes(entities)
        return self._enforce_and_log("quotes", as_of, df)

    def corporate_actions_as_of(self, as_of: pd.Timestamp, entities: list[str] | None = None) -> pd.DataFrame:
        """Splits/dividends (Alpaca) needed to correctly track position size
        and cash across corporate events during replay -- see
        build_corporate_actions for why available_at is pinned to ex_date."""
        df = load_corporate_actions(entities)
        return self._enforce_and_log("corporate_actions", as_of, df)

    def snapshot_as_of(self, ticker: str, as_of: pd.Timestamp,
                        price_lookback: int | None = None,
                        news_lookback: pd.Timedelta | None = None) -> dict:
        """Point-in-time-consistent bundle across all three sources for one
        ticker, all filtered to the same `as_of` cutoff -- this is the
        multi-source view an agent actually consumes. The raw datasets stay
        independently stored; this joins them at query time via the CIK
        crosswalk."""
        prices = self.prices_as_of(as_of, entities=[ticker])
        if price_lookback and not prices.empty:
            prices = prices.sort_values("date").tail(price_lookback)

        news = self.news_as_of(as_of, entities=[ticker])
        if news_lookback and not news.empty:
            news = news[news["published_at"] >= as_of - news_lookback]

        cik = ticker_to_cik(ticker)
        if cik is None:
            fundamentals = pd.DataFrame()
        else:
            fundamentals = self.fundamentals_as_of(as_of, entities=[cik])
            if not fundamentals.empty:
                # A fact can be refiled/amended after its original filing.
                # Keep only the most-recently-known value per (tag, ddate)
                # among rows already at/before as_of, so a restatement that
                # genuinely happened before as_of is reflected, but nothing
                # from after as_of ever leaks in.
                fundamentals = (fundamentals.sort_values("accepted")
                                 .drop_duplicates(subset=["tag", "ddate"], keep="last"))

        derived_features = {
            **compute_price_features(prices),
            **compute_fundamental_ratios(fundamentals),
        }

        return {"ticker": ticker, "cik": cik, "as_of": as_of,
                "prices": prices, "news": news, "fundamentals": fundamentals,
                "derived_features": derived_features}

    def universe_snapshot_as_of(self, as_of: pd.Timestamp, cash: float, positions: dict[str, float],
                                 tickers: list[str] | None = None,
                                 price_lookback: int | None = 10,
                                 news_lookback: pd.Timedelta | None = pd.Timedelta(days=3)) -> dict:
        """Portfolio-level sibling of snapshot_as_of: one bundle covering the
        whole tradeable universe at once, not one ticker. Needed because the
        benchmark's own metrics (Sharpe, Sortino, max drawdown, the
        Attribution regression) are computed on the *combined* portfolio's
        return series, not per-ticker -- a design where the agent only ever
        sees one ticker at a time structurally can't make a sensible
        portfolio-level risk decision, e.g. it could load up on five
        correlated names without any one of those five decisions ever
        having visibility into the other four.

        `cash` and `positions` (a {ticker: shares_held} dict) are required,
        not defaulted -- this class has no way to know a running portfolio's
        state on its own; it's dynamic state that changes with every fill,
        owned by whatever runs the episode loop (the Replay Engine side),
        and must be passed in explicitly rather than assumed.

        `tickers` defaults to EVALUATION_TICKERS, not
        EVALUATION_TICKERS + BENCHMARK_TICKERS -- benchmarks are passive
        comparison points, never something the agent is asked to trade.

        price_lookback/news_lookback default much tighter here (10 rows /
        3 days) than snapshot_as_of's own default (unlimited) -- bundling
        20 tickers' worth of full history into one prompt is a real token
        cost multiplier that a single-ticker call doesn't have to worry
        about."""
        if tickers is None:
            tickers = EVALUATION_TICKERS
        return {
            "as_of": as_of,
            "cash": cash,
            "positions": positions,
            "tickers": {t: self.snapshot_as_of(t, as_of, price_lookback=price_lookback,
                                                news_lookback=news_lookback)
                        for t in tickers},
        }

    def check(self, dataset: str, as_of: pd.Timestamp, requested_timestamps: pd.Series) -> tuple[bool, int]:
        """Explicit boundary check for a candidate access request (e.g. a tool
        call or inter-agent message payload timestamp), independent of the
        *_as_of loaders above. Returns (allowed, n_violations)."""
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        violations = int((requested_timestamps >= as_of).sum())
        allowed = violations == 0
        self.access_log.append({
            "dataset": dataset, "as_of": as_of,
            "rows_returned": len(requested_timestamps) - violations,
            "rows_blocked": violations,
        })
        if not allowed:
            log.warning("temporal integrity: %d/%d requested %s timestamps at/after as_of=%s",
                        violations, len(requested_timestamps), dataset, as_of)
        return allowed, violations

    def save_access_log(self, path: Path) -> None:
        """Persist the access log so temporal-integrity failures are an
        actual durable record, not just in-memory state that vanishes when
        the process exits -- matches the proposal's "logged, and recorded
        as a temporal integrity failure" requirement."""
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.access_log).to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Regime labeling: rolling, out-of-sample HMM
# ---------------------------------------------------------------------------

def _fit_best_hmm(train: np.ndarray, n_states: int, n_restarts: int, n_iter: int, seed0: int):
    """hmmlearn's single-init EM routinely lands in a degenerate local
    optimum on noisy daily-return-scale data (silent non-convergence, not
    just a cosmetic warning -- verified empirically: it materially changes
    which state absorbs known crisis days). Multiple random restarts, kept
    if they score better on training log-likelihood, fixes that."""
    import warnings

    from hmmlearn import hmm

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    best = None
    for k in range(n_restarts):
        try:
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                                     n_iter=n_iter, tol=1e-4, random_state=seed0 + k,
                                     min_covar=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train)
            score = model.score(train)
            if not np.isfinite(score):
                continue
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, model)
    if best is None:
        raise RuntimeError("all HMM restarts failed to converge to a finite likelihood")
    return best[1]


def label_regimes(
    returns: pd.Series,
    n_states: int = 3,
    min_train: int = 252,
    refit_every: int = 63,
    window: int | None = 252 * 3,
    vol_window: int = 21,
    n_restarts: int = 8,
    n_iter: int = 300,
    random_state: int = 0,
) -> pd.DataFrame:
    """Walk-forward regime labels for a daily return series.

    Fits a 2-feature Gaussian HMM (daily return, trailing `vol_window`-day
    realized volatility) rather than return alone. Raw return alone was
    tried first and empirically failed the obvious sanity check -- it missed
    the 2008 and 2020 crashes almost entirely, because a crisis is defined
    as much by *volatility clustering* over many days as by any single
    day's return, and a 1-feature model has too little to separate on.
    Adding realized vol as a second feature fixed that (verified against
    2008, 2011, 2018Q4, 2020, 2022 and three calm bull stretches).

    At each refit point t (every `refit_every` trading days, starting once
    `min_train` observations exist), fits on data strictly before t (the
    trailing `window` observations, or all history if `window` is None),
    then decodes regimes for the segment up to the next refit point using
    only that model. No observation after its own date is ever used to fit
    the model that labels it.

    Returns a DataFrame indexed like `returns` (minus the `vol_window`-day
    warmup) with columns:
      state       raw HMM state id (not comparable across refits)
      regime      one of "trending", "mean_reverting", "crisis"
    """
    returns = returns.dropna()
    vol = returns.rolling(vol_window).std()
    feat = pd.DataFrame({"ret": returns, "vol": vol}).dropna()
    idx = feat.index
    n = len(feat)
    states = np.full(n, -1, dtype=int)
    regimes = np.full(n, "", dtype=object)

    if n < min_train:
        raise ValueError(f"need at least {min_train} observations, got {n}")

    t = min_train
    while t < n:
        train_start = max(0, t - window) if window else 0
        train = feat.iloc[train_start:t][["ret", "vol"]].to_numpy()

        model = _fit_best_hmm(train, n_states, n_restarts, n_iter, random_state)

        seg_end = min(t + refit_every, n)
        seg = feat.iloc[t:seg_end][["ret", "vol"]].to_numpy()
        seg_states = model.predict(seg)
        states[t:seg_end] = seg_states

        means = model.means_  # (n_states, 2): columns are [ret, vol]
        vol_means, ret_means = means[:, 1], means[:, 0]
        crisis_state = int(np.argmax(vol_means))
        remaining = [s for s in range(n_states) if s != crisis_state]
        trending_state = max(remaining, key=lambda s: abs(ret_means[s]))
        mean_revert_state = [s for s in remaining if s != trending_state][0]
        label_map = {crisis_state: "crisis", trending_state: "trending",
                     mean_revert_state: "mean_reverting"}
        regimes[t:seg_end] = [label_map[s] for s in seg_states]

        t = seg_end

    out = pd.DataFrame({"state": states, "regime": regimes}, index=idx)
    return out[out["state"] >= 0]


# ---------------------------------------------------------------------------
# Episode window generator
# ---------------------------------------------------------------------------

def generate_episode_windows(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    window_len: pd.Timedelta,
    purge_gap: pd.Timedelta,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping [episode_start, episode_end) windows spanning
    [start, end], each `window_len` long and separated by `purge_gap` so no
    state/memory can leak between consecutive evaluation episodes."""
    # UTC, to match every other timestamp in this pipeline (available_at
    # columns, regime label index) -- naive timestamps here would silently
    # fail to compare against them downstream. Accepts either a naive input
    # (localized here) or an already-UTC-aware one (e.g. the EPISODE_RANGE_*
    # constants above) without raising -- pd.Timestamp(aware_ts, tz="UTC")
    # itself raises, so this can't just be the one-liner it used to be.
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    windows = []
    cursor = start
    while cursor + window_len <= end:
        windows.append((cursor, cursor + window_len))
        cursor = cursor + window_len + purge_gap
    return windows


def assign_episode_regimes(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each episode window a single market Regime Label, per the
    proposal's requirement that a regime is assigned to an episode prior to
    execution -- not derived afterward from what the agent saw. Uses the
    majority regime observed within the window from the walk-forward
    `label_regimes` output (which itself never looks past its own date, so
    this stays leakage-free too).

    Deliberate decision: this table's output is never passed into
    snapshot_as_of, universe_snapshot_as_of, or anything else agent-facing.
    "Assigned prior to execution" describes when the label gets computed
    (before the episode runs, so the labeling process itself can't cheat by
    looking at the outcome), not a promise that the agent is shown it. The
    whole point of Phase 3's regime-robustness check is whether the agent's
    real behavior degrades in a crisis it wasn't told about -- handing the
    agent an explicit "you are in: CRISIS" tag would turn that into a test
    of whether the agent can read a label, not whether its edge survives a
    real crash. The agent isn't left blind either way: realized_vol (the
    same raw volatility signal label_regimes itself trains on) is already
    in every derived_features dict -- raw signal in, the HMM's conclusion
    withheld. This table stays a Phase 3 bookkeeping input only."""
    rows = []
    for start, end in windows:
        seg = regimes.loc[(regimes.index >= start) & (regimes.index < end)]
        if seg.empty:
            rows.append({"episode_start": start, "episode_end": end,
                         "regime": None, "n_days": 0})
            continue
        rows.append({"episode_start": start, "episode_end": end,
                     "regime": seg["regime"].value_counts().idxmax(), "n_days": len(seg)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model training-cutoff registry
# ---------------------------------------------------------------------------
# The data-access layer can block environmental leakage (a query for future
# prices/news/fundamentals), but it can't fix *parametric* look-ahead bias:
# an LLM whose pretraining data already covers the evaluation window may
# "know" what happened independent of anything the store blocks. The
# proposal's mitigation for that is a documented recommendation -- only
# evaluate models whose training cutoff precedes the episode window.
#
# Deliberately left unseeded: exact training-cutoff dates are published and
# maintained by each model vendor and change with every release. Hardcoding
# a table of them here from memory risks being silently wrong, which would
# defeat the entire point of this check. Populate MODEL_TRAINING_CUTOFFS
# yourself from each model's official model card / system card before
# relying on it.

MODEL_TRAINING_CUTOFFS: dict[str, str] = {
    # "model-name": "YYYY-MM-DD",  # source: the model's official model card
}


def check_model_cutoff(model_name: str, episode_start: pd.Timestamp) -> bool:
    """True if `model_name`'s registered training cutoff precedes
    `episode_start` (safe to evaluate on that episode), False if it doesn't
    (flag as a parametric look-ahead risk). Raises KeyError for an
    unregistered model -- unregistered must mean unverified, not silently
    assumed safe."""
    if model_name not in MODEL_TRAINING_CUTOFFS:
        raise KeyError(
            f"{model_name!r} has no registered training cutoff. Add one to "
            "MODEL_TRAINING_CUTOFFS from the model's official model card "
            "before evaluating it -- this check can't tell you anything "
            "about an unregistered model."
        )
    cutoff = pd.Timestamp(MODEL_TRAINING_CUTOFFS[model_name], tz="UTC")
    if episode_start.tzinfo is None:
        episode_start = episode_start.tz_localize("UTC")
    safe = cutoff < episode_start
    if not safe:
        log.warning(
            "parametric look-ahead risk: %s training cutoff %s does not precede "
            "episode start %s", model_name, cutoff.date(), episode_start.date(),
        )
    return safe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-prices")
    p.add_argument("--limit", type=int, default=None, help="only process first N tickers")

    p = sub.add_parser("build-news")
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--with-text", action="store_true")

    p = sub.add_parser("build-fundamentals")
    p.add_argument("--quarters", nargs="*", default=None, help="e.g. 2023q4 2024q1")

    sub.add_parser("build-entity-map")
    sub.add_parser("build-news-entity-backfill")
    sub.add_parser("build-all")

    p = sub.add_parser("build-intraday-bars", help="Alpaca minute/hour bars, ~2016+ only")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", required=True, help="e.g. 2020-01-01")
    p.add_argument("--end", required=True)
    p.add_argument("--timeframe", default="1Min", choices=sorted(_ALPACA_TIMEFRAME_MINUTES))

    p = sub.add_parser("build-quotes", help="Alpaca NBBO bid/ask quotes, ~2016+ only")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)

    p = sub.add_parser("build-corporate-actions", help="Alpaca splits/dividends")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)

    args = parser.parse_args()

    if args.command == "build-prices":
        build_prices(limit_tickers=args.limit)
    elif args.command == "build-news":
        build_news(chunksize=args.chunksize, limit_rows=args.limit_rows, with_text=args.with_text)
    elif args.command == "build-fundamentals":
        build_fundamentals(quarters=args.quarters)
    elif args.command == "build-entity-map":
        build_entity_map()
    elif args.command == "build-news-entity-backfill":
        build_news_entity_backfill()
    elif args.command == "build-intraday-bars":
        build_intraday_bars(args.tickers, args.start, args.end, timeframe=args.timeframe)
    elif args.command == "build-quotes":
        build_quotes(args.tickers, args.start, args.end)
    elif args.command == "build-corporate-actions":
        build_corporate_actions(args.tickers, args.start, args.end)
    elif args.command == "build-all":
        build_prices()
        build_fundamentals()
        build_news()
        build_entity_map()
        build_news_entity_backfill()


if __name__ == "__main__":
    main()
