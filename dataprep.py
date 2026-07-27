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

Raw sources (already downloaded):
  ~/full_history.zip                    FNSPID prices, one CSV per ticker
  ~/nasdaq_exteral_data.csv             FNSPID news, ~9GB, one row per article
  ~/financial-statement-data-sets/*.zip SEC EDGAR quarterly financial
                                         statement data sets (sub/num/tag/pre)
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
            ticker = Path(member).stem
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

    total_tagged, total_matched, total_rows = 0, 0, 0
    for ydir in year_dirs:
        files = list(ydir.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

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
        log.info("entity backfill: %s -> %d tagged, %d recovered via title match, %d still untagged",
                 ydir.name, int(tagged_mask.sum()), len(matched_idx),
                 len(df) - int(tagged_mask.sum()) - len(matched_idx))

    log.info("entity backfill: done. %d/%d rows tagged (%d original + %d recovered)",
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
        safe = df[df["available_at"] < as_of]
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
    # fail to compare against them downstream.
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
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
    this stays leakage-free too)."""
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
    elif args.command == "build-all":
        build_prices()
        build_fundamentals()
        build_news()
        build_entity_map()
        build_news_entity_backfill()


if __name__ == "__main__":
    main()
