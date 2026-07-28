"""Data ingestion with on-disk caching.

Fixes carried over from the notebook:

* **Pinned ``end_date``** (bug #13). ``yf.download(start=...)`` with no ``end`` meant the
  dataset grew every day, so ratio-based split boundaries moved and no published metric
  could be reproduced.
* **``auto_adjust=True``** (bug #6). The notebook set ``auto_adjust=False``, took
  ``price = Adj Close``, but left ``High``/``Low`` raw. True range then differenced two
  price scales; a split or bonus issue injected a large spurious spike into ``atr_pct``,
  which fed the scaler and the RL observation.
* **Per-ticker forward fill** (bug #18). The notebook called ``data[num_cols].ffill()`` on
  a frame sorted by ``["ticker", "Date"]``, so the last row of one ticker filled into the
  first row of the next -- cross-ticker contamination in a pipeline that advertises
  per-ticker isolation.
* **Unused macro/sector downloads dropped** (bug #22). ``CL=F``, ``USDINR=X``, ``^TNX``,
  ``^CNXIT``, ``^NSEBANK`` were fetched and merged but appear in no feature set.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- cache


def _cache_key(kind: str, symbols: Sequence[str], cfg: DataConfig) -> str:
    payload = "|".join(
        [
            kind,
            ",".join(sorted(symbols)),
            str(cfg.start_date),
            str(cfg.end_date),
            str(cfg.auto_adjust),
        ]
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return f"{kind}__{cfg.start_date}__{cfg.end_date or 'live'}__{digest}"


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def _cache_path(key: str, cache_dir: Path) -> Path:
    """Parquet when an engine is installed, CSV otherwise.

    The cache is what makes a run reproducible, so it degrades rather than disabling
    itself when pyarrow is absent. CSV round-trips dates as strings, which is why the
    reader re-parses ``Date`` explicitly.
    """
    suffix = ".parquet" if _has_parquet_engine() else ".csv"
    return cache_dir / f"{key}{suffix}"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - corrupt cache
        warnings.warn(f"Cache read failed for {path.name}: {exc}")
        return None
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Cache write failed ({exc}); this run will not be reproducible.")


# ------------------------------------------------------------------- yfinance I/O


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _download_one(symbol: str, cfg: DataConfig) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests can run without network deps

    raw = yf.download(
        symbol,
        start=cfg.start_date,
        end=cfg.end_date,
        progress=False,
        auto_adjust=cfg.auto_adjust,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = _flatten_columns(raw).reset_index()
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw


def download_ohlcv(cfg: DataConfig, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Download adjusted OHLCV for ``tickers`` in long format.

    Returns columns ``[Date, Open, High, Low, Close, Volume, price, ticker]`` sorted by
    ``(ticker, Date)``. With ``auto_adjust=True`` every OHLC field is on the same
    adjusted scale, so ``price`` is simply ``Close``.
    """
    symbols = list(tickers) if tickers is not None else list(cfg.tickers)

    if cfg.use_cache:
        key = _cache_key("ohlcv", symbols, cfg)
        cached = _read_cache(_cache_path(key, cfg.cache_dir))
        if cached is not None:
            return cached

    frames = []
    for symbol in symbols:
        df = _download_one(symbol, cfg)
        if df.empty:
            warnings.warn(f"No data returned for {symbol}; excluded from the panel.")
            continue
        keep = ["Date"] + [c for c in OHLCV_COLUMNS if c in df.columns]
        df = df[keep].copy()
        # auto_adjust=True already folds dividends/splits into Close.
        df["price"] = df["Close"] if "Close" in df.columns else np.nan
        df["ticker"] = symbol
        frames.append(df)

    if not frames:
        raise RuntimeError("No tickers returned data; cannot build a panel.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "Date"]).reset_index(drop=True)

    if cfg.use_cache:
        _write_cache(out, _cache_path(_cache_key("ohlcv", symbols, cfg), cfg.cache_dir))
    return out


def download_series(symbol: str, column_name: str, cfg: DataConfig) -> pd.DataFrame:
    """Download a single auxiliary series as ``[Date, column_name]``."""
    if cfg.use_cache:
        key = _cache_key(f"series_{column_name}", [symbol], cfg)
        cached = _read_cache(_cache_path(key, cfg.cache_dir))
        if cached is not None:
            return cached

    df = _download_one(symbol, cfg)
    if df.empty:
        warnings.warn(f"No data for auxiliary series {symbol}.")
        return pd.DataFrame(columns=["Date", column_name])

    price_col = "Close" if "Close" in df.columns else df.columns[1]
    out = df[["Date", price_col]].rename(columns={price_col: column_name})

    if cfg.use_cache:
        _write_cache(out, _cache_path(_cache_key(f"series_{column_name}", [symbol], cfg), cfg.cache_dir))
    return out


# ------------------------------------------------------------------ panel assembly


def ffill_by_ticker(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Forward fill numeric columns **within each ticker**.

    The notebook's global ``ffill`` on a ticker-sorted frame leaked the last row of one
    ticker into the first row of the next (bug #18).
    """
    out = df.copy()
    cols = list(columns) if columns is not None else list(
        out.select_dtypes(include="number").columns
    )
    cols = [c for c in cols if c in out.columns]
    if not cols:
        return out
    out[cols] = out.groupby("ticker", sort=False)[cols].ffill()
    return out


def build_panel(cfg: DataConfig) -> pd.DataFrame:
    """Assemble the long-format panel: OHLCV + India VIX + benchmark return.

    Cross-sectional dispersion is computed here because it is a genuine cross-ticker
    aggregate (std of 5-day returns across the universe on each date) and therefore
    cannot be produced inside a per-ticker groupby.
    """
    prices = download_ohlcv(cfg)

    vix = download_series(cfg.vix_ticker, "india_vix", cfg)
    panel = prices.merge(vix, on="Date", how="left")

    bench = download_series(cfg.benchmark_ticker, "benchmark_close", cfg)
    if not bench.empty:
        bench = bench.sort_values("Date").reset_index(drop=True)
        bench["benchmark_return"] = bench["benchmark_close"].pct_change()
        panel = panel.merge(bench[["Date", "benchmark_close", "benchmark_return"]], on="Date", how="left")
    else:
        panel["benchmark_close"] = np.nan
        panel["benchmark_return"] = np.nan

    for symbol in cfg.macro_assets:
        col = symbol.replace("=", "_").replace("^", "")
        panel = panel.merge(download_series(symbol, col, cfg), on="Date", how="left")

    for symbol in cfg.sector_indices:
        col = symbol.replace("^", "") + "_close"
        panel = panel.merge(download_series(symbol, col, cfg), on="Date", how="left")

    panel = panel.sort_values(["ticker", "Date"]).reset_index(drop=True)
    panel = panel.merge(cross_sectional_dispersion(panel), on="Date", how="left")

    # Per-ticker fill for auxiliary series that use a slightly different calendar
    # (India VIX and ^NSEI occasionally differ from the equity calendar).
    panel = ffill_by_ticker(
        panel,
        [c for c in panel.columns if c not in ("Date", "ticker")],
    )
    return panel


def cross_sectional_dispersion(panel: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Std of ``window``-day returns across the universe, per date."""
    tmp = panel[["Date", "ticker", "price"]].copy()
    tmp["ret_w"] = tmp.groupby("ticker", sort=False)["price"].pct_change(window)
    disp = tmp.groupby("Date")["ret_w"].std().rename("cs_dispersion").reset_index()
    return disp


def chronological_split(df: pd.DataFrame, train_ratio: float, valid_ratio: float):
    """Split on **unique trading dates** so all tickers land in the same bands."""
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_dates = set(dates[:train_end])
    valid_dates = set(dates[train_end:valid_end])
    test_dates = set(dates[valid_end:])

    return (
        df[df["Date"].isin(train_dates)].reset_index(drop=True),
        df[df["Date"].isin(valid_dates)].reset_index(drop=True),
        df[df["Date"].isin(test_dates)].reset_index(drop=True),
    )
