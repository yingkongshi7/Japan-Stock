import argparse
import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
import yfinance as yf


JST = timezone(timedelta(hours=9))

FORWARD_FIELDS = [
    "price_1w",
    "return_1w_pct",
    "topix_return_1w_pct",
    "relative_return_1w_pct",
    "price_1m",
    "return_1m_forward_pct",
    "topix_return_1m_pct",
    "relative_return_1m_pct",
    "price_3m",
    "return_3m_forward_pct",
    "topix_return_3m_pct",
    "relative_return_3m_pct",
    "price_6m",
    "return_6m_forward_pct",
    "topix_return_6m_pct",
    "relative_return_6m_pct",
    "max_gain_1m_pct",
    "max_drawdown_1m_pct",
    "max_gain_3m_pct",
    "max_drawdown_3m_pct",
    "result_label",
]

HORIZONS = {
    "1w": 7,
    "1m": 30,
    "3m": 91,
    "6m": 182,
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fetch_price_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logging.warning("Failed to fetch %s: %s", ticker, exc)
        return None

    if data is None or data.empty:
        logging.warning("No price data for %s", ticker)
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna(subset=["Close"])
    if data.empty:
        return None

    if "Adj Close" in data.columns and data["Adj Close"].notna().any():
        base_close = data["Adj Close"]
    else:
        base_close = data["Close"]

    data["Signal Close"] = normalize_price_series(base_close, ticker)
    return data


def normalize_price_series(price: pd.Series, ticker: str) -> pd.Series:
    normalized = price.astype(float).copy()
    ratios = normalized / normalized.shift(1)
    split_like_ratios = ratios[(ratios > 0) & ((ratios < 0.5) | (ratios > 2.0))]

    for date, ratio in split_like_ratios.items():
        prior_mask = normalized.index < date
        if prior_mask.any():
            normalized.loc[prior_mask] = normalized.loc[prior_mask] * float(ratio)
            logging.warning("Normalized split-like gap for %s on %s by %.4f", ticker, date, float(ratio))

    return normalized


def read_signal_log(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_signal_log(path: str, columns: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_signal_date(row: Dict[str, str]) -> Optional[datetime]:
    value = row.get("signal_date", "")
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=JST)
    except ValueError:
        logging.warning("Invalid signal_date for %s: %s", row.get("signal_id", ""), value)
        return None


def price_on_or_after(data: pd.DataFrame, target: datetime) -> Optional[float]:
    target_ts = pd.Timestamp(target.date())
    subset = data[data.index >= target_ts]
    if subset.empty:
        return None
    return float(subset["Signal Close"].iloc[0])


def window_prices(data: pd.DataFrame, start: datetime, end: datetime) -> pd.Series:
    start_ts = pd.Timestamp(start.date())
    end_ts = pd.Timestamp(end.date())
    return data[(data.index >= start_ts) & (data.index <= end_ts)]["Signal Close"]


def pct_return(start_price: Optional[float], end_price: Optional[float]) -> Optional[float]:
    if start_price is None or end_price is None or start_price == 0:
        return None
    return (end_price / start_price - 1) * 100


def fmt_cell(value: Optional[float]) -> str:
    if value is None:
        return ""
    return str(value)


def should_fill(row: Dict[str, str], overwrite: bool) -> bool:
    if overwrite:
        return True
    return any(not row.get(field) for field in FORWARD_FIELDS)


def classify_result(row: Dict[str, str]) -> str:
    signal_type = row.get("signal_type", "")
    rel_1m = parse_float(row.get("relative_return_1m_pct"))
    rel_3m = parse_float(row.get("relative_return_3m_pct"))
    max_dd_1m = parse_float(row.get("max_drawdown_1m_pct"))

    risk_types = {"overheat_risk", "trend_weakness", "weak_deep_pullback"}
    candidate_types = {
        "pullback_watch",
        "deep_pullback_trend_intact",
        "pullback_but_overheated",
        "breakout_strength",
        "breakout_but_overheated",
    }

    if signal_type in risk_types:
        if rel_1m is not None and rel_1m < -5:
            return "avoided_loss"
        if max_dd_1m is not None and max_dd_1m <= -10:
            return "avoided_loss"
        if rel_3m is not None and rel_3m > 10:
            return "false_positive"
        return "neutral"

    if signal_type in candidate_types:
        if rel_3m is not None and rel_3m > 10:
            return "good_signal"
        if rel_1m is not None and rel_1m > 5:
            return "good_signal"
        if rel_3m is not None and rel_3m < -10:
            return "bad_signal"
        return "neutral"

    return "neutral"


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def update_forward_fields(
    row: Dict[str, str],
    stock_data: pd.DataFrame,
    topix_data: pd.DataFrame,
    signal_date: datetime,
    overwrite: bool,
) -> bool:
    signal_price = parse_float(row.get("price_at_signal"))
    if signal_price is None:
        signal_price = price_on_or_after(stock_data, signal_date)
    topix_signal_price = price_on_or_after(topix_data, signal_date)
    if signal_price is None or topix_signal_price is None:
        return False

    field_map = {
        "1w": ("price_1w", "return_1w_pct", "topix_return_1w_pct", "relative_return_1w_pct"),
        "1m": ("price_1m", "return_1m_forward_pct", "topix_return_1m_pct", "relative_return_1m_pct"),
        "3m": ("price_3m", "return_3m_forward_pct", "topix_return_3m_pct", "relative_return_3m_pct"),
        "6m": ("price_6m", "return_6m_forward_pct", "topix_return_6m_pct", "relative_return_6m_pct"),
    }

    updated = False
    now = datetime.now(JST)
    for label, days in HORIZONS.items():
        target = signal_date + timedelta(days=days)
        if target.date() > now.date():
            continue

        price_field, return_field, topix_field, relative_field = field_map[label]
        if not overwrite and row.get(price_field):
            continue

        stock_price = price_on_or_after(stock_data, target)
        topix_price = price_on_or_after(topix_data, target)
        stock_return = pct_return(signal_price, stock_price)
        topix_return = pct_return(topix_signal_price, topix_price)
        relative_return = None
        if stock_return is not None and topix_return is not None:
            relative_return = stock_return - topix_return

        row[price_field] = fmt_cell(stock_price)
        row[return_field] = fmt_cell(stock_return)
        row[topix_field] = fmt_cell(topix_return)
        row[relative_field] = fmt_cell(relative_return)
        updated = True

    for label, days in (("1m", 30), ("3m", 91)):
        end = signal_date + timedelta(days=days)
        if end.date() > now.date():
            continue
        gain_field = f"max_gain_{label}_pct"
        drawdown_field = f"max_drawdown_{label}_pct"
        if not overwrite and row.get(gain_field) and row.get(drawdown_field):
            continue
        prices = window_prices(stock_data, signal_date, end)
        if prices.empty:
            continue
        row[gain_field] = fmt_cell((float(prices.max()) / signal_price - 1) * 100)
        row[drawdown_field] = fmt_cell((float(prices.min()) / signal_price - 1) * 100)
        updated = True

    if overwrite or not row.get("result_label"):
        row["result_label"] = classify_result(row)
        updated = True

    return updated


def topix_candidates(config: Dict[str, Any]) -> List[str]:
    return config.get("topix_candidates", ["^TOPX", "998405.T", "1306.T", "1348.T"])


def fetch_topix(candidates: List[str], start: str, end: str) -> Tuple[str, pd.DataFrame]:
    for ticker in candidates:
        data = fetch_price_data(ticker, start, end)
        if data is not None and not data.empty:
            logging.info("Using %s as TOPIX benchmark", ticker)
            return ticker, data
        logging.warning("TOPIX candidate %s unavailable, trying next", ticker)
    raise RuntimeError("No usable TOPIX benchmark data found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill forward performance fields in stock_signal_log.csv.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--signal-log", default=None, help="Path to stock_signal_log.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing CSV")
    parser.add_argument("--overwrite", action="store_true", help="Recalculate fields even if already filled")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    signal_log_config = config.get("signal_log", {}) or {}
    signal_log_path = args.signal_log or signal_log_config.get("path", "stock_signal_log.csv")

    columns, rows = read_signal_log(signal_log_path)
    if not rows:
        logging.info("No signal rows found in %s", signal_log_path)
        return

    for field in FORWARD_FIELDS:
        if field not in columns:
            columns.append(field)

    signal_dates = [parse_signal_date(row) for row in rows]
    valid_dates = [date for date in signal_dates if date is not None]
    if not valid_dates:
        logging.info("No valid signal dates found")
        return

    start = (min(valid_dates) - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (datetime.now(JST) + timedelta(days=7)).strftime("%Y-%m-%d")
    topix_ticker, topix_data = fetch_topix(topix_candidates(config), start, end)

    cache: Dict[str, Optional[pd.DataFrame]] = {}
    updated_count = 0
    skipped_count = 0
    for row, signal_date in zip(rows, signal_dates):
        if signal_date is None or not should_fill(row, args.overwrite):
            skipped_count += 1
            continue

        ticker = row.get("ticker", "")
        if ticker not in cache:
            cache[ticker] = fetch_price_data(ticker, start, end)
        stock_data = cache[ticker]
        if stock_data is None:
            skipped_count += 1
            continue

        if update_forward_fields(row, stock_data, topix_data, signal_date, args.overwrite):
            updated_count += 1
        else:
            skipped_count += 1

    logging.info("Benchmark=%s, rows=%d, updated=%d, skipped=%d, dry_run=%s", topix_ticker, len(rows), updated_count, skipped_count, args.dry_run)
    if args.dry_run:
        return

    write_signal_log(signal_log_path, columns, rows)
    logging.info("Signal log updated: %s", signal_log_path)


if __name__ == "__main__":
    main()
