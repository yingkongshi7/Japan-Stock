import argparse
import json
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
import yfinance as yf


JST = timezone(timedelta(hours=9))


@dataclass
class StockInfo:
    ticker: str
    name: str
    sector: str


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("config.yaml is empty or invalid")

    required = ["stocks", "smtp", "email", "thresholds"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    return config


def flatten_stock_pool(config: Dict[str, Any]) -> List[StockInfo]:
    stocks: List[StockInfo] = []
    for sector, items in config["stocks"].items():
        for item in items:
            stocks.append(
                StockInfo(
                    ticker=item["ticker"],
                    name=item["name"],
                    sector=sector,
                )
            )
    return stocks


def fetch_price_data(ticker: str, period: str = "18mo") -> Optional[pd.DataFrame]:
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logging.exception("Failed to fetch %s: %s", ticker, exc)
        return None

    if data is None or data.empty:
        logging.warning("No price data for %s", ticker)
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna(subset=["Close"])
    if data.empty:
        logging.warning("No valid close data for %s", ticker)
        return None

    if "Adj Close" in data.columns and data["Adj Close"].notna().any():
        data["Signal Close"] = data["Adj Close"]
    else:
        data["Signal Close"] = data["Close"]

    return data


def get_price_at_or_before(data: pd.DataFrame, target_index: int) -> Optional[float]:
    if len(data) <= abs(target_index):
        return None
    value = data["Close"].iloc[target_index]
    if pd.isna(value) or value == 0:
        return None
    return float(value)


def pct_change_from_lookback(
    data: pd.DataFrame,
    lookback_days: int,
    price_col: str = "Signal Close",
) -> Optional[float]:
    if len(data) < lookback_days + 1:
        return None
    current = float(data[price_col].iloc[-1])
    past = float(data[price_col].iloc[-lookback_days - 1])
    if past == 0 or pd.isna(past):
        return None
    return (current / past - 1) * 100


def calculate_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    raw_close = data["Close"]
    signal_close = data["Signal Close"]
    volume = data["Volume"] if "Volume" in data.columns else pd.Series(dtype=float)

    ma20 = signal_close.rolling(20).mean()
    ma50 = signal_close.rolling(50).mean()
    ma200 = signal_close.rolling(200).mean()

    last_close = float(raw_close.iloc[-1])
    last_signal_close = float(signal_close.iloc[-1])
    high_52w = float(signal_close.tail(252).max())
    drawdown_pct = (high_52w - last_signal_close) / high_52w * 100 if high_52w else 0
    current_ma200 = ma200.iloc[-1]
    above_ma200 = bool(pd.notna(current_ma200) and last_signal_close > current_ma200)

    recent_cross_above_ma200 = False
    if len(data) >= 205:
        recent_close = signal_close.tail(6)
        recent_ma200 = ma200.tail(6)
        for i in range(1, len(recent_close)):
            prev_below_or_equal = recent_close.iloc[i - 1] <= recent_ma200.iloc[i - 1]
            now_above = recent_close.iloc[i] > recent_ma200.iloc[i]
            if pd.notna(recent_ma200.iloc[i - 1]) and pd.notna(recent_ma200.iloc[i]) and prev_below_or_equal and now_above:
                recent_cross_above_ma200 = True
                break

    avg_volume_20d = float(volume.tail(20).mean()) if not volume.empty else 0.0
    last_volume = float(volume.iloc[-1]) if not volume.empty and pd.notna(volume.iloc[-1]) else 0.0
    volume_spike = bool(avg_volume_20d > 0 and last_volume > avg_volume_20d * 1.5)
    avg_turnover_20d = float((raw_close * volume).tail(20).mean()) if not volume.empty else 0.0

    return {
        "current_price": last_close,
        "high_52w": high_52w,
        "is_52w_high": last_signal_close >= high_52w,
        "drawdown_pct": drawdown_pct,
        "ma20": float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else None,
        "ma50": float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else None,
        "ma200": float(current_ma200) if pd.notna(current_ma200) else None,
        "above_ma200": above_ma200,
        "recent_cross_above_ma200": recent_cross_above_ma200,
        "return_1m_pct": pct_change_from_lookback(data, 21),
        "return_3m_pct": pct_change_from_lookback(data, 63),
        "return_6m_pct": pct_change_from_lookback(data, 126),
        "avg_volume_20d": avg_volume_20d,
        "current_volume": last_volume,
        "volume_spike": volume_spike,
        "avg_turnover_20d": avg_turnover_20d,
        "last_date": data.index[-1].strftime("%Y-%m-%d"),
    }


def calculate_relative_strength(stock_indicators: Dict[str, Any], topix_data: pd.DataFrame) -> Dict[str, Any]:
    topix_return_3m = pct_change_from_lookback(topix_data, 63)
    stock_return_3m = stock_indicators.get("return_3m_pct")

    if topix_return_3m is None or stock_return_3m is None:
        relative_3m = None
    else:
        relative_3m = stock_return_3m - topix_return_3m

    return {
        "topix_return_3m_pct": topix_return_3m,
        "relative_topix_3m_pct": relative_3m,
    }


def load_state(path: str) -> Dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {"alerts": {}}

    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Failed to load state file %s: %s. Starting fresh.", path, exc)
        return {"alerts": {}}

    state.setdefault("alerts", {})
    return state


def save_state(state: Dict[str, Any], path: str) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def prune_old_state(state: Dict[str, Any], dedup_days: int) -> None:
    cutoff = datetime.now(JST).date() - timedelta(days=dedup_days * 2)
    alerts = state.get("alerts", {})
    for key in list(alerts.keys()):
        last_alert_date = alerts[key].get("last_alert_date")
        if not last_alert_date:
            continue
        try:
            parsed = datetime.strptime(last_alert_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        if parsed < cutoff:
            del alerts[key]


def should_send_alert(
    state: Dict[str, Any],
    ticker: str,
    alert_type: str,
    indicators: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> bool:
    alerts = state.setdefault("alerts", {})
    key = f"{ticker}|{alert_type}"
    existing = alerts.get(key)
    today = datetime.now(JST).date()
    dedup_days = int(thresholds.get("dedup_days", 30))
    extra_drawdown_pct = float(thresholds.get("extra_drawdown_pct", 10))

    if not existing:
        return True

    if indicators.get("is_52w_high") and alert_type in {"pullback_watch", "deep_pullback_trend_intact"}:
        return True

    last_alert_date_str = existing.get("last_alert_date")
    last_drawdown = float(existing.get("drawdown_pct", 0))
    current_drawdown = float(indicators.get("drawdown_pct", 0))

    if current_drawdown >= last_drawdown + extra_drawdown_pct:
        return True

    if not last_alert_date_str:
        return True

    try:
        last_alert_date = datetime.strptime(last_alert_date_str, "%Y-%m-%d").date()
    except ValueError:
        return True

    return (today - last_alert_date).days >= dedup_days


def record_alert_state(state: Dict[str, Any], ticker: str, alert_type: str, indicators: Dict[str, Any]) -> None:
    key = f"{ticker}|{alert_type}"
    state.setdefault("alerts", {})[key] = {
        "last_alert_date": datetime.now(JST).strftime("%Y-%m-%d"),
        "drawdown_pct": indicators.get("drawdown_pct"),
        "high_52w": indicators.get("high_52w"),
        "current_price": indicators.get("current_price"),
    }


def reset_drawdown_alerts_on_new_high(state: Dict[str, Any], ticker: str) -> None:
    for alert_type in ("pullback_watch", "deep_pullback_trend_intact"):
        state.get("alerts", {}).pop(f"{ticker}|{alert_type}", None)


def check_alert_conditions(
    stock: StockInfo,
    indicators: Dict[str, Any],
    thresholds: Dict[str, Any],
    state: Dict[str, Any],
) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    drawdown = indicators.get("drawdown_pct")
    relative_3m = indicators.get("relative_topix_3m_pct")
    avg_turnover_20d = indicators.get("avg_turnover_20d", 0)
    trend_ok = indicators.get("above_ma200") or indicators.get("recent_cross_above_ma200")
    liquidity_ok = avg_turnover_20d >= float(thresholds.get("min_avg_turnover_20d_jpy", 0))

    if indicators.get("is_52w_high"):
        reset_drawdown_alerts_on_new_high(state, stock.ticker)

    if drawdown is None or relative_3m is None:
        return alerts

    candidates = [
        (
            "pullback_watch",
            "回撤观察",
            drawdown >= float(thresholds["pullback_min_pct"])
            and drawdown <= float(thresholds["pullback_max_pct"])
            and trend_ok
            and relative_3m > float(thresholds["relative_3m_min_pct"])
            and liquidity_ok,
        ),
        (
            "breakout_strength",
            "强势突破观察",
            indicators.get("is_52w_high")
            and indicators.get("volume_spike")
            and relative_3m > float(thresholds["breakout_relative_3m_min_pct"]),
        ),
        (
            "deep_pullback_trend_intact",
            "深度回撤但趋势未坏",
            drawdown >= float(thresholds["deep_pullback_min_pct"]) and trend_ok,
        ),
        (
            "trend_weakness",
            "趋势转弱，谨慎观察",
            not indicators.get("above_ma200")
            and indicators.get("ma200") is not None
            and relative_3m < float(thresholds["risk_relative_3m_max_pct"]),
        ),
    ]

    for alert_type, title, triggered in candidates:
        if triggered and should_send_alert(state, stock.ticker, alert_type, indicators, thresholds):
            alerts.append({"type": alert_type, "title": title})

    return alerts


def fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def fmt_num(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:,.2f}"


def build_email_body(stock: StockInfo, indicators: Dict[str, Any], alert: Dict[str, str]) -> str:
    above_ma200 = "是" if indicators.get("above_ma200") else "否"
    recent_cross = "是" if indicators.get("recent_cross_above_ma200") else "否"
    volume_spike = "是" if indicators.get("volume_spike") else "否"

    return f"""提醒类型：{alert["title"]}

股票代码：{stock.ticker}
股票名称：{stock.name}
产业分类：{stock.sector}
数据日期：{indicators.get("last_date", "N/A")}

当前价格：{fmt_num(indicators.get("current_price"))}
52周最高收盘价：{fmt_num(indicators.get("high_52w"))}
从52周高点回撤：{fmt_pct(indicators.get("drawdown_pct"))}

20日均线：{fmt_num(indicators.get("ma20"))}
50日均线：{fmt_num(indicators.get("ma50"))}
200日均线：{fmt_num(indicators.get("ma200"))}
当前价格高于200日均线：{above_ma200}
最近5个交易日重新站上200日均线：{recent_cross}

过去1个月涨幅：{fmt_pct(indicators.get("return_1m_pct"))}
过去3个月涨幅：{fmt_pct(indicators.get("return_3m_pct"))}
过去6个月涨幅：{fmt_pct(indicators.get("return_6m_pct"))}
过去3个月TOPIX涨幅：{fmt_pct(indicators.get("topix_return_3m_pct"))}
过去3个月相对TOPIX超额收益：{fmt_pct(indicators.get("relative_topix_3m_pct"))}

当前成交量：{fmt_num(indicators.get("current_volume"))}
过去20日平均成交量：{fmt_num(indicators.get("avg_volume_20d"))}
成交量超过20日均量1.5倍：{volume_spike}
过去20日平均成交额（日元）：{fmt_num(indicators.get("avg_turnover_20d"))}

提醒：这不是自动交易，也不是买卖建议，只是观察名单提醒，需要人工确认。
"""


def send_email(config: Dict[str, Any], subject: str, body: str) -> None:
    smtp_config = config["smtp"]
    email_config = config["email"]
    password = os.environ.get("SMTP_PASSWORD")
    if not password:
        raise RuntimeError("SMTP_PASSWORD environment variable is not set")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = email_config["from"]
    msg["To"] = ", ".join(email_config["to"])

    host = smtp_config["host"]
    port = int(smtp_config.get("port", 587))
    use_tls = bool(smtp_config.get("use_tls", True))

    with smtplib.SMTP(host, port, timeout=30) as server:
        if use_tls:
            server.starttls()
        server.login(smtp_config["username"], password)
        server.sendmail(email_config["from"], email_config["to"], msg.as_string())


def fetch_topix_data(candidates: List[str]) -> Tuple[str, pd.DataFrame]:
    for ticker in candidates:
        data = fetch_price_data(ticker)
        if data is not None and len(data) >= 70:
            logging.info("Using %s as TOPIX benchmark", ticker)
            return ticker, data
        logging.warning("TOPIX candidate %s unavailable, trying next", ticker)
    raise RuntimeError("No usable TOPIX benchmark data found")


def build_test_email_body() -> str:
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"""日本股票监控脚本 SMTP 测试邮件

发送时间：{now}

如果你收到这封邮件，说明 SMTP 配置和 SMTP_PASSWORD 环境变量可用。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Japanese stock pullbacks and relative strength.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print alerts without sending email")
    parser.add_argument("--test-email", action="store_true", help="Send a test email and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("log_file", "monitor.log"))

    if args.test_email:
        send_email(config, "[日本股票监控] SMTP 测试", build_test_email_body())
        logging.info("Test email sent")
        return

    state_file = config.get("state_file", "alert_state.json")
    state = load_state(state_file)
    thresholds = config["thresholds"]
    prune_old_state(state, int(thresholds.get("dedup_days", 30)))

    topix_ticker, topix_data = fetch_topix_data(config.get("topix_candidates", ["^TOPX", "1306.T"]))
    stocks = flatten_stock_pool(config)
    sent_count = 0
    triggered_count = 0

    for stock in stocks:
        logging.info("Processing %s %s", stock.ticker, stock.name)
        data = fetch_price_data(stock.ticker)
        if data is None:
            continue

        try:
            indicators = calculate_indicators(data)
            indicators.update(calculate_relative_strength(indicators, topix_data))
            alerts = check_alert_conditions(stock, indicators, thresholds, state)
        except Exception as exc:
            logging.exception("Failed to process %s: %s", stock.ticker, exc)
            continue

        if not alerts:
            logging.info("No alert for %s", stock.ticker)
            continue

        for alert in alerts:
            triggered_count += 1
            subject = f"[日本股票监控] {alert['title']} - {stock.ticker} {stock.name}"
            body = build_email_body(stock, indicators, alert)

            if args.dry_run:
                print("=" * 80)
                print(subject)
                print(body)
            else:
                try:
                    send_email(config, subject, body)
                    sent_count += 1
                    logging.info("Email sent for %s %s", stock.ticker, alert["type"])
                except Exception as exc:
                    logging.exception("Failed to send email for %s: %s", stock.ticker, exc)
                    continue

            record_alert_state(state, stock.ticker, alert["type"], indicators)

    save_state(state, state_file)
    logging.info(
        "Finished. Benchmark=%s, triggered=%d, sent=%d, dry_run=%s",
        topix_ticker,
        triggered_count,
        sent_count,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
