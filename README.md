# Japan Stock Monitor

Python 3 script for post-close monitoring of Japanese stocks using `yfinance`.

This project is an investment watchlist reminder tool. It is not an automated trading system, does not connect to broker APIs, does not place orders, and does not provide deterministic buy/sell instructions. Every email is an observation or risk-review prompt that requires human confirmation.

## Local Usage

```bash
pip install -r requirements.txt
export SMTP_PASSWORD="your-smtp-app-password"
python stock_monitor.py
python stock_monitor.py --dry-run --report
python stock_monitor.py --summary-email
python stock_monitor.py --test-email
python stock_monitor.py --export-signals
python stock_monitor.py --signal-log stock_signal_log.csv
```

On Windows PowerShell:

```powershell
$env:SMTP_PASSWORD="your-smtp-app-password"
python stock_monitor.py --dry-run --report
python stock_monitor.py --dry-run --log-signals-dry-run
```

Edit `config.yaml` for the stock pool, sector classification, SMTP settings, recipients, and alert thresholds. The SMTP password is read from the `SMTP_PASSWORD` environment variable.

## GitHub Actions

The workflow file must be placed at:

```text
.github/workflows/japan-stock-monitor.yml
```

GitHub scheduled workflows only run on the default branch. GitHub Actions schedules can be delayed and may occasionally miss a run, so this project uses a primary schedule plus a backup schedule:

- JST 17:17 on weekdays
- JST 18:47 on weekdays as backup

The cron values are UTC:

```yaml
schedule:
  - cron: "17 8 * * 1-5"
  - cron: "47 9 * * 1-5"
```

Manual `workflow_dispatch` modes:

- `normal`: `python stock_monitor.py`
- `dry-run`: `python stock_monitor.py --dry-run --report`
- `report`: `python stock_monitor.py --report`
- `test-email`: `python stock_monitor.py --test-email`
- `summary-email`: `python stock_monitor.py --summary-email`

## Email Policy

The default email policy is summary-first to reduce inbox noise:

```yaml
email_policy:
  daily_summary_default: true
  send_individual_alerts: true
  notify_action_levels:
    - A
    - B
  send_summary_when_no_notify_alerts: false
  individual_alert_types:
    - weak_deep_pullback
    - pullback_but_overheated
    - breakout_but_overheated
    - pullback_watch
    - deep_pullback_trend_intact
  send_sector_heat_individual: false
```

Normal runs send one daily summary by default only when there are notify-level signals. Individual stock emails are sent only for higher-priority combined signals. `overheat_risk`, `trend_weakness`, and `sector_heat` are still recorded in the CSV and logs, but they do not send separate emails and do not appear in the ordinary daily summary by default.

The ordinary scheduled notification now focuses on A/B action levels. C/D/E signals are still recorded in `stock_signal_log.csv` and logs, but they do not appear in the normal daily notification. To restore C-level emails and summary entries, add `C` to `notify_action_levels`.

Daily runs are deduplicated in `alert_state.json` under `daily_runs` by JST date. The primary and backup schedules can both exist, but after one normal run completes, the backup run exits without sending mail. The workflow pulls the latest branch state before running the script and uses GitHub Actions concurrency to avoid overlapping monitor runs on the same branch.

## State Handling

`dry-run` does not save `alert_state.json` and will not pollute deduplication state.

The workflow commits `alert_state.json` back to the repository after normal runs. This is more stable for long-running monitoring than relying on GitHub Actions cache. `dry-run`, `report`, `test-email`, and `summary-email` do not commit state.

The workflow commits both `alert_state.json` and `stock_signal_log.csv` after normal runs. It still uploads `monitor.log`, `alert_state.json`, and `stock_signal_log.csv` as artifacts for debugging.

## Signal CSV Log

`stock_signal_log.csv` records combined stock alerts for later review. Each row is one script-generated combined stock signal, not one raw alert. For example, if `pullback_watch` and `overheat_risk` fire on the same stock, the CSV records one `pullback_but_overheated` row and keeps `raw_alerts` as `pullback_watch;overheat_risk`.

The CSV is meant for a 6-month review process:

- It records the signal date and technical context at the time of the alert.
- It leaves manual review columns blank for Google Sheets or Excel.
- It leaves future performance columns blank for a later backfill script.
- It is independent from email delivery, so a signal can be recorded even if SMTP fails.

Configuration:

```yaml
signal_log:
  enabled: true
  path: stock_signal_log.csv
  append_only: true
```

Commands:

```bash
python stock_monitor.py
python stock_monitor.py --dry-run --report
python stock_monitor.py --dry-run --log-signals-dry-run
python stock_monitor.py --export-signals
python stock_monitor.py --signal-log stock_signal_log.csv
```

`dry-run` does not write `stock_signal_log.csv` by default. Add `--log-signals-dry-run` only when you intentionally want to test CSV writing.

Suggested review labels:

- `good_signal`
- `bad_signal`
- `neutral`
- `avoided_loss`
- `false_positive`

How to evaluate later:

- For candidate/recovery signals, review 1-month and 3-month relative return versus TOPIX.
- For risk and weakness signals, check whether the stock continued to underperform TOPIX.
- For overheat signals, check whether the stock pulled back within 1 month.

## Backfill Signal Results

`backfill_signal_results.py` fills future performance fields in `stock_signal_log.csv` after enough time has passed.

```bash
python backfill_signal_results.py
python backfill_signal_results.py --dry-run
python backfill_signal_results.py --overwrite
python backfill_signal_results.py --signal-log stock_signal_log.csv
```

It backfills:

- 1 week, 1 month, 3 month, and 6 month forward price and return
- TOPIX return for the same windows
- Relative return versus TOPIX
- 1 month and 3 month maximum gain/drawdown
- `result_label`

`result_label` is a first-pass classification:

- Candidate/recovery signals become `good_signal` if later relative performance is strong, `bad_signal` if clearly weak, otherwise `neutral`.
- Risk/weakness/overheat signals become `avoided_loss` if the stock later underperforms or draws down, `false_positive` if it strongly outperforms, otherwise `neutral`.

The backfill script is for review only. It does not change alerts, send emails, connect to brokers, or trade.

## No Alert Email

Not receiving an alert email does not necessarily mean the job failed. It can also mean that no stock met the configured conditions.

To verify execution:

- Check the GitHub Actions log.
- Run `mode=report` to print the indicator table.
- Run `mode=summary-email` to receive a daily run summary, including "今日无触发提醒" when nothing triggered.

## Combined Alerts

Each stock sends at most one combined alert per run. Raw alert signals are preserved in the email body, but conflicting signals are merged into a single cautious conclusion.

Examples:

- `pullback_watch` + `overheat_risk` becomes `回撤后修复，但短线过热`
- `deep_pullback_trend_intact` + `trend_weakness` becomes `深度回撤但趋势转弱，谨慎复查`
- `breakout_strength` + `overheat_risk` becomes `强势突破但短线过热`

The goal is to avoid contradictory emails such as "buy candidate" and "avoid chasing" for the same stock on the same day.

## Safety Boundary

This project does not:

- Connect to broker APIs
- Place orders
- Perform automatic trading
- Provide definitive investment advice
- Encourage high-frequency trading

All outputs are observation, review, risk-warning, or candidate-research prompts. Human confirmation is required.
