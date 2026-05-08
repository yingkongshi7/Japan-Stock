# Japan Stock Monitor

Python 3 script for post-close monitoring of Japanese stocks using `yfinance`.

It only sends investment watchlist reminders. It does not trade, connect to broker APIs, or place orders.

## Local Usage

```bash
pip install -r requirements.txt
export SMTP_PASSWORD="your-smtp-app-password"
python stock_monitor.py --dry-run
python stock_monitor.py --test-email
python stock_monitor.py
```

On Windows PowerShell:

```powershell
$env:SMTP_PASSWORD="your-smtp-app-password"
python stock_monitor.py --dry-run
```

Edit `config.yaml` for stock pool, sector classification, SMTP settings, recipients, and alert thresholds.

## Cron Example

Run every weekday at 16:30 Japan time:

```cron
30 16 * * 1-5 cd /path/to/repo && /usr/bin/python3 stock_monitor.py >> monitor.log 2>&1
```

## GitHub Actions

Add `SMTP_PASSWORD` as a repository secret. The included workflow runs after the Japan close on weekdays and caches `alert_state.json` so duplicate alert logic can survive across runs.

The schedule uses UTC. `07:30 UTC` is `16:30 JST`.
