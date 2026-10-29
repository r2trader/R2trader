# R2trader

A crypto **paper-trading** research dashboard. It uses simulated money only: no real orders, no exchange keys, no financial advice.

## What it does
Every hour a GitHub Actions job:
1. Downloads hourly candles for 10 major coins from free public market-data endpoints.
2. Looks for breakouts: close above the prior 20-hour high, volume at least 2x average, price above the 50-hour trend line.
3. Applies fixed risk rules (1% risk per trade, max 3 open trades, max 25% of the account in one coin, trading halts for the day after a 3% loss).
4. Records simulated trades with 0.1% fees and 0.05% slippage, and updates the dashboard in `docs/`.
5. Replays the same rules over recent history as a backtest, shown next to buy-and-hold.

An optional AI summary uses any OpenAI-compatible endpoint (for example a free GLM or Qwen model). Add repository secrets `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL`; without them a plain template summary is used. The AI only writes the note and never makes trading decisions.

## Setup
1. Repository **Settings > Pages > Source: GitHub Actions**.
2. **Actions > R2trader update > Run workflow** for the first run.

## Limits
Hourly data only, so this is not a real-time system. GitHub scheduled runs can be delayed or skipped. Backtests are short and can overfit. Results are not evidence of future profit.
