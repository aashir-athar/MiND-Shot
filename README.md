<div align="center">

<h1>📡 MiND-Shot — Free Autonomous Crypto Trading Signal Bot</h1>

<p><strong>MiND-Shot is a free, open-source, serverless crypto trading signal bot that runs entirely on GitHub Actions and sends backtested Bitcoin &amp; Ethereum mean-reversion signals to Telegram — no server, no monthly cost, and zero Python dependencies.</strong></p>

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-View_Backtest_Report-8B5CF6?style=for-the-badge&logo=react&logoColor=white)](https://aashir-athar.github.io/MiND-Shot/)
[![Stars](https://img.shields.io/github/stars/aashir-athar/MiND-Shot?style=for-the-badge&logo=github&color=FFD33D)](https://github.com/aashir-athar/MiND-Shot/stargazers)
[![License](https://img.shields.io/github/license/aashir-athar/MiND-Shot?style=for-the-badge&color=blue)](./LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/aashir-athar/MiND-Shot/ci.yml?style=for-the-badge&label=ci)](https://github.com/aashir-athar/MiND-Shot/actions/workflows/ci.yml)
[![Deploy Dashboard](https://img.shields.io/github/actions/workflow/status/aashir-athar/MiND-Shot/pages.yml?style=for-the-badge&label=pages)](https://github.com/aashir-athar/MiND-Shot/actions/workflows/pages.yml)
[![Top language](https://img.shields.io/github/languages/top/aashir-athar/MiND-Shot?style=for-the-badge&logo=python&logoColor=white)](https://github.com/aashir-athar/MiND-Shot)

<a href="https://aashir-athar.github.io/MiND-Shot/"><strong>🔴 Live Dashboard</strong></a> ·
<a href="#-the-five-strategies"><strong>Strategies</strong></a> ·
<a href="#-getting-started"><strong>Getting Started</strong></a> ·
<a href="#-how-it-works"><strong>How It Works</strong></a> ·
<a href="#-faq"><strong>FAQ</strong></a> ·
<a href="https://github.com/aashir-athar/MiND-Shot/issues"><strong>Report Bug</strong></a>

</div>

---

## 📖 What is MiND-Shot?

**MiND-Shot** is an open-source **crypto trading signal engine** — an **algorithmic trading bot** for **Bitcoin (BTC)** and **Ethereum (ETH)** that runs as a **GitHub Actions cron job** with **$0 hosting**, no VPS, and **no third-party pip dependencies** (pure Python standard library). Its signal brain is a set of **five mean-reversion strategies**, each selected from a **1,620-config backtest sweep** and validated **out-of-sample** on real BTC/ETH market data. Around them sits a full quantitative-trading intelligence layer — a **self-learning machine-learning ensemble**, **whale-flow** order-flow context, a **Trade Verdict** score, and hard **risk-management** controls — and rich entry/exit **Telegram trading alerts** delivered via a webhook (Make.com / n8n / Pipedream) or the direct Telegram Bot API.

> 🚧 **Active quant research & trading-automation project.** MiND-Shot is a **decision-support tool, not financial advice** — read [Honest expectations](#-honest-expectations) before using it.

### ⚡ Quick facts

| | |
|---|---|
| **What it is** | Free, serverless crypto trading signal bot (BTC + ETH) |
| **How it runs** | GitHub Actions cron — no server, no VPS, $0/month |
| **Signals** | 5 backtested, ADX-gated mean-reversion strategies (4h chart, long & short) |
| **Backtest** | 326 trades · 72.7% overall win rate · +27.9% average return per strategy |
| **Delivery** | Telegram Bot API or webhook (Make.com / n8n / Pipedream) |
| **Dependencies** | None — pure Python 3.10+ standard library |
| **Dashboard** | Live React report → **[aashir-athar.github.io/MiND-Shot](https://aashir-athar.github.io/MiND-Shot/)** |
| **License** | MIT (free & open source) |

## 📊 Live Backtest Dashboard

**→ [aashir-athar.github.io/MiND-Shot](https://aashir-athar.github.io/MiND-Shot/)**

A **React + Vite** single-page **backtest reporting dashboard**, deployed to **GitHub Pages** and **fully automated**: every push regenerates the data from the live engine (`gen_report.py`) and rebuilds the site via GitHub Actions, so the numbers are never hand-typed. It shows:

- **KPI overview** — strategies validated, overall win rate vs the 60% gate, total trades, average return, worst drawdown
- **Win-rate & return charts** per strategy, with the 60% validation gate marked
- **Backtest-vs-playbook dumbbell** — how tightly out-of-sample results track the published expectation
- **Equity curves** for all five $100 accounts, plus a full **326-row trade blotter** (side, entry, exit, P&L, result) filterable by strategy
- **Sortable summary table** with a validated / failed verdict per strategy

Dark-first, theme-aware, accessible, colorblind-safe charts — built to match the engine's design system.

## ✨ Features

| | Feature | Description |
|---|---|---|
| 💸 | **$0 forever** | Runs on a public repo's free GitHub Actions minutes — no server, no VPS |
| 🐍 | **Zero dependencies** | Pure Python standard library — nothing to `pip install` |
| 🎯 | **5 backtested strategies** | Range-fading mean-reversion (VWAP · RSI-2 · Stochastic · Z-score), ADX-gated, both directions |
| 📊 | **Live dashboard** | Automated React/GitHub Pages backtest report — equity curves + full trade blotter |
| 🧪 | **Self-validating** | `python -m mind_shot.backtest` reproduces the documented win rates on live data; CI runs it |
| 🧠 | **Self-learning ML** | Calibrated online ensemble — Bayesian context buckets, FTRL-Proximal logistic, and Hedge expert weighting with drift detection — learning from every closed trade |
| 🎛 | **Trade Verdict score** | 0–100 score blending ML confidence, whale flow, funding, and session |
| 🐋 | **Whale-flow signals** | Binance Futures long/short ratio, open interest, taker buy/sell pressure |
| 🛡️ | **Risk controls** | Daily loss limit, max concurrent trades, post-SL cool-down |
| 🔁 | **Weekly retrain** | Walk-forward-validated challenger retrains every Sunday — published only if it beats the base-rate baseline |
| 📲 | **Telegram alerts** | Pre-formatted HTML alerts via webhook or direct Bot API |

## 🎯 The Five Strategies

All five share one edge — **fade an extreme back toward the mean, but only while the market is ranging (`ADX(14) < 25`)** — and each trades **both long and short** on the **4-hour** chart. Backtested at **$100 wallet · 10× leverage · 15%-of-wallet · cross margin · 0.10% round-trip fee**:

| Strategy | Coin | Win rate | Entry | Exit | Stop |
|---|---|---:|---|---|---|
| **VWAP-Reversion** | ETH | 78.4% | price ±2σ from VWAP(20) | TP 0.75×ATR | 1.5×ATR |
| **RSI-2 Reversion** | ETH | 72.3% | RSI(2) &lt; 10 / &gt; 90 | TP 0.75×ATR | 1.5×ATR |
| **VWAP-Reversion (revert)** | ETH | 70.0% | price ±2σ from VWAP(20) | back to VWAP | 2.0×ATR |
| **Stochastic Reversion** | ETH | 75.9% | %K(14) &lt; 20 / &gt; 80 | TP 0.75×ATR | 2.0×ATR |
| **Z-Score Reversion** | BTC | 65.1% | ±1.5σ from SMA(20) | back to mean | 3.0×ATR |

These numbers are **in-sample backtests, not promises.** Win rate alone is not edge — see [Honest expectations](#-honest-expectations). Explore them interactively on the **[live dashboard](https://aashir-athar.github.io/MiND-Shot/)**; the definitions live in [`mind_shot/strategies.py`](./mind_shot/strategies.py).

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Kraken](https://img.shields.io/badge/Kraken_API-5741D9?style=for-the-badge&logo=kraken&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)

| Layer | Choice |
|---|---|
| **Language** | Python 3.10+ (standard library only) |
| **Runtime** | GitHub Actions scheduled workflows (cron) |
| **Dashboard** | React 19 + Vite, deployed to GitHub Pages via Actions |
| **Market data** | Kraken public OHLC for the live feed (reachable from GitHub Actions; Binance geo-blocks the US runner IPs). The backtest validates against committed Binance 4h fixtures. |
| **Context** | Binance whale-flow · CoinGecko dominance · alternative.me Fear & Greed |
| **ML** | Online ensemble (Bayesian buckets · FTRL-Proximal · Hedge weighting · Platt calibration · drift detection) + weekly walk-forward challenger |
| **Delivery** | Telegram Bot API or Make.com / n8n / Pipedream webhook |
| **State** | Git-committed JSON (`state/state.json`, `state/trained_model.json`) |

## 🚀 Getting Started

Setup takes about 5 minutes. Keep your repo **public** for unlimited free Actions minutes.

### 1. Fork or clone
```bash
git clone https://github.com/aashir-athar/MiND-Shot.git
cd MiND-Shot
```

### 2. Add your delivery secret
**Settings → Secrets and variables → Actions → New repository secret**:
```text
# Option A — webhook (recommended)
WEBHOOK_URL = https://hook.eu1.make.com/...

# Option B — direct Telegram Bot API
TG_TOKEN    = 1234567890:AAEh...     # from @BotFather
TG_CHAT_ID  = 123456789
```
If both are set, `WEBHOOK_URL` wins. Optional repo **variables**: `LEVERAGE` (default `10`), `ACCOUNT_USD` (`100`), `ALLOC_PCT` (`15`).

### 3. Enable Actions
Open the **Actions** tab and enable workflows. Four are included:
- **MiND-Shot Engine** — polls every few minutes and ships signals
- **Weekly ML Retrain** — retrains the logistic model every Sunday
- **CI** — runs the test suite + strategy validation on every push
- **Deploy Dashboard** — regenerates the backtest data and publishes the GitHub Pages dashboard

### 4. Publish the dashboard (optional)
**Settings → Pages → Build and deployment → Source: GitHub Actions.** The **Deploy Dashboard** workflow then builds and publishes your own copy of the backtest report at `https://<you>.github.io/MiND-Shot/`.

## 📖 Usage

### Configuration (environment variables)
| Var | Default | Purpose |
|---|---|---|
| `WEBHOOK_URL` / `TG_TOKEN` + `TG_CHAT_ID` | — | delivery channel |
| `LEVERAGE` | `10` | leverage shown in alerts / PnL math |
| `ACCOUNT_USD` | `100` | account size for sizing display |
| `ALLOC_PCT` | `15` | % of wallet per trade |
| `ML_GATING_ENABLED` | `1` | let the Bayesian model veto low-confidence signals |
| `ML_MIN_TRADES` / `ML_MIN_CONF` | `12` / `0.40` | when/how strongly ML may veto |

The active strategy set is fixed to the five validated strategies (in `mind_shot/strategies.py`); there are no ad-hoc modes to misconfigure.

### Webhook payload
Each entry POSTs JSON; the `text` field is pre-formatted Telegram HTML ready to forward:
```json
{
  "type": "entry",
  "side": "LONG",
  "asset": "ETH",
  "tf": "4h",
  "strategy": "vwap_bracket_eth",
  "strategy_name": "VWAP-Reversion (bracket)",
  "ml_conf": 58.3,
  "leverage": 10,
  "entry": 1800.58,
  "sl": 1753.20,
  "tp": 1824.10,
  "target": null,
  "text": "🟢 ... MiND-Shot LONG ..."
}
```
TP / SL / exit events use `type: "event"` with `event: "tp" | "sl" | "exit"`. All dynamic text is HTML-escaped, so `parse_mode=HTML` delivery never breaks on characters like `<` or `>`.

## 🧠 How It Works

1. **Data** — every poll fetches recent **Kraken 4h** candles for ETH and BTC (Kraken is reachable from GitHub Actions runners; Binance returns HTTP 451 to their US IPs). The strategies are price-based, so the signals match the backtest.
2. **Signals** — each strategy checks, on the most recently *closed* bar, whether its oscillator is at an extreme **and** `ADX(14) < 25`. If so it proposes a long or short; the engine acts on the next bar's open.
3. **Second opinion** — a self-learning **online ensemble** scores every setup with a calibrated P(win): Bayesian context buckets and an FTRL-Proximal logistic model vote alongside the strategy's own base rate and the weekly directional model, weighted by their realised log-loss (Hedge / multiplicative weights — provably never much worse than the best expert in hindsight). A Page–Hinkley detector accelerates forgetting on regime breaks, and every closed trade updates a prequential honesty ledger (log-loss / Brier / accuracy vs baseline) committed to `state/`. Reproduce the evaluation yourself: `python -m mind_shot.ml_eval`.
4. **Management** — bracket strategies exit on a fixed take-profit / stop; revert strategies ride back to VWAP or the mean with a hard ATR stop. Stops are checked intrabar, stop-first.
5. **Delivery & learning** — entries and TP/SL/exit events ship to Telegram; every closed trade updates the ML, the streak heatmap, the journal, and the daily-R stats, all committed back to `state/`.

## 🧪 Validation

The strategies are **self-validating** — the same indicator/strategy code the live engine uses is replayed **offline** over committed fixtures of the original backtest window (`tests/fixtures/*_4h.csv`):
```bash
python -m mind_shot.backtest      # prints the validation report
python gen_report.py              # regenerates the dashboard data (web/public/backtest.json)
```
It prints each strategy's win rate, trade count, and $100→ result, and fails if any strategy drifts materially from its documented numbers. `gen_report.py` additionally **asserts its per-trade replay reproduces the official backtest** before writing the dashboard data, so the published blotter can never drift from the validated engine. Both need no network, so CI runs them deterministically on every push.

## 🧰 Development
```bash
python -m unittest discover -s tests -v      # unit tests (no network)
python -m mind_shot.backtest                 # strategy validation (live data)
OUTPUT_JSON=1 python mind_shot_engine.py      # one local dry-run (no secrets = no alerts sent)

cd web && npm install && npm run dev          # run the dashboard locally (Vite dev server)
```

<details>
<summary><strong>Project structure</strong></summary>

```text
mind_shot/
├── indicators.py     # pure-stdlib SMA/STD/z-score/RSI/ATR/ADX/Stochastic/VWAP
├── strategies.py     # the 5 backtested strategies (the registry)
├── market.py         # Kraken 4h klines (live feed)
├── trading.py        # trade lifecycle (bracket + revert exits)
├── ml.py             # Bayesian ensemble + trained-model application
├── context.py        # Fear & Greed / dominance / funding
├── whale.py          # whale-flow signals
├── intelligence.py   # Trade Verdict + analytics
├── notifier.py       # delivery + alert formatting (HTML-escaped)
├── state.py          # atomic JSON state
├── config.py         # env-driven configuration
├── engine.py         # poll orchestration
└── backtest.py       # in-repo validation backtest
web/                  # React + Vite backtest dashboard (GitHub Pages)
gen_report.py         # backtest -> web/public/backtest.json (trades + equity), replay-verified
mind_shot_engine.py   # entrypoint (used by the engine workflow / Electron host)
ml_trainer.py         # weekly walk-forward trainer
tests/                # unit tests + backtest fixtures (committed 4h playbook data)
.github/workflows/    # engine.yml · retrain.yml · ci.yml · pages.yml
```
</details>

## ❓ FAQ

**Is MiND-Shot free?**
Yes. MiND-Shot is 100% free and open source (MIT). It runs on a public repository's free GitHub Actions minutes, so there is no server bill, VPS, or subscription.

**Do I need a server or VPS to run this crypto signal bot?**
No. Everything runs serverless on GitHub Actions cron. Fork the repo, add one delivery secret, enable Actions — that's it.

**Which coins and timeframe does it trade?**
Bitcoin (BTC) and Ethereum (ETH) on the 4-hour chart, taking both long and short signals.

**How are the trading signals generated?**
Five mean-reversion strategies fade price extremes (VWAP, RSI-2, Stochastic, Z-score) back toward the mean, but only while the market is ranging (`ADX(14) < 25`). A self-learning ML ensemble acts as an advisory second opinion.

**How do I get the signals?**
As formatted alerts in Telegram — either through the direct Telegram Bot API or via a webhook automation platform such as Make.com, n8n, or Pipedream.

**Is this financial advice?**
No. MiND-Shot is an educational, decision-support tool. Backtested results are in-sample and do not guarantee future performance — paper-trade first and manage your own risk.

**Can I see the backtest results without installing anything?**
Yes — the live dashboard at [aashir-athar.github.io/MiND-Shot](https://aashir-athar.github.io/MiND-Shot/) shows the full backtest report, equity curves, and every trade.

**What are the dependencies?**
The engine has none — pure Python standard library. Only the optional dashboard uses Node/React to build.

## ⚠️ Honest Expectations

- **Win rate is not edge.** A high win rate with a wide stop can still lose money; these strategies are profitable only because their win rate clears the break-even implied by their reward:risk.
- The backtested numbers are **in-sample on one bear/chop regime**, selected from many configs. Expect **lower live win rates (~60–68%)** and **thin expectancy**, and **paper-trade before risking real capital**.
- The entire edge is *"ranges revert."* A real trend breaking out of the range produces a cluster of losses — the `ADX < 25` filter reduces but does not remove this.
- The ML adds roughly **52–62% out-of-sample directional accuracy** — a second opinion, not magic. Any tool promising "100% accuracy" is overfit and will lose money live.
- **This is not financial advice.** Past performance does not guarantee future results. Use responsibly and at your own risk.

## 🗺️ Roadmap
- [x] Five backtested, out-of-sample-validated strategies as the signal core
- [x] In-repo backtest + unit tests + CI
- [x] Self-learning ML ensemble + weekly walk-forward retrain
- [x] Telegram / webhook alert delivery
- [x] **Backtest reporting dashboard (React + GitHub Pages, fully automated)**
- [ ] Configurable strategy set via repo variables

## 🤝 Contributing
Contributions are welcome. For major changes, please open an issue first. Fork → branch (`git checkout -b feat/your-idea`) → commit → open a PR. CI must pass.

## 📄 License
Distributed under the **MIT License**. See [LICENSE](./LICENSE) for details.

## 👤 Author

**Aashir Athar**

[![GitHub](https://img.shields.io/badge/GitHub-aashir--athar-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aashir-athar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-aashirathar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aashirathar/)
[![X](https://img.shields.io/badge/X_(Twitter)-aashirathar-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/aashirathar)

<div align="center">
<sub>Built by <a href="https://github.com/aashir-athar">aashir-athar</a> · If MiND-Shot helped you, consider leaving a ⭐</sub>
<br/><br/>
<sub><strong>Keywords:</strong> free crypto trading bot · crypto trading signals · algorithmic trading bot · automated trading bot · Bitcoin trading bot · Ethereum trading signals · BTC ETH signals · mean-reversion strategy · backtesting dashboard · quantitative trading · quant trading Python · GitHub Actions trading bot · serverless trading bot · Telegram crypto signals · Telegram trading alerts · Make.com / n8n / Pipedream webhook · Kraken API · Binance whale flow · machine-learning trading · open-source trading bot · no-cost crypto signals</sub>
</div>
</div>
