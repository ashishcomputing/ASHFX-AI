# Aladdin-AI: Institutional Financial Intelligence & Risk Network

> **Enterprise Multi-Asset Portfolio Management, Black-Litterman Optimization & Stress Risk Platform**  
> *Architected after BlackRock Aladdin® and institutional quantitative risk systems.*

---

## 🏛️ Architecture Overview

**Aladdin-AI** brings institutional-grade portfolio construction, enterprise risk analytics, and quantitative AI reasoning into a cohesive system. It models an active **$100,000,000 USD** institutional multi-asset book across Equities, Fixed Income, Commodities, Digital Assets, and Cash.

```
aladdin-ai/
├── backend/
│   └── aladdin/
│       ├── models.py              # Pydantic schemas for portfolios, risk metrics, and views
│       ├── market_data.py         # Multi-asset universe, covariance calibration & return generator
│       ├── risk_engine.py         # VaR (Parametric, Historical, Monte Carlo), CVaR, Stress Testing
│       ├── portfolio_optimizer.py # Markowitz Frontier, Black-Litterman, Risk Parity (ERC)
│       ├── ai_copilot.py          # Quantitative Copilot, Macro Classifier, Committee Memo Generator
│       └── api.py                 # FastAPI endpoints & static UI routing
├── frontend/
│   └── static/
│       └── index.html             # Institutional dark Bloomberg/Aladdin Terminal UI
├── run.sh                         # Quick startup script
└── README.md
```

---

## 🔬 Core Quantitative Modules

### 1. Aladdin Enterprise Risk Analytics (`risk_engine.py`)
- **Value at Risk (VaR)**:
  - **Parametric VaR**: Higher-moment adjusted Cornish-Fisher expansion capturing skewness and kurtosis.
  - **Historical Simulation VaR**: Empirical quantile calculation over historical daily distributions.
  - **Monte Carlo VaR**: 10,000 correlated paths using Cholesky factorized covariance ($\Sigma = L L^T$).
- **Expected Shortfall / Conditional VaR (CVaR)**:
  - Measures expected tail loss conditional on breaching the 95% and 99% VaR thresholds ($E[L \mid L > \text{VaR}_\alpha]$).
- **Macro Stress Testing Matrix**:
  - **2008 Lehman Global Financial Crisis**: Systemic liquidity freeze, subprime liquidation, duration rally.
  - **2020 March COVID-19 Shock**: Global lockdown and emergency rate cuts.
  - **2022 Fed Aggressive Tightening**: 500+ bps rate surge, duration crash, stagflationary energy rally.
  - **Semiconductor & AI De-Rating**: Concentration mean-reversion.
  - **Geopolitical Energy Shock**: Crude supply shock and headline CPI surge.
- **Liquidity & Days-to-Liquidate Analysis**:
  - Almgren-Chriss square-root market impact slippage under normal (15% ADV) and stressed (5% ADV) conditions.

### 2. Portfolio Optimization & Black-Litterman (`portfolio_optimizer.py`)
- **Markowitz Modern Portfolio Theory**:
  - Continuous Quadratic Programming (SLSQP) mapping out the full Efficient Frontier.
  - Tangency portfolio (Maximum Sharpe Ratio) and Minimum Variance portfolio.
- **Black-Litterman Asset Allocation**:
  - Reverse-optimizes market equilibrium implied returns: $\Pi = \delta \Sigma w_{mkt}$.
  - Integrates subjective AI/Analyst views $P \cdot E(R) = Q + \varepsilon$ with Idzorek confidence weighting.
  - Calculates posterior expected returns and optimal weights.
- **Equal Risk Contribution (ERC / Risk Parity)**:
  - Solves for asset weights where each component contributes an equal share ($1/N$) of total portfolio volatility.

### 3. Aladdin AI Copilot & Macro Intelligence (`ai_copilot.py`)
- **Macroeconomic Regime Detection**: Classifies market into Goldilocks, Reflation, Stagflation, or Contraction regimes.
- **Tail-Risk Diagnostics**: Identifies equity concentration, beta drift, and duration exposure.
- **Tactical Hedging Engine**: Generates zero-cost collar structures, duration immunizations, and convexity hedges.
- **Formal Investment Committee Memoranda**: Executive markdown reports ready for Chief Investment Officers.

---

## 🚀 Getting Started

### Launch the Platform
```bash
/Users/admin/aladdin-ai/run.sh
```
Or directly with uvicorn:
```bash
PYTHONPATH=/Users/admin/aladdin-ai/backend /Users/admin/.venv/bin/uvicorn aladdin.api:app --host 0.0.0.0 --port 8888 --reload
```

### Access the Web Terminal
Open your browser to:
**`http://localhost:8888`**

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/portfolio` | `GET` | Current portfolio holdings, weights, and market values |
| `/api/portfolio/rebalance` | `POST` | Update asset weights and trigger instantaneous risk re-computation |
| `/api/risk/metrics` | `GET` | Multi-methodology VaR (95%/99%), CVaR, Sharpe, Sortino, Duration |
| `/api/risk/stress-test` | `GET` | Performance under 2008 GFC, 2020 COVID, 2022 Rate Shock, etc. |
| `/api/risk/distribution` | `GET` | 10,000-path Monte Carlo distribution histogram & cutoff lines |
| `/api/risk/factors` | `GET` | Fama-French factor exposures & risk contributions |
| `/api/risk/correlation` | `GET` | 15x15 asset correlation matrix |
| `/api/risk/liquidity` | `GET` | Days-to-liquidate and market impact slippage |
| `/api/optimize/frontier` | `GET` | Markowitz efficient frontier curve points & tangency portfolio |
| `/api/optimize/black-litterman` | `POST` | Execute Black-Litterman with subjective investor views |
| `/api/optimize/risk-parity` | `GET` | Equal Risk Contribution (ERC) weights |
| `/api/ai/query` | `POST` | Natural language queries to Aladdin Copilot |
| `/api/ai/macro-regime` | `GET` | Macro cycle signals, inflation trend, and recommended posture |
| `/api/ai/memo` | `GET` | Formal Chief Investment Officer Investment Committee Memorandum |
