"""
FastAPI application for Aladdin-AI enterprise platform.
"""
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Any
from pydantic import BaseModel

from aladdin.models import (
    PortfolioState,
    RiskMetrics,
    StressScenarioResult,
    FactorExposure,
    BlackLittermanView,
    AICopilotQuery
)
from aladdin.market_data import market_data
from aladdin.risk_engine import risk_engine
from aladdin.portfolio_optimizer import portfolio_optimizer
from aladdin.ai_copilot import aladdin_ai_copilot

app = FastAPI(
    title="ASHFX-AI Financial Risk & Intelligence Network",
    description="Enterprise multi-asset portfolio management, Black-Litterman optimization, and stress risk platform.",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory active portfolio state
current_portfolio = market_data.get_default_portfolio(total_aum=100_000_000.0)

@app.get("/api/portfolio", response_model=PortfolioState)
def get_portfolio():
    return current_portfolio

class RebalanceRequest(BaseModel):
    weights: Dict[str, float]  # ticker -> weight (0.0 - 1.0)
    total_aum: float = 100_000_000.0

@app.post("/api/portfolio/rebalance", response_model=PortfolioState)
def rebalance_portfolio(req: RebalanceRequest):
    global current_portfolio
    total_w = sum(req.weights.values())
    if abs(total_w - 1.0) > 0.05 and total_w > 0:
        # Normalize weights if slight rounding
        normalized_w = {k: v / total_w for k, v in req.weights.items()}
    else:
        normalized_w = req.weights

    new_holdings = []
    for ticker in market_data.tickers:
        w = normalized_w.get(ticker, 0.0)
        u = market_data.universe[ticker]
        mkt_val = req.total_aum * w
        shares = mkt_val / u["price"] if u["price"] > 0 else 0.0

        new_holdings.append({
            "ticker": ticker,
            "name": u["name"],
            "asset_class": u["class"],
            "sector": u["sector"],
            "weight": round(w, 4),
            "current_price": u["price"],
            "shares": round(shares, 4),
            "market_value": round(mkt_val, 2),
            "beta": u["beta"],
            "duration": u["duration"],
            "dividend_yield": u["yield"],
            "annual_volatility": u["vol"],
            "daily_volume": u["daily_volume"]
        })

    current_portfolio = PortfolioState(
        portfolio_name="ASHFX Global Alpha & Multi-Asset Parity Fund",
        base_currency="USD",
        total_aum=req.total_aum,
        cash_balance=req.total_aum * normalized_w.get("BIL", 0.03),
        holdings=new_holdings,
        benchmark="S&P 500 Total Return"
    )
    return current_portfolio

@app.get("/api/risk/metrics", response_model=RiskMetrics)
def get_risk_metrics():
    return risk_engine.compute_risk_metrics(current_portfolio)

@app.get("/api/risk/stress-test", response_model=List[StressScenarioResult])
def get_stress_tests():
    return risk_engine.run_stress_test_scenarios(current_portfolio)

@app.get("/api/risk/factors", response_model=List[FactorExposure])
def get_factor_exposures():
    return risk_engine.compute_factor_exposures(current_portfolio)

@app.get("/api/risk/liquidity")
def get_liquidity_profile():
    return risk_engine.compute_liquidity_profile(current_portfolio)

@app.get("/api/risk/correlation")
def get_correlation_matrix():
    tickers = market_data.tickers
    corr = market_data.corr_matrix
    return {
        "tickers": tickers,
        "matrix": [[round(float(corr[i, j]), 2) for j in range(len(tickers))] for i in range(len(tickers))]
    }

@app.get("/api/risk/distribution")
def get_return_distribution():
    """Returns Monte Carlo simulated daily returns histogram and VaR threshold lines."""
    weights, tickers = risk_engine.calculate_portfolio_weights_vector(current_portfolio)
    cov_matrix = market_data.cov_matrix / 252.0
    L = np.linalg.cholesky(cov_matrix)
    z = np.random.randn(10000, len(tickers))
    sim_returns = (z @ L.T) @ weights
    
    # Histogram counts and bin edges
    counts, bin_edges = np.histogram(sim_returns, bins=50, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    sorted_returns = np.sort(sim_returns)
    var_95 = float(-sorted_returns[int(len(sorted_returns) * 0.05)])
    var_99 = float(-sorted_returns[int(len(sorted_returns) * 0.01)])
    
    return {
        "bin_centers": [round(float(c) * 100, 3) for c in centers],
        "density": [round(float(cnt), 3) for cnt in counts],
        "var_95_pct": round(var_95 * 100, 3),
        "var_99_pct": round(var_99 * 100, 3)
    }

@app.get("/api/optimize/frontier")
def get_efficient_frontier():
    return portfolio_optimizer.generate_efficient_frontier()

class BLRequest(BaseModel):
    views: List[BlackLittermanView]
    delta: float = 2.5
    tau: float = 0.05

@app.post("/api/optimize/black-litterman")
def run_black_litterman(req: BLRequest):
    return portfolio_optimizer.run_black_litterman(views=req.views, delta=req.delta, tau=req.tau)

@app.get("/api/optimize/risk-parity")
def get_risk_parity():
    return portfolio_optimizer.optimize_risk_parity()

@app.get("/api/ai/macro-regime")
def get_macro_regime():
    return aladdin_ai_copilot.detect_macro_regime()

@app.get("/api/ai/diagnostic")
def get_portfolio_diagnostic():
    return aladdin_ai_copilot.generate_portfolio_diagnostic(current_portfolio)

@app.post("/api/ai/query")
def copilot_query(req: AICopilotQuery):
    answer = aladdin_ai_copilot.answer_query(req.query, current_portfolio)
    return {"query": req.query, "answer": answer}

@app.get("/api/ai/memo")
def get_investment_memo():
    memo = aladdin_ai_copilot.generate_investment_committee_memo(current_portfolio)
    return {"memo": memo}

# Mount static files for Aladdin Terminal frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"status": "Aladdin-AI API is running. UI index.html not found."}
