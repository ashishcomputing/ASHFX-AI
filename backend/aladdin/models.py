"""
Data models and schemas for Aladdin-AI enterprise finance platform.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class AssetHolding(BaseModel):
    ticker: str
    name: str
    asset_class: str  # Equity, Fixed Income, Commodity, Crypto, Cash
    sector: Optional[str] = "General"
    weight: float = Field(..., ge=0.0, le=1.0)
    current_price: float
    shares: float
    market_value: float
    beta: float = 1.0
    duration: float = 0.0  # For fixed income
    dividend_yield: float = 0.0
    annual_volatility: float = 0.20
    daily_volume: float = 10000000.0

class PortfolioState(BaseModel):
    portfolio_name: str = "ASHFX Global Alpha & Multi-Asset Parity Fund"
    base_currency: str = "USD"
    total_aum: float = 100_000_000.0  # $100M baseline AUM
    cash_balance: float = 5_000_000.0
    holdings: List[AssetHolding]
    benchmark: str = "S&P 500 (SPY)"

class RiskMetrics(BaseModel):
    portfolio_value: float
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    portfolio_beta: float
    max_drawdown_historical: float
    var_95_daily_parametric: float
    var_99_daily_parametric: float
    var_95_daily_historical: float
    var_99_daily_historical: float
    var_95_daily_monte_carlo: float
    var_99_daily_monte_carlo: float
    cvar_95_daily: float  # Expected Shortfall
    cvar_99_daily: float
    var_95_amount_usd: float
    cvar_95_amount_usd: float
    diversification_ratio: float
    effective_duration: float

class StressScenarioResult(BaseModel):
    scenario_id: str
    scenario_name: str
    description: str
    portfolio_return_pct: float
    portfolio_pnl_usd: float
    benchmark_return_pct: float
    relative_alpha_pct: float
    worst_asset: str
    worst_asset_return_pct: float
    best_asset: str
    best_asset_return_pct: float
    risk_assessment: str

class FactorExposure(BaseModel):
    factor_name: str
    exposure: float
    t_stat: float
    risk_contribution_pct: float
    benchmark_exposure: float

class BlackLittermanView(BaseModel):
    view_type: str = "absolute"  # 'absolute' or 'relative'
    asset: str
    relative_asset: Optional[str] = None
    expected_excess_return: float  # e.g. 0.08 for +8%
    confidence: float = Field(0.65, ge=0.01, le=1.0)  # Confidence between 1% and 100%

class OptimizationRequest(BaseModel):
    target_risk_free_rate: float = 0.045  # 4.5% US 3M rate
    allow_short: bool = False
    views: Optional[List[BlackLittermanView]] = None

class AICopilotQuery(BaseModel):
    query: str
    portfolio_context: Optional[Dict[str, Any]] = None
    include_stress_test: bool = True
