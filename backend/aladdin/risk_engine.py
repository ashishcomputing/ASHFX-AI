"""
Enterprise Risk Analytics Engine for Aladdin-AI.
Implements BlackRock Aladdin-style risk metrics:
- Multi-Method Value-at-Risk (Parametric, Historical, Monte Carlo)
- Expected Shortfall / Conditional VaR (CVaR)
- Institutional Scenario Stress Testing & Macro Shocks
- Factor Risk Decomposition
- Liquidity & Days-to-Liquidate Analysis
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any
from aladdin.models import (
    PortfolioState,
    RiskMetrics,
    StressScenarioResult,
    FactorExposure
)
from aladdin.market_data import market_data

class AladdinRiskEngine:
    def __init__(self):
        self.market_data = market_data

    def calculate_portfolio_weights_vector(self, portfolio: PortfolioState) -> Tuple[np.ndarray, List[str]]:
        """Extracts normalized weights vector aligned with universe tickers."""
        tickers = self.market_data.tickers
        w_dict = {h.ticker: h.weight for h in portfolio.holdings}
        weights = np.array([w_dict.get(t, 0.0) for t in tickers])
        total_w = np.sum(weights)
        if total_w > 0:
            weights = weights / total_w
        return weights, tickers

    def compute_risk_metrics(self, portfolio: PortfolioState) -> RiskMetrics:
        """
        Computes institutional risk metrics: VaR (all 3 methodologies), CVaR,
        Sharpe, Sortino, Drawdown, Duration, and Diversification.
        """
        weights, tickers = self.calculate_portfolio_weights_vector(portfolio)
        returns_df = self.market_data.historical_returns_df[tickers]
        
        # Portfolio historical daily return series
        daily_returns = returns_df.values @ weights
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns, ddof=1)
        annual_return = mean_daily * 252.0
        annual_vol = std_daily * np.sqrt(252.0)

        # Risk-free rate (assumed 4.5% annualized)
        rf_annual = 0.045
        rf_daily = rf_annual / 252.0

        # Sharpe Ratio
        excess_return = annual_return - rf_annual
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0.0

        # Downside Deviation & Sortino Ratio
        downside_returns = daily_returns[daily_returns < rf_daily] - rf_daily
        downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252.0) if len(downside_returns) > 0 else annual_vol
        sortino = excess_return / downside_std if downside_std > 0 else 0.0

        # Max Historical Drawdown
        cum_returns = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))

        # 1. PARAMETRIC VaR (Cornish-Fisher expansion for skewness & kurtosis)
        skew = stats.skew(daily_returns)
        kurt = stats.kurtosis(daily_returns)
        
        def cornish_fisher_z(alpha):
            z = stats.norm.ppf(1 - alpha)
            z_cf = z + (z**2 - 1)*skew/6.0 + (z**3 - 3*z)*kurt/24.0 - (2*z**3 - 5*z)*(skew**2)/36.0
            return z_cf

        z95 = cornish_fisher_z(0.95)
        z99 = cornish_fisher_z(0.99)
        q95 = mean_daily + z95 * std_daily
        q99 = mean_daily + z99 * std_daily
        var_95_param = float(max(0.0, -q95))
        var_99_param = float(max(0.0, -q99))

        # 2. HISTORICAL SIMULATION VaR & CVaR
        sorted_returns = np.sort(daily_returns)
        idx_95 = int(np.floor(len(sorted_returns) * 0.05))
        idx_99 = int(np.floor(len(sorted_returns) * 0.01))

        var_95_hist = float(max(0.0, -sorted_returns[idx_95]))
        var_99_hist = float(max(0.0, -sorted_returns[idx_99]))

        cvar_95 = float(max(0.0, -np.mean(sorted_returns[:idx_95]))) if idx_95 > 0 else var_95_hist
        cvar_99 = float(max(0.0, -np.mean(sorted_returns[:idx_99]))) if idx_99 > 0 else var_99_hist

        # 3. MONTE CARLO VaR (10,000 paths with Cholesky correlated covariance)
        n_mc_sims = 10000
        cov_matrix = self.market_data.cov_matrix / 252.0
        L = np.linalg.cholesky(cov_matrix)
        z = np.random.randn(n_mc_sims, len(tickers))
        mc_daily_returns = (z @ L.T) @ weights + mean_daily
        sorted_mc = np.sort(mc_daily_returns)
        mc_idx_95 = int(np.floor(n_mc_sims * 0.05))
        mc_idx_99 = int(np.floor(n_mc_sims * 0.01))
        var_95_mc = float(max(0.0, -sorted_mc[mc_idx_95]))
        var_99_mc = float(max(0.0, -sorted_mc[mc_idx_99]))

        # Portfolio Beta to Benchmark (S&P 500 approximated via Equity Holdings)
        betas = np.array([self.market_data.universe[t]["beta"] for t in tickers])
        portfolio_beta = float(np.sum(weights * betas))

        # Effective Duration for Fixed Income components
        durations = np.array([self.market_data.universe[t]["duration"] for t in tickers])
        effective_duration = float(np.sum(weights * durations))

        # Diversification Ratio: Weighted avg asset volatility / Portfolio volatility
        vols = np.array([self.market_data.universe[t]["vol"] for t in tickers])
        weighted_vol = np.sum(weights * vols)
        div_ratio = float(weighted_vol / annual_vol) if annual_vol > 0 else 1.0

        aum = portfolio.total_aum
        var_95_usd = var_95_hist * aum
        cvar_95_usd = cvar_95 * aum

        return RiskMetrics(
            portfolio_value=round(aum, 2),
            expected_annual_return=round(annual_return, 4),
            annual_volatility=round(annual_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            portfolio_beta=round(portfolio_beta, 3),
            max_drawdown_historical=round(max_drawdown, 4),
            var_95_daily_parametric=round(var_95_param, 4),
            var_99_daily_parametric=round(var_99_param, 4),
            var_95_daily_historical=round(var_95_hist, 4),
            var_99_daily_historical=round(var_99_hist, 4),
            var_95_daily_monte_carlo=round(var_95_mc, 4),
            var_99_daily_monte_carlo=round(var_99_mc, 4),
            cvar_95_daily=round(cvar_95, 4),
            cvar_99_daily=round(cvar_99, 4),
            var_95_amount_usd=round(var_95_usd, 2),
            cvar_95_amount_usd=round(cvar_95_usd, 2),
            diversification_ratio=round(div_ratio, 3),
            effective_duration=round(effective_duration, 2)
        )

    def run_stress_test_scenarios(self, portfolio: PortfolioState) -> List[StressScenarioResult]:
        """
        Executes institutional stress testing across legendary market crises and macro shocks.
        """
        weights, tickers = self.calculate_portfolio_weights_vector(portfolio)
        aum = portfolio.total_aum

        # Macro shocks calibration: Asset class return shocks under historical crisis conditions
        scenarios = [
            {
                "id": "gfc_2008",
                "name": "2008 Lehman Global Financial Crisis",
                "description": "Subprime mortgage systemic collapse, acute credit liquidity freeze, and global equities liquidation.",
                "benchmark_shock": -0.42,
                "shocks": {
                    "NVDA": -0.58, "AAPL": -0.45, "MSFT": -0.40, "GOOGL": -0.44,
                    "AMZN": -0.48, "JPM": -0.62, "XOM": -0.32, "LLY": -0.22,
                    "TLT": 0.28,  # Huge flight to quality rally in Treasuries
                    "LQD": -0.14, "HYG": -0.36, "GLD": 0.21, "USO": -0.65,
                    "BTC": -0.68, "BIL": 0.015
                }
            },
            {
                "id": "covid_2020",
                "name": "March 2020 COVID-19 Liquidity Crunch",
                "description": "Global economic lockdown, panic-selling of risk assets followed by emergency central bank intervention.",
                "benchmark_shock": -0.34,
                "shocks": {
                    "NVDA": -0.22, "AAPL": -0.24, "MSFT": -0.21, "GOOGL": -0.28,
                    "AMZN": -0.15, "JPM": -0.38, "XOM": -0.52, "LLY": -0.14,
                    "TLT": 0.18, "LQD": -0.06, "HYG": -0.22, "GLD": 0.08,
                    "USO": -0.72, "BTC": -0.45, "BIL": 0.005
                }
            },
            {
                "id": "rate_shock_2022",
                "name": "2022 Fed Aggressive Tightening / Stagflation",
                "description": "Highest inflation in 40 years triggering 500+ bps Fed rate hikes. Simultaneous equity and bond bear market.",
                "benchmark_shock": -0.19,
                "shocks": {
                    "NVDA": -0.50, "AAPL": -0.27, "MSFT": -0.28, "GOOGL": -0.39,
                    "AMZN": -0.50, "JPM": -0.15, "XOM": 0.68,  # Energy surged
                    "LLY": 0.32,   # Defensive pharma outperformance
                    "TLT": -0.31,  # Historic duration crash
                    "LQD": -0.18, "HYG": -0.11, "GLD": -0.01, "USO": 0.35,
                    "BTC": -0.64, "BIL": 0.04
                }
            },
            {
                "id": "tech_unwind",
                "name": "Semiconductor & AI Multiple De-Rating",
                "description": "Tech concentration mean-reversion, enterprise Capex pause, and rotation to value and defensive dividend assets.",
                "benchmark_shock": -0.14,
                "shocks": {
                    "NVDA": -0.38, "AAPL": -0.18, "MSFT": -0.16, "GOOGL": -0.20,
                    "AMZN": -0.18, "JPM": 0.05, "XOM": 0.08, "LLY": 0.06,
                    "TLT": 0.08, "LQD": 0.03, "HYG": -0.02, "GLD": 0.12,
                    "USO": 0.02, "BTC": -0.25, "BIL": 0.02
                }
            },
            {
                "id": "geopolitical_oil_shock",
                "name": "Middle East Energy Escalation & Strait of Hormuz Blockade",
                "description": "Stagflationary crude oil supply spike above $140/bbl, driving headline CPI inflation and delaying rate cuts.",
                "benchmark_shock": -0.12,
                "shocks": {
                    "NVDA": -0.14, "AAPL": -0.10, "MSFT": -0.09, "GOOGL": -0.11,
                    "AMZN": -0.16, "JPM": -0.05, "XOM": 0.45, "LLY": -0.02,
                    "TLT": -0.09, "LQD": -0.05, "HYG": -0.07, "GLD": 0.24,
                    "USO": 0.85, "BTC": -0.08, "BIL": 0.02
                }
            }
        ]

        results = []
        for s in scenarios:
            shock_map = s["shocks"]
            asset_returns = np.array([shock_map.get(t, -0.10) for t in tickers])
            port_ret = float(np.sum(weights * asset_returns))
            port_pnl = port_ret * aum
            bench_ret = s["benchmark_shock"]
            alpha = port_ret - bench_ret

            # Best and worst impacted assets in portfolio
            active_mask = weights > 0.001
            active_tickers = [tickers[i] for i in range(len(tickers)) if active_mask[i]]
            active_shocks = [asset_returns[i] for i in range(len(tickers)) if active_mask[i]]

            worst_idx = np.argmin(active_shocks)
            best_idx = np.argmax(active_shocks)

            worst_asset = active_tickers[worst_idx]
            worst_ret = active_shocks[worst_idx]
            best_asset = active_tickers[best_idx]
            best_ret = active_shocks[best_idx]

            if port_ret > -0.10:
                risk_status = "RESILIENT: Portfolio exhibits exceptional multi-asset hedging and defensive duration buffer."
            elif port_ret > -0.22:
                risk_status = "MODERATE: Portfolio sustains manageable drawdown buffered by decorrelated assets."
            else:
                risk_status = "VULNERABLE: Significant drawdown driven by growth equity and cyclical beta concentration."

            results.append(StressScenarioResult(
                scenario_id=s["id"],
                scenario_name=s["name"],
                description=s["description"],
                portfolio_return_pct=round(port_ret * 100, 2),
                portfolio_pnl_usd=round(port_pnl, 2),
                benchmark_return_pct=round(bench_ret * 100, 2),
                relative_alpha_pct=round(alpha * 100, 2),
                worst_asset=f"{worst_asset} ({self.market_data.universe[worst_asset]['name']})",
                worst_asset_return_pct=round(worst_ret * 100, 2),
                best_asset=f"{best_asset} ({self.market_data.universe[best_asset]['name']})",
                best_asset_return_pct=round(best_ret * 100, 2),
                risk_assessment=risk_status
            ))

        return results

    def compute_factor_exposures(self, portfolio: PortfolioState) -> List[FactorExposure]:
        """
        Calculates Fama-French multi-factor sensitivities & risk contribution.
        """
        weights, tickers = self.calculate_portfolio_weights_vector(portfolio)

        factor_loadings = {
            "Market Beta (Equity Risk Premium)": {
                "AAPL": 1.12, "MSFT": 1.18, "NVDA": 1.75, "GOOGL": 1.15, "AMZN": 1.25,
                "JPM": 0.95, "XOM": 0.65, "LLY": 0.70, "TLT": -0.25, "LQD": 0.15,
                "HYG": 0.45, "GLD": 0.05, "USO": 0.35, "BTC": 1.40, "BIL": 0.01,
                "benchmark": 1.00
            },
            "Momentum (High vs Low Past Return)": {
                "NVDA": 1.60, "LLY": 1.10, "MSFT": 0.65, "AAPL": 0.40, "GOOGL": 0.35,
                "AMZN": 0.45, "JPM": 0.30, "XOM": 0.10, "TLT": -0.65, "LQD": -0.20,
                "HYG": 0.15, "GLD": 0.45, "USO": -0.10, "BTC": 0.85, "BIL": 0.0,
                "benchmark": 0.15
            },
            "Quality / High Profitability": {
                "AAPL": 1.45, "MSFT": 1.55, "GOOGL": 1.30, "NVDA": 1.40, "AMZN": 0.90,
                "JPM": 0.85, "XOM": 0.75, "LLY": 1.25, "TLT": 1.00, "LQD": 0.80,
                "HYG": -0.70, "GLD": 0.0, "USO": -0.40, "BTC": -0.80, "BIL": 1.0,
                "benchmark": 0.45
            },
            "Value vs Growth (HML)": {
                "XOM": 1.45, "JPM": 0.95, "HYG": 0.40, "LQD": 0.20, "AAPL": -0.40,
                "MSFT": -0.65, "NVDA": -1.20, "GOOGL": -0.30, "AMZN": -0.55, "LLY": -0.85,
                "TLT": 0.10, "GLD": 0.0, "USO": 0.60, "BTC": -0.75, "BIL": 0.0,
                "benchmark": 0.05
            },
            "Interest Rate Duration Exposure": {
                "TLT": 1.70, "LQD": 0.85, "HYG": 0.35, "AAPL": -0.20, "MSFT": -0.25,
                "NVDA": -0.45, "GOOGL": -0.20, "AMZN": -0.30, "JPM": -0.15, "XOM": -0.05,
                "LLY": -0.10, "GLD": 0.35, "USO": -0.10, "BTC": -0.30, "BIL": 0.02,
                "benchmark": -0.10
            }
        }

        results = []
        for factor_name, loading_dict in factor_loadings.items():
            loadings = np.array([loading_dict.get(t, 0.0) for t in tickers])
            port_exposure = float(np.sum(weights * loadings))
            t_stat = port_exposure * np.sqrt(252) / 1.5
            bench_exp = loading_dict.get("benchmark", 0.0)
            risk_contrib = abs(port_exposure) / (abs(port_exposure) + 1.0) * 100

            results.append(FactorExposure(
                factor_name=factor_name,
                exposure=round(port_exposure, 3),
                t_stat=round(t_stat, 2),
                risk_contribution_pct=round(risk_contrib, 1),
                benchmark_exposure=round(bench_exp, 3)
            ))

        return results

    def compute_liquidity_profile(self, portfolio: PortfolioState) -> Dict[str, Any]:
        """
        Aladdin Days-to-Liquidate Analysis:
        Calculates how many trading days are required to liquidate each position
        under normal market conditions (assuming max 15% daily participation rate)
        and under stressed market conditions (5% participation rate).
        """
        holdings_liquidity = []
        total_aum = portfolio.total_aum

        for h in portfolio.holdings:
            pos_usd = h.market_value
            daily_vol = h.daily_volume
            
            # Days to liquidate under 15% ADV participation
            days_normal = pos_usd / (0.15 * daily_vol) if daily_vol > 0 else 999.0
            # Days under 5% ADV stressed participation
            days_stressed = pos_usd / (0.05 * daily_vol) if daily_vol > 0 else 999.0
            
            # Almgren-Chriss estimated market impact slippage in bps
            # Slippage ~ sigma * sqrt(pos / ADV)
            vol = h.annual_volatility
            impact_bps = vol * np.sqrt(pos_usd / daily_vol) * 10000.0 if daily_vol > 0 else 100.0

            holdings_liquidity.append({
                "ticker": h.ticker,
                "name": h.name,
                "market_value": h.market_value,
                "weight_pct": round(h.weight * 100, 2),
                "daily_volume_usd": h.daily_volume,
                "days_to_liquidate_normal": round(days_normal, 2),
                "days_to_liquidate_stressed": round(days_stressed, 2),
                "estimated_impact_bps": round(impact_bps, 1),
                "liquidity_tier": "Tier 1 (< 1 Day)" if days_normal < 1.0 else ("Tier 2 (1-3 Days)" if days_normal < 3.0 else "Tier 3 (> 3 Days)")
            })

        # Portfolio weighted days to liquidate
        total_val = sum(item["market_value"] for item in holdings_liquidity)
        weighted_days_normal = sum(item["days_to_liquidate_normal"] * item["market_value"] for item in holdings_liquidity) / total_val if total_val > 0 else 0.0

        return {
            "portfolio_weighted_days_normal": round(weighted_days_normal, 2),
            "total_liquidatable_24h_pct": round(sum(item["weight_pct"] for item in holdings_liquidity if item["days_to_liquidate_normal"] <= 1.0), 1),
            "positions": holdings_liquidity
        }

# Global risk engine singleton
risk_engine = AladdinRiskEngine()
