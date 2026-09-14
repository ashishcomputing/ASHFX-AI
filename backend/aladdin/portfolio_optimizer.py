"""
Portfolio Optimization Engine for Aladdin-AI.
Implements:
1. Markowitz Modern Portfolio Theory (Efficient Frontier, Max Sharpe, Min Volatility)
2. Black-Litterman Asset Allocation with subjective AI/Analyst views & confidence calibration
3. Risk Parity / Equal Risk Contribution (ERC)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Optional
from aladdin.market_data import market_data
from aladdin.models import BlackLittermanView

class AladdinPortfolioOptimizer:
    def __init__(self):
        self.market_data = market_data
        self.tickers = market_data.tickers
        self.n_assets = len(self.tickers)
        self.cov_matrix = market_data.cov_matrix
        self.expected_returns = np.array([market_data.universe[t]["expected_return"] for t in self.tickers])
        self.rf = 0.045  # 4.5% risk free rate

    def _portfolio_metrics(self, weights: np.ndarray, custom_returns: Optional[np.ndarray] = None, custom_cov: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Calculates expected return, volatility, and Sharpe ratio for given weights."""
        returns = custom_returns if custom_returns is not None else self.expected_returns
        cov = custom_cov if custom_cov is not None else self.cov_matrix

        p_return = float(np.sum(weights * returns))
        p_var = float(weights.T @ cov @ weights)
        p_vol = float(np.sqrt(max(1e-8, p_var)))
        sharpe = float((p_return - self.rf) / p_vol) if p_vol > 0 else 0.0
        return p_return, p_vol, sharpe

    def optimize_min_volatility(self, custom_returns: Optional[np.ndarray] = None, custom_cov: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculates minimum variance portfolio (long-only)."""
        cov = custom_cov if custom_cov is not None else self.cov_matrix

        def objective(w):
            return np.sqrt(w.T @ cov @ w)

        init_w = np.ones(self.n_assets) / self.n_assets
        bounds = tuple((0.0, 0.40) for _ in range(self.n_assets))  # Cap single asset at 40%
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = res.x / np.sum(res.x)
        ret, vol, sharpe = self._portfolio_metrics(opt_w, custom_returns, custom_cov)

        return {
            "type": "Minimum Volatility",
            "weights": {self.tickers[i]: round(float(opt_w[i]), 4) for i in range(self.n_assets)},
            "expected_return": round(ret, 4),
            "volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 3)
        }

    def optimize_max_sharpe(self, custom_returns: Optional[np.ndarray] = None, custom_cov: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculates Maximum Sharpe ratio (Tangency) portfolio."""
        returns = custom_returns if custom_returns is not None else self.expected_returns
        cov = custom_cov if custom_cov is not None else self.cov_matrix

        def neg_sharpe(w):
            p_ret = np.sum(w * returns)
            p_vol = np.sqrt(max(1e-8, w.T @ cov @ w))
            return -(p_ret - self.rf) / p_vol

        init_w = np.ones(self.n_assets) / self.n_assets
        bounds = tuple((0.0, 0.35) for _ in range(self.n_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(neg_sharpe, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = res.x / np.sum(res.x)
        ret, vol, sharpe = self._portfolio_metrics(opt_w, custom_returns, custom_cov)

        return {
            "type": "Maximum Sharpe Ratio",
            "weights": {self.tickers[i]: round(float(opt_w[i]), 4) for i in range(self.n_assets)},
            "expected_return": round(ret, 4),
            "volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 3)
        }

    def generate_efficient_frontier(self, n_points: int = 25) -> Dict[str, Any]:
        """Generates the Markowitz Efficient Frontier curve."""
        min_vol_res = self.optimize_min_volatility()
        max_sharpe_res = self.optimize_max_sharpe()

        min_ret = min_vol_res["expected_return"]
        max_ret = max(self.expected_returns) * 0.90
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_points = []
        bounds = tuple((0.0, 0.40) for _ in range(self.n_assets))

        for target_r in target_returns:
            def objective(w):
                return np.sqrt(w.T @ self.cov_matrix @ w)

            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sum(w * self.expected_returns) - target_r}
            )

            res = minimize(objective, np.ones(self.n_assets) / self.n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
            if res.success:
                opt_w = res.x / np.sum(res.x)
                p_vol = float(np.sqrt(opt_w.T @ self.cov_matrix @ opt_w))
                sharpe = float((target_r - self.rf) / p_vol) if p_vol > 0 else 0.0
                frontier_points.append({
                    "return": round(float(target_r), 4),
                    "volatility": round(p_vol, 4),
                    "sharpe": round(sharpe, 3)
                })

        return {
            "frontier": frontier_points,
            "min_volatility": min_vol_res,
            "max_sharpe": max_sharpe_res
        }

    def run_black_litterman(self, views: List[BlackLittermanView], delta: float = 2.5, tau: float = 0.05) -> Dict[str, Any]:
        """
        Executes Black-Litterman Asset Allocation.
        Combines Market Equilibrium Prior (reverse-engineered via CAPM) with subjective Views.
        """
        sigma = self.cov_matrix
        # Baseline market capitalization weights proxy
        w_mkt = np.array([
            0.15, 0.15, 0.14, 0.10, 0.08, 0.06, 0.04, 0.04,  # Equities
            0.10, 0.06, 0.03,                                # Bonds
            0.03, 0.01,                                      # Commodities
            0.01,                                            # BTC
            0.00                                             # Cash
        ])
        w_mkt = w_mkt / np.sum(w_mkt)

        # 1. Implied Market Equilibrium Returns: Pi = delta * Sigma * w_mkt
        pi = delta * (sigma @ w_mkt)

        if not views:
            # If no views, posterior is simply the market equilibrium
            opt_bl = self.optimize_max_sharpe(custom_returns=pi, custom_cov=sigma)
            return {
                "posterior_expected_returns": {self.tickers[i]: round(float(pi[i]), 4) for i in range(self.n_assets)},
                "prior_implied_returns": {self.tickers[i]: round(float(pi[i]), 4) for i in range(self.n_assets)},
                "optimal_weights": opt_bl["weights"],
                "expected_return": opt_bl["expected_return"],
                "volatility": opt_bl["volatility"],
                "sharpe_ratio": opt_bl["sharpe_ratio"],
                "views_applied": 0
            }

        k = len(views)
        P = np.zeros((k, self.n_assets))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, v in enumerate(views):
            if v.asset in self.tickers:
                idx1 = self.tickers.index(v.asset)
                if v.view_type == "relative" and v.relative_asset and v.relative_asset in self.tickers:
                    idx2 = self.tickers.index(v.relative_asset)
                    P[i, idx1] = 1.0
                    P[i, idx2] = -1.0
                else:
                    P[i, idx1] = 1.0
                
                Q[i] = v.expected_excess_return
                # Idzorek uncertainty calibration: Omega_ii = P_i * (tau * Sigma) * P_i^T * ((1 - conf) / conf)
                p_row = P[i, :]
                var_view = p_row.T @ (tau * sigma) @ p_row
                conf_factor = (1.0 - v.confidence) / max(0.01, v.confidence)
                omega_diag[i] = max(1e-6, var_view * conf_factor)

        Omega = np.diag(omega_diag)

        # Black-Litterman Master Formula:
        # E[R] = [ (tau * Sigma)^-1 + P^T Omega^-1 P ]^-1 * [ (tau * Sigma)^-1 Pi + P^T Omega^-1 Q ]
        tau_sigma_inv = np.linalg.inv(tau * sigma)
        omega_inv = np.linalg.inv(Omega)

        M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        posterior_returns = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        posterior_cov = sigma + M

        # Optimize for Max Sharpe under Black-Litterman posterior distribution
        opt_bl = self.optimize_max_sharpe(custom_returns=posterior_returns, custom_cov=posterior_cov)

        return {
            "posterior_expected_returns": {self.tickers[i]: round(float(posterior_returns[i]), 4) for i in range(self.n_assets)},
            "prior_implied_returns": {self.tickers[i]: round(float(pi[i]), 4) for i in range(self.n_assets)},
            "optimal_weights": opt_bl["weights"],
            "expected_return": opt_bl["expected_return"],
            "volatility": opt_bl["volatility"],
            "sharpe_ratio": opt_bl["sharpe_ratio"],
            "views_applied": k
        }

    def optimize_risk_parity(self) -> Dict[str, Any]:
        """
        Equal Risk Contribution (ERC) / Risk Parity.
        Allocates risk equally across all universe assets so that each asset
        contributes 1/N to the total portfolio volatility.
        """
        sigma = self.cov_matrix
        target_risk_budget = np.ones(self.n_assets) / self.n_assets

        def erc_objective(w):
            port_vol = np.sqrt(w.T @ sigma @ w)
            marginal_contrib = (sigma @ w) / port_vol
            risk_contrib = w * marginal_contrib
            rc_pct = risk_contrib / port_vol
            # Minimize sum of squared deviations from target budget
            return np.sum((rc_pct - target_risk_budget)**2)

        init_w = 1.0 / np.sqrt(np.diag(sigma))
        init_w = init_w / np.sum(init_w)
        bounds = tuple((0.01, 0.35) for _ in range(self.n_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(erc_objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = res.x / np.sum(res.x)
        ret, vol, sharpe = self._portfolio_metrics(opt_w)

        return {
            "type": "Equal Risk Contribution (Risk Parity)",
            "weights": {self.tickers[i]: round(float(opt_w[i]), 4) for i in range(self.n_assets)},
            "expected_return": round(ret, 4),
            "volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 3)
        }

# Global optimizer singleton
portfolio_optimizer = AladdinPortfolioOptimizer()
