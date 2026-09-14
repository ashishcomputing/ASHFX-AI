"""
Market data engine for Aladdin-AI: Multi-asset universe, covariance calibration,
and synthetic historical time series generation for institutional risk modeling.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from aladdin.models import AssetHolding, PortfolioState

# Asset Universe Definition with institutional market parameters
DEFAULT_UNIVERSE = {
    "NVDA": {
        "name": "Nvidia Corporation",
        "class": "Equity",
        "sector": "Information Technology",
        "price": 128.50,
        "vol": 0.48,
        "expected_return": 0.28,
        "beta": 1.75,
        "duration": 0.0,
        "yield": 0.0003,
        "daily_volume": 45_000_000_000.0,
    },
    "AAPL": {
        "name": "Apple Inc.",
        "class": "Equity",
        "sector": "Information Technology",
        "price": 224.20,
        "vol": 0.22,
        "expected_return": 0.14,
        "beta": 1.12,
        "duration": 0.0,
        "yield": 0.0045,
        "daily_volume": 12_000_000_000.0,
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "class": "Equity",
        "sector": "Information Technology",
        "price": 448.30,
        "vol": 0.24,
        "expected_return": 0.16,
        "beta": 1.18,
        "duration": 0.0,
        "yield": 0.0070,
        "daily_volume": 9_500_000_000.0,
    },
    "GOOGL": {
        "name": "Alphabet Inc. (Class A)",
        "class": "Equity",
        "sector": "Communication Services",
        "price": 178.90,
        "vol": 0.26,
        "expected_return": 0.15,
        "beta": 1.15,
        "duration": 0.0,
        "yield": 0.0045,
        "daily_volume": 6_200_000_000.0,
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "class": "Equity",
        "sector": "Consumer Discretionary",
        "price": 186.40,
        "vol": 0.27,
        "expected_return": 0.17,
        "beta": 1.25,
        "duration": 0.0,
        "yield": 0.0,
        "daily_volume": 8_100_000_000.0,
    },
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "class": "Equity",
        "sector": "Financials",
        "price": 215.60,
        "vol": 0.19,
        "expected_return": 0.11,
        "beta": 0.95,
        "duration": 0.0,
        "yield": 0.022,
        "daily_volume": 4_800_000_000.0,
    },
    "XOM": {
        "name": "Exxon Mobil Corp.",
        "class": "Equity",
        "sector": "Energy",
        "price": 118.40,
        "vol": 0.23,
        "expected_return": 0.09,
        "beta": 0.65,
        "duration": 0.0,
        "yield": 0.033,
        "daily_volume": 3_500_000_000.0,
    },
    "LLY": {
        "name": "Eli Lilly and Company",
        "class": "Equity",
        "sector": "Healthcare",
        "price": 945.00,
        "vol": 0.28,
        "expected_return": 0.22,
        "beta": 0.70,
        "duration": 0.0,
        "yield": 0.0055,
        "daily_volume": 4_100_000_000.0,
    },
    "TLT": {
        "name": "iShares 20+ Year Treasury Bond ETF",
        "class": "Fixed Income",
        "sector": "Sovereign Debt",
        "price": 96.80,
        "vol": 0.15,
        "expected_return": 0.048,
        "beta": -0.25,
        "duration": 16.8,
        "yield": 0.043,
        "daily_volume": 3_200_000_000.0,
    },
    "LQD": {
        "name": "iShares iBoxx $ Investment Grade Corp Bond",
        "class": "Fixed Income",
        "sector": "Corporate IG Debt",
        "price": 111.50,
        "vol": 0.09,
        "expected_return": 0.054,
        "beta": 0.15,
        "duration": 8.2,
        "yield": 0.051,
        "daily_volume": 1_800_000_000.0,
    },
    "HYG": {
        "name": "iShares iBoxx $ High Yield Corporate Bond",
        "class": "Fixed Income",
        "sector": "High Yield Debt",
        "price": 78.40,
        "vol": 0.08,
        "expected_return": 0.068,
        "beta": 0.45,
        "duration": 3.7,
        "yield": 0.065,
        "daily_volume": 2_100_000_000.0,
    },
    "GLD": {
        "name": "SPDR Gold Shares ETF",
        "class": "Commodity",
        "sector": "Precious Metals",
        "price": 238.50,
        "vol": 0.14,
        "expected_return": 0.085,
        "beta": 0.05,
        "duration": 0.0,
        "yield": 0.0,
        "daily_volume": 2_400_000_000.0,
    },
    "USO": {
        "name": "United States Oil Fund LP",
        "class": "Commodity",
        "sector": "Energy Commodities",
        "price": 74.20,
        "vol": 0.32,
        "expected_return": 0.06,
        "beta": 0.35,
        "duration": 0.0,
        "yield": 0.0,
        "daily_volume": 1_200_000_000.0,
    },
    "BTC": {
        "name": "Bitcoin (Digital Gold)",
        "class": "Crypto",
        "sector": "Digital Assets",
        "price": 64800.00,
        "vol": 0.58,
        "expected_return": 0.35,
        "beta": 1.40,
        "duration": 0.0,
        "yield": 0.0,
        "daily_volume": 28_000_000_000.0,
    },
    "BIL": {
        "name": "SPDR Bloomberg 1-3 Month T-Bill ETF",
        "class": "Cash",
        "sector": "Cash Equivalents",
        "price": 91.60,
        "vol": 0.008,
        "expected_return": 0.048,
        "beta": 0.01,
        "duration": 0.15,
        "yield": 0.048,
        "daily_volume": 900_000_000.0,
    }
}

# Initial baseline institutional weights ($100M AUM)
DEFAULT_WEIGHTS = {
    "NVDA": 0.12,
    "AAPL": 0.10,
    "MSFT": 0.10,
    "GOOGL": 0.08,
    "AMZN": 0.06,
    "JPM": 0.07,
    "XOM": 0.05,
    "LLY": 0.06,
    "TLT": 0.12,
    "LQD": 0.08,
    "HYG": 0.04,
    "GLD": 0.05,
    "USO": 0.02,
    "BTC": 0.02,
    "BIL": 0.03
}

class MarketDataEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.universe = DEFAULT_UNIVERSE
        self.tickers = list(DEFAULT_UNIVERSE.keys())
        self.n_assets = len(self.tickers)
        self.corr_matrix = self._build_realistic_correlation_matrix()
        self.cov_matrix = self._build_covariance_matrix()
        self.historical_returns_df = self._generate_historical_returns(n_days=504)

    def _build_realistic_correlation_matrix(self) -> np.ndarray:
        """
        Builds an empirically realistic correlation matrix for multi-asset institutional portfolios.
        """
        n = self.n_assets
        corr = np.eye(n)

        # Realistic pairwise correlation guidelines based on real empirical asset dynamics
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = self.tickers[i], self.tickers[j]
                c1, c2 = self.universe[t1]["class"], self.universe[t2]["class"]
                s1, s2 = self.universe[t1]["sector"], self.universe[t2]["sector"]

                val = 0.25 # baseline cross-asset correlation

                if c1 == "Equity" and c2 == "Equity":
                    if s1 == s2:
                        val = 0.82  # Tech peers
                    else:
                        val = 0.58  # Equity cross-sector
                elif c1 == "Fixed Income" and c2 == "Fixed Income":
                    val = 0.72  # Treasuries & Corporate bond co-movement
                elif (c1 == "Equity" and t2 == "TLT") or (t1 == "TLT" and c2 == "Equity"):
                    val = -0.28  # Flight to safety / duration hedge
                elif (c1 == "Equity" and t2 == "LQD") or (t1 == "LQD" and c2 == "Equity"):
                    val = 0.20  # IG credit has small equity beta
                elif (c1 == "Equity" and t2 == "HYG") or (t1 == "HYG" and c2 == "Equity"):
                    val = 0.65  # High yield strongly tracks equity credit cycle
                elif t1 == "GLD" or t2 == "GLD":
                    if t1 == "TLT" or t2 == "TLT":
                        val = 0.38  # Gold & Treasuries both benefit from rate cuts
                    elif c1 == "Equity" or c2 == "Equity":
                        val = 0.05  # Gold essentially uncorrelated to broad equities
                    elif t1 == "BTC" or t2 == "BTC":
                        val = 0.22  # Store of value / alternative hedge
                elif t1 == "USO" or t2 == "USO":
                    if t1 == "XOM" or t2 == "XOM":
                        val = 0.74  # Oil and Exxon high correlation
                    elif c1 == "Equity" or c2 == "Equity":
                        val = 0.28
                elif t1 == "BTC" or t2 == "BTC":
                    if c1 == "Equity" or c2 == "Equity":
                        val = 0.45  # High beta tech correlation
                    elif c1 == "Fixed Income" or c2 == "Fixed Income":
                        val = -0.10
                elif t1 == "BIL" or t2 == "BIL":
                    val = 0.01  # Cash/T-bills has zero correlation to risk assets

                corr[i, j] = val
                corr[j, i] = val

        # Ensure positive semi-definiteness via eigenvalue thresholding
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Rescale diagonal to 1.0
        d = np.sqrt(np.diag(corr_clean))
        corr_clean = corr_clean / np.outer(d, d)
        return corr_clean

    def _build_covariance_matrix(self) -> np.ndarray:
        vols = np.array([self.universe[t]["vol"] for t in self.tickers])
        cov = np.outer(vols, vols) * self.corr_matrix
        return cov

    def _generate_historical_returns(self, n_days: int = 504) -> pd.DataFrame:
        """
        Generates 2-year realistic multi-asset daily return series with Student-t fat tails
        and realistic jump risks using Cholesky decomposition of calibrated covariance.
        """
        daily_cov = self.cov_matrix / 252.0
        daily_expected_returns = np.array([self.universe[t]["expected_return"] / 252.0 for t in self.tickers])

        # Cholesky decomposition L such that L L^T = daily_cov
        L = np.linalg.cholesky(daily_cov)

        # Generate correlated student-t innovations (nu=5 degrees of freedom for fat tails)
        nu = 5.0
        standard_normals = np.random.randn(n_days, self.n_assets)
        chi2_samples = np.random.chisquare(nu, size=(n_days, 1)) / nu
        t_samples = standard_normals / np.sqrt(chi2_samples)

        # Correlated daily innovations
        raw_innovations = (L @ t_samples.T).T
        # Mean center to 0
        innovations = raw_innovations - np.mean(raw_innovations, axis=0)
        # Rescale to target daily standard deviations
        target_daily_stds = np.array([self.universe[t]["vol"] / np.sqrt(252.0) for t in self.tickers])
        sample_stds = np.std(innovations, axis=0, ddof=1)
        innovations = innovations * (target_daily_stds / np.maximum(sample_stds, 1e-8))

        correlated_returns = daily_expected_returns + innovations

        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="B")
        df = pd.DataFrame(correlated_returns, index=dates, columns=self.tickers)
        return df

    def get_default_portfolio(self, total_aum: float = 100_000_000.0) -> PortfolioState:
        """
        Builds the institutional PortfolioState object populated with current weights and market values.
        """
        holdings = []
        for ticker in self.tickers:
            w = DEFAULT_WEIGHTS.get(ticker, 0.0)
            u = self.universe[ticker]
            mkt_val = total_aum * w
            shares = mkt_val / u["price"]

            holdings.append(AssetHolding(
                ticker=ticker,
                name=u["name"],
                asset_class=u["class"],
                sector=u["sector"],
                weight=w,
                current_price=u["price"],
                shares=round(shares, 4),
                market_value=round(mkt_val, 2),
                beta=u["beta"],
                duration=u["duration"],
                dividend_yield=u["yield"],
                annual_volatility=u["vol"],
                daily_volume=u["daily_volume"]
            ))

        cash_bal = total_aum * DEFAULT_WEIGHTS.get("BIL", 0.03)
        return PortfolioState(
            portfolio_name="ASHFX Global Alpha & Multi-Asset Parity Fund",
            base_currency="USD",
            total_aum=total_aum,
            cash_balance=cash_bal,
            holdings=holdings,
            benchmark="S&P 500 Total Return"
        )

# Global engine singleton
market_data = MarketDataEngine()
