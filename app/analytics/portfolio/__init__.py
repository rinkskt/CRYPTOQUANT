from .data_loader import get_binance_data, get_current_price, get_binance_klines, get_portfolio_data
from .performance import compute_portfolio_value, compute_portfolio_returns, compute_returns, compute_volatility, compute_drawdown, calculate_sharpe_ratio
from .risk import compute_var, compute_cvar, calculate_volatility, calculate_correlations, calculate_var, calculate_expected_shortfall, calculate_drawdown, calculate_beta, calculate_risk_contribution, calculate_conditional_var
from .optimization import optimize_max_sharpe, risk_parity, optimize_equal_weight, optimize_minimum_variance, calculate_efficient_frontier, rebalance_portfolio
from .fundamentals import get_fundamental_data, calculate_fundamental_ratios
from .metrics import summarize_portfolio, compute_portfolio_metrics_over_time, compare_portfolios
from .portfolio_spread import compute_portfolio_spread, compute_correlation_metrics, compute_cointegration_beta, compute_zscore, analyze_spread_full

__all__ = [
    # Data Loading
    'get_binance_data',
    'get_current_price',
    'get_binance_klines',
    'get_portfolio_data',

    # Performance
    'compute_portfolio_value',
    'compute_portfolio_returns',
    'compute_returns',
    'compute_volatility',
    'compute_drawdown',
    'calculate_sharpe_ratio',

    # Risk
    'compute_var',
    'compute_cvar',
    'calculate_volatility',
    'calculate_correlations',
    'calculate_var',
    'calculate_expected_shortfall',
    'calculate_drawdown',
    'calculate_beta',
    'calculate_risk_contribution',
    'calculate_conditional_var',

    # Optimization
    'optimize_max_sharpe',
    'risk_parity',
    'optimize_equal_weight',
    'optimize_minimum_variance',
    'calculate_efficient_frontier',
    'rebalance_portfolio',

    # Fundamentals
    'get_fundamental_data',
    'calculate_fundamental_ratios',

    # Metrics
    'summarize_portfolio',
    'compute_portfolio_metrics_over_time',
    'compare_portfolios',

    # Spread Analysis
    'compute_portfolio_spread',
    'compute_correlation_metrics',
    'compute_cointegration_beta',
    'compute_zscore',
    'analyze_spread_full'
]
