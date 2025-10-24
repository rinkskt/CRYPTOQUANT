"""
Portfolio Performance Module

Implementa métricas de performance do portfólio:
- Retornos ajustados ao risco (Sharpe, Sortino)
- Medidas relativas ao benchmark (Alpha, Information Ratio)
- Eficiência e consistência
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import stats


def compute_portfolio_value(prices_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Calcula valor do portfólio ao longo do tempo.

    Args:
        prices_df: DataFrame com preços dos ativos
        weights: Dict com pesos do portfólio

    Returns:
        Série com valor do portfólio
    """
    if prices_df.empty or not weights:
        return pd.Series(dtype=float)

    # Filtra ativos disponíveis
    available_assets = [asset for asset in weights.keys() if asset in prices_df.columns]
    if not available_assets:
        return pd.Series(dtype=float)

    # Normaliza pesos
    total_weight = sum(weights[asset] for asset in available_assets)
    normalized_weights = {asset: weights[asset] / total_weight for asset in available_assets}

    # Calcula valor ponderado
    portfolio_value = pd.Series(0.0, index=prices_df.index, dtype=float)
    for asset in available_assets:
        portfolio_value += prices_df[asset] * normalized_weights[asset]

    return portfolio_value


def compute_portfolio_returns(portfolio_value_series: pd.Series) -> pd.Series:
    """
    Calcula retornos do portfólio.

    Args:
        portfolio_value_series: Série com valor do portfólio

    Returns:
        Série com retornos logarítmicos
    """
    if portfolio_value_series.empty or len(portfolio_value_series) < 2:
        return pd.Series(dtype=float)

    # Retornos logarítmicos
    returns = np.log(portfolio_value_series / portfolio_value_series.shift(1))
    return returns.dropna()


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona coluna de retornos ao DataFrame.

    Args:
        df: DataFrame com coluna 'close'

    Returns:
        DataFrame com coluna 'returns' adicionada
    """
    if 'close' not in df.columns:
        return df

    df_copy = df.copy()
    df_copy['returns'] = df_copy['close'].pct_change()
    df_copy['returns_daily'] = df_copy['close'].pct_change()
    df_copy['returns_cumulative'] = (1 + df_copy['returns_daily']).cumprod() - 1
    return df_copy


def compute_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Calcula volatilidade móvel.

    Args:
        df: DataFrame com coluna 'returns'
        window: Janela para cálculo

    Returns:
        Série com volatilidade anualizada
    """
    if 'returns' not in df.columns:
        df = compute_returns(df)

    return df['returns'].rolling(window).std() * np.sqrt(365)


def compute_drawdown(df: pd.DataFrame) -> pd.Series:
    """
    Calcula drawdown do portfólio.

    Args:
        df: DataFrame com preços

    Returns:
        Série com drawdown
    """
    if 'close' in df.columns:
        prices = df['close']
    else:
        prices = df

    cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
    max_cumulative = cumulative.cummax()
    drawdown = (cumulative - max_cumulative) / max_cumulative

    return drawdown


def calculate_sharpe_ratio(returns: pd.Series,
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 365) -> Dict[str, float]:
    """
    Calcula Sharpe Ratio do portfólio.

    S = (μ_p - r_f) / σ_p

    Args:
        returns: Série de retornos
        risk_free_rate: Taxa livre de risco anual
        periods_per_year: Períodos por ano

    Returns:
        Dict com Sharpe e componentes
    """
    # Anualiza retorno e volatilidade
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    return {
        'sharpe_ratio': sharpe,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'risk_free_rate': risk_free_rate
    }


def calculate_sortino_ratio(returns: pd.Series,
                         risk_free_rate: float = 0.0,
                         periods_per_year: int = 365) -> Dict[str, float]:
    """
    Calcula Sortino Ratio do portfólio.

    So = (μ_p - r_f) / σ_down

    Args:
        returns: Série de retornos
        risk_free_rate: Taxa livre de risco anual
        periods_per_year: Períodos por ano

    Returns:
        Dict com Sortino e componentes
    """
    # Anualiza retorno
    ann_return = returns.mean() * periods_per_year

    # Downside deviation
    negative_returns = returns[returns < 0]
    ann_downside_std = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0

    # Sortino Ratio
    sortino = (ann_return - risk_free_rate) / ann_downside_std if ann_downside_std > 0 else 0

    return {
        'sortino_ratio': sortino,
        'annualized_return': ann_return,
        'downside_volatility': ann_downside_std,
        'risk_free_rate': risk_free_rate
    }


def calculate_calmar_ratio(returns: pd.Series,
                         prices: Optional[pd.Series] = None,
                         periods_per_year: int = 365) -> Dict[str, float]:
    """
    Calcula Calmar Ratio do portfólio.

    Ca = Retorno anualizado / Max Drawdown

    Args:
        returns: Série de retornos
        prices: Série de preços (opcional)
        periods_per_year: Períodos por ano

    Returns:
        Dict com Calmar e componentes
    """
    # Retorno anualizado
    ann_return = returns.mean() * periods_per_year

    # Máximo drawdown
    if prices is not None:
        drawdown = (prices / prices.cummax() - 1)
    else:
        cumulative_returns = (1 + returns).cumprod()
        drawdown = (cumulative_returns / cumulative_returns.cummax() - 1)

    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar = -ann_return / max_drawdown if max_drawdown < 0 else 0

    return {
        'calmar_ratio': calmar,
        'annualized_return': ann_return,
        'max_drawdown': max_drawdown
    }


def calculate_alpha_beta(returns: pd.Series,
                       market_returns: pd.Series,
                       risk_free_rate: float = 0.0,
                       periods_per_year: int = 365) -> Dict[str, float]:
    """
    Calcula Alpha e Beta do portfólio.

    α = μ_p - [r_f + β_p(μ_m - r_f)]

    Args:
        returns: Retornos do portfólio
        market_returns: Retornos do benchmark
        risk_free_rate: Taxa livre de risco anual
        periods_per_year: Períodos por ano

    Returns:
        Dict com Alpha, Beta e R²
    """
    # Calcula Beta
    covariance = returns.cov(market_returns)
    market_variance = market_returns.var()
    beta = covariance / market_variance if market_variance > 0 else 1.0

    # Retornos anualizados
    portfolio_return = returns.mean() * periods_per_year
    market_return = market_returns.mean() * periods_per_year

    # Alpha de Jensen
    alpha = portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))

    # R² (coeficiente de determinação)
    correlation = returns.corr(market_returns)
    r_squared = correlation ** 2

    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'correlation': correlation
    }


def calculate_information_ratio(returns: pd.Series,
                             benchmark_returns: pd.Series,
                             periods_per_year: int = 365) -> Dict[str, float]:
    """
    Calcula Information Ratio do portfólio.

    IR = (μ_p - μ_m) / σ_(p-m)

    Args:
        returns: Retornos do portfólio
        benchmark_returns: Retornos do benchmark
        periods_per_year: Períodos por ano

    Returns:
        Dict com IR e tracking error
    """
    # Retorno diferencial
    active_returns = returns - benchmark_returns

    # Information Ratio
    active_return = active_returns.mean() * periods_per_year
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)

    info_ratio = active_return / tracking_error if tracking_error > 0 else 0

    return {
        'information_ratio': info_ratio,
        'active_return': active_return,
        'tracking_error': tracking_error
    }


def calculate_rolling_metrics(returns: pd.Series,
                          benchmark_returns: Optional[pd.Series] = None,
                          window: int = 60,
                          risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Calcula métricas de performance em janela móvel.

    Args:
        returns: Retornos do portfólio
        benchmark_returns: Retornos do benchmark (opcional)
        window: Tamanho da janela em períodos
        risk_free_rate: Taxa livre de risco anual

    Returns:
        DataFrame com métricas móveis
    """
    # Métricas básicas
    rolling_ret = returns.rolling(window).mean() * 365
    rolling_vol = returns.rolling(window).std() * np.sqrt(365)
    rolling_sharpe = (rolling_ret - risk_free_rate) / rolling_vol

    metrics = pd.DataFrame({
        'return': rolling_ret,
        'volatility': rolling_vol,
        'sharpe_ratio': rolling_sharpe
    })

    # Métricas vs benchmark
    if benchmark_returns is not None:
        rolling_alpha = pd.Series(index=returns.index)
        rolling_beta = pd.Series(index=returns.index)
        rolling_ir = pd.Series(index=returns.index)

        for i in range(window, len(returns)+1):
            slice_ret = returns[i-window:i]
            slice_bench = benchmark_returns[i-window:i]

            # Alpha e Beta
            ab_metrics = calculate_alpha_beta(slice_ret, slice_bench, risk_free_rate)
            rolling_alpha.iloc[i-1] = ab_metrics['alpha']
            rolling_beta.iloc[i-1] = ab_metrics['beta']

            # Information Ratio
            ir_metrics = calculate_information_ratio(slice_ret, slice_bench)
            rolling_ir.iloc[i-1] = ir_metrics['information_ratio']

        metrics['alpha'] = rolling_alpha
        metrics['beta'] = rolling_beta
        metrics['information_ratio'] = rolling_ir

    return metrics


def calculate_capture_ratios(returns: pd.Series,
                          benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Calcula Up/Down Capture Ratios.

    Args:
        returns: Retornos do portfólio
        benchmark_returns: Retornos do benchmark

    Returns:
        Dict com up/down capture ratios
    """
    # Separa mercados de alta e baixa
    up_market = benchmark_returns > 0
    down_market = benchmark_returns < 0

    # Up Capture
    up_capture = (returns[up_market].mean() / benchmark_returns[up_market].mean()) \
        if len(returns[up_market]) > 0 else 0

    # Down Capture
    down_capture = (returns[down_market].mean() / benchmark_returns[down_market].mean()) \
        if len(returns[down_market]) > 0 else 0

    return {
        'up_capture': up_capture,
        'down_capture': down_capture,
        'capture_spread': up_capture - down_capture
    }
