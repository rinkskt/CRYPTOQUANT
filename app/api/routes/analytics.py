"""
Analytics API Routes

This module provides API endpoints for accessing analytics data including
correlations, cointegration, rolling correlations, and pairs trading analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

from app.analytics.correlation import compute_correlation_matrix
from app.analytics.cointegration import test_cointegration
from app.analytics.rolling import latest_rolling_corr_matrix
from app.analytics.pairs import score_all_pairs
from app.analytics.persist import get_analytics_data, save_analytics_data
from app.db.engine import get_db_session
from app.db.models import Asset, Ohlcv, Analytics

router = APIRouter()


@router.get("/correlation/matrix")
async def get_correlation_matrix(
    limit: int = Query(100, description="Number of recent records to use")
) -> Dict[str, Any]:
    """
    Get current correlation matrix for all assets.

    Returns correlation matrix with asset symbols and values.
    """
    try:
        with get_db_session() as session:
            # Get all active assets
            assets = session.query(Asset).filter(Asset.active == True).all()
            if not assets:
                raise HTTPException(status_code=404, detail="No active assets found")

            asset_symbols = [asset.symbol for asset in assets]

            # Get recent OHLCV data for correlation calculation
            cutoff_date = datetime.now() - timedelta(days=limit)

            correlations = {}
            matrix_data = []

            for i, asset1 in enumerate(assets):
                row = []
                for j, asset2 in enumerate(assets):
                    if i == j:
                        # Diagonal is always 1.0
                        row.append(1.0)
                    elif i < j:
                        # Calculate correlation for upper triangle
                        ohlcv1 = session.query(Ohlcv).filter(
                            Ohlcv.asset_id == asset1.id,
                            Ohlcv.timestamp >= cutoff_date
                        ).order_by(Ohlcv.timestamp).all()

                        ohlcv2 = session.query(Ohlcv).filter(
                            Ohlcv.asset_id == asset2.id,
                            Ohlcv.timestamp >= cutoff_date
                        ).order_by(Ohlcv.timestamp).all()

                        if ohlcv1 and ohlcv2:
                            # Convert to DataFrames
                            df1 = pd.DataFrame([(o.timestamp, o.close) for o in ohlcv1],
                                             columns=['timestamp', 'close'])
                            df2 = pd.DataFrame([(o.timestamp, o.close) for o in ohlcv2],
                                             columns=['timestamp', 'close'])

                            # Merge on timestamp
                            merged = pd.merge(df1, df2, on='timestamp', suffixes=('_1', '_2'))

                            if len(merged) > 10:
                                corr = merged['close_1'].corr(merged['close_2'])
                                correlations[f"{asset1.symbol}_{asset2.symbol}"] = corr
                                row.append(corr)
                            else:
                                row.append(0.0)
                        else:
                            row.append(0.0)
                    else:
                        # Use symmetry for lower triangle
                        row.append(correlations.get(f"{asset2.symbol}_{asset1.symbol}", 0.0))

                matrix_data.append(row)

            return {
                "assets": asset_symbols,
                "matrix": matrix_data,
                "timestamp": datetime.now().isoformat(),
                "description": f"Correlation matrix using last {limit} days of data"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing correlation matrix: {str(e)}")


@router.get("/cointegration")
async def get_cointegration_results(
    top_n: int = Query(50, description="Number of top pairs to return")
) -> List[Dict[str, Any]]:
    """
    Get cointegration test results for asset pairs.

    Returns list of cointegrated pairs with test statistics.
    """
    try:
        # Try to get from database first
        with get_db_session() as session:
            analytics_data = session.query(Analytics).filter(
                Analytics.metric.like('cointegration_%')
            ).order_by(Analytics.ts.desc()).limit(1000).all()

            if analytics_data:
                # Parse stored cointegration results
                results = []
                for record in analytics_data:
                    if 'cointegration_' in record.metric:
                        try:
                            # Extract pair info from metric name
                            parts = record.metric.split('_')
                            if len(parts) >= 4:
                                asset1 = parts[1]
                                asset2 = parts[2]
                                metric_type = '_'.join(parts[3:])

                                # Find existing result or create new
                                existing = next((r for r in results if r.get('asset1') == asset1 and r.get('asset2') == asset2), None)
                                if existing:
                                    existing[metric_type] = record.value
                                else:
                                    results.append({
                                        'asset1': asset1,
                                        'asset2': asset2,
                                        metric_type: record.value
                                    })
                        except:
                            continue

                # Filter and sort results
                valid_results = [r for r in results if 'p_value' in r and 'correlation' in r]
                valid_results.sort(key=lambda x: x.get('p_value', 1.0))

                return valid_results[:top_n]

        # Fallback: compute on the fly
        with get_db_session() as session:
            assets = session.query(Asset).filter(Asset.active == True).limit(20).all()  # Limit for performance

            if len(assets) < 2:
                return []

            # Get recent data
            cutoff_date = datetime.now() - timedelta(days=365)
            results = []

            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i >= j:  # Skip symmetric pairs and self
                        continue

                    try:
                        # Get price data
                        data1 = session.query(Ohlcv).filter(
                            Ohlcv.asset_id == asset1.id,
                            Ohlcv.timestamp >= cutoff_date
                        ).order_by(Ohlcv.timestamp).all()

                        data2 = session.query(Ohlcv).filter(
                            Ohlcv.asset_id == asset2.id,
                            Ohlcv.timestamp >= cutoff_date
                        ).order_by(Ohlcv.timestamp).all()

                        if len(data1) > 30 and len(data2) > 30:
                            prices1 = pd.Series([d.close for d in data1])
                            prices2 = pd.Series([d.close for d in data2])

                            # Test cointegration
                            coint_result = test_cointegration(prices1, prices2)

                            if coint_result:
                                results.append({
                                    'asset1': asset1.symbol,
                                    'asset2': asset2.symbol,
                                    'cointegrated': coint_result.get('cointegrated', False),
                                    'p_value': coint_result.get('p_value', 1.0),
                                    'correlation': prices1.corr(prices2),
                                    'beta': coint_result.get('beta', 0.0),
                                    'half_life': coint_result.get('half_life', 0.0),
                                    'latest_zscore': coint_result.get('latest_zscore', 0.0)
                                })

                    except Exception as e:
                        continue

            # Sort by p-value and return top results
            results.sort(key=lambda x: x.get('p_value', 1.0))
            return results[:top_n]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing cointegration: {str(e)}")


@router.get("/rolling_correlation")
async def get_rolling_correlation(
    window: int = Query(30, description="Rolling window in days"),
    format: str = Query("json", description="Response format: 'json' or 'csv'")
) -> Any:
    """
    Get rolling correlation matrix.

    Returns rolling correlation analysis for asset pairs.
    """
    try:
        with get_db_session() as session:
            assets = session.query(Asset).filter(Asset.active == True).limit(15).all()  # Limit for performance

            if len(assets) < 2:
                if format == "json":
                    return {"error": "Insufficient assets for correlation analysis"}
                else:
                    return "Insufficient assets for correlation analysis"

            # Get rolling correlation data
            corr_data = latest_rolling_corr_matrix(assets, window=window, session=session)

            if format == "csv":
                # Convert to CSV format
                if corr_data and 'correlation_matrix' in corr_data:
                    df = pd.DataFrame(
                        corr_data['correlation_matrix'],
                        index=corr_data['assets'],
                        columns=corr_data['assets']
                    )
                    return df.to_csv()
                else:
                    return "No correlation data available"
            else:
                return corr_data or {"error": "No correlation data available"}

    except Exception as e:
        if format == "json":
            raise HTTPException(status_code=500, detail=f"Error computing rolling correlation: {str(e)}")
        else:
            return f"Error computing rolling correlation: {str(e)}"


@router.get("/cointegrated_pairs")
async def get_cointegrated_pairs(
    top_n: int = Query(50, description="Number of top pairs to return"),
    format: str = Query("json", description="Response format: 'json' or 'csv'")
) -> Any:
    """
    Get cointegrated pairs analysis.

    Returns detailed analysis of cointegrated asset pairs.
    """
    try:
        with get_db_session() as session:
            assets = session.query(Asset).filter(Asset.active == True).limit(20).all()

            if len(assets) < 2:
                if format == "json":
                    return []
                else:
                    return "Insufficient assets for pairs analysis"

            # Get pairs analysis
            pairs_data = score_all_pairs(assets, session=session)

            if not pairs_data:
                if format == "json":
                    return []
                else:
                    return "No pairs data available"

            # Sort by p-value and limit results
            sorted_pairs = sorted(pairs_data, key=lambda x: x.get('p_value', 1.0))[:top_n]

            if format == "csv":
                # Convert to CSV
                df = pd.DataFrame(sorted_pairs)
                return df.to_csv(index=False)
            else:
                return sorted_pairs

    except Exception as e:
        if format == "json":
            raise HTTPException(status_code=500, detail=f"Error computing pairs analysis: {str(e)}")
        else:
            return f"Error computing pairs analysis: {str(e)}"


@router.post("/run")
async def run_analytics_pipeline() -> Dict[str, Any]:
    """
    Trigger the analytics pipeline run.

    Executes all analytics computations and saves results to database.
    """
    try:
        from app.analytics.run_analytics import run_analytics_pipeline

        result = run_analytics_pipeline()

        return {
            "status": "success",
            "message": "Analytics pipeline completed",
            "results": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running analytics pipeline: {str(e)}")


@router.get("/")
async def get_analytics(
    asset_id: Optional[int] = None,
    symbol: Optional[str] = None,
    metric: Optional[str] = None,
    limit: int = Query(1000, description="Maximum number of records to return")
) -> List[Dict[str, Any]]:
    """
    Get analytics data with optional filtering.

    Returns analytics records from the database.
    """
    try:
        with get_db_session() as session:
            query = session.query(Analytics)

            if asset_id:
                query = query.filter(Analytics.asset_id == asset_id)
            if symbol:
                # Find asset by symbol
                asset = session.query(Asset).filter(Asset.symbol == symbol).first()
                if asset:
                    query = query.filter(Analytics.asset_id == asset.id)
            if metric:
                query = query.filter(Analytics.metric.like(f"%{metric}%"))

            # Order by timestamp descending and limit
            analytics_records = query.order_by(Analytics.ts.desc()).limit(limit).all()

            return [
                {
                    "id": record.id,
                    "asset_id": record.asset_id,
                    "metric": record.metric,
                    "value": record.value,
                    "ts": record.ts.isoformat() if record.ts else None
                }
                for record in analytics_records
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics data: {str(e)}")
