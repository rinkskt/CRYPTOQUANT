from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from sqlalchemy.orm import Session
from app.db.engine import engine
from app.db.models import Asset
from pydantic import BaseModel

router = APIRouter()

class AssetResponse(BaseModel):
    id: int
    symbol: str
    name: str
    exchange: str
    active: bool

@router.get("/assets", response_model=List[AssetResponse])
def get_assets(
    active_only: bool = Query(True, description="Return only active assets"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    limit: int = Query(100, description="Maximum number of assets to return")
):
    """
    Get list of assets with optional filtering.
    """
    with Session(engine) as session:
        query = session.query(Asset)

        if active_only:
            query = query.filter(Asset.active == True)

        if exchange:
            query = query.filter(Asset.exchange == exchange)

        assets = query.limit(limit).all()

        return [
            AssetResponse(
                id=asset.id,
                symbol=asset.symbol,
                name=asset.name,
                exchange=asset.exchange,
                active=asset.active
            )
            for asset in assets
        ]

@router.get("/assets/{asset_id}", response_model=AssetResponse)
def get_asset(asset_id: int):
    """
    Get a specific asset by ID.
    """
    with Session(engine) as session:
        asset = session.query(Asset).filter(Asset.id == asset_id).first()

        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        return AssetResponse(
            id=asset.id,
            symbol=asset.symbol,
            name=asset.name,
            exchange=asset.exchange,
            active=asset.active
        )

@router.get("/assets/symbol/{symbol}", response_model=AssetResponse)
def get_asset_by_symbol(symbol: str):
    """
    Get a specific asset by symbol.
    """
    with Session(engine) as session:
        asset = session.query(Asset).filter(Asset.symbol == symbol).first()

        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with symbol '{symbol}' not found")

        return AssetResponse(
            id=asset.id,
            symbol=asset.symbol,
            name=asset.name,
            exchange=asset.exchange,
            active=asset.active
        )
