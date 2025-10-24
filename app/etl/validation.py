from typing import Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

class OhlcvData(BaseModel):
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    
    @validator('high')
    def high_must_be_max(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high must be greater than or equal to low')
        if 'open' in values and v < values['open']:
            raise ValueError('high must be greater than or equal to open')
        if 'close' in values and v < values['close']:
            raise ValueError('high must be greater than or equal to close')
        return v
    
    @validator('low')
    def low_must_be_min(cls, v, values):
        if 'open' in values and v > values['open']:
            raise ValueError('low must be less than or equal to open')
        if 'close' in values and v > values['close']:
            raise ValueError('low must be less than or equal to close')
        return v

class AssetData(BaseModel):
    symbol: str
    name: str
    exchange: str
    active: bool = True
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.split('/')) != 2:
            raise ValueError('symbol must be in format BASE/QUOTE')
        return v