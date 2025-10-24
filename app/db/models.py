from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String(20), default='user')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Asset(Base):
    __tablename__ = 'assets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    exchange = Column(String(50), nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_asset_symbol', 'symbol'),
        Index('idx_asset_exchange', 'exchange'),
    )



class Ohlcv(Base):
    __tablename__ = 'ohlcv'

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint('asset_id', 'timestamp', name='uq_ohlcv_asset_timestamp'),
        Index('idx_ohlcv_asset_timestamp', 'asset_id', 'timestamp'),
    )

class AvailableSymbols(Base):
    __tablename__ = 'available_symbols'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False, unique=True)
    name = Column(String(100))
    base_currency = Column(String(20), nullable=False)
    quote_currency = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    last_price = Column(Float)
    last_update = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_available_symbols_symbol', 'symbol'),
        Index('idx_available_symbols_base', 'base_currency'),
        Index('idx_available_symbols_quote', 'quote_currency'),
    )

class Portfolio(Base):
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    qty = Column(Float, nullable=False)
    price_entry = Column(Float, nullable=False)
    price_current = Column(Float)
    stop_loss = Column(Float)
    target_1 = Column(Float)
    target_2 = Column(Float)
    target_3 = Column(Float)
    realized_1 = Column(Float, default=0)
    realized_2 = Column(Float, default=0)
    realized_3 = Column(Float, default=0)
    allocation_target = Column(Float)
    allocation_actual = Column(Float)
    last_update = Column(DateTime(timezone=True))
    notes = Column(String(500))

    __table_args__ = (
        Index('idx_portfolio_symbol', 'symbol'),
    )

class Analytics(Base):
    __tablename__ = 'analytics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    metric = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)

    __table_args__ = (
        Index('idx_analytics_asset_metric', 'asset_id', 'metric'),
        Index('idx_analytics_ts', 'ts'),
    )
