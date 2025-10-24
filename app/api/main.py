from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import crypto, assets, ohlcv, analytics, auth, portfolio

app = FastAPI(
    title="Crypto Quant API",
    version="1.0.0",
    description="API for cryptocurrency quantitative analysis"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
app.include_router(crypto.router, prefix="/api/v1", tags=["crypto"])
app.include_router(assets.router, prefix="/api/v1", tags=["assets"])
app.include_router(ohlcv.router, prefix="/api/v1", tags=["ohlcv"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])

@app.get("/")
def read_root():
    return {"message": "Crypto Quant API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
