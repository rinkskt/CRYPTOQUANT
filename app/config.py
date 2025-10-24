from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///crypto.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "your-secret-key-here"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Crypto Quant"
    
    # JWT Settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Celery Settings
    CELERY_BROKER_URL: str = REDIS_URL
    CELERY_RESULT_BACKEND: str = REDIS_URL
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Dashboard Settings
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 8501
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
REDIS_URL = settings.REDIS_URL