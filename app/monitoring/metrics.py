from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# Métricas customizadas
API_REQUESTS = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of currently active users'
)

ETL_JOBS = Counter(
    'etl_jobs_total',
    'Total number of ETL jobs',
    ['status']
)

DB_CONNECTIONS = Gauge(
    'db_connections',
    'Number of active database connections'
)

def init_metrics(app):
    # Configurar instrumentação básica
    instrumentator = Instrumentator()
    
    # Adicionar métricas padrão
    instrumentator.add(metrics.request_size())
    instrumentator.add(metrics.response_size())
    instrumentator.add(metrics.latency())
    instrumentator.add(metrics.requests())
    
    # Iniciar instrumentação
    instrumentator.instrument(app)
    
    return instrumentator