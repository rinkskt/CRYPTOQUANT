from celery import Celery
from celery.schedules import crontab
from app.config import REDIS_URL

# Criar aplicação Celery
celery = Celery(
    'crypto_quant',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        'app.workers.tasks.etl',
        'app.workers.tasks.analytics'
    ]
)

# Configurações do Celery
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hora
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,
)

# Configurar tarefas periódicas
celery.conf.beat_schedule = {
    'fetch-ohlcv-data': {
        'task': 'app.workers.tasks.etl.fetch_all_ohlcv',
        'schedule': crontab(minute='*/15'),  # A cada 15 minutos
    },
    'update-analytics': {
        'task': 'app.workers.tasks.analytics.update_analytics',
        'schedule': crontab(minute=0, hour='*/1'),  # A cada hora
    },
}

if __name__ == '__main__':
    celery.start()