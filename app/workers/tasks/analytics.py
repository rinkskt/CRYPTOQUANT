from app.workers.celery_app import celery
from app.analytics.run_analytics import calculate_all_analytics
from app.analytics.persist import save_all_results
import logging

logger = logging.getLogger(__name__)

@celery.task(
    bind=True,
    max_retries=3,
    retry_backoff=True
)
def update_analytics(self):
    """Tarefa Celery para atualizar todas as análises"""
    try:
        # Calcular análises
        results = calculate_all_analytics()
        
        # Salvar resultados
        save_all_results(results)
        
        return {
            'status': 'success',
            'message': 'Analytics updated successfully'
        }
    except Exception as exc:
        logger.error(f'Error in update_analytics: {exc}')
        raise self.retry(exc=exc)

@celery.task(
    bind=True,
    max_retries=3,
    retry_backoff=True
)
def calculate_pair_analytics(self, symbol1: str, symbol2: str):
    """Tarefa Celery para calcular análises para um par específico"""
    try:
        # Implementar cálculos específicos para o par
        pass
    except Exception as exc:
        logger.error(f'Error calculating analytics for {symbol1}/{symbol2}: {exc}')
        raise self.retry(exc=exc)