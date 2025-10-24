"""
Módulo de inicialização do banco de dados.
"""
from app.db.engine import engine
from app.db.models import Base

def init_db() -> None:
    """
    Inicializa o banco de dados e suas tabelas usando SQLAlchemy.
    """
    print(f"Inicializando banco de dados em: {engine.url}")
    
    try:
        # Cria todas as tabelas definidas nos modelos
        Base.metadata.create_all(engine)
        print("Banco de dados inicializado com sucesso!")
        
    except Exception as e:
        print(f"Erro ao inicializar banco de dados: {e}")
        raise

if __name__ == "__main__":
    init_db()