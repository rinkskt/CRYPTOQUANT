"""
Script para inicialização completa do ambiente.
"""
from app.db.init_db import init_db
from app.etl.sync_symbols import sync_binance_symbols

def main():
    """
    Inicializa todo o ambiente necessário para a aplicação.
    """
    print("Iniciando setup do ambiente...")
    
    # Inicializa o banco de dados
    print("\n1. Inicializando banco de dados...")
    init_db()
    
    # Sincroniza símbolos da Binance
    print("\n2. Sincronizando símbolos da Binance...")
    sync_binance_symbols()
    
    print("\nSetup concluído com sucesso!")

if __name__ == "__main__":
    main()