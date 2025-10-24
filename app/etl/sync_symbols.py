"""
Script para sincronizar símbolos disponíveis da Binance com o banco de dados local.
"""
from binance.client import Client
from app.db.engine import get_db_session
from app.db.models import AvailableSymbols
import pandas as pd
from datetime import datetime
from sqlalchemy.dialects.sqlite import insert

def sync_binance_symbols():
    """
    Busca todos os símbolos disponíveis na Binance e atualiza o banco de dados local.
    """
    try:
        # Inicializa o cliente da Binance
        client = Client()
        
        # Busca informações de todos os símbolos
        exchange_info = client.get_exchange_info()
        
        # Converte para DataFrame
        symbols_df = pd.DataFrame(exchange_info['symbols'])
        
        # Filtra apenas pares ativos e spot
        active_symbols = symbols_df[
            (symbols_df['status'] == 'TRADING') & 
            (symbols_df['isSpotTradingAllowed'] == True)
        ]
        
        # Prepara os dados para inserção
        symbol_data = []
        for _, row in active_symbols.iterrows():
            symbol_data.append({
                'symbol': row['symbol'],
                'base_currency': row['baseAsset'],
                'quote_currency': row['quoteAsset'],
                'is_active': True,
                'last_update': datetime.now()
            })
        
        # Conecta ao banco de dados usando SQLAlchemy
        session = get_db_session()
        
        try:
            # Upsert dos símbolos usando SQLAlchemy
            for data in symbol_data:
                stmt = insert(AvailableSymbols).values(**data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol'],
                    set_=data
                )
                session.execute(stmt)
            
            session.commit()
            print(f"Sincronizados {len(symbol_data)} símbolos com sucesso!")
            
        except Exception as e:
            session.rollback()
            raise e
            
        finally:
            session.close()
            
    except Exception as e:
        print(f"Erro ao sincronizar símbolos: {str(e)}")
        raise

if __name__ == "__main__":
    sync_binance_symbols()