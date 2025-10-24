"""
Funções auxiliares para o formulário do portfólio.
"""
import sqlite3
from typing import List, Dict, Optional
import os

def get_db_path() -> str:
    """Retorna o caminho do banco de dados."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'crypto.db')

def search_symbols(search_term: str) -> List[Dict[str, str]]:
    """
    Busca símbolos no banco de dados que correspondam ao termo de pesquisa.
    Retorna uma lista de dicionários com os símbolos encontrados.
    """
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        # Busca por correspondência parcial no símbolo ou moeda base
        cursor.execute("""
            SELECT symbol, base_currency, quote_currency
            FROM available_symbols 
            WHERE symbol LIKE ? 
            OR base_currency LIKE ? 
            OR quote_currency LIKE ?
            AND is_active = 1
            LIMIT 10
        """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
        
        results = cursor.fetchall()
        
        return [{
            'symbol': row[0],
            'base_currency': row[1],
            'quote_currency': row[2]
        } for row in results]
        
    except Exception as e:
        print(f"Erro ao buscar símbolos: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()
            
def get_binance_symbol_info(symbol: str) -> Optional[Dict]:
    """
    Busca informações detalhadas sobre um símbolo específico no banco de dados.
    """
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, base_currency, quote_currency, last_price, last_update
            FROM available_symbols 
            WHERE symbol = ? AND is_active = 1
        """, (symbol,))
        
        result = cursor.fetchone()
        
        if result:
            return {
                'symbol': result[0],
                'base_currency': result[1],
                'quote_currency': result[2],
                'last_price': result[3],
                'last_update': result[4]
            }
        return None
        
    except Exception as e:
        print(f"Erro ao buscar informações do símbolo: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()