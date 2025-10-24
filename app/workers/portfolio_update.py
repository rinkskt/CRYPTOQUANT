"""
Portfolio Update Service

Serviço para atualização automática dos dados do portfólio.
"""

import os
import sys
import time
import json
import ccxt
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from app.dashboard.pages.portfolio.portfolio_form import PortfolioPosition

class PortfolioUpdateService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.exchange = ccxt.binance()
        self.setup_database()
    
    def setup_database(self):
        """
        Configura o banco de dados para armazenar histórico de preços.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Tabela de histórico de preços
        c.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            symbol TEXT,
            price REAL,
            timestamp DATETIME,
            PRIMARY KEY (symbol, timestamp)
        )
        ''')
        
        # Tabela de portfólio
        c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            symbol TEXT,
            qty REAL,
            price_entry REAL,
            price_current REAL,
            stop_loss REAL,
            target_1 REAL,
            target_2 REAL,
            target_3 REAL,
            realized_1 REAL,
            realized_2 REAL,
            realized_3 REAL,
            allocation_target REAL,
            allocation_actual REAL,
            last_update DATETIME,
            notes TEXT,
            PRIMARY KEY (symbol)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Busca preço atual na Binance.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except:
            return None
    
    def update_prices(self, portfolio: List[Dict]):
        """
        Atualiza preços e salva histórico.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now()
        
        for position in portfolio:
            symbol = position['symbol']
            price = self.get_current_price(symbol)
            
            if price:
                # Salva histórico
                c.execute(
                    'INSERT INTO price_history (symbol, price, timestamp) VALUES (?, ?, ?)',
                    (symbol, price, now)
                )
                
                # Atualiza posição
                c.execute('''
                UPDATE portfolio 
                SET price_current = ?, last_update = ?
                WHERE symbol = ?
                ''', (price, now, symbol))
        
        conn.commit()
        conn.close()
    
    def load_portfolio(self) -> List[Dict]:
        """
        Carrega portfólio do banco de dados.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT * FROM portfolio')
        columns = [col[0] for col in c.description]
        portfolio = [dict(zip(columns, row)) for row in c.fetchall()]
        
        conn.close()
        return portfolio
    
    def run(self, update_interval: int = 300):
        """
        Executa o serviço de atualização continuamente.
        
        Args:
            update_interval: Intervalo em segundos entre atualizações (padrão: 5 minutos)
        """
        print(f"Iniciando serviço de atualização com intervalo de {update_interval} segundos")
        
        while True:
            try:
                portfolio = self.load_portfolio()
                if portfolio:
                    print(f"Atualizando {len(portfolio)} posições...")
                    self.update_prices(portfolio)
                    print(f"Atualização concluída em {datetime.now()}")
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"Erro durante atualização: {str(e)}")
                time.sleep(60)  # Espera 1 minuto em caso de erro

if __name__ == "__main__":
    db_path = os.path.join(root_dir, "crypto.db")
    service = PortfolioUpdateService(db_path)
    service.run()