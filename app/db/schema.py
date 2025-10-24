"""
Database Schema Module

Define e gerencia o esquema do banco de dados SQLite.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("crypto.db")

def create_tables():
    """
    Cria as tabelas necessárias no banco de dados.
    Se já existirem, ignora.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Tabela de símbolos disponíveis
    c.execute('''
        CREATE TABLE IF NOT EXISTS available_symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            name TEXT,
            base_currency TEXT,
            quote_currency TEXT,
            is_active BOOLEAN DEFAULT 1,
            last_price REAL,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabela de portfólio
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            price_entry REAL NOT NULL,
            price_current REAL,
            stop_loss REAL,
            target_1 REAL,
            target_2 REAL,
            target_3 REAL,
            realized_1 REAL DEFAULT 0,
            realized_2 REAL DEFAULT 0,
            realized_3 REAL DEFAULT 0,
            allocation_target REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabela de histórico de preços
    c.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
    ''')
    
    # Tabela de histórico de trades
    c.execute('''
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            trade_type TEXT NOT NULL CHECK(trade_type IN ('buy', 'sell')),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabela de configurações
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Índices para otimização
    c.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_price_history_symbol ON price_history(symbol)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key)')
    
    conn.commit()
    conn.close()

def reset_database():
    """
    Apaga e recria todas as tabelas.
    CUIDADO: Isso apagará todos os dados!
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Lista todas as tabelas
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    
    # Apaga cada tabela
    for table in tables:
        if table[0] not in ('sqlite_sequence'):
            c.execute(f'DROP TABLE IF EXISTS {table[0]}')
    
    conn.commit()
    conn.close()
    
    # Recria as tabelas
    create_tables()

if __name__ == '__main__':
    create_tables()