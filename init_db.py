"""
Inicializa o banco de dados e cria estrutura inicial.
"""

from app.db.schema import create_tables

if __name__ == '__main__':
    print("Criando estrutura do banco de dados...")
    create_tables()
    print("Banco de dados inicializado com sucesso!")