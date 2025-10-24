"""
Dashboard Entry Point
"""

import os
import sys

# Adicionar o diretório raiz ao PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Importar e executar o dashboard
from app.dashboard.app import main

if __name__ == "__main__":
    main()