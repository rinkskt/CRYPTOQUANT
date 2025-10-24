"""
Script de inicialização do dashboard.
"""

import os
import sys

def main():
    # Adiciona o diretório raiz ao PYTHONPATH
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root_dir)
    
    # Importa o módulo do dashboard e executa
    import streamlit.cli
    dashboard_path = os.path.join(root_dir, "app", "dashboard", "app.py")
    sys.argv = ["streamlit", "run", dashboard_path]
    streamlit.cli.main()

if __name__ == "__main__":
    main()