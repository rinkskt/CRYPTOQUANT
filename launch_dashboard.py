"""
Script de inicialização do dashboard com configuração de ambiente.
"""

import os
import sys
import subprocess

def main():
    # Configura o ambiente Python
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = root_dir
    
    # Caminho para o arquivo do dashboard
    dashboard_path = os.path.join(root_dir, "app", "dashboard", "app.py")
    
    # Comando para executar o Streamlit
    cmd = [
        os.path.join(root_dir, "venv", "Scripts", "python.exe"),
        "-m",
        "streamlit",
        "run",
        dashboard_path
    ]
    
    # Executa o comando
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    main()