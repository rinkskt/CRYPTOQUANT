"""
Inicializa o serviço de atualização do portfólio.
"""

import os
import sys
import subprocess

def main():
    # Configura o ambiente Python
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = root_dir
    
    # Caminho para o script do serviço
    service_path = os.path.join(root_dir, "app", "workers", "portfolio_update.py")
    
    # Comando para executar o serviço
    cmd = [
        os.path.join(root_dir, "venv", "Scripts", "python.exe"),
        service_path
    ]
    
    # Executa o comando
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    main()