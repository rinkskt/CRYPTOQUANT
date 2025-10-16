# Crypto Quant

Este projeto é uma plataforma de análise quantitativa para criptomoedas, incluindo ETL, analytics, API e dashboard.

## Visão Geral

O projeto visa fornecer ferramentas para análise de dados de criptomoedas, incluindo:

- ETL para coleta e processamento de dados OHLCV
- Módulos de analytics para cálculos de retornos, correlação, cointegration, etc.
- API FastAPI para acesso aos dados
- Dashboard Streamlit para visualização
- Orquestração com Celery/Prefect
- Monitoramento com Prometheus/Grafana
- CI/CD com GitHub Actions
- Deploy via Docker Compose/K8s

## Estrutura do Projeto

- `app/`: Código principal da aplicação
  - `api/`: API FastAPI
  - `analytics/`: Módulos de análise
  - `etl/`: Extração, Transformação e Carga de dados
  - `db/`: Modelos de banco de dados e migrations
  - `dashboard/`: Aplicação Streamlit
  - `workers/`: Workers para tarefas assíncronas
  - `utils/`: Utilitários
- `tests/`: Testes
- `docker-compose.yml`: Configuração Docker Compose
- `Dockerfile`: Imagem Docker
- `pyproject.toml`: Dependências e configurações Python

## Pré-requisitos

- VS Code instalado
- Docker & Docker Compose
- Python 3.11+
- Git

## Instalação e Execução

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt` ou `poetry install`
3. Execute o Docker Compose: `docker-compose up`
4. Acesse a API em http://localhost:8000
5. Acesse o dashboard em http://localhost:8501

## Desenvolvimento

Para desenvolvimento local:

1. Crie um ambiente virtual: `python -m venv venv`
2. Ative o venv: `venv\Scripts\activate` (Windows)
3. Instale dependências: `pip install -e .`
4. Execute a aplicação: `uvicorn app.api.main:app --reload`

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT.
