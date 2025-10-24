# 🚀 CryptoQuant Dashboard

Uma plataforma completa de análise quantitativa para criptomoedas com dashboard interativo, API FastAPI e deploy na nuvem.

![CryptoQuant](https://img.shields.io/badge/CryptoQuant-v2.0-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-orange)

## 🌟 Funcionalidades

### 📊 Análise de Mercado
- **Visão Geral**: Ranking de criptos, métricas globais e indicadores técnicos
- **Correlações**: Mapa de calor e análise de cointegração para pairs trading
- **Detalhes do Ativo**: Análise técnica completa com Z-score e sinais

### 💼 Gestão de Portfólio
- **Performance**: Retornos, Sharpe ratio, beta e drawdown
- **Risco**: VaR, Expected Shortfall e análise de correlação
- **Otimização**: Fronteira eficiente e rebalanceamento automático
- **Rebalanceamento**: Sugestões de trades com impacto de custos

### 🔧 Ferramentas Avançadas
- **Laboratório Quant**: Backtesting e simulações personalizadas
- **Dados em Tempo Real**: Análise de correlação rolling

### ⚠️ Monitoramento
- **Sistema de Alertas**: Notificações automáticas baseadas em regras

## 🚀 Deploy na Nuvem (GRÁTIS)

### 🎯 Opções Gratuitas Recomendadas

#### 1. **Streamlit Cloud** (Mais Fácil - GRÁTIS)
```bash
# 1. Faça push do código para GitHub
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main

# 2. Deploy no Streamlit Cloud
# - Acesse share.streamlit.io
# - Conecte sua conta GitHub
# - Selecione o repositório cryptoquant-dashboard
# - Defina o arquivo principal: app/dashboard/app.py
# - Clique em "Deploy"
```

**Vantagens:**
- ✅ 100% gratuito
- ✅ Deploy em 2 minutos
- ✅ Integração direta com GitHub
- ✅ Sem configuração complexa

#### 2. **Railway** (Plano Gratuito Disponível)
```bash
# 1. Faça push do código para GitHub
git add .
git commit -m "Deploy to Railway"
git push origin main

# 2. Conecte no Railway
# - Acesse railway.app
# - Faça login com GitHub
# - Clique em "New Project"
# - Selecione "Deploy from GitHub repo"
# - Escolha seu repositório
# - Railway detectará automaticamente o railway.json
```

#### 3. **Render** (Plano Gratuito Disponível)
```bash
# 1. Faça push do código para GitHub
git add .
git commit -m "Deploy to Render"
git push origin main

# 2. Conecte no Render
# - Acesse render.com
# - Faça login com GitHub
# - Clique em "New +"
# - Selecione "Web Service"
# - Conecte seu repositório GitHub
# - Configure:
#   - Runtime: Python 3
#   - Build Command: pip install -r requirements.txt
#   - Start Command: streamlit run app/dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

#### 4. **Docker Local** (Para Desenvolvimento)
```bash
# Para desenvolvimento local
docker-compose up --build
```

## 📋 Pré-requisitos

- Python 3.11+
- Docker & Docker Compose (opcional)
- Conta GitHub
- Conta na plataforma de deploy (Railway/Render)

## 🛠️ Instalação Local

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/cryptoquant-dashboard.git
cd cryptoquant-dashboard

# 2. Crie ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instale dependências
pip install -r requirements.txt

# 4. Execute localmente
# API
uvicorn app.api.main:app --reload

# Dashboard (em outro terminal)
streamlit run app/dashboard/app.py
```

## 🌐 Acesso

Após deploy, você terá:
- **API**: `https://sua-api.railway.app` ou `https://sua-api.onrender.com`
- **Dashboard**: `https://sua-dashboard.railway.app` ou `https://sua-dashboard.onrender.com`

## 📁 Estrutura do Projeto

```
cryptoquant-dashboard/
├── app/
│   ├── api/                 # FastAPI endpoints
│   ├── dashboard/           # Streamlit app
│   ├── analytics/           # Quantitative analysis modules
│   ├── db/                  # Database models & migrations
│   └── etl/                 # Data extraction & processing
├── docker-compose.yml       # Local development
├── Dockerfile              # Container configuration
├── railway.json            # Railway deployment
├── render.yaml             # Render deployment
└── requirements.txt        # Python dependencies
```

## 🔧 Configuração

### Variáveis de Ambiente
```env
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379
API_BASE_URL=https://your-api-url.com
```

### Banco de Dados
- **Desenvolvimento**: SQLite automático
- **Produção**: PostgreSQL + Redis (via Docker Compose)

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Suporte

- 📧 Email: seu-email@exemplo.com
- 💬 Issues: [GitHub Issues](https://github.com/seu-usuario/cryptoquant-dashboard/issues)
- 📖 Docs: [Documentação Completa](docs/)

---

**⭐ Star este repositório se achou útil!**
