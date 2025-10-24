"""
Portfolio Form Module

Interface para entrada e edição dos dados do portfólio.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import ccxt
from sqlalchemy import select
from app.db.engine import get_db_session
from app.db.models import AvailableSymbols, Portfolio
from app.etl.sync_symbols import sync_binance_symbols
from .utils import search_symbols, get_binance_symbol_info

@dataclass
class PortfolioPosition:
    symbol: str
    qty: float
    price_entry: float
    price_current: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    realized_1: Optional[float] = None  # % realizada no target 1
    realized_2: Optional[float] = None  # % realizada no target 2
    realized_3: Optional[float] = None  # % realizada no target 3
    allocation_target: Optional[float] = None
    allocation_actual: Optional[float] = None
    last_update: Optional[datetime] = None
    notes: Optional[str] = None

def get_symbol_info(symbol: str) -> Optional[Dict]:
    """
    Busca informações do símbolo no banco de dados.
    """
    try:
        session = get_db_session()
        stmt = select(AvailableSymbols).where(AvailableSymbols.symbol == symbol)
        result = session.execute(stmt).scalar_one_or_none()
        
        if result:
            return {
                'symbol': result.symbol,
                'base_currency': result.base_currency,
                'quote_currency': result.quote_currency,
                'last_price': result.last_price,
                'last_update': result.last_update
            }
        return None
        
    except Exception as e:
        st.error(f"Erro ao buscar informações do símbolo: {e}")
        return None
    finally:
        session.close()

def get_binance_price(symbol: str) -> float:
    """
    Busca preço atual na Binance.
    """
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        
        # Atualiza o preço no banco de dados
        session = get_db_session()
        stmt = select(AvailableSymbols).where(AvailableSymbols.symbol == symbol)
        symbol_record = session.execute(stmt).scalar_one_or_none()
        
        if symbol_record:
            symbol_record.last_price = ticker['last']
            symbol_record.last_update = datetime.now()
            session.commit()
        
        return ticker['last']
    except Exception as e:
        st.error(f"Erro ao buscar preço: {e}")
        return None
    finally:
        if 'session' in locals():
            session.close()

def update_position_data(position: PortfolioPosition) -> PortfolioPosition:
    """
    Atualiza dados da posição com informações da exchange.
    """
    current_price = get_binance_price(position.symbol)
    if current_price:
        position.price_current = current_price
        position.last_update = datetime.now()
    return position

def calculate_position_metrics(position: PortfolioPosition) -> Dict:
    """
    Calcula métricas da posição.
    """
    if not position.price_current:
        return {}
    
    return {
        "unrealized_pl": (position.price_current - position.price_entry) * position.qty,
        "unrealized_pl_pct": (position.price_current - position.price_entry) / position.price_entry,
        "risk": (position.price_entry - position.stop_loss) / position.price_entry if position.stop_loss else None,
        "to_target_1": (position.target_1 - position.price_current) / position.price_current if position.target_1 else None,
        "to_target_2": (position.target_2 - position.price_current) / position.price_current if position.target_2 else None,
        "to_target_3": (position.target_3 - position.price_current) / position.price_current if position.target_3 else None,
    }

def save_portfolio(portfolio: List[PortfolioPosition]):
    """
    Salva o portfólio no banco de dados e no session_state do Streamlit.
    """
    # Salva no session_state para a sessão atual
    st.session_state['portfolio'] = [vars(pos) for pos in portfolio]
    
    try:
        # Salva no banco de dados usando SQLAlchemy
        session = get_db_session()
        
        # Remove posições existentes
        session.query(Portfolio).delete()
        
        # Insere novas posições usando o modelo SQLAlchemy
        for pos in portfolio:
            new_pos = Portfolio(
                symbol=pos.symbol,
                qty=pos.qty,
                price_entry=pos.price_entry,
                price_current=pos.price_current,
                stop_loss=pos.stop_loss,
                target_1=pos.target_1,
                target_2=pos.target_2,
                target_3=pos.target_3,
                realized_1=pos.realized_1,
                realized_2=pos.realized_2,
                realized_3=pos.realized_3,
                allocation_target=pos.allocation_target,
                allocation_actual=pos.allocation_actual,
                last_update=pos.last_update,
                notes=pos.notes
            )
            session.add(new_pos)
        
        session.commit()
        
    except Exception as e:
        st.error(f"Erro ao salvar no banco de dados: {str(e)}")
        if 'session' in locals():
            session.rollback()
    finally:
        if 'session' in locals():
            session.close()

def load_portfolio() -> List[PortfolioPosition]:
    """
    Carrega o portfólio do banco de dados ou session_state do Streamlit.
    """
    # Lista de campos válidos do PortfolioPosition
    valid_fields = set(PortfolioPosition.__dataclass_fields__.keys())
    
    # Tenta carregar do session_state primeiro
    if 'portfolio' in st.session_state:
        # Filtra apenas os campos válidos para cada posição
        portfolio_data = []
        for pos in st.session_state['portfolio']:
            filtered_pos = {k: v for k, v in pos.items() if k in valid_fields}
            portfolio_data.append(PortfolioPosition(**filtered_pos))
        return portfolio_data
    
    try:
        # Se não existir no session_state, carrega do banco usando SQLAlchemy
        session = get_db_session()
        portfolio_records = session.query(Portfolio).all()
        
        if not portfolio_records:
            return []
            
        positions = []
        for record in portfolio_records:
            # Converte registro para dicionário
            record_dict = {
                'symbol': record.symbol,
                'qty': record.qty,
                'price_entry': record.price_entry,
                'price_current': record.price_current,
                'stop_loss': record.stop_loss,
                'target_1': record.target_1,
                'target_2': record.target_2,
                'target_3': record.target_3,
                'realized_1': record.realized_1,
                'realized_2': record.realized_2,
                'realized_3': record.realized_3,
                'allocation_target': record.allocation_target,
                'allocation_actual': record.allocation_actual if hasattr(record, 'allocation_actual') else None,
                'last_update': record.last_update,
                'notes': record.notes
            }
            
            # Remove valores None para usar os padrões da classe
            record_dict = {k: v for k, v in record_dict.items() if v is not None}
            
            positions.append(PortfolioPosition(**record_dict))
        
        # Salva no session_state para futuras consultas
        st.session_state['portfolio'] = [vars(pos) for pos in positions]
        
        return positions
        
    except Exception as e:
        st.error(f"Erro ao carregar do banco de dados: {str(e)}")
        return []
        
    finally:
        if 'session' in locals():
            session.close()

def show_portfolio_form():
    """
    Mostra formulário de edição do portfólio.
    """
    st.subheader("📝 Editar Portfólio")
    
    # Carrega portfólio atual
    portfolio = load_portfolio()
    
    # Botão para sincronizar símbolos
    if st.button("🔄 Sincronizar Símbolos"):
        try:
            with st.spinner('Sincronizando símbolos da Binance...'):
                sync_binance_symbols()
            st.success("Símbolos sincronizados com sucesso!")
        except Exception as e:
            st.error(f"Erro ao sincronizar símbolos: {e}")
    
    # Form para adicionar nova posição
    with st.form("nova_posicao"):
        st.write("Adicionar Nova Posição")
        
        # Campo de busca
        symbol_search = st.text_input("Buscar Symbol (ex: BTC)", key="symbol_search")
        
        symbol = None
        price = None
        
        # Busca símbolos que correspondam à pesquisa
        if symbol_search:
            session = get_db_session()
            query = select(AvailableSymbols).where(
                (AvailableSymbols.symbol.ilike(f"%{symbol_search.upper()}%")) |
                (AvailableSymbols.base_currency.ilike(f"%{symbol_search.upper()}%")) |
                (AvailableSymbols.quote_currency.ilike(f"%{symbol_search.upper()}%"))
            ).where(AvailableSymbols.is_active == True)
            
            matches = session.execute(query).scalars().all()
            session.close()
            
            if matches:
                # Cria lista de opções formatadas
                options = [f"{m.symbol} ({m.base_currency}/{m.quote_currency})" for m in matches]
                selected = st.selectbox("Selecione o par de trading:", options)
                
                if selected:
                    # Extrai o símbolo da opção selecionada
                    symbol = selected.split()[0]  # Pega apenas o símbolo
                    # Busca informações do símbolo
                    symbol_info = get_symbol_info(symbol)
                    if symbol_info:
                        price = symbol_info['last_price']
                        if price:
                            st.success(f"Preço atual: ${price:.2f}")
                        else:
                            # Se não tiver preço no banco, busca na Binance
                            price = get_binance_price(symbol)
                            if price:
                                st.success(f"Preço atual: ${price:.2f}")
                            else:
                                st.warning("Não foi possível obter o preço atual")
            else:
                st.warning("Nenhum símbolo encontrado. Tente outro termo de busca.")
        
        col1, col2 = st.columns(2)
        with col1:
            qty = st.number_input("Quantidade", min_value=0.0, key="new_qty", help="Quantidade do ativo a ser comprada")
            price_entry = st.number_input("Preço de Entrada", 
                                        min_value=0.0, 
                                        value=price if price else 0.0,
                                        key="new_price",
                                        help="Preço de entrada da posição")
            
            # Ajuda visual para validar o preço de entrada
            if price and price_entry > 0:
                price_diff = ((price_entry - price) / price) * 100
                if abs(price_diff) > 5:  # Se diferença maior que 5%
                    st.warning(f"Preço de entrada está {price_diff:.1f}% {'acima' if price_diff > 0 else 'abaixo'} do preço atual")
                elif abs(price_diff) > 0:
                    st.info(f"Preço de entrada está {price_diff:.1f}% {'acima' if price_diff > 0 else 'abaixo'} do preço atual")
                        
            stop_loss = st.number_input("Stop Loss", 
                                      min_value=0.0, 
                                      key="new_stop",
                                      help="Preço para stop loss (opcional)")
            
            # Validação do stop loss
            if stop_loss > 0 and price_entry > 0:
                stop_pct = ((price_entry - stop_loss) / price_entry) * 100
                st.info(f"Stop Loss definido em {stop_pct:.1f}% abaixo do preço de entrada")
            
        with col2:
            # Configuração dos targets com feedback visual
            target_1 = st.number_input("Target 1 (Primeiro Alvo)", 
                                     min_value=0.0, 
                                     key="new_target1",
                                     help="Primeiro alvo de lucro (opcional)")
            
            # Feedback do target 1
            if target_1 > 0 and price_entry > 0:
                t1_pct = ((target_1 - price_entry) / price_entry) * 100
                st.info(f"Target 1: {t1_pct:.1f}% acima do preço de entrada")
            
            target_2 = st.number_input("Target 2 (Segundo Alvo)", 
                                     min_value=0.0, 
                                     key="new_target2",
                                     help="Segundo alvo de lucro (opcional)")
            
            # Feedback do target 2
            if target_2 > 0 and price_entry > 0:
                t2_pct = ((target_2 - price_entry) / price_entry) * 100
                st.info(f"Target 2: {t2_pct:.1f}% acima do preço de entrada")
            
            target_3 = st.number_input("Target 3 (Terceiro Alvo)", 
                                     min_value=0.0, 
                                     key="new_target3",
                                     help="Terceiro alvo de lucro (opcional)")
            
            # Feedback do target 3
            if target_3 > 0 and price_entry > 0:
                t3_pct = ((target_3 - price_entry) / price_entry) * 100
                st.info(f"Target 3: {t3_pct:.1f}% acima do preço de entrada")
            
        col3, col4 = st.columns(2)
        with col3:
            # Configuração das realizações parciais
            realized_1 = st.number_input("% Realizada Target 1", 
                                       min_value=0.0, 
                                       max_value=100.0, 
                                       key="new_real1",
                                       help="Porcentagem a ser realizada no Target 1")
            
            realized_2 = st.number_input("% Realizada Target 2", 
                                       min_value=0.0, 
                                       max_value=100.0, 
                                       key="new_real2",
                                       help="Porcentagem a ser realizada no Target 2")
            
            realized_3 = st.number_input("% Realizada Target 3", 
                                       min_value=0.0, 
                                       max_value=100.0, 
                                       key="new_real3",
                                       help="Porcentagem a ser realizada no Target 3")
            
            # Validação das realizações
            total_realized = realized_1 + realized_2 + realized_3
            if total_realized > 0:
                st.info(f"Total a ser realizado: {total_realized:.1f}%")
                if total_realized > 100:
                    st.warning("⚠️ Total de realização acima de 100%")
                elif total_realized < 100:
                    st.warning("⚠️ Total de realização abaixo de 100%")
                else:
                    st.success("✅ Realização total de 100%")
            
        with col4:
            allocation_target = st.number_input("Alocação Alvo (%)", 
                                              min_value=0.0, 
                                              max_value=100.0, 
                                              key="new_alloc",
                                              help="Porcentagem alvo do portfólio para esta posição")
            
            notes = st.text_area("Observações", 
                               key="new_notes",
                               help="Notas adicionais sobre a posição")
        
        submitted = st.form_submit_button("Adicionar Posição")
        if submitted:
            if symbol and qty > 0 and price_entry > 0:
                new_position = PortfolioPosition(
                    symbol=symbol,
                    qty=qty,
                    price_entry=price_entry,
                    price_current=price if price else None,
                    stop_loss=stop_loss if stop_loss > 0 else None,
                    target_1=target_1 if target_1 > 0 else None,
                    target_2=target_2 if target_2 > 0 else None,
                    target_3=target_3 if target_3 > 0 else None,
                    realized_1=realized_1/100 if realized_1 > 0 else None,
                    realized_2=realized_2/100 if realized_2 > 0 else None,
                    realized_3=realized_3/100 if realized_3 > 0 else None,
                    allocation_target=allocation_target/100 if allocation_target > 0 else None,
                    notes=notes if notes else None,
                    last_update=datetime.now()
                )
                portfolio.append(new_position)
                save_portfolio(portfolio)
                st.success(f"Posição em {symbol} adicionada com sucesso!")
    
    # Mostra posições atuais
    if portfolio:
        st.write("### Posições Atuais")
        
        # Botão para atualizar preços
        if st.button("🔄 Atualizar Preços"):
            updated_portfolio = []
            for pos in portfolio:
                updated_pos = update_position_data(pos)
                updated_portfolio.append(updated_pos)
            portfolio = updated_portfolio
            save_portfolio(portfolio)
            st.success("Preços atualizados!")
        
        # Prepara dados para visualização
        positions_data = []
        for pos in portfolio:
            # Primeiro coletamos apenas os campos do PortfolioPosition
            pos_data = {k: v for k, v in vars(pos).items() if k in PortfolioPosition.__dataclass_fields__}
            
            # Depois adicionamos as métricas calculadas em um dicionário separado
            metrics = calculate_position_metrics(pos)
            display_data = pos_data.copy()  # Cria uma cópia para exibição
            display_data.update(metrics)
            positions_data.append(display_data)
        
        df = pd.DataFrame(positions_data)
        
        # Adiciona coluna de exclusão
        df['excluir'] = False

        # Configuração das colunas do editor
        column_config = {
            "symbol": "Symbol",
            "qty": st.column_config.NumberColumn("Quantidade", format="%.4f"),
            "price_entry": st.column_config.NumberColumn("Preço de Entrada", format="%.2f"),
            "price_current": st.column_config.NumberColumn("Preço Atual", format="%.2f"),
            "stop_loss": st.column_config.NumberColumn("Stop Loss", format="%.2f"),
            "target_1": st.column_config.NumberColumn("Target 1", format="%.2f"),
            "target_2": st.column_config.NumberColumn("Target 2", format="%.2f"),
            "target_3": st.column_config.NumberColumn("Target 3", format="%.2f"),
            "realized_1": st.column_config.NumberColumn("% Real. T1", format="%.1%"),
            "realized_2": st.column_config.NumberColumn("% Real. T2", format="%.1%"),
            "realized_3": st.column_config.NumberColumn("% Real. T3", format="%.1%"),
            "allocation_target": st.column_config.NumberColumn("Aloc. Alvo", format="%.1%"),
            "allocation_actual": st.column_config.NumberColumn("Aloc. Atual", format="%.1%"),
            "unrealized_pl": st.column_config.NumberColumn("P&L ($)", format="%.2f"),
            "unrealized_pl_pct": st.column_config.NumberColumn("P&L %", format="%.1%"),
            "risk": st.column_config.NumberColumn("Risco %", format="%.1%"),
            "to_target_1": st.column_config.NumberColumn("Até T1", format="%.1%"),
            "to_target_2": st.column_config.NumberColumn("Até T2", format="%.1%"),
            "to_target_3": st.column_config.NumberColumn("Até T3", format="%.1%"),
            "last_update": "Última Atualização",
            "notes": "Observações",
            "excluir": st.column_config.CheckboxColumn("Excluir", help="Marque para excluir esta posição")
        }
        
        # Permite edição do DataFrame
        edited_df = st.data_editor(
            df,
            hide_index=True,
            column_config=column_config,
            width='stretch'
        )
        
        # Verifica se há posições marcadas para exclusão
        positions_to_delete = edited_df[edited_df['excluir'] == True]

        if not positions_to_delete.empty:
            # Remove as posições marcadas para exclusão
            symbols_to_delete = positions_to_delete['symbol'].tolist()
            new_portfolio = [pos for pos in portfolio if pos.symbol not in symbols_to_delete]

            if len(new_portfolio) < len(portfolio):
                save_portfolio(new_portfolio)
                deleted_count = len(portfolio) - len(new_portfolio)
                st.success(f"{deleted_count} posição(ões) excluída(s) com sucesso!")
                st.rerun()  # Recarrega a página para atualizar a tabela
                return

        # Atualiza o portfólio se houver outras mudanças (excluindo a coluna 'excluir')
        df_no_excluir = df.drop(columns=['excluir'])
        edited_df_no_excluir = edited_df.drop(columns=['excluir'])

        if not df_no_excluir.equals(edited_df_no_excluir):
            try:
                # Filtra apenas os campos válidos e cria novas posições
                valid_fields = set(PortfolioPosition.__dataclass_fields__.keys())
                new_portfolio = []

                for row in edited_df.to_dict('records'):
                    # Filtra apenas campos válidos do PortfolioPosition
                    filtered_data = {k: v for k, v in row.items() if k in valid_fields}
                    # Converte campos None para seus valores padrão
                    position = PortfolioPosition(**filtered_data)
                    new_portfolio.append(position)

                save_portfolio(new_portfolio)
                st.success("Portfólio atualizado!")

            except Exception as e:
                st.error(f"Erro ao atualizar portfólio: {str(e)}")
    
    else:
        st.info("Nenhuma posição cadastrada. Adicione sua primeira posição usando o formulário acima.")