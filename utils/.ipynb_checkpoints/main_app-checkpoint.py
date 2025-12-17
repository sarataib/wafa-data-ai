"""
📊 MarketSense Morocco - Application de Trading et Analyse de Marché
Application principale Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sqlite3

# Style CSS personnalisé
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #0D47A1;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #1E88E5;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .alert-low {
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .positive-change {
            color: #4caf50;
            font-weight: bold;
        }
        .negative-change {
            color: #f44336;
            font-weight: bold;
        }
        .stock-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Fonctions utilitaires de base (pour remplacer les imports manquants)
def get_db_connection():
    """Crée une connexion à la base de données"""
    try:
        conn = sqlite3.connect('database/market_data.db', check_same_thread=False)
        return conn
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        return None

def load_instruments():
    """Charge tous les instruments"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = "SELECT DISTINCT ticker FROM daily_prices"
        df = pd.read_sql_query(query, conn)
        return df
    except:
        return pd.DataFrame()

# Fonctions principales de l'application
def main():
    """Fonction principale de l'application"""
    apply_custom_css()
    
    # Titre principal
    st.markdown('<h1 class="main-header">📈 MarketSense Morocco</h1>', unsafe_allow_html=True)
    st.markdown("### Plateforme d'Analyse et de Prédiction des Marchés Marocains")

    # Sidebar - Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=80)
        st.title("Navigation")
        
        menu = st.radio(
            "Sélectionnez une section:",
            ["🏠 Tableau de Bord", "📊 Analyse d'Instrument", "🤖 Prédictions ML", "🚨 Alertes", "📈 Comparaison", "⚙️ Recherche"]
        )
        
        st.markdown("---")
        st.markdown("### Filtres")
        
        # Filtre par secteur
        sectors = ["Tous", "Banque", "Assurance", "Immobilier", "Énergie", "Télécom", "Industrie", "Services"]
        selected_sector = st.selectbox("Secteur:", sectors)
        
        st.markdown("---")
        st.markdown("### 📡 Mise à jour")
        if st.button("🔄 Actualiser les données"):
            st.cache_data.clear()
            st.success("Données actualisées!")
            time.sleep(1)
            st.rerun()

    # Navigation principale
    if menu == "🏠 Tableau de Bord":
        show_dashboard()
    elif menu == "📊 Analyse d'Instrument":
        show_analysis()
    elif menu == "🤖 Prédictions ML":
        show_predictions()
    elif menu == "🚨 Alertes":
        show_alerts()
    elif menu == "📈 Comparaison":
        show_comparison()
    elif menu == "⚙️ Recherche":
        show_search()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>📊 <b>MarketSense Morocco</b> - Plateforme d'Analyse des Marchés | Données mises à jour quotidiennement</p>
            <p>⚠️ <i>Les analyses et prédictions sont fournies à titre indicatif seulement.</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_dashboard():
    """Affiche le tableau de bord"""
    st.markdown('<h2 class="sub-header">🏠 Tableau de Bord du Marché</h2>', unsafe_allow_html=True)
    
    # Connexion à la base de données
    conn = get_db_connection()
    
    if conn is None:
        st.warning("Base de données non disponible")
        return
    
    # Récupérer les métriques de base
    try:
        # Nombre d'instruments
        query = "SELECT COUNT(DISTINCT ticker) as count FROM daily_prices"
        count_df = pd.read_sql_query(query, conn)
        instrument_count = count_df['count'].iloc[0]
        
        # Dernières données
        query = """
        SELECT ticker, close, volume, date 
        FROM daily_prices 
        WHERE date = (SELECT MAX(date) FROM daily_prices)
        """
        latest_df = pd.read_sql_query(query, conn)
        
        # Calculer les métriques
        total_volume = latest_df['volume'].sum() if not latest_df.empty else 0
        avg_price = latest_df['close'].mean() if not latest_df.empty else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Instruments", f"{instrument_count}")
        
        with col2:
            st.metric("💰 Volume Total", f"{total_volume:,.0f}")
        
        with col3:
            st.metric("📈 Prix Moyen", f"{avg_price:.2f} MAD")
        
        with col4:
            st.metric("📅 Dernière MAJ", latest_df['date'].iloc[0] if not latest_df.empty else "N/A")
        
        # Afficher les instruments disponibles
        st.markdown("### 📋 Instruments Disponibles")
        if not latest_df.empty:
            # Sélectionner quelques colonnes
            display_df = latest_df[['ticker', 'close', 'volume']].copy()
            display_df.columns = ['Ticker', 'Dernier Prix (MAD)', 'Volume']
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Aucune donnée disponible dans la base de données")
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
    finally:
        conn.close()

def show_analysis():
    """Affiche l'analyse d'instrument"""
    st.markdown('<h2 class="sub-header">📊 Analyse d\'Instrument</h2>', unsafe_allow_html=True)
    
    # Récupérer les instruments disponibles
    instruments_df = load_instruments()
    
    if instruments_df.empty:
        st.warning("Aucun instrument disponible dans la base de données")
        st.info("Exécutez init_database.py pour créer des données de test")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_ticker = st.selectbox(
            "Sélectionnez un instrument:",
            instruments_df['ticker'].tolist()
        )
    
    with col2:
        period = st.selectbox(
            "Période d'analyse:",
            ["7 jours", "30 jours", "90 jours", "1 an"],
            index=1
        )
    
    if selected_ticker:
        # Connexion à la base de données
        conn = get_db_connection()
        
        if conn is None:
            return
        
        try:
            # Récupérer les données historiques
            periods = {"7 jours": 7, "30 jours": 30, "90 jours": 90, "1 an": 365}
            days = periods[period]
            
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM daily_prices 
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(selected_ticker, days))
            
            if df.empty:
                st.warning(f"Aucune donnée disponible pour {selected_ticker}")
                return
            
            # Convertir la date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')  # Trier par date croissante
            
            # Informations de base
            latest = df.iloc[-1]  # Dernière ligne (la plus récente)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prix Actuel", f"{latest['close']:.2f} MAD")
            with col2:
                price_change = ((latest['close'] - latest['open']) / latest['open'] * 100)
                st.metric("Change Journalier", f"{price_change:.2f}%")
            with col3:
                st.metric("Volume", f"{latest['volume']:,.0f}")
            with col4:
                st.metric("Prix Haut", f"{latest['high']:.2f} MAD")
            
            # Onglets
            tab1, tab2 = st.tabs(["📈 Graphique des Prix", "📊 Données"])
            
            with tab1:
                # Graphique de prix
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['close'],
                    mode='lines',
                    name='Prix de clôture',
                    line=dict(color='#1E88E5', width=2)
                ))
                
                # Ajouter les prix d'ouverture
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['open'],
                    mode='lines',
                    name='Prix d\'ouverture',
                    line=dict(color='#FF9800', width=1, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Évolution du prix - {selected_ticker}',
                    xaxis_title='Date',
                    yaxis_title='Prix (MAD)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Graphique de volume
                fig2 = go.Figure()
                
                # Barres de volume colorées selon le mouvement de prix
                colors = ['red' if close < open else 'green' 
                         for close, open in zip(df['close'], df['open'])]
                
                fig2.add_trace(go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors
                ))
                
                fig2.update_layout(
                    title=f'Volume d\'échanges - {selected_ticker}',
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    hovermode='x unified',
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                # Afficher les données
                st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
                
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {e}")
        finally:
            conn.close()

def show_predictions():
    """Affiche les prédictions ML"""
    st.markdown('<h2 class="sub-header">🤖 Prédictions Machine Learning</h2>', unsafe_allow_html=True)
    
    st.info("🚀 Module de prédictions en cours de développement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Modèles disponibles")
        st.write("""
        - Modèle de régression linéaire
        - Random Forest Regressor
        - LSTM (Deep Learning)
        - XGBoost
        """)
    
    with col2:
        st.markdown("### 🎯 Précision cible")
        st.write("""
        - Prédiction J+1: 70-80%
        - Prédiction J+5: 60-70%
        - Prédiction direction: 75-85%
        """)
    
    # Simulation de prédiction
    st.markdown("### 🔮 Simulation de prédiction")
    
    instruments_df = load_instruments()
    if not instruments_df.empty:
        selected_ticker = st.selectbox(
            "Instrument pour simulation:",
            instruments_df['ticker'].tolist()
        )
        
        if selected_ticker:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prédiction J+1", "102.45 MAD", "+2.15%")
            with col2:
                st.metric("Prédiction J+5", "105.78 MAD", "+4.78%")
            with col3:
                st.metric("Confiance", "78%", "FORT")

def show_alerts():
    """Affiche les alertes"""
    st.markdown('<h2 class="sub-header">🚨 Alertes du Marché</h2>', unsafe_allow_html=True)
    
    # Alertes de démonstration
    alerts = [
        {"ticker": "ATTIJARIWAFA", "type": "🟢 OVERSOLD", "message": "RSI à 28.5 - Opportunité d'achat", "priority": "HIGH"},
        {"ticker": "IAM", "type": "📊 VOLUME_SPIKE", "message": "Volume 2.5x la moyenne - Forte activité", "priority": "MEDIUM"},
        {"ticker": "BMCE", "type": "📈 MACD_BULLISH", "message": "Signal d'achat technique - Croisement haussier", "priority": "MEDIUM"},
        {"ticker": "LHM", "type": "🔴 OVERBOUGHT", "message": "RSI à 72.3 - Possible correction", "priority": "HIGH"},
        {"ticker": "COSUMAR", "type": "📉 VOLUME_LOW", "message": "Volume faible - Faible intérêt", "priority": "LOW"},
    ]
    
    # Statistiques
    high_alerts = len([a for a in alerts if a['priority'] == 'HIGH'])
    medium_alerts = len([a for a in alerts if a['priority'] == 'MEDIUM'])
    low_alerts = len([a for a in alerts if a['priority'] == 'LOW'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Haute Priorité", high_alerts)
    with col2:
        st.metric("🟡 Moyenne Priorité", medium_alerts)
    with col3:
        st.metric("🟢 Basse Priorité", low_alerts)
    
    # Filtres
    st.markdown("### 🔍 Filtres")
    col1, col2 = st.columns(2)
    
    with col1:
        show_high = st.checkbox("Haute priorité", value=True)
        show_medium = st.checkbox("Moyenne priorité", value=True)
    
    with col2:
        show_low = st.checkbox("Basse priorité", value=True)
    
    # Afficher les alertes filtrées
    st.markdown(f"### 📋 Alertes ({len(alerts)})")
    
    for alert in alerts:
        if (alert['priority'] == 'HIGH' and not show_high) or \
           (alert['priority'] == 'MEDIUM' and not show_medium) or \
           (alert['priority'] == 'LOW' and not show_low):
            continue
        
        priority_class = f"alert-{alert['priority'].lower()}"
        
        st.markdown(f"""
        <div class="{priority_class}">
            <strong>{alert['ticker']}</strong> - {alert['type']}<br>
            {alert['message']}<br>
            <small>Priorité: {alert['priority']}</small>
        </div>
        """, unsafe_allow_html=True)

def show_comparison():
    """Affiche la comparaison"""
    st.markdown('<h2 class="sub-header">📈 Comparaison d\'Instruments</h2>', unsafe_allow_html=True)
    
    # Récupérer les instruments disponibles
    instruments_df = load_instruments()
    
    if instruments_df.empty:
        st.warning("Aucun instrument disponible")
        return
    
    # Sélection multiple
    selected_tickers = st.multiselect(
        "Sélectionnez les instruments à comparer (max 5):",
        instruments_df['ticker'].tolist(),
        default=instruments_df['ticker'].tolist()[:3] if len(instruments_df) >= 3 else instruments_df['ticker'].tolist(),
        max_selections=5
    )
    
    if not selected_tickers:
        st.info("Veuillez sélectionner au moins un instrument")
        return
    
    # Connexion à la base de données
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        # Récupérer les données pour chaque instrument
        all_data = {}
        
        for ticker in selected_tickers:
            query = """
            SELECT date, close 
            FROM daily_prices 
            WHERE ticker = ? 
            ORDER BY date DESC 
            LIMIT 30
            """
            df = pd.read_sql_query(query, conn, params=(ticker,))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                all_data[ticker] = df
        
        if not all_data:
            st.warning("Aucune donnée disponible pour les instruments sélectionnés")
            return
        
        # Graphique comparatif
        fig = go.Figure()
        
        for ticker, df in all_data.items():
            # Normaliser les prix pour une comparaison équitable
            normalized = df['close'] / df['close'].iloc[0] * 100
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=normalized,
                mode='lines',
                name=ticker
            ))
        
        fig.update_layout(
            title="Comparaison des performances (normalisé à 100)",
            xaxis_title="Date",
            yaxis_title="Performance (%)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau comparatif
        st.markdown("### 📊 Tableau comparatif")
        
        comparison_data = []
        for ticker, df in all_data.items():
            if not df.empty:
                latest = df.iloc[0]
                first = df.iloc[-1]  # Le plus ancien (inversé)
                change_pct = ((latest['close'] - first['close']) / first['close'] * 100)
                
                comparison_data.append({
                    'Ticker': ticker,
                    'Prix Initial': f"{first['close']:.2f}",
                    'Prix Actuel': f"{latest['close']:.2f}",
                    'Variation': f"{change_pct:+.2f}%"
                })
        
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la comparaison: {e}")
    finally:
        conn.close()

def show_search():
    """Affiche la recherche"""
    st.markdown('<h2 class="sub-header">⚙️ Recherche d\'Instruments</h2>', unsafe_allow_html=True)
    
    # Barre de recherche
    search_term = st.text_input("🔍 Rechercher un instrument (ticker):", "")
    
    # Récupérer tous les instruments
    instruments_df = load_instruments()
    
    if instruments_df.empty:
        st.warning("Aucun instrument disponible")
        return
    
    # Recherche
    if search_term:
        results = instruments_df[instruments_df['ticker'].str.contains(search_term, case=False)]
        st.markdown(f"### 📊 Résultats ({len(results)} trouvés)")
        
        if not results.empty:
            st.dataframe(results, use_container_width=True)
        else:
            st.info(f"Aucun instrument trouvé pour '{search_term}'")
    
    # Tous les instruments
    st.markdown("### 📋 Tous les Instruments")
    
    # Pagination
    items_per_page = 10
    total_items = len(instruments_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        st.write(f"Affichage des instruments {start_idx + 1} à {end_idx} sur {total_items}")
        
        # Afficher la page courante
        current_page_df = instruments_df.iloc[start_idx:end_idx]
        st.dataframe(current_page_df, use_container_width=True)
        
        # Navigation de page
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page > 1:
                if st.button("← Page précédente"):
                    st.session_state.page = page - 1
                    st.rerun()
        with col3:
            if page < total_pages:
                if st.button("Page suivante →"):
                    st.session_state.page = page + 1
                    st.rerun()
    else:
        st.dataframe(instruments_df, use_container_width=True)