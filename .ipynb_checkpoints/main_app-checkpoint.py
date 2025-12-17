"""
📊 MarketSense Morocco - Application Complète avec Prédictions Interactives
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sqlite3
import os
import sys

# Ajouter le répertoire courant pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        .prediction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .signal-buy {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        .signal-sell {
            background-color: #F44336 !important;
            color: white !important;
        }
        .signal-neutral {
            background-color: #FF9800 !important;
            color: white !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 5px 5px 0px 0px;
            padding: 10px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

# Fonctions utilitaires de base
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
        query = "SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des instruments: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_instrument_name(ticker):
    """Récupère le nom d'un instrument"""
    conn = get_db_connection()
    if conn is None:
        return ticker
    
    try:
        query = "SELECT name FROM instruments WHERE ticker = ?"
        result = pd.read_sql_query(query, conn, params=(ticker,))
        if not result.empty:
            return result.iloc[0]['name']
        return ticker
    except:
        return ticker
    finally:
        conn.close()

def get_available_models():
    """Récupère la liste des modèles disponibles"""
    models_dir = 'models'
    if os.path.exists(models_dir):
        available_models = [f.split('_model.joblib')[0] for f in os.listdir(models_dir) 
                          if f.endswith('_model.joblib')]
        return sorted(available_models)
    return []

# Fonctions principales de l'application
def main():
    """Fonction principale de l'application"""
    apply_custom_css()
    
    # Titre principal
    st.markdown('<h1 class="main-header">📈 MarketSense Morocco</h1>', unsafe_allow_html=True)
    st.markdown("### Plateforme d'Analyse et de Prédiction des Marchés Marocains")

    # Sidebar - Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/?size=100&id=112613&format=png&color=000000", width=80)
        st.title("Navigation")
        
        menu_options = [
            "🏠 Tableau de Bord",
            "📊 Analyse d'Instrument", 
            "🤖 Prédictions ML",
            "🎮 Prédictions Interactives",
            "🚨 Alertes",
            "📈 Comparaison",
            "⚙️ Recherche"
        ]
        
        menu = st.radio("Sélectionnez une section:", menu_options)
        
    # Navigation principale
    if menu == "🏠 Tableau de Bord":
        show_dashboard()
    elif menu == "📊 Analyse d'Instrument":
        show_analysis()
    elif menu == "🤖 Prédictions ML":
        show_predictions()
    elif menu == "🎮 Prédictions Interactives":
        show_interactive_predictions()
    elif menu == "🚨 Alertes":
        show_alerts()
    elif menu == "📈 Comparaison":
        show_comparison()
    elif menu == "⚙️ Recherche":
        show_search()

    # Footer
    st.markdown("---")

# ============================================
# TABLEAU DE BORD
# ============================================

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
        instrument_count = count_df['count'].iloc[0] if not count_df.empty else 0
        
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
            # Ajouter les noms
            latest_df['name'] = latest_df['ticker'].apply(get_instrument_name)
            
            # Sélectionner quelques colonnes
            display_df = latest_df[['ticker', 'name', 'close', 'volume']].copy()
            display_df.columns = ['Ticker', 'Nom', 'Dernier Prix (MAD)', 'Volume']
            
            # Formater les nombres
            display_df['Dernier Prix (MAD)'] = display_df['Dernier Prix (MAD)'].apply(lambda x: f"{x:.2f}")
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune donnée disponible dans la base de données")
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
    finally:
        conn.close()

# ============================================
# ANALYSE D'INSTRUMENT
# ============================================

def show_analysis():
    """Affiche l'analyse d'instrument"""
    st.markdown('<h2 class="sub-header">📊 Analyse d\'Instrument</h2>', unsafe_allow_html=True)
    
    # Récupérer les instruments disponibles
    instruments_df = load_instruments()
    
    if instruments_df.empty:
        st.warning("Aucun instrument disponible dans la base de données")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_ticker = st.selectbox(
            "Sélectionnez un instrument:",
            instruments_df['ticker'].tolist(),
            format_func=lambda x: f"{x} - {get_instrument_name(x)}"
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
                display_df = df.sort_values('date', ascending=False).copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df['Volume'] = display_df['volume'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(display_df[['date', 'open', 'high', 'low', 'close', 'Volume']], 
                           use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {e}")
        finally:
            conn.close()

# ============================================
# PRÉDICTIONS ML (AUTOMATIQUES)
# ============================================

def show_predictions():
    """Affiche les prédictions ML automatiques"""
    st.markdown('<h2 class="sub-header">🤖 Prédictions Machine Learning</h2>', unsafe_allow_html=True)
    
    # Vérifier les modèles disponibles
    available_models = get_available_models()
    
    if not available_models:
        st.warning("""
        ⚠️ **Aucun modèle ML disponible**
        
        Pour utiliser cette fonctionnalité :
        1. Exécutez `python train_models.py` pour créer les modèles
        2. Assurez-vous que le dossier `models/` contient des fichiers `.joblib`
        3. Redémarrez l'application
        """)
        
        # Mode démonstration
        show_demo_predictions()
        return
    
    # SUPPRIMÉ: Onglets retirés - affichage direct des prédictions auto
    show_auto_predictions(available_models)

def show_auto_predictions(available_models):
    """Affiche les prédictions automatiques"""
    st.markdown("### 📊 Prédictions Automatiques")
    st.markdown("Basées sur les dernières données disponibles")
    
    # Sélection de l'instrument
    selected_ticker = st.selectbox(
        "Sélectionnez un instrument:",
        available_models,
        format_func=lambda x: f"{x} - {get_instrument_name(x)}",
        help="Choisissez un instrument pour voir les prédictions"
    )
    
    if selected_ticker:
        try:
            # Essayer d'importer le module de prédictions
            from utils.predictions import load_predictions
            
            with st.spinner("🤖 Chargement des prédictions..."):
                prediction = load_predictions(selected_ticker)
                
                if prediction:
                    display_prediction_results(prediction)
                else:
                    st.error("Impossible de générer des prédictions pour cet instrument")
                    
        except ImportError:
            # Mode dégradé
            show_fallback_predictions(selected_ticker)
        except Exception as e:
            st.error(f"Erreur: {e}")
            show_fallback_predictions(selected_ticker)

def display_prediction_results(prediction):
    """Affiche les résultats de prédiction"""
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prix Actuel", f"{prediction['current_price']:.2f} MAD")
    
    with col2:
        change_1d = ((prediction['day1'] - prediction['current_price']) / prediction['current_price']) * 100
        st.metric("Prédiction J+1", f"{prediction['day1']:.2f} MAD", f"{change_1d:+.2f}%")
    
    with col3:
        change_5d = ((prediction['day5'] - prediction['current_price']) / prediction['current_price']) * 100
        st.metric("Prédiction J+5", f"{prediction['day5']:.2f} MAD", f"{change_5d:+.2f}%")
    
    with col4:
        st.metric("Direction", prediction['direction'], f"Confiance: {prediction['confidence']*100:.1f}%")
    
    # Graphique
    fig = go.Figure()
    
    periods = ['Actuel', 'J+1', 'J+5', 'J+10']
    prices = [
        prediction['current_price'],
        prediction['day1'],
        prediction['day5'],
        prediction['day10']
    ]
    
    fig.add_trace(go.Scatter(
        x=periods,
        y=prices,
        mode='lines+markers',
        name='Prix prédit',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Prédictions de prix',
        xaxis_title='Horizon',
        yaxis_title='Prix (MAD)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal d'investissement
    signal = get_prediction_signal(prediction)
    if signal:
        signal_class = f"signal-{signal['signal'].lower().split()[0]}"
        st.markdown(f"""
        <div class="{signal_class}" style="padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3>🎯 Signal: {signal['signal']}</h3>
            <p><b>Rendement attendu (5 jours):</b> {signal['expected_return']:.2f}%</p>
            <p><b>Confiance du signal:</b> {signal['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

def get_prediction_signal(prediction):
    """Génère un signal d'achat/vente"""
    if not prediction:
        return None
    
    change_5d = ((prediction['day5'] - prediction['current_price']) / prediction['current_price']) * 100
    confidence = prediction['confidence']
    
    if change_5d > 3 and confidence > 0.7:
        signal = "ACHAT FORT"
    elif change_5d > 1 and confidence > 0.6:
        signal = "ACHAT"
    elif change_5d < -3 and confidence > 0.7:
        signal = "VENTE FORT"
    elif change_5d < -1 and confidence > 0.6:
        signal = "VENTE"
    else:
        signal = "NEUTRE"
    
    return {
        'signal': signal,
        'expected_return': change_5d,
        'confidence': confidence * 100
    }

def show_fallback_predictions(ticker):
    """Mode dégradé quand le module n'est pas disponible"""
    
    # Simulation de prédiction
    conn = get_db_connection()
    if conn:
        try:
            query = "SELECT close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1"
            result = pd.read_sql_query(query, conn, params=(ticker,))
            
            if not result.empty:
                current_price = result.iloc[0]['close']
                
                # Simulation aléatoire
                np.random.seed(hash(ticker) % 10000)
                
                pred_1d = current_price * (1 + np.random.uniform(-0.03, 0.03))
                pred_5d = current_price * (1 + np.random.uniform(-0.05, 0.08))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prix Actuel", f"{current_price:.2f} MAD")
                with col2:
                    change_1d = ((pred_1d - current_price) / current_price) * 100
                    st.metric("Simulation J+1", f"{pred_1d:.2f} MAD", f"{change_1d:+.2f}%")
                with col3:
                    change_5d = ((pred_5d - current_price) / current_price) * 100
                    st.metric("Simulation J+5", f"{pred_5d:.2f} MAD", f"{change_5d:+.2f}%")
        except:
            pass
        finally:
            conn.close()

# ============================================
# PRÉDICTIONS INTERACTIVES (NOUVEAU)
# ============================================

def show_interactive_predictions():
    """Page dédiée aux prédictions interactives"""
    st.markdown('<h2 class="sub-header">🎮 Prédictions Interactives</h2>', unsafe_allow_html=True)
    
    # Vérifier si le module est disponible
    try:
        from utils.predictions import interactive_prediction_demo
        
        # Exécuter la démo interactive
        interactive_prediction_demo()
        
    except ImportError as e:
        st.error(f"❌ Module non disponible: {e}")
        
        # Mode démonstration
        show_interactive_demo()

def show_interactive_demo():
    """Mode démonstration des prédictions interactives"""
    st.markdown("---")
    st.markdown("### 🔮 Mode Démonstration Interactive")
    
    # Interface simplifiée
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Paramètres")
        
        price = st.slider("Prix de clôture (MAD)", 50.0, 500.0, 100.0, 1.0)
        rsi = st.slider("RSI", 0.0, 100.0, 50.0, 1.0)
        volume = st.slider("Volume (milliers)", 100, 10000, 1000, 100)
    
    with col2:
        st.markdown("#### 📈 Indicateurs")
        
        sma_20 = st.number_input("SMA 20 jours", value=price * 0.98)
        bb_upper = st.number_input("Bollinger Upper", value=price * 1.02)
        bb_lower = st.number_input("Bollinger Lower", value=price * 0.98)
    
    if st.button("🚀 Simuler la prédiction", type="primary", use_container_width=True):
        # Simulation simple
        np.random.seed(int(price * 100))
        
        pred_1d = price * (1 + (rsi - 50) / 1000 + np.random.uniform(-0.02, 0.02))
        pred_5d = price * (1 + (rsi - 50) / 500 + np.random.uniform(-0.05, 0.08))
        
        st.markdown("---")
        st.markdown("### 📊 Résultats de Simulation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prix Saisi", f"{price:.2f} MAD")
        with col2:
            change_1d = ((pred_1d - price) / price) * 100
            st.metric("Simulation J+1", f"{pred_1d:.2f} MAD", f"{change_1d:+.2f}%")
        with col3:
            change_5d = ((pred_5d - price) / price) * 100
            st.metric("Simulation J+5", f"{pred_5d:.2f} MAD", f"{change_5d:+.2f}%")
        
        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=['Actuel', 'J+1', 'J+5'],
            y=[price, pred_1d, pred_5d],
            mode='lines+markers',
            line=dict(color='#1E88E5', width=3)
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_demo_predictions():
    """Affiche des prédictions de démonstration"""
    st.markdown("### 🔮 Mode Démonstration")
    
    instruments = ["ATW", "IAM", "BMCE", "LHM"]
    selected = st.selectbox("Instrument:", instruments)
    
    # Données de démonstration
    demo_data = {
        "ATW": {"price": 450.25, "trend": "📈"},
        "IAM": {"price": 128.75, "trend": "📉"},
        "BMCE": {"price": 89.30, "trend": "📈"},
        "LHM": {"price": 215.50, "trend": "📊"}
    }
    
    if selected in demo_data:
        data = demo_data[selected]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prix Actuel", f"{data['price']:.2f} MAD")
        with col2:
            st.metric("Prédiction J+1", f"{data['price'] * 1.02:.2f} MAD", "+2.0%")
        with col3:
            st.metric("Signal", data['trend'])

# ============================================
# ALERTES
# ============================================

def show_alerts():
    """Affiche les alertes dynamiques basées sur les données réelles"""
    st.markdown('<h2 class="sub-header">🚨 Alertes du Marché</h2>', unsafe_allow_html=True)
    
    # Récupérer les alertes dynamiques
    alerts = generate_dynamic_alerts()
    
    if not alerts:
        st.info("⚠️ Aucune alerte à afficher pour le moment")
        return
    
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
            <strong>{alert['ticker']} - {get_instrument_name(alert['ticker'])}</strong><br>
            <strong>{alert['type']}:</strong> {alert['message']}<br>
            <small>Priorité: {alert['priority']} | Date: {alert.get('date', 'N/A')}</small>
        </div>
        """, unsafe_allow_html=True)

def generate_dynamic_alerts():
    """Génère des alertes dynamiques basées sur les données réelles"""
    alerts = []
    
    # Liste de vos instruments
    your_instruments = {
        'ATW': 'ATTIJARIWAFA BANK',
        'BCI': 'BMCI',
        'BOA': 'BANK OF AFRICA',
        'CFG': 'CFG BANK',
        'CIH': 'CIH',
        'SAH': 'SANLAM MAROC',
        'WAA': 'WAFA ASSURANCE'
    }
    
    conn = get_db_connection()
    if conn is None:
        return alerts
    
    try:
        for ticker, name in your_instruments.items():
            # Récupérer les dernières données pour cet instrument
            query = """
            SELECT date, open, high, low, close, volume 
            FROM daily_prices 
            WHERE ticker = ? 
            ORDER BY date DESC 
            LIMIT 30
            """
            df = pd.read_sql_query(query, conn, params=(ticker,))
            
            if df.empty or len(df) < 10:
                continue
            
            # Convertir les dates
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculer les indicateurs
            latest = df.iloc[0]  # Dernière donnée
            prev_day = df.iloc[1] if len(df) > 1 else latest
            
            # Prix actuel et variation
            current_price = latest['close']
            prev_price = prev_day['close']
            daily_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
            
            # Volume moyen sur 20 jours
            avg_volume_20 = df['volume'].head(20).mean()
            
            # Prix moyen sur 20 jours
            avg_price_20 = df['close'].head(20).mean()
            
            # 1. Alerte de forte variation journalière
            if abs(daily_change) > 5:
                alerts.append({
                    'ticker': ticker,
                    'type': '📈 FORTE VARIATION' if daily_change > 0 else '📉 FORTE CHUTE',
                    'message': f'Variation de {daily_change:+.2f}% ({prev_price:.2f} → {current_price:.2f} MAD)',
                    'priority': 'HIGH',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
            
            # 2. Alerte de volume anormal
            volume_ratio = latest['volume'] / avg_volume_20 if avg_volume_20 > 0 else 1
            if volume_ratio > 2:
                alerts.append({
                    'ticker': ticker,
                    'type': '📊 VOLUME ÉLEVÉ',
                    'message': f'Volume {volume_ratio:.1f}x supérieur à la moyenne ({latest["volume"]:,.0f} vs {avg_volume_20:,.0f})',
                    'priority': 'MEDIUM',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
            elif volume_ratio < 0.3:
                alerts.append({
                    'ticker': ticker,
                    'type': '📉 VOLUME FAIBLE',
                    'message': f'Volume {volume_ratio:.1f}x inférieur à la moyenne ({latest["volume"]:,.0f} vs {avg_volume_20:,.0f})',
                    'priority': 'LOW',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
            
            # 3. Alerte de franchissement de moyenne mobile
            if current_price > avg_price_20 * 1.05:
                alerts.append({
                    'ticker': ticker,
                    'type': '🚀 AU-DESSUS DE LA MM20',
                    'message': f'Prix {current_price:.2f}MAD > MM20 {avg_price_20:.2f}MAD (+{(current_price/avg_price_20-1)*100:.1f}%)',
                    'priority': 'MEDIUM',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
            elif current_price < avg_price_20 * 0.95:
                alerts.append({
                    'ticker': ticker,
                    'type': '📉 EN-DESSOUS DE LA MM20',
                    'message': f'Prix {current_price:.2f}MAD < MM20 {avg_price_20:.2f}MAD (-{(1-current_price/avg_price_20)*100:.1f}%)',
                    'priority': 'MEDIUM',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
            
            # 4. Alerte de tendance
            if len(df) >= 5:
                # Variation sur 5 jours
                price_5d_ago = df.iloc[4]['close'] if len(df) > 4 else current_price
                change_5d = ((current_price - price_5d_ago) / price_5d_ago * 100) if price_5d_ago > 0 else 0
                
                if change_5d > 8:
                    alerts.append({
                        'ticker': ticker,
                        'type': '📈 FORTE HAUSSE (5j)',
                        'message': f'Hausse de {change_5d:+.1f}% sur 5 jours',
                        'priority': 'HIGH',
                        'date': latest['date'].strftime('%Y-%m-%d')
                    })
                elif change_5d < -8:
                    alerts.append({
                        'ticker': ticker,
                        'type': '📉 FORTE BAISSE (5j)',
                        'message': f'Baisse de {change_5d:+.1f}% sur 5 jours',
                        'priority': 'HIGH',
                        'date': latest['date'].strftime('%Y-%m-%d')
                    })
            
            # 5. Alerte de volatilité (écart entre high et low)
            daily_range = (latest['high'] - latest['low']) / latest['close'] * 100 if latest['close'] > 0 else 0
            if daily_range > 6:
                alerts.append({
                    'ticker': ticker,
                    'type': '⚡ VOLATILITÉ ÉLEVÉE',
                    'message': f'Fourchette journalière de {daily_range:.1f}% ({latest["low"]:.2f}-{latest["high"]:.2f} MAD)',
                    'priority': 'MEDIUM',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
            
            # 6. Alerte de support/résistance (prix proche du high ou low du mois)
            monthly_high = df['high'].head(20).max()
            monthly_low = df['low'].head(20).min()
            
            if current_price >= monthly_high * 0.98:
                alerts.append({
                    'ticker': ticker,
                    'type': '🎯 PROCHE DU HIGH MENSUEL',
                    'message': f'Prix {current_price:.2f}MAD proche du high mensuel {monthly_high:.2f}MAD',
                    'priority': 'MEDIUM',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
            elif current_price <= monthly_low * 1.02:
                alerts.append({
                    'ticker': ticker,
                    'type': '🎯 PROCHE DU LOW MENSUEL',
                    'message': f'Prix {current_price:.2f}MAD proche du low mensuel {monthly_low:.2f}MAD',
                    'priority': 'MEDIUM',
                    'date': latest['date'].strftime('%Y-%m-%d')
                })
        
        # Trier les alertes par priorité (HIGH d'abord) puis par date
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        alerts.sort(key=lambda x: (priority_order[x['priority']], x.get('date', ''), x['ticker']))
        
        return alerts
        
    except Exception as e:
        st.error(f"Erreur lors de la génération des alertes: {e}")
        # Retourner des alertes de démonstration en cas d'erreur
        return generate_fallback_alerts(your_instruments)
    finally:
        conn.close()

def generate_fallback_alerts(instruments):
    """Génère des alertes de démonstration si la base de données n'est pas disponible"""
    alerts = []
    
    import random
    from datetime import datetime, timedelta
    
    alert_types = [
        ("📈 FORTE HAUSSE", "HIGH"),
        ("📉 FORTE BAISSE", "HIGH"),
        ("📊 VOLUME ÉLEVÉ", "MEDIUM"),
        ("🎯 PROCHE DU HIGH", "MEDIUM"),
        ("⚡ VOLATILITÉ", "MEDIUM"),
        ("📉 VOLUME FAIBLE", "LOW")
    ]
    
    for ticker, name in instruments.items():
        # Générer 1-2 alertes aléatoires par instrument
        num_alerts = random.randint(1, 2)
        for _ in range(num_alerts):
            alert_type, priority = random.choice(alert_types)
            
            # Générer un message approprié
            if "HAUSSE" in alert_type:
                change = random.uniform(5.0, 10.0)
                message = f"Hausse de {change:.1f}% sur la journée"
            elif "BAISSE" in alert_type:
                change = random.uniform(-10.0, -5.0)
                message = f"Baisse de {change:.1f}% sur la journée"
            elif "VOLUME ÉLEVÉ" in alert_type:
                multiplier = random.uniform(2.0, 4.0)
                message = f"Volume {multiplier:.1f}x supérieur à la moyenne"
            elif "VOLUME FAIBLE" in alert_type:
                multiplier = random.uniform(0.1, 0.3)
                message = f"Volume {multiplier:.1f}x inférieur à la moyenne"
            elif "HIGH" in alert_type:
                price = random.uniform(100.0, 500.0)
                high = price * random.uniform(1.02, 1.05)
                message = f"Prix {price:.2f}MAD proche du high {high:.2f}MAD"
            else:  # VOLATILITÉ
                volatility = random.uniform(6.0, 12.0)
                message = f"Fourchette journalière de {volatility:.1f}%"
            
            alerts.append({
                'ticker': ticker,
                'type': alert_type,
                'message': message,
                'priority': priority,
                'date': (datetime.now() - timedelta(days=random.randint(0, 2))).strftime('%Y-%m-%d')
            })
    
    return alerts

# ============================================
# COMPARAISON
# ============================================

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
        max_selections=5,
        format_func=lambda x: f"{x} - {get_instrument_name(x)}"
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
                name=f"{ticker} - {get_instrument_name(ticker)}"
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
                    'Nom': get_instrument_name(ticker),
                    'Prix Initial': f"{first['close']:.2f}",
                    'Prix Actuel': f"{latest['close']:.2f}",
                    'Variation': f"{change_pct:+.2f}%"
                })
        
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la comparaison: {e}")
    finally:
        conn.close()

# ============================================
# RECHERCHE
# ============================================

def show_search():
    """Affiche la recherche"""
    st.markdown('<h2 class="sub-header">⚙️ Recherche d\'Instruments</h2>', unsafe_allow_html=True)
    
    # Barre de recherche
    search_term = st.text_input("🔍 Rechercher un instrument (ticker ou nom):", "")
    
    # Récupérer tous les instruments avec noms
    instruments_df = load_instruments()
    if not instruments_df.empty:
        instruments_df['name'] = instruments_df['ticker'].apply(get_instrument_name)
    
    if instruments_df.empty:
        st.warning("Aucun instrument disponible")
        return
    
    # Recherche
    if search_term:
        search_lower = search_term.lower()
        mask = (
            instruments_df['ticker'].str.lower().str.contains(search_lower) |
            instruments_df['name'].str.lower().str.contains(search_lower)
        )
        results = instruments_df[mask]
        
        st.markdown(f"### 📊 Résultats ({len(results)} trouvés)")
        
        if not results.empty:
            display_df = results.copy()
            display_df.columns = ['Ticker', 'Nom']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"Aucun instrument trouvé pour '{search_term}'")
    
    # Tous les instruments
    st.markdown("### 📋 Tous les Instruments")
    
    # Pagination
    items_per_page = 10
    total_items = len(instruments_df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    if total_pages > 1:
        page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        st.write(f"Affichage des instruments {start_idx + 1} à {end_idx} sur {total_items}")
        
        # Afficher la page courante
        current_page_df = instruments_df.iloc[start_idx:end_idx].copy()
        current_page_df.columns = ['Ticker', 'Nom']
        st.dataframe(current_page_df, use_container_width=True, hide_index=True)
        
        # Navigation de page
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page > 1 and st.button("← Précédent"):
                pass  # La page se met à jour via le number_input
        with col3:
            if page < total_pages and st.button("Suivant →"):
                pass  # La page se met à jour via le number_input
    else:
        display_df = instruments_df.copy()
        display_df.columns = ['Ticker', 'Nom']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    main()