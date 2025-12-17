"""
Système de prédictions ML avec interface utilisateur interactive
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data(ttl=3600)
def load_model(ticker):
    """
    Charge le modèle ML pour un instrument
    Args:
        ticker (str): Code de l'instrument
    Returns:
        dict: Données du modèle ou None
    """
    model_path = f'models/{ticker}_model.joblib'
    
    if not os.path.exists(model_path):
        st.warning(f"Modèle non trouvé pour {ticker}. Créez d'abord le modèle avec train_models.py")
        return None
    
    try:
        model_data = joblib.load(model_path)
        st.success(f"✅ Modèle chargé pour {ticker}")
        return model_data
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def get_user_input_features():
    """
    Interface utilisateur pour saisir les features manuellement
    Returns:
        dict: Features saisies par l'utilisateur
    """
    st.markdown("### 🔧 Paramètres d'entrée pour la prédiction")
    
    with st.expander("📊 Indicateurs de prix", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            close_price = st.number_input(
                "Prix de clôture (MAD)",
                min_value=0.0,
                value=100.0,
                step=0.1,
                help="Dernier cours de l'instrument"
            )
            open_price = st.number_input(
                "Prix d'ouverture (MAD)",
                min_value=0.0,
                value=99.5,
                step=0.1
            )
            high_price = st.number_input(
                "Plus haut (MAD)",
                min_value=0.0,
                value=101.0,
                step=0.1
            )
        with col2:
            low_price = st.number_input(
                "Plus bas (MAD)",
                min_value=0.0,
                value=98.0,
                step=0.1
            )
            volume = st.number_input(
                "Volume d'échanges",
                min_value=0,
                value=1000000,
                step=1000,
                help="Nombre de titres échangés"
            )
            market_cap = st.number_input(
                "Capitalisation (MAD)",
                min_value=0.0,
                value=1000000000.0,
                step=1000000.0
            )
    
    with st.expander("📈 Indicateurs techniques", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            rsi = st.slider(
                "RSI (Relative Strength Index)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.5,
                help="Indicateur de momentum (30=survente, 70=surachat)"
            )
            macd = st.number_input(
                "MACD",
                value=0.0,
                step=0.01,
                help="Différence entre EMA12 et EMA26"
            )
            macd_signal = st.number_input(
                "Signal MACD",
                value=0.0,
                step=0.01
            )
        with col2:
            bb_upper = st.number_input(
                "Bande de Bollinger Supérieure",
                min_value=0.0,
                value=close_price * 1.02,
                step=0.1
            )
            bb_lower = st.number_input(
                "Bande de Bollinger Inférieure",
                min_value=0.0,
                value=close_price * 0.98,
                step=0.1
            )
            atr = st.number_input(
                "ATR (Average True Range)",
                min_value=0.0,
                value=close_price * 0.02,
                step=0.01,
                help="Indicateur de volatilité"
            )
    
    with st.expander("📊 Moyennes mobiles"):
        col1, col2 = st.columns(2)
        with col1:
            sma_7 = st.number_input(
                "SMA 7 jours",
                min_value=0.0,
                value=close_price,
                step=0.1
            )
            sma_21 = st.number_input(
                "SMA 21 jours",
                min_value=0.0,
                value=close_price,
                step=0.1
            )
        with col2:
            sma_50 = st.number_input(
                "SMA 50 jours",
                min_value=0.0,
                value=close_price,
                step=0.1
            )
            ema_12 = st.number_input(
                "EMA 12 jours",
                min_value=0.0,
                value=close_price,
                step=0.1
            )
            ema_26 = st.number_input(
                "EMA 26 jours",
                min_value=0.0,
                value=close_price,
                step=0.1
            )
    
    with st.expander("📈 Autres indicateurs"):
        volume_ratio = st.slider(
            "Ratio Volume / Moyenne",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Volume actuel / moyenne mobile volume"
        )
        daily_return = st.number_input(
            "Rendement quotidien (%)",
            value=0.0,
            step=0.1
        )
        volatility_20d = st.slider(
            "Volatilité 20 jours (%)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1
        )
    
    # Calcul automatique de certains indicateurs
    bb_width = ((bb_upper - bb_lower) / close_price) * 100 if close_price > 0 else 0
    
    features = {
        'close': close_price,
        'volume': volume,
        'sma_7': sma_7,
        'sma_21': sma_21,
        'sma_50': sma_50,
        'ema_12': ema_12,
        'ema_26': ema_26,
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd_signal,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_width': bb_width,
        'atr': atr,
        'volume_ratio': volume_ratio,
        'daily_return': daily_return,
        'volatility_20d': volatility_20d,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'market_cap': market_cap
    }
    
    # Aperçu des features
    st.markdown("### 📋 Aperçu des paramètres saisis")
    preview_df = pd.DataFrame([features])
    st.dataframe(preview_df.T.rename(columns={0: 'Valeur'}), use_container_width=True)
    
    return features

def prepare_features_from_user_input(user_features):
    """
    Prépare les features pour la prédiction à partir des inputs utilisateur
    Args:
        user_features (dict): Features saisies par l'utilisateur
    Returns:
        np.array: Features préparées pour le modèle
    """
    features = [
        user_features.get('close', 0),
        user_features.get('volume', 0),
        user_features.get('sma_7', user_features.get('close', 0)),
        user_features.get('sma_21', user_features.get('close', 0)),
        user_features.get('sma_50', user_features.get('close', 0)),
        user_features.get('ema_12', user_features.get('close', 0)),
        user_features.get('ema_26', user_features.get('close', 0)),
        user_features.get('rsi', 50),
        user_features.get('macd', 0),
        user_features.get('macd_signal', 0),
        user_features.get('bb_upper', user_features.get('close', 0) * 1.02),
        user_features.get('bb_lower', user_features.get('close', 0) * 0.98),
        user_features.get('bb_width', 0.02),
        user_features.get('atr', user_features.get('close', 0) * 0.02),
        user_features.get('volume_ratio', 1),
        user_features.get('daily_return', 0),
        user_features.get('volatility_20d', 2)
    ]
    
    # Remplacer les NaN par des valeurs par défaut
    features = [0 if pd.isna(f) else f for f in features]
    
    return np.array([features])

def predict_from_user_input(ticker, user_features):
    """
    Fait une prédiction à partir des inputs utilisateur
    Args:
        ticker (str): Code de l'instrument
        user_features (dict): Features saisies par l'utilisateur
    Returns:
        dict: Prédictions
    """
    model_data = load_model(ticker)
    
    if model_data is None:
        return None
    
    try:
        # Préparer les features
        features = prepare_features_from_user_input(user_features)
        
        # Normaliser
        features_scaled = model_data['price_scaler'].transform(features)
        
        # Prédiction de prix
        pred_price_1d = model_data['price_model_1d'].predict(features_scaled)[0]
        
        # Calculer la tendance
        current_price = float(user_features['close'])
        trend = (pred_price_1d - current_price) / current_price
        
        # Prédictions à 5 et 10 jours
        pred_price_5d = pred_price_1d * (1 + trend * 3)
        pred_price_10d = pred_price_1d * (1 + trend * 6)
        
        # Prédiction de direction
        direction_proba = model_data['direction_model_1d'].predict_proba(features_scaled)[0]
        confidence = float(max(direction_proba))
        direction = "HAUSSE" if direction_proba[1] > direction_proba[0] else "BAISSE"
        
        # Informations sur le modèle
        model_info = {
            'trained_at': model_data.get('trained_at', 'N/A'),
            'version': model_data.get('version', '1.0'),
            'features_used': model_data.get('features', [])
        }
        
        return {
            'current_price': current_price,
            'day1': float(pred_price_1d),
            'day5': float(pred_price_5d),
            'day10': float(pred_price_10d),
            'direction': direction,
            'confidence': confidence,
            'direction_proba': {
                'baisse': float(direction_proba[0]),
                'hausse': float(direction_proba[1])
            },
            'model_info': model_info,
            'features_used': user_features
        }
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None

def create_prediction_visualization(prediction, user_features):
    """
    Crée une visualisation des prédictions
    Args:
        prediction (dict): Résultats de prédiction
        user_features (dict): Features utilisateur
    Returns:
        plotly.graph_objects.Figure: Graphique
    """
    # Données pour le graphique
    periods = ['Actuel', 'J+1', 'J+5', 'J+10']
    prices = [
        prediction['current_price'],
        prediction['day1'],
        prediction['day5'],
        prediction['day10']
    ]
    
    # Calcul des variations
    variations = [
        0,
        ((prediction['day1'] - prediction['current_price']) / prediction['current_price']) * 100,
        ((prediction['day5'] - prediction['current_price']) / prediction['current_price']) * 100,
        ((prediction['day10'] - prediction['current_price']) / prediction['current_price']) * 100
    ]
    
    # Création du graphique
    fig = go.Figure()
    
    # Ligne des prix
    fig.add_trace(go.Scatter(
        x=periods,
        y=prices,
        mode='lines+markers+text',
        name='Prix prédit',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=10),
        text=[f'{p:.2f} MAD' for p in prices],
        textposition='top center'
    ))
    
    # Annotations des variations
    for i, (period, price, var) in enumerate(zip(periods, prices, variations)):
        if i > 0:  # Ne pas annoter le point actuel
            color = 'green' if var > 0 else 'red'
            fig.add_annotation(
                x=period,
                y=price,
                text=f'{var:+.1f}%',
                showarrow=False,
                yshift=20,
                font=dict(color=color, size=12, weight='bold')
            )
    
    # Mise en forme
    fig.update_layout(
        title='📈 Prédictions de prix',
        xaxis_title='Horizon temporel',
        yaxis_title='Prix (MAD)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_direction_probability_chart(prediction):
    """
    Crée un graphique des probabilités de direction
    Args:
        prediction (dict): Résultats de prédiction
    Returns:
        plotly.graph_objects.Figure: Graphique en camembert
    """
    labels = ['Baisse', 'Hausse']
    values = [
        prediction['direction_proba']['baisse'] * 100,
        prediction['direction_proba']['hausse'] * 100
    ]
    colors = ['#EF5350', '#4CAF50']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=colors,
        textinfo='label+percent',
        hoverinfo='label+value',
        textposition='inside'
    )])
    
    fig.update_layout(
        title='🎯 Probabilités de direction',
        height=400,
        showlegend=True,
        annotations=[
            dict(
                text=f"Confiance: {prediction['confidence']*100:.1f}%",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )
        ]
    )
    
    return fig

def display_prediction_results(prediction):
    """
    Affiche les résultats de prédiction de manière interactive
    Args:
        prediction (dict): Résultats de prédiction
    """
    if not prediction:
        st.error("❌ Aucune prédiction disponible")
        return
    
    # Métriques principales
    st.markdown("### 📊 Résultats de la prédiction")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prix Actuel",
            f"{prediction['current_price']:.2f} MAD"
        )
    
    with col2:
        change_1d = ((prediction['day1'] - prediction['current_price']) / 
                    prediction['current_price'] * 100)
        st.metric(
            "Prédiction J+1",
            f"{prediction['day1']:.2f} MAD",
            f"{change_1d:+.2f}%"
        )
    
    with col3:
        change_5d = ((prediction['day5'] - prediction['current_price']) / 
                    prediction['current_price'] * 100)
        st.metric(
            "Prédiction J+5",
            f"{prediction['day5']:.2f} MAD",
            f"{change_5d:+.2f}%"
        )
    
    with col4:
        st.metric(
            "Direction",
            prediction['direction'],
            f"Confiance: {prediction['confidence']*100:.1f}%"
        )
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price = create_prediction_visualization(prediction, prediction.get('features_used', {}))
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_direction = create_direction_probability_chart(prediction)
        st.plotly_chart(fig_direction, use_container_width=True)
    
    # Signal d'achat/vente
    signal = get_prediction_signal(prediction)
    if signal:
        st.markdown(f"""
        <div style="background-color: {signal['color']}20; padding: 20px; 
                    border-radius: 10px; border-left: 5px solid {signal['color']}; 
                    margin: 10px 0;">
            <h3>🎯 Signal: {signal['signal']}</h3>
            <p><b>Rendement attendu (5 jours):</b> {signal['expected_return']:.2f}%</p>
            <p><b>Confiance du signal:</b> {signal['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Informations détaillées
    with st.expander("📋 Détails techniques"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Probabilités de direction")
            prob_df = pd.DataFrame({
                'Direction': ['Baisse', 'Hausse'],
                'Probabilité (%)': [
                    prediction['direction_proba']['baisse'] * 100,
                    prediction['direction_proba']['hausse'] * 100
                ]
            })
            st.dataframe(prob_df, use_container_width=True)
        
        with col2:
            st.markdown("#### Informations du modèle")
            if 'model_info' in prediction:
                st.write(f"**Version:** {prediction['model_info'].get('version', 'N/A')}")
                st.write(f"**Entraîné le:** {prediction['model_info'].get('trained_at', 'N/A')}")
                if 'features_used' in prediction['model_info']:
                    st.write(f"**Features utilisées:** {len(prediction['model_info']['features_used'])}")

def interactive_prediction_demo():
    """
    Démo interactive complète de prédiction
    """
    st.title("🤖 Démo Interactive de Prédiction ML")
    st.markdown("""
    Cette interface vous permet de saisir manuellement les paramètres d'un instrument
    et d'obtenir des prédictions basées sur nos modèles Machine Learning.
    """)
    
    # Sélection du ticker
    st.markdown("### 1. Sélection de l'instrument")
    
    # Liste des modèles disponibles
    models_dir = 'models'
    if os.path.exists(models_dir):
        available_models = [f.split('_model.joblib')[0] for f in os.listdir(models_dir) 
                          if f.endswith('_model.joblib')]
    else:
        available_models = []
    
    if not available_models:
        st.warning("⚠️ Aucun modèle ML n'est disponible. Exécutez d'abord train_models.py")
        return
    
    selected_ticker = st.selectbox(
        "Sélectionnez un instrument:",
        available_models,
        help="Choisissez un instrument pour lequel un modèle ML existe"
    )
    
    # Charger les données historiques pour référence
    try:
        from .database import load_instrument_detail
        historical_data = load_instrument_detail(selected_ticker, days=5)
        if not historical_data.empty:
            st.info(f"📊 Données historiques disponibles pour {selected_ticker}")
            st.dataframe(historical_data[['date', 'open', 'high', 'low', 'close', 'volume']].head(), 
                        use_container_width=True)
    except:
        pass
    
    # Options d'entrée
    st.markdown("### 2. Mode d'entrée des données")
    
    input_mode = st.radio(
        "Comment souhaitez-vous entrer les données?",
        ["📝 Saisie manuelle", "📊 Utiliser les dernières données"]
    )
    
    if input_mode == "📝 Saisie manuelle":
        user_features = get_user_input_features()
    else:
        # Charger les dernières données
        try:
            from .database import load_instrument_detail
            latest_data = load_instrument_detail(selected_ticker, days=1)
            if not latest_data.empty:
                latest_row = latest_data.iloc[0]
                user_features = {
                    'close': latest_row.get('close', 0),
                    'volume': latest_row.get('volume', 0),
                    'sma_7': latest_row.get('sma_7', latest_row.get('close', 0)),
                    'sma_21': latest_row.get('sma_21', latest_row.get('close', 0)),
                    'sma_50': latest_row.get('sma_50', latest_row.get('close', 0)),
                    'ema_12': latest_row.get('ema_12', latest_row.get('close', 0)),
                    'ema_26': latest_row.get('ema_26', latest_row.get('close', 0)),
                    'rsi': latest_row.get('rsi', 50),
                    'macd': latest_row.get('macd', 0),
                    'macd_signal': latest_row.get('macd_signal', 0),
                    'bb_upper': latest_row.get('bb_upper', latest_row.get('close', 0) * 1.02),
                    'bb_lower': latest_row.get('bb_lower', latest_row.get('close', 0) * 0.98),
                    'bb_width': latest_row.get('bb_width', 0.02),
                    'atr': latest_row.get('atr', latest_row.get('close', 0) * 0.02),
                    'volume_ratio': latest_row.get('volume_ratio', 1),
                    'daily_return': latest_row.get('daily_return', 0),
                    'volatility_20d': latest_row.get('volatility_20d', 2)
                }
                st.success(f"✅ Données chargées pour {selected_ticker}")
            else:
                st.warning("Aucune donnée disponible, basculez vers la saisie manuelle")
                user_features = get_user_input_features()
        except:
            st.warning("Impossible de charger les données, basculez vers la saisie manuelle")
            user_features = get_user_input_features()
    
    # Bouton de prédiction
    st.markdown("### 3. Lancez la prédiction")
    
    if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
        with st.spinner("🤖 Calcul des prédictions en cours..."):
            prediction = predict_from_user_input(selected_ticker, user_features)
            
            if prediction:
                display_prediction_results(prediction)
                
                # Option de sauvegarde
                if st.button("💾 Sauvegarder cette prédiction"):
                    save_prediction_to_history(selected_ticker, prediction, user_features)
                    st.success("Prédiction sauvegardée!")
            else:
                st.error("❌ Impossible de générer une prédiction")

def save_prediction_to_history(ticker, prediction, features):
    """
    Sauvegarde une prédiction dans l'historique
    Args:
        ticker (str): Code de l'instrument
        prediction (dict): Prédiction
        features (dict): Features utilisées
    """
    history_file = 'data/prediction_history.csv'
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    # Créer l'entrée
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'current_price': prediction['current_price'],
        'prediction_1d': prediction['day1'],
        'prediction_5d': prediction['day5'],
        'prediction_10d': prediction['day10'],
        'direction': prediction['direction'],
        'confidence': prediction['confidence'],
        'features': str(features)
    }
    
    # Sauvegarder
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, pd.DataFrame([entry])], ignore_index=True)
    else:
        history_df = pd.DataFrame([entry])
    
    history_df.to_csv(history_file, index=False)

# Fonctions existantes (conservées pour compatibilité)
def prepare_features(latest_data):
    """
    Prépare les features pour la prédiction (version historique)
    """
    features = [
        latest_data.get('close', 0),
        latest_data.get('volume', 0),
        latest_data.get('sma_7', latest_data.get('close', 0)),
        latest_data.get('sma_21', latest_data.get('close', 0)),
        latest_data.get('sma_50', latest_data.get('close', 0)),
        latest_data.get('ema_12', latest_data.get('close', 0)),
        latest_data.get('ema_26', latest_data.get('close', 0)),
        latest_data.get('rsi', 50),
        latest_data.get('macd', 0),
        latest_data.get('macd_signal', 0),
        latest_data.get('bb_upper', latest_data.get('close', 0) * 1.02),
        latest_data.get('bb_lower', latest_data.get('close', 0) * 0.98),
        latest_data.get('bb_width', 0.02),
        latest_data.get('atr', latest_data.get('close', 0) * 0.02),
        latest_data.get('volume_ratio', 1),
        latest_data.get('daily_return', 0),
        latest_data.get('volatility_20d', 2)
    ]
    
    features = [0 if pd.isna(f) else f for f in features]
    
    return np.array([features])

def predict_instrument(ticker, df_detail):
    """
    Fait une prédiction pour un instrument (version historique)
    """
    if df_detail.empty:
        return None
    
    model_data = load_model(ticker)
    
    if model_data is None:
        return None
    
    try:
        latest = df_detail.iloc[0]
        
        features = prepare_features(latest)
        features_scaled = model_data['price_scaler'].transform(features)
        
        pred_price_1d = model_data['price_model_1d'].predict(features_scaled)[0]
        
        current_price = float(latest['close'])
        trend = (pred_price_1d - current_price) / current_price
        
        pred_price_5d = pred_price_1d * (1 + trend * 3)
        pred_price_10d = pred_price_1d * (1 + trend * 6)
        
        direction_proba = model_data['direction_model_1d'].predict_proba(features_scaled)[0]
        confidence = float(max(direction_proba))
        direction = "HAUSSE" if direction_proba[1] > direction_proba[0] else "BAISSE"
        
        return {
            'current_price': current_price,
            'day1': float(pred_price_1d),
            'day5': float(pred_price_5d),
            'day10': float(pred_price_10d),
            'direction': direction,
            'confidence': confidence,
            'direction_proba': {
                'baisse': float(direction_proba[0]),
                'hausse': float(direction_proba[1])
            },
            'model_version': model_data.get('version', '1.0'),
            'trained_at': model_data.get('trained_at')
        }
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None

def load_predictions(ticker):
    """
    Charge les prédictions pour un instrument (fonction de compatibilité)
    """
    from .database import load_instrument_detail
    
    df_detail = load_instrument_detail(ticker, days=1)
    
    if df_detail.empty:
        return None
    
    return predict_instrument(ticker, df_detail)

def get_prediction_accuracy(ticker, days=30):
    """
    Calcule la précision des prédictions passées
    """
    from .database import load_instrument_detail
    
    df = load_instrument_detail(ticker, days=days+10)
    
    if len(df) < days + 5:
        return None
    
    model_data = load_model(ticker)
    
    if model_data is None:
        return None
    
    try:
        predictions = []
        actuals = []
        
        df = df.iloc[::-1].reset_index(drop=True)
        
        for i in range(len(df) - 5):
            current = df.iloc[i]
            actual_next = df.iloc[i + 1]['close']
            
            features = prepare_features(current)
            features_scaled = model_data['price_scaler'].transform(features)
            
            pred = model_data['price_model_1d'].predict(features_scaled)[0]
            
            predictions.append(pred)
            actuals.append(actual_next)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        pred_direction = predictions > actuals[:-1] if len(predictions) > 1 else []
        actual_direction = actuals[1:] > actuals[:-1] if len(actuals) > 1 else []
        
        if len(pred_direction) > 0 and len(actual_direction) > 0:
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
        else:
            directional_accuracy = 0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy),
            'num_predictions': len(predictions)
        }
    
    except Exception as e:
        st.error(f"Erreur lors du calcul de précision: {e}")
        return None

def compare_predictions(tickers):
    """
    Compare les prédictions de plusieurs instruments
    """
    results = []
    
    for ticker in tickers:
        pred = load_predictions(ticker)
        
        if pred:
            change_1d = ((pred['day1'] - pred['current_price']) / pred['current_price']) * 100
            change_5d = ((pred['day5'] - pred['current_price']) / pred['current_price']) * 100
            
            results.append({
                'Ticker': ticker,
                'Prix Actuel': f"{pred['current_price']:.2f}",
                'Préd. J+1': f"{pred['day1']:.2f}",
                'Var. J+1': f"{change_1d:+.2f}%",
                'Préd. J+5': f"{pred['day5']:.2f}",
                'Var. J+5': f"{change_5d:+.2f}%",
                'Direction': pred['direction'],
                'Confiance': f"{pred['confidence']*100:.1f}%"
            })
    
    return pd.DataFrame(results)

def get_prediction_signal(prediction):
    """
    Génère un signal d'achat/vente basé sur la prédiction
    """
    if not prediction:
        return None
    
    change_5d = ((prediction['day5'] - prediction['current_price']) / prediction['current_price']) * 100
    confidence = prediction['confidence']
    
    if change_5d > 3 and confidence > 0.7:
        signal = "ACHAT FORT"
        color = "green"
    elif change_5d > 1 and confidence > 0.6:
        signal = "ACHAT"
        color = "lightgreen"
    elif change_5d < -3 and confidence > 0.7:
        signal = "VENTE FORT"
        color = "red"
    elif change_5d < -1 and confidence > 0.6:
        signal = "VENTE"
        color = "orange"
    else:
        signal = "NEUTRE"
        color = "gray"
    
    return {
        'signal': signal,
        'color': color,
        'expected_return': change_5d,
        'confidence': confidence * 100
    }