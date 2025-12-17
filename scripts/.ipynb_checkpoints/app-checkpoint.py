# app.py
import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np
from datetime import datetime

st.set_page_config(page_title="MarketSense Morocco", layout="wide")
st.title("MarketSense Morocco")
st.subheader("Intelligence Boursière Augmentée par IA")

# --- Connexion à la base SQLite ---
conn = sqlite3.connect('database/market_data.db')

# --- Charger les instruments ---
df_instruments = pd.read_sql("SELECT ticker, name FROM instruments", conn)

# --- Barre latérale pour sélectionner l'instrument ---
st.sidebar.title("Sélection de l'instrument")
ticker = st.sidebar.selectbox("Choisir un instrument", df_instruments['ticker'].tolist())
name = df_instruments[df_instruments['ticker'] == ticker]['name'].values[0]

# --- Charger les données historiques et indicateurs ---
query = f"""
    SELECT dp.*, ti.SMA_7, ti.SMA_21, ti.SMA_50, ti.EMA_12, ti.EMA_26,
           ti.RSI, ti.MACD, ti.ATR, ti.Volume_MA_20, ti.Volume_Ratio,
           ti.OBV, ti.Daily_Return, ti.Volatility_20d
    FROM daily_prices dp
    LEFT JOIN technical_indicators ti
    ON dp.ticker = ti.ticker AND dp.date = ti.date
    WHERE dp.ticker = '{ticker}'
    ORDER BY dp.date ASC
"""
df_prices = pd.read_sql(query, conn)

st.subheader(f"{name} - Données Historiques")
st.dataframe(df_prices.tail(5))

# --- Charger le modèle ML ---
model_path = f"models/{ticker}_lr_model.pkl"  # adapter le nom selon ton modèle
try:
    model = joblib.load(model_path)
    st.success(f"✅ Modèle ML chargé pour {name}")
except FileNotFoundError:
    st.error(f"❌ Modèle ML non trouvé pour {name}")
    model = None

# --- Liste des features attendues par le modèle ---
features = [
    'SMA_7','SMA_21','SMA_50','EMA_12','EMA_26','RSI','MACD',
    'ATR','Volume_MA_20','Volume_Ratio','OBV','Daily_Return','Volatility_20d',
    'Open','High','Low','Close','Volume','Turnover'
]

# --- Remplir les colonnes manquantes ---
for col in features:
    if col not in df_prices.columns:
        df_prices[col] = 0.0  # ou np.nan et faire un fillna plus tard

# --- Faire une prédiction ML ---
if model is not None:
    try:
        X_latest = df_prices[features].tail(1)
        pred_price = model.predict(X_latest)[0]
        st.subheader("Prédiction ML")
        st.write(f"🤖 Prix prédit dans 5 jours : {pred_price:.2f} MAD")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# --- Affichage des indicateurs techniques ---
st.subheader("Indicateurs Techniques (dernières 5 lignes)")
indicators_to_show = ['date','RSI','MACD','SMA_50','EMA_12','EMA_26']
available_cols = [c for c in indicators_to_show if c in df_prices.columns]
st.dataframe(df_prices[available_cols].tail(5))

# --- Alertes simples ---
st.subheader("Alertes")
alerts = []

# RSI faible
if 'RSI' in df_prices.columns and df_prices['RSI'].iloc[-1] < 30:
    alerts.append(f"RSI < 30 - Opportunité achat pour {name}")

# MACD haussier
if 'MACD' in df_prices.columns and df_prices['MACD'].iloc[-1] > 0:
    alerts.append(f"Tendance haussière détectée pour {name}")

# Volume inhabituel
if 'Volume' in df_prices.columns:
    avg_volume = df_prices['Volume'].mean()
    if df_prices['Volume'].iloc[-1] > 1.5 * avg_volume:
        alerts.append(f"Volume inhabituel détecté pour {name}")

# Affichage des alertes
if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("Aucune alerte active pour le moment.")

# --- Fermer la connexion ---
conn.close()
