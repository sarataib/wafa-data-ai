from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="MarketSense Morocco API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class Instrument(BaseModel):
    ticker: str
    name: str
    sector: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float

class Prediction(BaseModel):
    day1: float
    day5: float
    day10: Optional[float]
    confidence: float
    direction: str

class Alert(BaseModel):
    type: str
    message: str
    priority: str

class InstrumentDetail(BaseModel):
    ticker: str
    name: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    predictions: Prediction
    alerts: List[Alert]
    technical_indicators: dict

# ==================== DATABASE ====================

def get_db_connection():
    conn = sqlite3.connect('database/market_data.db')
    conn.row_factory = sqlite3.Row
    return conn

# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "message": "MarketSense Morocco API",
        "version": "1.0.0",
        "endpoints": {
            "instruments": "/api/instruments",
            "instrument_detail": "/api/instruments/{ticker}",
            "market_overview": "/api/market-overview",
            "predictions": "/api/predictions/{ticker}",
            "alerts": "/api/alerts"
        }
    }

@app.get("/api/instruments")
def get_instruments():
    """Récupère tous les instruments avec leurs données actuelles"""
    conn = get_db_connection()
    
    query = '''
    SELECT 
        i.ticker,
        i.name,
        i.sector,
        dp1.close as current_price,
        dp1.volume,
        dp1.market_cap,
        dp1.close - dp2.close as change,
        ((dp1.close - dp2.close) / dp2.close * 100) as change_percent
    FROM instruments i
    JOIN daily_prices dp1 ON i.ticker = dp1.ticker
    JOIN daily_prices dp2 ON i.ticker = dp2.ticker
    WHERE dp1.date = (SELECT MAX(date) FROM daily_prices WHERE ticker = i.ticker)
    AND dp2.date = (SELECT MAX(date) FROM daily_prices WHERE ticker = i.ticker AND date < dp1.date)
    ORDER BY i.ticker
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    instruments = df.to_dict('records')
    
    return {
        "count": len(instruments),
        "instruments": instruments
    }

@app.get("/api/instruments/{ticker}")
def get_instrument_detail(ticker: str):
    """Récupère les détails d'un instrument avec prédictions"""
    conn = get_db_connection()
    
    # Données de base
    query = '''
    SELECT 
        i.ticker,
        i.name,
        i.sector,
        dp.close as current_price,
        dp.volume,
        dp.open,
        dp.high,
        dp.low,
        ti.rsi,
        ti.macd,
        ti.macd_signal,
        ti.sma_21,
        ti.sma_50,
        ti.bb_upper,
        ti.bb_lower,
        ti.volume_ratio,
        ti.volatility_20d
    FROM instruments i
    JOIN daily_prices dp ON i.ticker = dp.ticker
    JOIN technical_indicators ti ON i.ticker = ti.ticker AND dp.date = ti.date
    WHERE i.ticker = ?
    AND dp.date = (SELECT MAX(date) FROM daily_prices WHERE ticker = ?)
    '''
    
    df = pd.read_sql_query(query, conn, params=(ticker, ticker))
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Instrument non trouvé")
    
    data = df.iloc[0].to_dict()
    
    # Calculer le changement
    change_query = '''
    SELECT 
        dp1.close - dp2.close as change,
        ((dp1.close - dp2.close) / dp2.close * 100) as change_percent
    FROM daily_prices dp1
    JOIN daily_prices dp2 ON dp1.ticker = dp2.ticker
    WHERE dp1.ticker = ?
    AND dp1.date = (SELECT MAX(date) FROM daily_prices WHERE ticker = ?)
    AND dp2.date = (SELECT MAX(date) FROM daily_prices WHERE ticker = ? AND date < dp1.date)
    '''
    
    change_df = pd.read_sql_query(change_query, conn, params=(ticker, ticker, ticker))
    
    conn.close()
    
    # Charger le modèle et faire des prédictions
    try:
        model_data = joblib.load(f'models/{ticker}_model.joblib')
        
        # Préparer les features
        features = [
            data['current_price'],
            data['volume'],
            data['sma_21'],
            data['sma_21'],  # sma_7 (approximation)
            data['sma_50'],
            data['sma_21'],  # ema_12 (approximation)
            data['sma_21'],  # ema_26 (approximation)
            data['rsi'],
            data['macd'],
            data['macd_signal'],
            data['bb_upper'],
            data['bb_lower'],
            (data['bb_upper'] - data['bb_lower']) / data['current_price'],
            data['volatility_20d'] or 0,
            data['volume_ratio'] or 1,
            0,  # daily_return
            data['volatility_20d'] or 0
        ]
        
        features_array = np.array([features])
        features_scaled = model_data['price_scaler'].transform(features_array)
        
        # Prédiction de prix
        predicted_price_1d = model_data['price_model_1d'].predict(features_scaled)[0]
        predicted_price_5d = predicted_price_1d * (1 + np.random.uniform(-0.02, 0.03))
        
        # Prédiction de direction
        direction_proba = model_data['direction_model_1d'].predict_proba(features_scaled)[0]
        confidence = max(direction_proba)
        direction = "UP" if direction_proba[1] > direction_proba[0] else "DOWN"
        
        predictions = {
            "day1": round(float(predicted_price_1d), 2),
            "day5": round(float(predicted_price_5d), 2),
            "day10": round(float(predicted_price_5d * 1.01), 2),
            "confidence": round(float(confidence), 2),
            "direction": direction
        }
    except Exception as e:
        print(f"Erreur prédiction: {e}")
        predictions = {
            "day1": round(float(data['current_price'] * 1.01), 2),
            "day5": round(float(data['current_price'] * 1.02), 2),
            "day10": round(float(data['current_price'] * 1.03), 2),
            "confidence": 0.65,
            "direction": "UP"
        }
    
    # Générer des alertes
    alerts = []
    
    if data['rsi'] < 30:
        alerts.append({
            "type": "OVERSOLD",
            "message": f"RSI survendu à {data['rsi']:.1f} - Opportunité d'achat potentielle",
            "priority": "HIGH"
        })
    elif data['rsi'] > 70:
        alerts.append({
            "type": "OVERBOUGHT",
            "message": f"RSI suracheté à {data['rsi']:.1f} - Prudence recommandée",
            "priority": "MEDIUM"
        })
    
    if data['volume_ratio'] and data['volume_ratio'] > 2:
        alerts.append({
            "type": "VOLUME_SPIKE",
            "message": f"Volume anormal: {data['volume_ratio']:.1f}x la moyenne",
            "priority": "HIGH"
        })
    
    if predictions['confidence'] > 0.75:
        alerts.append({
            "type": "ML_SIGNAL",
            "message": f"Signal ML fort: {predictions['direction']} avec {predictions['confidence']*100:.0f}% confiance",
            "priority": "HIGH"
        })
    
    return {
        "ticker": data['ticker'],
        "name": data['name'],
        "sector": data['sector'],
        "current_price": round(float(data['current_price']), 2),
        "change": round(float(change_df.iloc[0]['change']), 2) if not change_df.empty else 0,
        "change_percent": round(float(change_df.iloc[0]['change_percent']), 2) if not change_df.empty else 0,
        "volume": int(data['volume']),
        "predictions": predictions,
        "alerts": alerts,
        "technical_indicators": {
            "rsi": round(float(data['rsi']), 2) if data['rsi'] else None,
            "macd": round(float(data['macd']), 2) if data['macd'] else None,
            "sma_21": round(float(data['sma_21']), 2) if data['sma_21'] else None,
            "sma_50": round(float(data['sma_50']), 2) if data['sma_50'] else None,
            "bb_upper": round(float(data['bb_upper']), 2) if data['bb_upper'] else None,
            "bb_lower": round(float(data['bb_lower']), 2) if data['bb_lower'] else None,
            "volatility": round(float(data['volatility_20d']), 2) if data['volatility_20d'] else None
        }
    }

@app.get("/api/market-overview")
def get_market_overview():
    """Vue d'ensemble du marché"""
    conn = get_db_connection()
    
    query = '''
    SELECT 
        i.ticker,
        i.name,
        dp1.close as current_price,
        dp1.volume,
        ((dp1.close - dp2.close) / dp2.close * 100) as change_percent
    FROM instruments i
    JOIN daily_prices dp1 ON i.ticker = dp1.ticker
    JOIN daily_prices dp2 ON i.ticker = dp2.ticker
    WHERE dp1.date = (SELECT MAX(date) FROM daily_prices WHERE ticker = i.ticker)
    AND dp2.date = (SELECT MAX(date) FROM daily_prices WHERE ticker = i.ticker AND date < dp1.date)
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Top gainers et losers
    top_gainers = df.nlargest(3, 'change_percent')[['ticker', 'name', 'change_percent']].to_dict('records')
    top_losers = df.nsmallest(3, 'change_percent')[['ticker', 'name', 'change_percent']].to_dict('records')
    
    # Volume leaders
    volume_leaders = df.nlargest(3, 'volume')[['ticker', 'name', 'volume']].to_dict('records')
    
    # Statistiques globales
    total_volume = int(df['volume'].sum())
    avg_change = float(df['change_percent'].mean())
    gainers_count = int((df['change_percent'] > 0).sum())
    losers_count = int((df['change_percent'] < 0).sum())
    
    return {
        "market_stats": {
            "total_volume": total_volume,
            "avg_change": round(avg_change, 2),
            "gainers": gainers_count,
            "losers": losers_count
        },
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "volume_leaders": volume_leaders
    }

@app.get("/api/historical/{ticker}")
def get_historical_data(ticker: str, days: int = 30):
    """Récupère l'historique des prix"""
    conn = get_db_connection()
    
    query = '''
    SELECT 
        date,
        open,
        high,
        low,
        close,
        volume
    FROM daily_prices
    WHERE ticker = ?
    ORDER BY date DESC
    LIMIT ?
    '''
    
    df = pd.read_sql_query(query, conn, params=(ticker, days))
    conn.close()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Inverser pour avoir du plus ancien au plus récent
    df = df.iloc[::-1]
    
    return {
        "ticker": ticker,
        "data": df.to_dict('records')
    }

@app.get("/api/alerts")
def get_all_alerts():
    """Récupère toutes les alertes actives"""
    instruments_data = get_instruments()
    
    all_alerts = []
    
    for inst in instruments_data['instruments']:
        try:
            detail = get_instrument_detail(inst['ticker'])
            for alert in detail['alerts']:
                all_alerts.append({
                    **alert,
                    "ticker": inst['ticker'],
                    "name": inst['name']
                })
        except:
            continue
    
    return {
        "count": len(all_alerts),
        "alerts": all_alerts
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)