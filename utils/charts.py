"""
Fonctions de visualisation et graphiques
"""

import plotly.graph_objects as go
import pandas as pd

def create_price_chart(df, ticker):
    """Crée un graphique de prix"""
    fig = go.Figure()
    
    if df.empty:
        fig.update_layout(title=f"Aucune donnée disponible pour {ticker}")
        return fig
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Prix de clôture',
        line=dict(color='#1E88E5', width=2)
    ))
    
    fig.update_layout(
        title=f'Évolution du prix - {ticker}',
        xaxis_title='Date',
        yaxis_title='Prix (MAD)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_candlestick_chart(df, ticker):
    """Crée un graphique en chandeliers"""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Aucune donnée disponible pour {ticker}")
        return fig
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    )])
    
    fig.update_layout(
        title=f'Graphique en chandeliers - {ticker}',
        xaxis_title='Date',
        yaxis_title='Prix (MAD)',
        height=400
    )
    
    return fig

def create_volume_chart(df, ticker):
    """Crée un graphique de volume"""
    fig = go.Figure()
    
    if df.empty:
        fig.update_layout(title=f"Aucune donnée disponible pour {ticker}")
        return fig
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        name='Volume'
    ))
    
    fig.update_layout(
        title=f'Volume d\'échanges - {ticker}',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300
    )
    
    return fig

def create_indicator_chart(df, indicator, ticker, indicator_name):
    """Crée un graphique pour un indicateur technique"""
    fig = go.Figure()
    fig.update_layout(title=f"{indicator_name} - {ticker}", height=300)
    return fig

def create_prediction_chart(df, prediction, ticker):
    """Crée un graphique avec les prédictions"""
    fig = go.Figure()
    fig.update_layout(title=f"Prédictions - {ticker}", height=400)
    return fig