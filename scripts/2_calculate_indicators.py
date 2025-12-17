import os
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

def calculate_technical_indicators(df):
    """
    Calcule tous les indicateurs techniques pour chaque instrument
    """
    print("\n🔧 Calcul des indicateurs techniques...")
    
    # Créer une copie
    df = df.copy()
    
    # Grouper par ticker pour calculer les indicateurs séparément
    result_dfs = []
    
    for ticker in df['Ticker'].unique():
        print(f"   Traitement de {ticker}...")
        
        # Filtrer les données du ticker
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
        
        # ========== INDICATEURS DE TENDANCE ==========
        
        # Moyennes Mobiles Simples (SMA)
        ticker_df['SMA_7'] = SMAIndicator(close=ticker_df['Close'], window=7).sma_indicator()
        ticker_df['SMA_21'] = SMAIndicator(close=ticker_df['Close'], window=21).sma_indicator()
        ticker_df['SMA_50'] = SMAIndicator(close=ticker_df['Close'], window=50).sma_indicator()
        
        # Moyennes Mobiles Exponentielles (EMA)
        ticker_df['EMA_12'] = EMAIndicator(close=ticker_df['Close'], window=12).ema_indicator()
        ticker_df['EMA_26'] = EMAIndicator(close=ticker_df['Close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=ticker_df['Close'])
        ticker_df['MACD'] = macd.macd()
        ticker_df['MACD_Signal'] = macd.macd_signal()
        ticker_df['MACD_Diff'] = macd.macd_diff()
        
        # ========== INDICATEURS DE MOMENTUM ==========
        
        # RSI (Relative Strength Index)
        ticker_df['RSI'] = RSIIndicator(close=ticker_df['Close'], window=14).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=ticker_df['High'],
            low=ticker_df['Low'],
            close=ticker_df['Close'],
            window=14
        )
        ticker_df['Stoch_K'] = stoch.stoch()
        ticker_df['Stoch_D'] = stoch.stoch_signal()
        
        # ========== INDICATEURS DE VOLATILITÉ ==========
        
        # Bollinger Bands
        bb = BollingerBands(close=ticker_df['Close'], window=20, window_dev=2)
        ticker_df['BB_Upper'] = bb.bollinger_hband()
        ticker_df['BB_Middle'] = bb.bollinger_mavg()
        ticker_df['BB_Lower'] = bb.bollinger_lband()
        ticker_df['BB_Width'] = (ticker_df['BB_Upper'] - ticker_df['BB_Lower']) / ticker_df['BB_Middle']
        
        # ATR (Average True Range)
        ticker_df['ATR'] = AverageTrueRange(
            high=ticker_df['High'],
            low=ticker_df['Low'],
            close=ticker_df['Close'],
            window=14
        ).average_true_range()
        
        # ========== INDICATEURS DE VOLUME ==========
        
        # Volume Moyen
        ticker_df['Volume_MA_20'] = ticker_df['Volume'].rolling(window=20).mean()
        ticker_df['Volume_Ratio'] = ticker_df['Volume'] / ticker_df['Volume_MA_20']
        
        # OBV (On-Balance Volume)
        ticker_df['OBV'] = (np.sign(ticker_df['Close'].diff()) * ticker_df['Volume']).fillna(0).cumsum()
        
        # ========== FEATURES ADDITIONNELLES ==========
        
        # Rendements
        ticker_df['Daily_Return'] = ticker_df['Close'].pct_change() * 100
        ticker_df['Return_5d'] = ticker_df['Close'].pct_change(periods=5) * 100
        ticker_df['Return_10d'] = ticker_df['Close'].pct_change(periods=10) * 100
        ticker_df['Return_20d'] = ticker_df['Close'].pct_change(periods=20) * 100
        
        # Volatilité
        ticker_df['Volatility_20d'] = ticker_df['Daily_Return'].rolling(window=20).std()
        
        # Momentum
        ticker_df['Momentum_5d'] = ticker_df['Close'] - ticker_df['Close'].shift(5)
        ticker_df['Momentum_10d'] = ticker_df['Close'] - ticker_df['Close'].shift(10)
        
        # Distance des extremes
        ticker_df['Pct_from_High'] = (ticker_df['Close'] - ticker_df['High']) / ticker_df['High'] * 100
        ticker_df['Pct_from_Low'] = (ticker_df['Close'] - ticker_df['Low']) / ticker_df['Low'] * 100
        
        # Range
        ticker_df['Daily_Range'] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Close'] * 100
        
        # ========== LABELS POUR ML (FUTURES VALUES) ==========
        
        # Prix futurs (pour prédiction)
        ticker_df['Target_Close_1d'] = ticker_df['Close'].shift(-1)
        ticker_df['Target_Close_5d'] = ticker_df['Close'].shift(-5)
        ticker_df['Target_Close_10d'] = ticker_df['Close'].shift(-10)
        
        # Direction future (1 = hausse, 0 = baisse)
        ticker_df['Target_Direction_1d'] = (ticker_df['Target_Close_1d'] > ticker_df['Close']).astype(int)
        ticker_df['Target_Direction_5d'] = (ticker_df['Target_Close_5d'] > ticker_df['Close']).astype(int)
        
        # Amplitude du mouvement futur
        ticker_df['Target_Change_1d'] = (ticker_df['Target_Close_1d'] - ticker_df['Close']) / ticker_df['Close'] * 100
        ticker_df['Target_Change_5d'] = (ticker_df['Target_Close_5d'] - ticker_df['Close']) / ticker_df['Close'] * 100
        
        result_dfs.append(ticker_df)
    
    # Combiner tous les résultats
    final_df = pd.concat(result_dfs, ignore_index=True)
    
    print(f"✅ Indicateurs calculés: {len(final_df.columns)} colonnes")
    
    return final_df


def save_processed_data(df, output_path):
    """
    Sauvegarde les données avec indicateurs
    """
    df.to_csv(output_path, index=False)
    print(f"\n💾 Données avec indicateurs sauvegardées: {output_path}")
    
    # Afficher les nouvelles colonnes
    print("\n📊 NOUVELLES COLONNES CRÉÉES:")
    new_columns = [col for col in df.columns if col not in [
        'Date', 'Name', 'Ticker', 'Open', 'High', 'Low', 'Close', 
        'Volume', 'Turnover', 'Trades', 'MarketCap', 'AdjClose'
    ]]
    
    for i, col in enumerate(new_columns, 1):
        print(f"   {i}. {col}")
    
    return output_path


def main():
    """
    Fonction principale
    """
    print("=" * 60)
    print("CALCUL DES INDICATEURS TECHNIQUES")
    print("=" * 60)
    
    # Spécifier le chemin dynamique du fichier
    base_path = os.path.dirname(os.path.realpath(__file__))  # Répertoire actuel du script
    input_path = os.path.join(base_path, '..', 'data', 'processed', 'combined_market_data.csv')  # chemin relatif
    output_path = os.path.join(base_path, '..', 'data', 'processed', 'market_data_with_indicators.csv')
    
    # Vérifier si le fichier existe avant de charger
    if not os.path.exists(input_path):
        print(f"❌ Le fichier {input_path} n'a pas été trouvé.")
        return
    
    # Charger les données combinées
    print("\n📂 Chargement des données combinées...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ {len(df)} lignes chargées")
    
    # Calculer les indicateurs
    df_with_indicators = calculate_technical_indicators(df)
    
    # Sauvegarder
    save_processed_data(df_with_indicators, output_path)
    
    print("\n✅ TERMINÉ!")
    
    return df_with_indicators


if __name__ == "__main__":
    # Installer les dépendances si nécessaire
    # pip install ta pandas numpy
    
    df = main()
