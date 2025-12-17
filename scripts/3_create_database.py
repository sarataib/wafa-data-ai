import pandas as pd
import sqlite3
from datetime import datetime
import os

# ======== Chemins ========
DB_PATH = 'database/market_data.db'
CSV_PATH = 'data/processed/market_data_with_indicators.csv'

# ======== Création de la base ========
def create_database():
    """
    Crée la structure de la base SQLite
    """
    print("="*60)
    print("CRÉATION DE LA BASE DE DONNÉES")
    print("="*60)

    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table instruments
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instruments (
        ticker TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        sector TEXT,
        last_update TEXT
    )
    ''')

    # Table daily_prices
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        turnover REAL,
        trades INTEGER,
        market_cap REAL,
        adj_close REAL,
        FOREIGN KEY (ticker) REFERENCES instruments(ticker),
        UNIQUE(ticker, date)
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_date ON daily_prices(ticker, date)')

    # Table technical_indicators
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS technical_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        sma_7 REAL,
        sma_21 REAL,
        sma_50 REAL,
        ema_12 REAL,
        ema_26 REAL,
        rsi REAL,
        macd REAL,
        macd_signal REAL,
        macd_diff REAL,
        bb_upper REAL,
        bb_middle REAL,
        bb_lower REAL,
        bb_width REAL,
        atr REAL,
        volume_ma_20 REAL,
        volume_ratio REAL,
        obv REAL,
        daily_return REAL,
        volatility_20d REAL,
        FOREIGN KEY (ticker) REFERENCES instruments(ticker),
        UNIQUE(ticker, date)
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_ticker_date ON technical_indicators(ticker, date)')

    # Table ml_predictions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prediction_date DATE NOT NULL,
        target_date DATE NOT NULL,
        predicted_price REAL,
        predicted_direction INTEGER,
        confidence REAL,
        model_version TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ticker) REFERENCES instruments(ticker)
    )
    ''')

    # Table alerts
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        message TEXT NOT NULL,
        priority TEXT,
        triggered_at TEXT DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        FOREIGN KEY (ticker) REFERENCES instruments(ticker)
    )
    ''')

    conn.commit()
    print("✅ Structure de la base créée")
    return conn

# ======== Remplissage instruments ========
def populate_instruments(conn, df):
    """
    Insère les instruments uniques depuis le CSV
    """
    cursor = conn.cursor()
    instruments = df[['Ticker', 'Name']].drop_duplicates()
    instruments['Sector'] = 'Banques'  # Modifier selon besoin
    instruments['last_update'] = datetime.now().isoformat()
    tuples = [tuple(x) for x in instruments[['Ticker', 'Name', 'Sector', 'last_update']].values]

    cursor.executemany('''
        INSERT OR REPLACE INTO instruments (ticker, name, sector, last_update)
        VALUES (?, ?, ?, ?)
    ''', tuples)

    conn.commit()
    print(f"✅ {len(tuples)} instruments insérés dans 'instruments'")

# ======== Remplissage données ========
def populate_data(conn, df):
    """
    Insère daily_prices et technical_indicators
    """
    # daily_prices
    daily_prices = df[[
        'Ticker', 'Date', 'Open', 'High', 'Low', 'Close',
        'Volume', 'Turnover', 'Trades', 'MarketCap', 'AdjClose'
    ]].copy()
    daily_prices.columns = [
        'ticker', 'date', 'open', 'high', 'low', 'close',
        'volume', 'turnover', 'trades', 'market_cap', 'adj_close'
    ]
    daily_prices.to_sql('daily_prices', conn, if_exists='replace', index=False)
    print(f"✅ {len(daily_prices)} enregistrements insérés dans 'daily_prices'")

    # technical_indicators
    indicators = df[[
        'Ticker', 'Date', 'SMA_7', 'SMA_21', 'SMA_50', 'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'BB_Upper', 'BB_Middle',
        'BB_Lower', 'BB_Width', 'ATR', 'Volume_MA_20', 'Volume_Ratio', 'OBV',
        'Daily_Return', 'Volatility_20d'
    ]].copy()
    indicators.columns = [
        'ticker', 'date', 'sma_7', 'sma_21', 'sma_50', 'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_middle',
        'bb_lower', 'bb_width', 'atr', 'volume_ma_20', 'volume_ratio', 'obv',
        'daily_return', 'volatility_20d'
    ]
    indicators.to_sql('technical_indicators', conn, if_exists='replace', index=False)
    print(f"✅ {len(indicators)} enregistrements insérés dans 'technical_indicators'")

# ======== Vérification ========
def verify_database(conn):
    """
    Vérifie le contenu de la base
    """
    cursor = conn.cursor()
    tables = ['instruments', 'daily_prices', 'technical_indicators']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"📊 Table '{table}': {count} enregistrements")

    # Exemple dernières lignes
    query = '''
    SELECT dp.ticker, dp.date, dp.close, ti.rsi, ti.macd, ti.sma_21
    FROM daily_prices dp
    JOIN technical_indicators ti ON dp.ticker = ti.ticker AND dp.date = ti.date
    ORDER BY dp.date DESC
    LIMIT 5
    '''
    df_sample = pd.read_sql_query(query, conn)
    print("\n📋 Dernières 5 lignes combinées:")
    print(df_sample.to_string())

# ======== MAIN ========
def main():
    os.makedirs('database', exist_ok=True)

    # Charger CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ Fichier introuvable : {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])

    # Créer base
    conn = create_database()

    # Remplir instruments
    populate_instruments(conn, df)

    # Remplir données
    populate_data(conn, df)

    # Vérifier
    verify_database(conn)

    conn.close()
    print("\n✅ BASE DE DONNÉES CRÉÉE ET PRÊTE !")
    print(f"📍 Emplacement: {DB_PATH}")

if __name__ == "__main__":
    main()
