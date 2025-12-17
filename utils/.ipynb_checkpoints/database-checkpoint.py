"""
Gestion de la base de données et requêtes
"""

import sqlite3
import pandas as pd

def get_db_connection():
    """Crée une connexion à la base de données"""
    try:
        conn = sqlite3.connect('database/market_data.db', check_same_thread=False)
        return conn
    except Exception as e:
        print(f"Erreur de connexion à la base de données: {e}")
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
    finally:
        conn.close()