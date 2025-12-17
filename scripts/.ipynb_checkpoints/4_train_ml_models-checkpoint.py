import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score
)
import xgboost as xgb
import joblib
import os
from datetime import datetime


# ============================================================
# LOAD DATA
# ============================================================

def load_data_from_db():
    print("📂 Chargement des données depuis la base de données...")

    conn = sqlite3.connect('database/market_data.db')

    query = """
    SELECT dp.*, ti.*
    FROM daily_prices dp
    JOIN technical_indicators ti
        ON dp.ticker = ti.ticker
        AND dp.date = ti.date
    ORDER BY dp.ticker, dp.date
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df = df.loc[:, ~df.columns.duplicated()]
    print(f"✅ {len(df)} lignes chargées")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_ml_features(df, ticker):
    ticker_df = df[df['ticker'] == ticker].copy()
    ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

    feature_columns = [
        'close', 'volume',
        'sma_7', 'sma_21', 'sma_50',
        'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr', 'volume_ratio',
        'daily_return', 'volatility_20d'
    ]

    ticker_df = ticker_df.dropna(subset=feature_columns)

    ticker_df['target_1d'] = ticker_df['close'].shift(-1)
    ticker_df['target_5d'] = ticker_df['close'].shift(-5)
    ticker_df['target_10d'] = ticker_df['close'].shift(-10)

    ticker_df['target_direction_1d'] = (
        ticker_df['target_1d'] > ticker_df['close']
    ).astype(int)

    ticker_df['target_direction_5d'] = (
        ticker_df['target_5d'] > ticker_df['close']
    ).astype(int)

    ticker_df = ticker_df[:-10]

    return ticker_df, feature_columns


# ============================================================
# PRICE PREDICTION
# ============================================================

def train_price_prediction_model(df, ticker, horizon='1d'):
    print(f"\n🤖 Entraînement du modèle de prix pour {ticker} ({horizon})")

    ticker_df, feature_columns = prepare_ml_features(df, ticker)

    X = ticker_df[feature_columns].values

    if horizon == '1d':
        y = ticker_df['target_1d'].values
    elif horizon == '5d':
        y = ticker_df['target_5d'].values
    else:
        y = ticker_df['target_10d'].values

    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results[name] = {
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        }

        print(f"   {name} → RMSE={rmse:.2f} | MAE={mae:.2f} | MAPE={mape:.2f}%")

    best_name = min(results, key=lambda x: results[x]['rmse'])
    print(f"   ✅ Meilleur modèle : {best_name}")

    return (
        results[best_name]['model'],
        scaler,
        results[best_name],
        feature_columns
    )


# ============================================================
# DIRECTION PREDICTION
# ============================================================

def train_direction_prediction_model(df, ticker, horizon='1d'):
    print(f"\n🎯 Entraînement direction pour {ticker} ({horizon})")

    ticker_df, feature_columns = prepare_ml_features(df, ticker)

    X = ticker_df[feature_columns].values
    y = (
        ticker_df['target_direction_1d'].values
        if horizon == '1d'
        else ticker_df['target_direction_5d'].values
    )

    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   ✅ Accuracy: {accuracy * 100:.2f}%")

    return model, scaler, accuracy, feature_columns


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(
    ticker,
    price_model,
    price_scaler,
    direction_model,
    direction_scaler,
    feature_columns
):
    os.makedirs("models", exist_ok=True)

    model_data = {
        "ticker": ticker,
        "price_model_1d": price_model,
        "price_scaler": price_scaler,
        "direction_model_1d": direction_model,
        "direction_scaler": direction_scaler,
        "features": feature_columns,
        "trained_at": datetime.now(),
        "version": "1.0"
    }

    path = f"models/{ticker}_model.joblib"
    joblib.dump(model_data, path)
    print(f"   💾 Modèle sauvegardé → {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES ML")
    print("=" * 60)

    df = load_data_from_db()
    tickers = df['ticker'].unique()

    print(f"\n📊 {len(tickers)} instruments : {', '.join(tickers)}")

    summary = []

    for ticker in tickers:
        print("\n" + "=" * 60)
        print(f"TICKER: {ticker}")
        print("=" * 60)

        try:
            price_model, price_scaler, price_metrics, features = \
                train_price_prediction_model(df, ticker, '1d')

            direction_model, direction_scaler, accuracy, _ = \
                train_direction_prediction_model(df, ticker, '1d')

            save_models(
                ticker,
                price_model,
                price_scaler,
                direction_model,
                direction_scaler,
                features
            )

            summary.append({
                "ticker": ticker,
                "rmse": price_metrics['rmse'],
                "mape": price_metrics['mape'],
                "accuracy": accuracy * 100
            })

        except Exception as e:
            print(f"❌ Erreur pour {ticker}: {e}")

    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    if summary_df.empty:
        print("❌ Aucun modèle entraîné.")
        return

    print(f"\n✅ RMSE moyen: {summary_df['rmse'].mean():.2f}")
    print(f"✅ MAPE moyen: {summary_df['mape'].mean():.2f}%")
    print(f"✅ Accuracy moyenne: {summary_df['accuracy'].mean():.2f}%")


if __name__ == "__main__":
    main()
