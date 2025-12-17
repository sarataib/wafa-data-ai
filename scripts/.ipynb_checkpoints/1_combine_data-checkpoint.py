import pandas as pd
import os
import glob

def load_and_combine_instruments():
    """
    Charge tous les fichiers Excel et les combine
    """

    # 📌 Chemin ABSOLU vers data/raw
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(BASE_DIR, "data", "raw")

    all_data = []

    excel_files = glob.glob(os.path.join(data_folder, "*.xlsx"))

    print(f"📂 {len(excel_files)} fichiers trouvés dans {data_folder}")

    if len(excel_files) == 0:
        raise ValueError("❌ Aucun fichier Excel trouvé dans data/raw")

    for file_path in excel_files:
        try:
            # ✅ Lecture Excel
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()

            print(f"✅ Chargé {os.path.basename(file_path)} : {len(df)} lignes")
            all_data.append(df)

        except Exception as e:
            print(f"❌ Erreur avec {file_path} : {e}")

    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"\n✨ Total combiné : {len(combined_df)} lignes")
    return combined_df


def clean_data(df):
    print("\n🧹 Nettoyage des données...")

    column_mapping = {
        'Séance': 'Date',
        'Instrument': 'Name',
        'Ticker': 'Ticker',
        'Ouverture': 'Open',
        'Dernier Cours': 'Close',
        '+haut du jour': 'High',
        '+bas du jour': 'Low',
        'Nombre de titres échangés': 'Volume',
        'Volume des échanges': 'Turnover',
        'Nombre de contrats': 'Trades',
        'Capitalisation': 'MarketCap',
        'Cours ajusté': 'AdjClose'
    }

    df = df.rename(columns=column_mapping)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    numeric_columns = [
        'Open', 'High', 'Low', 'Close',
        'Volume', 'Turnover', 'Trades',
        'MarketCap', 'AdjClose'
    ]

    for col in numeric_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

    print(f"✅ Nettoyage terminé : {len(df)} lignes")
    return df


def save_combined_data(df):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "combined_market_data.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n💾 Données sauvegardées : {output_path}")
    return output_path


def main():
    print("=" * 60)
    print(" COMBINAISON DES DONNÉES DE MARCHÉ ")
    print("=" * 60)

    df = load_and_combine_instruments()
    df = clean_data(df)
    save_combined_data(df)

    print("\n✅ TERMINÉ AVEC SUCCÈS")
    return df


if __name__ == "__main__":
    main()
