# app.py - Fichier principal
import streamlit as st
import os

# Configuration de la page DOIT ÊTRE LA PREMIÈRE COMMANDE
st.set_page_config(
    page_title="MarketSense Morocco",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importer l'application principale
try:
    from main_app import main
    main()
except Exception as e:
    st.error(f"Erreur: {e}")
    st.info("Vérifiez que tous les fichiers sont présents.")