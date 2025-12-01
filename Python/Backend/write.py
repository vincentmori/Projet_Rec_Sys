import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Dict, List, Any
import os 
from dotenv import load_dotenv
import streamlit as st

# --- Configuration de la connexion (Doit être lue via .env dans le script principal) ---
# Load .env from project root (standard location, ignored by git)
_env_path = os.path.join('.env')
load_dotenv(_env_path)

DB_CONNECTION_STRING = os.environ.get("DATABASE_URL")

def _get_db_engine(db_connection_string: str) -> Engine:

    if not db_connection_string:
        raise ValueError("La chaîne de connexion DB_CONNECTION_STRING est manquante.")
    return create_engine(db_connection_string)

def _sync_dataframe_to_table(df: pd.DataFrame, table_name: str) -> bool:
    engine = _get_db_engine(DB_CONNECTION_STRING)
    try:
        # Utilise 'replace' pour effacer les anciennes données et insérer les nouvelles
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        return True
    except Exception as e:
        st.error(f"ÉCHEC: Erreur critique lors de la synchronisation de la table '{table_name}': {e}")
        return False

def update_users_table(df_users: pd.DataFrame) -> bool:
    return _sync_dataframe_to_table(df_users, "users_generated")

def update_travel_table(df_travel: pd.DataFrame) -> bool:
    return _sync_dataframe_to_table(df_travel, "travel_generated")



"""
Exemple d'utilisation:
Dans un premier temps tu mets a jour le df puis apres tu mets a jours la bdd ca sert a avoir la base de données toujours aligné au df
En gros je recup juste le df et je le renvoies dans la base de donnes solutions la plus simple 

# --- Dans votre script principal (où DB_CONNECTION_STRING est lu depuis .env) ---
from write import update_users_table, update_travel_table 

# ... (Après avoir créé et validé df_users et df_travel) ...

# Synchronisation de la table USERS
update_users_table(df_users, DB_CONNECTION_STRING)

# Synchronisation de la table TRAVEL
update_travel_table(df_travel, DB_CONNECTION_STRING)

"""