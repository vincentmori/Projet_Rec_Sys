import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from typing import Optional

# *************************************************************************
# CLÉ DE CONNEXION HEROKU (DATABASE_URL)
# Cette chaîne permet la connexion à la base de données cloud PostgreSQL
# *************************************************************************
# Load .env from project root (standard location, ignored by git)
_env_path = os.path.join('.env')
load_dotenv(_env_path)

DB_CONNECTION_STRING = os.environ.get("DATABASE_URL")

def _execute_query_and_get_df(sql_query: str, table_name: str) -> pd.DataFrame:
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        df = pd.read_sql(sql_query, engine)
        
        print(f"Succès : Données de la table '{table_name}' chargées. ({len(df)} lignes)")
        return df

    except Exception as e:
        print(f" Échec de la récupération de la table '{table_name}'. Erreur: {e}")
        return pd.DataFrame() 

# FONCTIONS DE RECUPERATION USER ET TRAVEL
def recup_users() -> pd.DataFrame:
    table_name = "users_generated"
    query = f"SELECT * FROM {table_name}"
    return _execute_query_and_get_df(query, table_name)

def recup_travel() -> pd.DataFrame:
    """
    Récupère l'intégralité du dataset des voyages ('travel_generated')
    depuis la base de données Heroku et le retourne en DataFrame.
    """
    table_name = "travel_generated"
    query = f"SELECT * FROM {table_name}"
    return _execute_query_and_get_df(query, table_name)


# --- EXEMPLE D'UTILISATION ---
if __name__ == '__main__':
    # 1. Récupération des données utilisateurs
    df_users = recup_users()
    if not df_users.empty:
        print("\n--- Aperçu des Données Utilisateurs (Users) ---")
        print(df_users.head())
        print(f"Colonnes: {list(df_users.columns)}")

    print("-" * 55)

    # 2. Récupération des données voyages
    df_travel = recup_travel()
    if not df_travel.empty:
        print("\n--- Aperçu des Données Voyages (Travel) ---")
        print(df_travel.head())
        print(f"Colonnes: {list(df_travel.columns)}")
        
 