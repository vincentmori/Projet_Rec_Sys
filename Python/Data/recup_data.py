import pandas as pd
from sqlalchemy import create_engine
from typing import Optional

# *************************************************************************
# CLÉ DE CONNEXION HEROKU (DATABASE_URL)
# Cette chaîne permet la connexion à votre base de données cloud PostgreSQL
# *************************************************************************
DB_CONNECTION_STRING = "postgresql+psycopg2://ucbkaif701fjf1:p82d260bcd738f493c3bec4bce418296879d7b558a0528685cfcad52a4632d69a@cet8gijgk7sjl9.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d2i3gudil1qocc" 

# --- Fonction Générique de Connexion et d'Extraction ---
def _execute_query_and_get_df(sql_query: str, table_name: str) -> pd.DataFrame:
    """
    Se connecte à la base de données PostgreSQL Cloud, exécute une requête SQL,
    et retourne le résultat en tant que DataFrame Pandas.
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        df = pd.read_sql(sql_query, engine)
        
        print(f"✅ Succès : Données de la table '{table_name}' chargées. ({len(df)} lignes)")
        return df

    except Exception as e:
        print(f"❌ Échec de la récupération de la table '{table_name}'. Erreur: {e}")
        return pd.DataFrame() 

# =========================================================================
# FONCTIONS PUBLIQUES POUR LES ML ENGINEERS
# =========================================================================

def recup_users() -> pd.DataFrame:
    """
    Récupère l'intégralité du dataset des utilisateurs ('users_generated')
    depuis la base de données Heroku et le retourne en DataFrame.
    """
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


# --- EXEMPLE D'UTILISATION (pour vos collègues) ---
if __name__ == '__main__':
    print("=========================================================")
    print("             TEST DES FONCTIONS D'EXTRACTION             ")
    print("=========================================================")

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
        
    print("=========================================================")