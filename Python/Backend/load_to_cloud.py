import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
import os

# --- Configuration de la connexion CLOUD ---
load_dotenv() 
DB_CONNECTION_STRING = os.environ.get("DATABASE_URL")
if not DB_CONNECTION_STRING:
    print("ERREUR : La variable 'DATABASE_URL' est manquante.")
    exit(1)

# --- Noms des fichiers sources ---
USERS_CSV = "users_generated.csv"
TRAVEL_CSV = "travel_generated.csv"

# --- Noms des tables cibles dans PostgreSQL ---
TABLE_USERS = "users_generated"
TABLE_TRAVEL = "travel_generated"

# --- Fonction utilitaire de nettoyage des noms de colonnes ---
def clean_column_names(df: pd.DataFrame) -> None:
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()

# --- Fonction pour supprimer les anciennes tables (Nettoyage de la BDD) ---
def clean_old_tables(engine: Engine, table_names_to_drop: list):
    print("🧹 Début du nettoyage des tables existantes...")
    for table in table_names_to_drop:
        try:
            with engine.connect() as connection:
                connection.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
                connection.commit()
            print(f"Table '{table}' supprimée.")
        except Exception as e:
            print(f"Erreur lors de la suppression de la table '{table}': {e}")
    print("Nettoyage terminé.")

# --- Fonction principale de Transformation et Chargement ---
def normalize_and_import(engine: Engine):    
    try:
        # Lecture robuste pour la gestion des guillemets dans les destinations
        df_users_brut = pd.read_csv(USERS_CSV, sep=',', quotechar='"', encoding='utf-8')
        df_travel = pd.read_csv(TRAVEL_CSV, sep=',', quotechar='"', encoding='utf-8')
    except Exception as e:
        print(f"Erreur critique de lecture du CSV: {e}")
        return

    clean_column_names(df_users_brut)

    total_rows = len(df_users_brut) + len(df_travel)
    if total_rows > 9900:
        print(f"ERREUR DE VOLUME: {total_rows} lignes. Dépasse la limite de 10k du plan gratuit. Avorté.")
        return
    print(f"Volume total ({total_rows} lignes) est conforme au plan gratuit.")


    # 2. CRÉATION DE LA TABLE USERS 
    df_users = df_users_brut.copy()
    
    # Renommage de la clé primaire
    df_users = df_users.rename(columns={'user_id': 'traveler_user_id'})
    
    # NOUVELLE COLONNE : Création du mot de passe selon la formule nom_âge
    df_users['mot_de_passe'] = df_users['traveler_name'].astype(str) + '_' + df_users['traveler_age'].astype(str)

    user_info_columns_to_drop = [
        'traveler_name'
    ]
    
    datasets_to_load = {
        TABLE_USERS: df_users, 
        TABLE_TRAVEL: df_travel
    }
    
    for table_name, df_data in datasets_to_load.items():
        try:
            df_data.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Table '{table_name}' chargée avec succès : {len(df_data)} lignes.")
        except Exception as e:
            print(f"Échec critique du chargement de la table '{table_name}': {e}")


# --- Bloc d'exécution principal ---
if __name__ == '__main__':
    # Vérification de sécurité CRITIQUE
    if not DB_CONNECTION_STRING:
        print("ERREUR : La variable d'environnement 'DATABASE_URL' n'est pas définie.")
        exit(1)
        
    tables_to_drop = [TABLE_USERS, TABLE_TRAVEL, "travel_details", "users_generated", "travel_generated"]
    
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        
        clean_old_tables(engine, tables_to_drop)
        
        normalize_and_import(engine)
        
    except Exception as e:
        print(f"Erreur fatale de connexion ou d'exécution : {e}")