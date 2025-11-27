import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import csv
import os
'''
# --- Configuration de la connexion CLOUD ---
# Utilisez la chaîne de connexion Heroku mise à jour avec le dialecte psycopg2
DB_CONNECTION_STRING = "postgresql+psycopg2://ucbkaif701fjf1:p82d260bcd738f493c3bec4bce418296879d7b558a0528685cfcad52a4632d69a@cet8gijgk7sjl9.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d2i3gudil1qocc" 

# --- Nouveaux fichiers et noms de tables ---
TABLE_FILES = {
    "users_generated": "users_generated.csv",
    "travel_generated": "travel_generated.csv"
}

# --- Fonction pour nettoyer les noms de colonnes pour PostgreSQL ---
def clean_column_names(df: pd.DataFrame) -> None:
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()

# --- Fonction pour supprimer les anciennes tables (Nettoyage de la BDD) ---
def clean_old_tables(engine: Engine, table_names_to_drop: list):
    print("🧹 Début du nettoyage des anciennes tables...")
    for table in table_names_to_drop:
        try:
            # Utilisation de 'text' pour exécuter du SQL direct et sécurisé
            with engine.connect() as connection:
                connection.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
                connection.commit()
            print(f"   ✅ Table '{table}' supprimée.")
        except Exception as e:
            print(f"   ❌ Erreur lors de la suppression de la table '{table}': {e}")
    print("Nettoyage terminé.")

# --- Fonction principale pour l'importation ---
def import_datasets(engine: Engine):
    print("🚀 Début de l'importation des nouveaux datasets...")
    for table_name, csv_file in TABLE_FILES.items():
        if not os.path.exists(csv_file):
            print(f"   ❌ Erreur: Le fichier CSV '{csv_file}' est introuvable. Ignoré.")
            continue
            
        print(f"   ⏳ Importation de '{csv_file}' dans la table '{table_name}'...")
        
        try:
            # 1. Lecture et nettoyage du CSV
            df = pd.read_csv(csv_file, delimiter=',', quoting=csv.QUOTE_NONE, encoding='utf-8')
            clean_column_names(df)  # Nettoie les noms de colonnes (minuscules, pas d'espaces)

            # 2. Écriture dans PostgreSQL
            # 'if_exists=replace' est utilisé pour recréer la table si elle existe.
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            
            print(f"   ✅ Données chargées dans '{table_name}': {len(df)} lignes.")

        except Exception as e:
            print(f"   ❌ Échec de l'importation de '{table_name}': {e}")

# --- Bloc d'exécution principal ---
if __name__ == '__main__':
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        print("Connexion à la base de données Heroku établie.")
        
        # 1. Nettoyage des anciennes tables (inclut l'ancienne 'travel_details')
        tables_to_drop = ["travel_details", "users_generated", "travel_generated"]
        clean_old_tables(engine, tables_to_drop)
        
        # 2. Importation des nouveaux datasets
        import_datasets(engine)
        
        print("\n🎉 Processus Data Engineering terminé avec succès.")
        
    except Exception as e:
        print(f"\n🛑 Erreur fatale de connexion ou d'exécution : {e}")'''