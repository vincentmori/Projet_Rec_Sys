import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_and_process_data(csv_path):
    print(f"--- [Data Engineer] Chargement et Réparation de {csv_path} ---")

    # 1. LECTURE
    try:
        df = pd.read_csv(csv_path, sep=',')
        if df.shape[1] < 5:
            print("⚠️ Séparateur virgule inefficace, passage en tabulation...")
            df = pd.read_csv(csv_path, sep='\t')
    except:
        df = pd.read_csv(csv_path, sep=None, engine='python')

    df.columns = df.columns.str.strip()
    initial_len = len(df)

    # ==============================================================================
    # 2. NETTOYAGE (AVANT DE CRÉER LES IDs)
    # ==============================================================================
    # C'est CRUCIAL de le faire maintenant. Si on supprime des lignes après avoir
    # donné les numéros, on crée des trous (0, 2, 5...) et le modèle plante.

    num_cols = ['Accommodation cost', 'Transportation cost', 'Duration (days)', 'Traveler age']

    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Suppression des lignes sales
    df.dropna(subset=[c for c in num_cols if c in df.columns], inplace=True)

    print(f"Nettoyage : {initial_len} -> {len(df)} lignes conservées.")
    if len(df) == 0:
        print("❌ Erreur : Plus aucune donnée après nettoyage.")
        exit()

    # Conversion finale float
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')

    # ==============================================================================
    # 3. MAPPING ET RE-NUMÉROTATION (OBLIGATOIRE POUR PYTORCH)
    # ==============================================================================
    # On ignore les colonnes 'uid', 'dest_id' du CSV car elles ont des trous maintenant.
    # On recrée des index parfaits 0..N basés sur les Noms/Villes.

    mappings = {}

    # Users : On refait l'ID basé sur le Nom unique
    user_names = df['Traveler name'].unique()
    mappings['User'] = {name: i for i, name in enumerate(user_names)}
    df['uid'] = df['Traveler name'].map(mappings['User'])

    # Destinations
    dest_names = df['Destination'].unique()
    mappings['Destination'] = {name: i for i, name in enumerate(dest_names)}
    df['dest_id'] = df['Destination'].map(mappings['Destination'])

    # Categories (On refait aussi pour être sûr)
    # Accommodation
    acc_names = df['Accommodation type'].unique()
    mappings['Accommodation type'] = {name: i for i, name in enumerate(acc_names)}
    df['acc_id'] = df['Accommodation type'].map(mappings['Accommodation type'])

    # Transport
    trans_names = df['Transportation type'].unique()
    mappings['Transportation type'] = {name: i for i, name in enumerate(trans_names)}
    df['trans_id'] = df['Transportation type'].map(mappings['Transportation type'])

    # Gender
    gender_names = df['Traveler gender'].unique()
    mappings['Traveler gender'] = {name: i for i, name in enumerate(gender_names)}
    df['gender_id'] = df['Traveler gender'].map(mappings['Traveler gender'])

    # Nationality (On le crée proprement)
    nat_names = df['Traveler nationality'].unique()
    mappings['Traveler nationality'] = {name: i for i, name in enumerate(nat_names)}
    df['nationality_id'] = df['Traveler nationality'].map(mappings['Traveler nationality'])

    # Saison
    # Pour la saison, on peut garder la logique de calcul si la colonne n'existe pas
    if 'travel_season' not in df.columns:
        def get_season(date):
            try:
                m = date.month
                if m in [12, 1, 2]:
                    return 'Winter'
                elif m in [3, 4, 5]:
                    return 'Spring'
                elif m in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Autumn'
            except:
                return 'Autumn'

        df['Start date'] = pd.to_datetime(df['Start date'])
        df['travel_season'] = df['Start date'].apply(get_season)

    season_names = df['travel_season'].unique()
    mappings['season'] = {name: i for i, name in enumerate(season_names)}
    df['season_id'] = df['travel_season'].map(mappings['season'])

    print("IDs régénérés et contigus (0 à N). Prêt pour le Graphe.")
    return df, mappings


def build_graph(df):
    print("--- [Data Engineer] Construction du Graphe ---")

    # 1. NODE USER
    # On est sûr que uid est propre (0..N)
    user_cols = ['uid', 'gender_id', 'nationality_id', 'Traveler age']
    user_df = df[user_cols].drop_duplicates('uid').sort_values('uid')

    # Sécurité : Vérifier que les UIDs sont bien 0, 1, 2... sans trou
    assert len(user_df) == user_df['uid'].max() + 1, "Erreur: Les User IDs ne sont pas contigus !"

    scaler_age = StandardScaler()
    age_scaled = scaler_age.fit_transform(user_df[['Traveler age']].values)

    user_x = torch.cat([
        torch.tensor(user_df['gender_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['nationality_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(age_scaled, dtype=torch.float)
    ], dim=1)

    # 2. NODE DESTINATION
    dest_df = df[['dest_id']].drop_duplicates('dest_id').sort_values('dest_id')
    assert len(dest_df) == dest_df['dest_id'].max() + 1, "Erreur: Les Dest IDs ne sont pas contigus !"

    dest_x = torch.ones((len(dest_df), 1), dtype=torch.float)

    # 3. EDGES
    src = torch.tensor(df['uid'].values, dtype=torch.long)
    dst = torch.tensor(df['dest_id'].values, dtype=torch.long)

    # 4. CONTEXTE
    edge_attr_cat = torch.stack([
        torch.tensor(df['acc_id'].values, dtype=torch.long),
        torch.tensor(df['trans_id'].values, dtype=torch.long),
        torch.tensor(df['season_id'].values, dtype=torch.long)
    ], dim=1)

    scaler_edges = StandardScaler()
    edge_nums = df[['Accommodation cost', 'Transportation cost', 'Duration (days)']].values
    edge_attr_num = torch.tensor(scaler_edges.fit_transform(edge_nums), dtype=torch.float)

    # 5. HETERODATA
    data = HeteroData()

    data['user'].num_nodes = len(user_df)
    data['user'].x = user_x

    data['destination'].num_nodes = len(dest_df)
    data['destination'].x = dest_x

    data['user', 'visits', 'destination'].edge_index = torch.stack([src, dst], dim=0)
    data['user', 'visits', 'destination'].edge_attr_cat = edge_attr_cat
    data['user', 'visits', 'destination'].edge_attr_num = edge_attr_num

    data['destination', 'rev_visits', 'user'].edge_index = torch.stack([dst, src], dim=0)

    return data