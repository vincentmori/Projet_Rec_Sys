import pandas as pd
import os
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from Python.Backend import recup_data  
from Python.Backend.genV2 import TRANSPORT_MODES

from typing import Optional

def process_new_data(df_users, df_travel):
    """
    Étape 1 : Nettoyage brut, calculs de coûts et de dates.
    Ne génère PAS les IDs graph (uid, dest_id) pour éviter les incohérences après dropna.
    """
    print("--- [Preprocessing] Nettoyage et Feature Engineering ---")

    # Copy travel dataframe and normalize column names (snake_case) for robust processing
    df = df_travel.copy()
    # Normalize column names: lower, strip, replace spaces and special chars with underscore
    def normalize_col(c):
        return c.strip().lower().replace(' ', '_').replace('-', '_').replace('/', '_')

    new_cols = {c: normalize_col(c) for c in df.columns}
    df.rename(columns=new_cols, inplace=True)

    # 1. NETTOYAGE DES TEXTES (Guillemets et Espaces)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r'["\']', '', regex=True).str.strip()

    # 2. CALCULS MANQUANTS (Coûts et Dates)

    # Coûts : S'assurer que c'est numérique
    for col in ['total_cost', 'accommodation_cost']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calcul Transport si possible -> use normalized column names (total_cost/accommodation_cost) and keep normalized key
    if 'total_cost' in df.columns and 'accommodation_cost' in df.columns:
        df['transportation_cost'] = (df['total_cost'] - df['accommodation_cost']).clip(lower=0)
    else:
        df['transportation_cost'] = 0

    # Dates: parse normalized 'start_date' and 'end_date' if present
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

    # Durée (operate on normalized keys)
    if 'duration_days' in df.columns:
        df['duration_(days)'] = pd.to_numeric(df['duration_days'], errors='coerce').fillna(1)
    elif 'start_date' in df.columns and 'end_date' in df.columns:
        df['duration_(days)'] = (df['end_date'] - df['start_date']).dt.days
    else:
        df['duration_(days)'] = 1  # Valeur par défaut

    # 3. SAISON (Basée sur la date de début)
    def get_season(date_obj):
        if pd.isnull(date_obj): return "Unknown"
        try:
            m = date_obj.month
            d = date_obj.day
            if (m == 3 and d >= 20) or m in [4, 5] or (m == 6 and d < 21):
                return "Spring"
            elif (m == 6 and d >= 21) or m in [7, 8] or (m == 9 and d < 23):
                return "Summer"
            elif (m == 9 and d >= 23) or m in [10, 11] or (m == 12 and d < 21):
                return "Autumn"
            else:
                return "Winter"
        except:
            return "Unknown"

    if 'start_date' in df.columns:
        df['travel_season'] = df['start_date'].apply(get_season)
    else:
        df['travel_season'] = "Unknown"

    # 4. PAYS (Extraction depuis "Ville, Pays")
    if 'destination' in df.columns:
        df['dest_country'] = df['destination'].apply(lambda x: x.split(',')[-1].strip() if ',' in x else x)
    else:
        df['dest_country'] = "Unknown"

    # 5. RENOMMAGE (Standardisation des noms de colonnes). Map normalized keys to canonical column names
    rename_map = {
        'destination': 'Destination',
        'traveler_name': 'Traveler name',
        'traveler_age': 'Traveler age',
        'traveler_gender': 'Traveler gender',
        'traveler_nationality': 'Traveler nationality',
        'accommodation_type': 'Accommodation type',
        'accommodation_cost': 'Accommodation cost',
        'transportation_cost': 'Transportation cost',
        'local_transport_mode': 'Transportation type',
        'local_transport_options': 'Transportation options',
        'user_id': 'User ID',
        'start_date': 'Start date',
        'end_date': 'End date',
        'duration_(days)': 'Duration (days)'
    }
    df = df.rename(columns=rename_map)

    # Vérification que les colonnes clés existent (sinon on met 'Unknown' ou 0)
    expected_cols = ['Destination', 'Accommodation type', 'Transportation type',
                     'Traveler gender', 'Traveler nationality']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    if 'Traveler age' not in df.columns:
        df['Traveler age'] = 30  # Age par défaut
        
    # Suppression des colonnes présentes également dans df_users qui ne doivent pas être dupliques sur les arêtes 
    node_features_to_drop = [
        'Traveler name', 
        'Traveler age', 
        'Traveler gender', 
        'Traveler nationality',
    ]
    df.drop(columns=node_features_to_drop, errors='ignore', inplace=True)
    
    #------------------------------
    # Traitement sur df_user
    # -----------------------------
    df_users_copy = df_users.copy()  
    
    # 1. NETTOYAGE DES TEXTES (Guillemets et Espaces)
    for col in df_users_copy.columns:
        if df_users_copy[col].dtype == object:
            df_users_copy[col] = df_users_copy[col].astype(str).str.replace(r'["\']', '', regex=True).str.strip()
     
    rename_map_user = {
        'traveler_user_id': 'Traveler User ID',
        'traveler_name': 'Traveler Name',
        'traveler_age': 'Traveler Age',
        'traveler_gender': 'Traveler Gender',
        'traveler_nationality': 'Traveler Nationality',
        'profile_type': 'Profile Type',
        'climate_pref': 'Climate Pref',
        'primary_dest_type': 'Primary Dest Type',
        'acc_pref': 'Accomodation Pref',
        'transport_core_modes': 'Transport modes',
        'traveler_continent': 'Traveler Continent'
    }
      
    df_users_cleaned = df_users_copy.rename(columns=rename_map_user, errors='ignore')

    return df, df_users_cleaned

def load_and_process_data(static_mappings: Optional[dict] = None):
    """
    Étape 2 : Chargement, Filtrage des NaN, et Création des IDs (Mapping).
    C'est la fonction principale à appeler.
    
    If csv_path is provided and exists, loads from CSV.
    Otherwise, tries to load from database.
    If database fails, falls back to local CSV files in Data folder.
    """

    df_users = recup_data.recup_users()
    df_travel = recup_data.recup_travel()

    # 2. Pré-traitement (Nettoyage texte/dates)
    df_travel_cleaned, df_users_cleaned = process_new_data(df_users, df_travel)

    initial_len = len(df_travel_cleaned)
    initial_len_user = len(df_users_cleaned)
    
    def nettoyage_numerique(df_num, col_num, col_imp_nn_num):
        for col in col_num:
            col = str(col)
            # If any expected numeric column is missing, fill with zeros to avoid KeyError
            if col not in df_num.columns:
                df_num[col] = 0
            # If the column exists but maps to a DataFrame slice (rare), reduce to a single Series
            val = df_num[col]
            if isinstance(val, pd.DataFrame):
                if val.shape[1] >= 1:
                    # Take the first sub-column if multiple; preserve name
                    val = val.iloc[:, 0]
                else:
                    val = val.squeeze()
                df_num[col] = val
            # Nettoyage des symboles monétaires si restants
            if df_num[col].dtype == object:
                df_num[col] = df_num[col].astype(str).str.replace(r'[$,]', '', regex=True)
            # Conversion
            df_num[col] = pd.to_numeric(df_num[col], errors='coerce')

        # Suppression des lignes avec des valeurs manquantes dans les colonnes critiques
        df_num.dropna(subset=col_num + col_imp_nn_num, inplace=True)

        # Conversion float32 pour PyTorch
        for col in col_num:
            df_num[col] = df_num[col].astype('float32')
            
        return df_num

    # 3. NETTOYAGE NUMÉRIQUE (CRUCIAL AVANT LE MAPPING)
    # df
    num_cols_travel = ['Accommodation cost', 'Transportation cost', 'Duration (days)']

    col_imp_travel = ['Destination', 'User ID']
    
    df_travel_cleaned = nettoyage_numerique(df_travel_cleaned, num_cols_travel, col_imp_travel)

    print(f"Nettoyage Df_travel: {initial_len_user} -> {len(df_users_cleaned)} lignes de travel conservées pour l'entraînement.")
    
    # df_user
    num_cols_user = ['Traveler Age']
    
    col_imp_user = [
        'Traveler User ID', 'Traveler Gender', 'Traveler Nationality',
        'Profile Type', 'Climate Pref', 'Primary Dest Type', 
        'Accomodation Pref', 'Transport modes', 'Traveler Continent'
        ]
    
    df_users_cleaned = nettoyage_numerique(df_users_cleaned, num_cols_user, col_imp_user)

    print(f"Nettoyage Df_user: {initial_len} -> {len(df_travel_cleaned)} user conservées pour l'entraînement.")

    if len(df_travel_cleaned) == 0:
        raise ValueError(f"❌ Erreur : Plus aucune donnée travel après le nettoyage. Vérifiez vos données sources.")
    
    if len(df_users_cleaned) == 0:
        raise ValueError(f"❌ Erreur : Plus aucune donnée user après le nettoyage. Vérifiez vos données sources.")

    # 4. MAPPING ET CRÉATION DES IDs (0 à N, sans trous)
    mappings = {}

    # Helper pour créer un mapping
    def create_mapping(df_map, column_name, mapping_key):
        if column_name not in df_map.columns and column_name.lower() in df_map.columns:
            column_name = column_name.lower()
        unique_vals = df_map[column_name].unique()

        mapp = {name: i for i, name in enumerate(unique_vals)}
        mappings[mapping_key] = mapp
        
        return df_map[column_name].map(mapp)


    # Application du mapping et creation du maping Traveler ID ensuite applique au df
    # Pour la gestion des nouvelles personnes et des personnes sans historique de voyage
    df_users_cleaned['uid'] = create_mapping(df_users_cleaned, 'Traveler User ID', 'User')
    
    # Application des mappings sur df
    df_travel_cleaned['uid'] = df_travel_cleaned['User ID'].map(mappings['User'])

    
    df_travel_cleaned['dest_id'] = create_mapping(df_travel_cleaned, 'Destination', 'Destination')
    
    # Helper pour appliquer un mapping EXISTANT    
    def apply_mapping(df_map, column_name, mapping_dict):
        # Gestion des noms de colonnes et application du mapping
        if column_name not in df_map.columns and column_name.lower() in df_map.columns:
            column_name = column_name.lower()
            
        # Le .map() va remplacer les chaînes non trouvées (nouveaux modes) par NaN.
        return df_map[column_name].map(mapping_dict)

    # Application des mappings statiques pour les attributs d'arête
    if static_mappings:
        # 1. Hébergement (Statique)
        df_travel_cleaned['acc_id'] = apply_mapping(df_travel_cleaned, 'Accommodation type', static_mappings['Accommodation type'])
        
        # 2. Transport (Statique)
        df_travel_cleaned['trans_id'] = apply_mapping(df_travel_cleaned, 'Transportation type', static_mappings['Transportation type'])
        
        # 3. Saison (Statique)
        df_travel_cleaned['season_id'] = apply_mapping(df_travel_cleaned, 'travel_season', static_mappings['season'])

        df_travel_cleaned['trans_id'] = pd.to_numeric(df_travel_cleaned['trans_id'], errors='coerce')
        df_travel_cleaned['trans_id'] = df_travel_cleaned['trans_id'].fillna(-1).astype(int) 

        # Correction d'index > 12
        df_travel_cleaned.loc[df_travel_cleaned['trans_id'] > 12, 'trans_id'] = 0 
        df_travel_cleaned.loc[df_travel_cleaned['trans_id'] < 0, 'trans_id'] = 0 # Corrige les NaN mappés à -1
    
    # Nettoyage des lignes sans uid et dest_id 
    df_travel_cleaned.dropna(subset=['uid', 'dest_id'], inplace=True) 
    
    # Application des mappings sur df_user
    df_users_cleaned['gender_id'] = create_mapping(df_users_cleaned, 'Traveler Gender', 'Traveler Gender')
    df_users_cleaned['nationality_id'] = create_mapping(df_users_cleaned, 'Traveler Nationality', 'Traveler Nationality')
    df_users_cleaned['profile_type_id'] = create_mapping(df_users_cleaned, 'Profile Type', 'Profile Type')
    df_users_cleaned['climate_pref_id'] = create_mapping(df_users_cleaned, 'Climate Pref', 'Climate Pref')
    df_users_cleaned['acc_pref_id'] = create_mapping(df_users_cleaned, 'Accomodation Pref', 'Accommodation Pref')
    df_users_cleaned['transport_modes_id'] = create_mapping(df_users_cleaned, 'Transport modes', 'Transport Core Modes')
    df_users_cleaned['continent_id'] = create_mapping(df_users_cleaned, 'Traveler Continent', 'Traveler Continent')

    print("[OK] Données prêtes et IDs générés.")
    return df_travel_cleaned, df_users_cleaned, mappings

def build_graph(df_travel, df_users, num_users_override: int = None, num_dests_override: int = None):
    """
    Étape 3 : Conversion du DataFrame propre en HeteroData (PyTorch Geometric).
    Utilise df_users pour les features des nœuds utilisateurs (user.x).
    """
    print("--- [Data Engineer] Construction du Graphe ---")

    # ----------------------------------------------------
    # 1. NODE USER (Source : df_users)
    # ----------------------------------------------------
    
    # Préparation du DataFrame utilisateur 
    # On filtre les colonnes ID et on s'assure de l'alignement par 'uid'
    user_df = df_users.dropna(subset=['uid']).drop_duplicates('uid').sort_values('uid')
    
    num_users_to_check = num_users_override if num_users_override is not None else len(user_df)

    # Normalisation de l'âge (Feature Numérique)
    scaler_age = StandardScaler()
    
    # La colonne 'Traveler Age' a été renommée et nettoyée dans les étapes précédentes.
    age_scaled = scaler_age.fit_transform(user_df[['Traveler Age']].values)

    # Construction du tenseur X (Features User)
    user_x_tensors = [
        # Features Catégorielles Mappées (IDs numériques)
        torch.tensor(user_df['gender_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['nationality_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['profile_type_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['climate_pref_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['acc_pref_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['transport_modes_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['continent_id'].values, dtype=torch.float).unsqueeze(1),
        
        # Feature Numérique (Âge normalisé)
        torch.tensor(age_scaled, dtype=torch.float)
    ]
    
    user_x = torch.cat(user_x_tensors, dim=1)
    
    # Gestion du padding/override pour s'assurer que la matrice a la taille attendue par l'Embedding Layer
    if user_x.size(0) < num_users_to_check:
        pad = torch.zeros((num_users_to_check - user_x.size(0), user_x.size(1)), dtype=user_x.dtype)
        user_x = torch.cat([user_x, pad], dim=0)

    # ----------------------------------------------------
    # 2. NODE DESTINATION (Source : df_travel)
    # ----------------------------------------------------
    # Utilise df_travel pour la destination
    dest_df = df_travel[['dest_id']].drop_duplicates('dest_id').sort_values('dest_id')
    
    if num_dests_override is None:
        assert len(dest_df) == dest_df['dest_id'].max() + 1, "Erreur critique : Les Dest IDs ne sont pas contigus !"
    else:
        if dest_df['dest_id'].max() >= num_dests_override:
            raise AssertionError("Some dest ids are out of range of num_dests_override")

    # Features Destination (Vecteur identité)
    dest_x = torch.ones((len(dest_df), 1), dtype=torch.float)
    if num_dests_override is not None and dest_x.size(0) < num_dests_override:
        pad = torch.zeros((num_dests_override - dest_x.size(0), dest_x.size(1)), dtype=dest_x.dtype)
        dest_x = torch.cat([dest_x, pad], dim=0)

    # ----------------------------------------------------
    # 3. EDGES & 4. ATTRIBUTES (Source : df_travel)
    # ----------------------------------------------------
    # 3. EDGES
    src = torch.tensor(df_travel['uid'].values, dtype=torch.long)
    dst = torch.tensor(df_travel['dest_id'].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # 4. EDGE ATTRIBUTES
    # Catégoriels 
    edge_attr_cat = torch.stack([
        torch.tensor(df_travel['acc_id'].values, dtype=torch.long),
        torch.tensor(df_travel['trans_id'].values, dtype=torch.long),
        torch.tensor(df_travel['season_id'].values, dtype=torch.long)
    ], dim=1)

    # Numériques 
    scaler_edges = StandardScaler()
    # Utiliser les noms de colonnes dans df_travel après nettoyage
    edge_nums = df_travel[['Accommodation cost', 'Transportation cost', 'Duration (days)']].values
    edge_attr_num = torch.tensor(scaler_edges.fit_transform(edge_nums), dtype=torch.float)

    # ----------------------------------------------------
    # 5. CRÉATION DE L'OBJET HETERODATA
    # ----------------------------------------------------
    data = HeteroData()

    # Noeuds
    data['user'].num_nodes = len(user_df) 
    data['user'].x = user_x

    data['destination'].num_nodes = len(dest_df)
    data['destination'].x = dest_x

    # Arêtes (User -> Visits -> Destination)
    data['user', 'visits', 'destination'].edge_index = edge_index
    data['user', 'visits', 'destination'].edge_attr_cat = edge_attr_cat
    data['user', 'visits', 'destination'].edge_attr_num = edge_attr_num

    # Arêtes inverses (Destination -> Rev_Visits -> User)
    data['destination', 'rev_visits', 'user'].edge_index = torch.stack([dst, src], dim=0)

    print(f"[OK] Graphe construit : {data}")
    return data