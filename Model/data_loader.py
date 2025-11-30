import pandas as pd
import os
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from Python.Backend import recup_data  # Ton module d'import


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

    return df


def load_and_process_data(csv_path=None):
    """
    Étape 2 : Chargement, Filtrage des NaN, et Création des IDs (Mapping).
    C'est la fonction principale à appeler.
    """
    # 1. Chargement
    if csv_path is not None and os.path.exists(csv_path):
        # Load travel data from CSV file
        df_travel = pd.read_csv(csv_path)
        # Users may not be available via CSV; try to extract users from the csv if present
        df_users = None
        # Normalize column names like we do in process_new_data so we can find columns consistently
        cols = {c: c.strip() for c in df_travel.columns}
        lower_cols = {c.lower().strip().replace(' ', '_'): c for c in df_travel.columns}
        if 'traveler_name' in lower_cols and 'user_id' in lower_cols:
            users_df = df_travel[[lower_cols['user_id'], lower_cols['traveler_name']]].drop_duplicates()
            users_df.columns = ['user_id', 'traveler_name']
            df_users = users_df
    else:
        df_users = recup_data.recup_users()
        df_travel = recup_data.recup_travel()

    # 2. Pré-traitement (Nettoyage texte/dates)
    df = process_new_data(df_users, df_travel)

    initial_len = len(df)

    # 3. NETTOYAGE NUMÉRIQUE (CRUCIAL AVANT LE MAPPING)
    num_cols = ['Accommodation cost', 'Transportation cost', 'Duration (days)', 'Traveler age']
    # Ensure list elements are strings in case of unexpected transformations
    num_cols = [str(x) for x in num_cols]

    for col in num_cols:
        col = str(col)
        # If any expected numeric column is missing, fill with zeros to avoid KeyError
        if col not in df.columns:
            df[col] = 0
        # If the column exists but maps to a DataFrame slice (rare), reduce to a single Series
        val = df[col]
        if isinstance(val, pd.DataFrame):
            if val.shape[1] >= 1:
                # Take the first sub-column if multiple; preserve name
                val = val.iloc[:, 0]
            else:
                val = val.squeeze()
            df[col] = val
        # Nettoyage des symboles monétaires si restants
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
        # Conversion
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Suppression des lignes avec des valeurs manquantes dans les colonnes critiques
    df.dropna(subset=num_cols + ['Destination', 'Traveler name'], inplace=True)

    # Conversion float32 pour PyTorch
    for col in num_cols:
        df[col] = df[col].astype('float32')

    print(f"Nettoyage : {initial_len} -> {len(df)} lignes conservées pour l'entraînement.")

    if len(df) == 0:
        raise ValueError("❌ Erreur : Plus aucune donnée après le nettoyage (dropna). Vérifiez vos données sources.")

    # 4. MAPPING ET CRÉATION DES IDs (0 à N, sans trous)
    mappings = {}

    # Helper pour créer un mapping
    def create_mapping(column_name, mapping_key):
        # Accept either canonical column names or lowercased keys (e.g., 'Traveler name' or 'traveler_name')
        if column_name not in df.columns and column_name.lower() in df.columns:
            column_name = column_name.lower()
        unique_vals = df[column_name].unique()
        # Création du dico {Nom: ID}
        mapp = {name: i for i, name in enumerate(unique_vals)}
        mappings[mapping_key] = mapp
        return df[column_name].map(mapp)

    # Application des mappings
    df['uid'] = create_mapping('Traveler name', 'User')
    df['dest_id'] = create_mapping('Destination', 'Destination')
    df['acc_id'] = create_mapping('Accommodation type', 'Accommodation type')
    df['trans_id'] = create_mapping('Transportation type', 'Transportation type')
    df['gender_id'] = create_mapping('Traveler gender', 'Traveler gender')
    df['nationality_id'] = create_mapping('Traveler nationality', 'Traveler nationality')
    df['season_id'] = create_mapping('travel_season', 'season')

    # On peut aussi mapper le pays si on veut l'utiliser plus tard
    df['dest_country_id'] = create_mapping('dest_country', 'Country')

    print("[OK] Données prêtes et IDs générés.")
    return df, mappings


def build_graph(df, num_users_override: int = None, num_dests_override: int = None):
    """
    Étape 3 : Conversion du DataFrame propre en HeteroData (PyTorch Geometric).
    """
    print("--- [Data Engineer] Construction du Graphe ---")

    # 1. NODE USER
    # On récupère les attributs uniques par utilisateur
    # On trie par uid pour être sûr que l'index 0 correspond à l'utilisateur 0
    user_cols = ['uid', 'gender_id', 'nationality_id', 'Traveler age']
    user_df = df[user_cols].drop_duplicates('uid').sort_values('uid')

    # Sécurité : Vérifier que les UIDs sont soit contigus, or fit within the provided override.
    if num_users_override is None:
        assert len(user_df) == user_df['uid'].max() + 1, "Erreur critique : Les User IDs ne sont pas contigus !"
    else:
        # Ensure no uid exceeds override range
        if user_df['uid'].max() >= num_users_override:
            raise AssertionError("Some user ids are out of range of num_users_override")

    # Normalisation de l'âge
    scaler_age = StandardScaler()
    age_scaled = scaler_age.fit_transform(user_df[['Traveler age']].values)

    # Construction du tenseur X (Features User)
    # On utilise unsqueeze(1) pour avoir des dimensions (N, 1) et pouvoir concaténer
    user_x = torch.cat([
        torch.tensor(user_df['gender_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['nationality_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(age_scaled, dtype=torch.float)
    ], dim=1)
    # Expand user_x to num_users_override if requested
    if num_users_override is not None and user_x.size(0) < num_users_override:
        pad = torch.zeros((num_users_override - user_x.size(0), user_x.size(1)), dtype=user_x.dtype)
        user_x = torch.cat([user_x, pad], dim=0)

    # 2. NODE DESTINATION
    dest_df = df[['dest_id']].drop_duplicates('dest_id').sort_values('dest_id')
    if num_dests_override is None:
        assert len(dest_df) == dest_df['dest_id'].max() + 1, "Erreur critique : Les Dest IDs ne sont pas contigus !"
    else:
        if dest_df['dest_id'].max() >= num_dests_override:
            raise AssertionError("Some dest ids are out of range of num_dests_override")

    # Features Destination (Pour l'instant un vecteur de 1, ou embedding identité)
    dest_x = torch.ones((len(dest_df), 1), dtype=torch.float)
    if num_dests_override is not None and dest_x.size(0) < num_dests_override:
        pad = torch.zeros((num_dests_override - dest_x.size(0), dest_x.size(1)), dtype=dest_x.dtype)
        dest_x = torch.cat([dest_x, pad], dim=0)

    # 3. EDGES (Les voyages)
    src = torch.tensor(df['uid'].values, dtype=torch.long)
    dst = torch.tensor(df['dest_id'].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # 4. EDGE ATTRIBUTES (Contexte du voyage)
    # Catégoriels (Type logement, Transport, Saison)
    edge_attr_cat = torch.stack([
        torch.tensor(df['acc_id'].values, dtype=torch.long),
        torch.tensor(df['trans_id'].values, dtype=torch.long),
        torch.tensor(df['season_id'].values, dtype=torch.long)
    ], dim=1)

    # Numériques (Coûts, Durée) -> Normalisés
    scaler_edges = StandardScaler()
    edge_nums = df[['Accommodation cost', 'Transportation cost', 'Duration (days)']].values
    edge_attr_num = torch.tensor(scaler_edges.fit_transform(edge_nums), dtype=torch.float)

    # 5. CRÉATION DE L'OBJET HETERODATA
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

    # Arêtes inverses (Destination -> Rev_Visits -> User) - Pour le message passing bidirectionnel
    data['destination', 'rev_visits', 'user'].edge_index = torch.stack([dst, src], dim=0)

    print(f"[OK] Graphe construit : {data}")
    return data