"""
Lightweight inference module to provide simple `get_recommendation(user_id)` API

This script loads saved artifacts (config, mappings, model weights), rebuilds the PyG
HeteroData graph from the database (or optionally a CSV file), computes deterministic 
embeddings and returns top-K destination names for a given user.
"""
import os
import json
import pickle
from typing import List, Union, Optional

import pandas as pd
import streamlit as st
import torch

from Model.model import HGIB_Context_Model
from Model.data_loader import load_and_process_data, build_graph


DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
# CSV path is now optional - when None, data is loaded from database
DEFAULT_CSV = None


def load_artifacts(artifacts_dir: str = None):
    artifacts_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    config_path = os.path.join(artifacts_dir, 'config.json')
    mappings_path = os.path.join(artifacts_dir, 'mappings.pkl')
    model_path = os.path.join(artifacts_dir, 'hgib_model.pth')

    if not os.path.exists(config_path) or not os.path.exists(mappings_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"One or more artifacts not found in {artifacts_dir}. Make sure training saved artifacts there.")

    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    return config, mappings, model_path


def map_external_id_to_traveler_name(external_user_id: str, csv_path: Optional[str] = None) -> Optional[str]:
    """Map an external 'User ID' (like 'U0001') to the Traveler name.

    Returns the traveler name if found, otherwise None.
    This helper is lightweight and doesn't require the model or torch, making it
    suitable for backend integration and unit testing.
    
    If csv_path is provided, loads from CSV. Otherwise loads from database.
    """
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Load from database
        from Python.Backend import recup_data
        df = recup_data.recup_travel()
    
    key_col = 'User ID' if 'User ID' in df.columns else 'user_id'
    name_col = 'Traveler name' if 'Traveler name' in df.columns else 'traveler_name'
    row = df[df[key_col].astype(str) == str(external_user_id)]
    if len(row) == 0:
        return None
    return row.iloc[0][name_col]


def recommend_by_external_user_id(external_user_id: str, top_k: int = 3, artifacts_dir: str = None, csv_path: str = None) -> List[str]:
    """Convenience wrapper that resolves an external user id (U0001) to a Traveler name
    then calls the core `get_recommendation` function.
    """
    traveler_name = map_external_id_to_traveler_name(external_user_id, csv_path=csv_path)
    if traveler_name is None:
        raise ValueError(f"No traveler found for external user id: {external_user_id}")
    return get_recommendation(traveler_name, top_k=top_k, artifacts_dir=artifacts_dir, csv_path=csv_path)


class Predictor:
    """Initialize once, uses CPU by default. Loads model and data on first use.
    Use in a server to keep the model loaded in memory.
    """

    def __init__(self, artifacts_dir: str = None, csv_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.artifacts_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
        self.csv_path = csv_path or DEFAULT_CSV
        self.device = device or torch.device('cpu')
        self._loaded = False

    def load(self):
        config, static_mappings, model_path = load_artifacts(self.artifacts_dir)

        df_travel_clean, df_users_clean, dynamic_maps = load_and_process_data()
    
        maps = static_mappings.copy()

        # 2. Fusion des Mappings (User et Destination doivent être les versions dynamiques)
        if 'User' in dynamic_maps:
            maps['User'] = dynamic_maps['User']
        if 'Destination' in dynamic_maps:
            maps['Destination'] = dynamic_maps['Destination']

        # 3. Nettoyage final des NaNs sur les Arêtes (df_travel_clean)
        required_ids = ['uid', 'dest_id', 'acc_id', 'trans_id', 'season_id']
        required_ids = [c for c in required_ids if c in df_travel_clean.columns]
        
        initial_len = len(df_travel_clean)
        df_travel_clean.dropna(subset=required_ids, inplace=True)
        if len(df_travel_clean) < initial_len:
            print(f"[WARN] {initial_len - len(df_travel_clean)} lignes supprimées car elles contenaient des IDs inconnus (nouvelle catégorie non vue lors de l'entraînement).")

        data = build_graph(df_travel_clean, df_users_clean)

        num_users = len(maps['User'])
        num_dests = len(maps['Destination'])

        model = HGIB_Context_Model(
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            metadata=data.metadata(),
            num_acc=config['num_acc'],
            num_trans=config['num_trans'],
            num_season=config['num_season'],
            num_users=num_users,
            num_dests=num_dests
        ).to(self.device)

        try:
            state = torch.load(model_path, map_location=self.device)
        except Exception:
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            
        # Capture le dictionnaire de poids actuel du modèle (taille complète, initialisée).
        model_dict = model.state_dict() 
        
        # Extrait les embeddings entraînés pour les utilisateurs historiques depuis le checkpoint.
        user_emb_checkpoint = state['user_emb.weight'] 
        
        # Obtient une référence modifiable à la matrice d'embeddings utilisateur actuelle.
        user_emb_current = model_dict['user_emb.weight'] 
        
        # INJECTION CRITIQUE : Copie les poids entraînés dans les premières lignes de la nouvelle matrice.
        user_emb_current[:user_emb_checkpoint.size(0), :] = user_emb_checkpoint 
    
        # Met à jour le dictionnaire de poids du modèle avec la matrice fusionnée.     
        model_dict['user_emb.weight'] = user_emb_current 
        
        # === 2. Gestion de dest_emb (si la taille change) ===
        if 'dest_emb.weight' in state and state['dest_emb.weight'].shape != model_dict['dest_emb.weight'].shape:
            dest_emb_checkpoint = state['dest_emb.weight']
            dest_emb_current = model_dict['dest_emb.weight']
            
            # Remplacer les N premières lignes par les poids entraînés
            dest_emb_current[:dest_emb_checkpoint.size(0), :] = dest_emb_checkpoint
            model_dict['dest_emb.weight'] = dest_emb_current
            
        # === 3. Chargement Final ===
        # Charger les poids dans le modèle, en utilisant strict=False pour ignorer les autres incohérences 
        # (par exemple, si la taille des biais a changé)
        model.load_state_dict(model_dict, strict=False)

        model.eval()

        self.model = model
        self.data = data
        self.maps = maps
        self._loaded = True

    def recommend(self, user_identifier: Union[int, str], top_k: int = 5) -> List[str]:
        if not self._loaded:
            self.load()

        # resolve user id
        if isinstance(user_identifier, str):
            if user_identifier not in self.maps['User']:
                raise ValueError(f"User ID '{user_identifier}' not found in mappings")
            uid = self.maps['User'][user_identifier]
        else:
            uid = int(user_identifier)
            if uid < 0 or uid >= len(self.maps['User']):
                raise ValueError(f"User id {uid} is out of range (0..{len(self.maps['User'])-1})")

        x_dict = self.data.x_dict
        edge_index_dict = self.data.edge_index_dict
        edge_attr_cat_dict = {('user', 'visits', 'destination'): self.data['user', 'visits', 'destination'].edge_attr_cat}
        edge_attr_num_dict = {('user', 'visits', 'destination'): self.data['user', 'visits', 'destination'].edge_attr_num}

        with torch.no_grad():
            mu, logstd = self.model.encode(x_dict, edge_index_dict, edge_attr_cat_dict, edge_attr_num_dict)

        z_user = mu['user']
        z_dest = mu['destination']

        num_dests = z_dest.size(0)
        edge_label_index = torch.stack([
            torch.tensor([uid] * num_dests, dtype=torch.long),
            torch.arange(num_dests, dtype=torch.long)
        ])
        scores = self.model.decode(z_user, z_dest, edge_label_index).cpu().numpy()

        # compute visited destinations for the user
        ei = self.data['user', 'visits', 'destination'].edge_index
        srcs = ei[0].cpu().numpy()
        dsts = ei[1].cpu().numpy()
        visited = set(dsts[srcs == uid].tolist())

        # filter and rank
        candidates = [(i, float(s)) for i, s in enumerate(scores) if i not in visited]
        candidates.sort(key=lambda x: x[1], reverse=True)

        id_to_dest = {v: k for k, v in self.maps['Destination'].items()}
        top = candidates[:top_k]
        recommended_names = [id_to_dest[d] for d, _ in top]
        return recommended_names

_GLOBAL_PREDICTOR: Optional[Predictor] = None

def reset_predictor():
    """Forces the global predictor to be reloaded on next call to get_recommendation."""
    global _GLOBAL_PREDICTOR
    _GLOBAL_PREDICTOR = None
    print("[INFO] Predictor has been reset and will reload data on next call.")


def get_recommendation(user_identifier: Union[int, str], top_k: int = 5, artifacts_dir: str = None, csv_path: str = None) -> List[str]:
    """Simple wrapper function used by backend (SE)."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = Predictor(artifacts_dir=artifacts_dir, csv_path=csv_path)
    # If artifacts/csv_path were passed in earlier, the Predictor will honor them on first load
    return _GLOBAL_PREDICTOR.recommend(user_identifier, top_k=top_k)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python predict.py <user_id_or_name> [top_k]')
        sys.exit(1)
    user = sys.argv[1]
    try:
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    except Exception:
        top_k = 5
    try:
        uid = int(user)
    except Exception:
        uid = user
    recs = get_recommendation(uid, top_k=top_k)
    print('Recommendations:', recs)
