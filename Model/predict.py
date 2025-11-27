"""
Lightweight inference module to provide simple `get_recommendation(user_id)` API

This script loads saved artifacts (config, mappings, model weights), rebuilds the PyG
HeteroData graph from `Travel_details_ready.csv`, computes deterministic embeddings
and returns top-K destination names for a given user.
"""
import os
import json
import pickle
from typing import List, Union, Optional

import pandas as pd
import numpy as np
import torch

from model import HGIB_Context_Model
from data_loader import load_and_process_data, build_graph


DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DEFAULT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Travel_details_ready.csv'))


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
        config, mappings, model_path = load_artifacts(self.artifacts_dir)

        if self.csv_path is None or not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV dataset not found at {self.csv_path}. Provide valid csv_path to Predictor.")

        df, _ = load_and_process_data(self.csv_path)
        maps = mappings

        # Overwrite the generated IDs in df using saved mappings to prevent mismatch
        if 'Traveler name' in df.columns:
            df['uid'] = df['Traveler name'].map(maps['User'])
        if 'Destination' in df.columns:
            df['dest_id'] = df['Destination'].map(maps['Destination'])
        if 'Accommodation type' in df.columns and 'Accommodation type' in maps:
            df['acc_id'] = df['Accommodation type'].map(maps['Accommodation type'])
        if 'Transportation type' in df.columns and 'Transportation type' in maps:
            df['trans_id'] = df['Transportation type'].map(maps['Transportation type'])
        if 'Traveler gender' in df.columns and 'Traveler gender' in maps:
            df['gender_id'] = df['Traveler gender'].map(maps['Traveler gender'])
        if 'Traveler nationality' in df.columns and 'Traveler nationality' in maps:
            df['nationality_id'] = df['Traveler nationality'].map(maps['Traveler nationality'])
        if 'travel_season' in maps and 'travel_season' not in df.columns:
            df['Start date'] = pd.to_datetime(df['Start date'])
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
            df['travel_season'] = df['Start date'].apply(get_season)
        if 'travel_season' in maps:
            df['season_id'] = df['travel_season'].map(maps['season'])

        # Sanity: ensure mapping worked
        if df['uid'].isna().any():
            missing = df[df['uid'].isna()]['Traveler name'].unique()[:5]
            raise ValueError(f"Some traveler names are missing from saved mappings. Examples: {missing}")
        if df['dest_id'].isna().any():
            missing = df[df['dest_id'].isna()]['Destination'].unique()[:5]
            raise ValueError(f"Some Destination names are missing from saved mappings. Examples: {missing}")

        data = build_graph(df)

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

        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
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
                raise ValueError(f"User name '{user_identifier}' not found in mappings")
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
