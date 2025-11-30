"""
Evaluation harness for offline metrics (NDCG@k).

Usage:
    python -m Model.eval --artifacts_dir Model/artifacts --csv_path ../Data/Travel_details_ready.csv --k 10

This script:
 - loads model artifacts (config, mappings, weights)
 - loads CSV and re-applies saved mappings
 - samples a per-user test edge (leave-one-out) and keeps the rest as train edges
 - computes embeddings using the model encoded on train-only graph
 - scores all destination candidates per user and computes NDCG@k

It's implemented to avoid retraining and to be deterministic and reproducible.
"""
import argparse
import json
import os
import pickle
from copy import deepcopy
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch

from Model.predict import load_artifacts
from Model.data_loader import load_and_process_data, build_graph
from Model.model import HGIB_Context_Model


def _choose_test_edges(df: pd.DataFrame, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each user with >=2 interactions, sample one as test (leave-one-out per user).
    Users with only one interaction are left with that interaction in train only.
    Returns (train_df, test_df).
    """
    rng = np.random.RandomState(random_seed)
    df = df.copy()
    groups = df.groupby('uid')
    test_idxs = []
    for uid, group in groups:
        if len(group) >= 2:
            idx = rng.choice(group.index.values)
            test_idxs.append(idx)
    test_df = df.loc[test_idxs].copy()
    train_df = df.drop(test_idxs).copy()
    return train_df, test_df


def _dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute DCG@k for a single binary relevance vector (1 for relevant, 0 otherwise)
    where the vector is ordered by predicted ranking (descending relevance).
    """
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    # DCG formula for binary relevance: sum(rel_i / log2(i+2)) with i starting at 0
    discounts = 1.0 / np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance * discounts))


def _idcg_at_k(num_relevant: int, k: int) -> float:
    n = min(num_relevant, k)
    if n <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    return float(np.sum(discounts))


def compute_ndcg_at_k_for_saved_model(artifacts_dir: str, csv_path: str, k: int = 10, device: str = 'cpu') -> float:
    # Load artifacts
    config, maps, model_path = load_artifacts(artifacts_dir)

    # Load and process CSV, then overwrite uids/dest_ids with saved mappings for alignment
    df, _ = load_and_process_data(csv_path)
    # Apply saved mappings if present
    if 'User' in maps and 'Traveler name' in df.columns:
        df['uid'] = df['Traveler name'].map(maps['User'])
    if 'Destination' in maps and 'Destination' in df.columns:
        df['dest_id'] = df['Destination'].map(maps['Destination'])

    # Map categorical/contextual columns to saved mapping vocabulary to ensure consistency with the model
    if 'Accommodation type' in maps and 'Accommodation type' in df.columns:
        df['acc_id'] = df['Accommodation type'].map(maps['Accommodation type'])
    if 'Transportation type' in maps and 'Transportation type' in df.columns:
        df['trans_id'] = df['Transportation type'].map(maps['Transportation type'])
    if 'Traveler gender' in maps and 'Traveler gender' in df.columns:
        df['gender_id'] = df['Traveler gender'].map(maps['Traveler gender'])
    if 'Traveler nationality' in maps and 'Traveler nationality' in df.columns:
        df['nationality_id'] = df['Traveler nationality'].map(maps['Traveler nationality'])
    if 'season' in maps and 'travel_season' in df.columns:
        df['season_id'] = df['travel_season'].map(maps['season'])

    # Drop rows with user/destination values not present in saved mappings and warn
    missing_user_rows = df['uid'].isna().sum()
    missing_dest_rows = df['dest_id'].isna().sum()
    if missing_user_rows > 0 or missing_dest_rows > 0:
        print(f"[WARNING] Dropping {missing_user_rows} rows with unknown users and {missing_dest_rows} rows with unknown destinations to align with saved mappings.")
        df = df.dropna(subset=['uid', 'dest_id']).copy()

    # Drop rows where contextual categorical values are not present in saved mappings (avoid OOB indices)
    cat_columns_to_check = ['acc_id', 'trans_id', 'gender_id', 'nationality_id', 'season_id']
    missing_cat_total = 0
    for c in cat_columns_to_check:
        if c in df.columns:
            missing_cat = df[c].isna().sum()
            if missing_cat > 0:
                print(f"[WARNING] Dropping {missing_cat} rows with unknown categorical mapping in column {c}.")
                df = df.dropna(subset=[c]).copy()
                missing_cat_total += missing_cat

    df['uid'] = df['uid'].astype(int)
    df['dest_id'] = df['dest_id'].astype(int)

    # Build the full graph (for node features)
    num_users = len(maps['User'])
    num_dests = len(maps['Destination'])
    full_graph = build_graph(df, num_users_override=num_users, num_dests_override=num_dests)

    # Split train/test per user
    train_df, test_df = _choose_test_edges(df)

    # Create train_graph by copying the node features from full_graph and replacing edges
    train_graph = deepcopy(full_graph)
    src = torch.tensor(train_df['uid'].values, dtype=torch.long)
    dst = torch.tensor(train_df['dest_id'].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    # build edge attrs compatible with full_graph arrays; try to reuse edge attribute columns
    # prepare categorical edge attributes if present
    def _get_edge_attr_cols(df_local):
        cat_cols = []
        num_cols = []
        if 'acc_id' in df_local.columns:
            cat_cols.append('acc_id')
        if 'trans_id' in df_local.columns:
            cat_cols.append('trans_id')
        if 'season_id' in df_local.columns:
            cat_cols.append('season_id')
        if 'Accommodation cost' in df_local.columns and 'Transportation cost' in df_local.columns and 'Duration (days)' in df_local.columns:
            num_cols = ['Accommodation cost', 'Transportation cost', 'Duration (days)']
        return cat_cols, num_cols

    cat_cols, num_cols = _get_edge_attr_cols(train_df)
    # Build edge attr tensors
    if len(cat_cols) > 0:
        edge_attr_cat = torch.stack([torch.tensor(train_df[c].values, dtype=torch.long) for c in cat_cols], dim=1)
    else:
        # fallback to empty tensor with shape (num_edges, 0)
        edge_attr_cat = torch.zeros((edge_index.size(1), 0), dtype=torch.long)
    if len(num_cols) > 0:
        # normalize using train_df only (this is a simple approach; more robust would reuse scaler)
        nums = train_df[num_cols].fillna(0).values
        # Avoid depending on sklearn at runtime here
        nums_mean = nums.mean(axis=0)
        nums_std = np.std(nums, axis=0) + 1e-9
        nums_norm = (nums - nums_mean) / nums_std
        edge_attr_num = torch.tensor(nums_norm, dtype=torch.float)
    else:
        edge_attr_num = torch.zeros((edge_index.size(1), 0), dtype=torch.float)

    train_graph['user', 'visits', 'destination'].edge_index = edge_index
    train_graph['user', 'visits', 'destination'].edge_attr_cat = edge_attr_cat
    train_graph['user', 'visits', 'destination'].edge_attr_num = edge_attr_num
    # update reverse edges too
    train_graph['destination', 'rev_visits', 'user'].edge_index = torch.stack([dst, src], dim=0)

    # Load model
    device = torch.device(device)
    num_users = len(maps['User'])
    num_dests = len(maps['Destination'])
    model = HGIB_Context_Model(
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        metadata=full_graph.metadata(),
        num_acc=config.get('num_acc', 1),
        num_trans=config.get('num_trans', 1),
        num_season=config.get('num_season', 1),
        num_users=num_users,
        num_dests=num_dests
    ).to(device)
    try:
        state = torch.load(model_path, map_location=device)
    except Exception:
        # Try again with weights_only=False for newer PyTorch versions or when the checkpoint needs full pickling
        state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # Encode embeddings on train_graph
    x_dict = train_graph.x_dict
    edge_index_dict = train_graph.edge_index_dict
    edge_attr_cat_dict = {('user', 'visits', 'destination'): train_graph['user', 'visits', 'destination'].edge_attr_cat}
    edge_attr_num_dict = {('user', 'visits', 'destination'): train_graph['user', 'visits', 'destination'].edge_attr_num}
    with torch.no_grad():
        mu, _ = model.encode(x_dict, edge_index_dict, edge_attr_cat_dict, edge_attr_num_dict)

    z_user = mu['user'].cpu().numpy()
    z_dest = mu['destination'].cpu().numpy()
    num_dests = z_dest.shape[0]

    # Build train edge set per user to exclude from ranking
    train_edges = set(zip(train_df['uid'].astype(int).tolist(), train_df['dest_id'].astype(int).tolist()))
    test_edges = set(zip(test_df['uid'].astype(int).tolist(), test_df['dest_id'].astype(int).tolist()))

    ndcgs = []
    users_with_test = sorted(test_df['uid'].unique())
    for uid in users_with_test:
        # Skip if uid is not in range
        if uid < 0 or uid >= num_users:
            continue
        # compute scores (dot product) w/ numpy
        scores = z_user[uid].dot(z_dest.T)
        # mask train visited destinations
        visited_mask = np.zeros(num_dests, dtype=bool)
        for dest in range(num_dests):
            if (uid, dest) in train_edges:
                visited_mask[dest] = True
        scores_masked = scores.copy()
        scores_masked[visited_mask] = -np.inf
        # ranking indices
        ranked = np.argsort(scores_masked)[::-1]
        # binary relevance vector: 1 if dest is in test set for this user
        test_set = set([d for (u, d) in test_edges if u == uid])
        rel = np.array([1 if idx in test_set else 0 for idx in ranked], dtype=int)
        dcg = _dcg_at_k(rel, k)
        idcg = _idcg_at_k(len(test_set), k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    # global mean NDCG
    if len(ndcgs) == 0:
        return 0.0
    return float(np.mean(ndcgs))


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved model offline (NDCG@k)')
    parser.add_argument('--artifacts_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'artifacts'))
    parser.add_argument('--csv_path', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'Travel_details_ready.csv')))
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ndcg = compute_ndcg_at_k_for_saved_model(args.artifacts_dir, args.csv_path, k=args.k, device=args.device)
    print(f"Mean NDCG@{args.k}: {ndcg:.6f}")


if __name__ == '__main__':
    main()
