import importlib.util
import pytest
import os

spec = importlib.util.find_spec('torch')
if spec is None:
    pytest.skip("Torch not installed; skipping evaluation tests.", allow_module_level=True)

from Model.eval import compute_ndcg_at_k_for_saved_model
from Model.data_loader import load_and_process_data, build_graph
from Model.model import HGIB_Context_Model
import torch
import json
import pickle


def test_ndcg_range_and_print():
    artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'synthetic_travel_data_daily_cost_coherent.csv'))
    # The function may raise FileNotFoundError if artifacts missing; let that fail here
    ndcg = compute_ndcg_at_k_for_saved_model(artifacts_dir, csv_path=csv_path, k=10)
    print(f"Computed NDCG@10: {ndcg}")
    assert 0.0 <= ndcg <= 1.0


def test_ndcg_toy_dataset(tmp_path):
    # Build small toy dataset with 4 users and 6 destinations
    csv_path = tmp_path / 'toy.csv'
    rows = [
        # user U1 visits dest A,B
        {'Trip ID': 'T1', 'User ID': 'UX1', 'Destination': 'A, TC', 'Start date': '2023-01-01', 'End date': '2023-01-03', 'Duration (days)': 2, 'Traveler name': 'Alice', 'Traveler age': 30, 'Traveler gender': 'Female', 'Traveler nationality': 'American', 'Accommodation type': 'Hotel', 'Accommodation cost': 100, 'Transportation type': 'Plane', 'Transportation cost': 50},
        {'Trip ID': 'T2', 'User ID': 'UX1', 'Destination': 'B, TC', 'Start date': '2023-02-01', 'End date': '2023-02-02', 'Duration (days)': 1, 'Traveler name': 'Alice', 'Traveler age': 30, 'Traveler gender': 'Female', 'Traveler nationality': 'American', 'Accommodation type': 'Hostel', 'Accommodation cost': 50, 'Transportation type': 'Train', 'Transportation cost': 20},
        # user U2 visits A, C
        {'Trip ID': 'T3', 'User ID': 'UX2', 'Destination': 'A, TC', 'Start date': '2023-03-01', 'End date': '2023-03-03', 'Duration (days)': 2, 'Traveler name': 'Bob', 'Traveler age': 40, 'Traveler gender': 'Male', 'Traveler nationality': 'British', 'Accommodation type': 'Hotel', 'Accommodation cost': 110, 'Transportation type': 'Plane', 'Transportation cost': 60},
        {'Trip ID': 'T4', 'User ID': 'UX2', 'Destination': 'C, TC', 'Start date': '2023-04-01', 'End date': '2023-04-02', 'Duration (days)': 1, 'Traveler name': 'Bob', 'Traveler age': 40, 'Traveler gender': 'Male', 'Traveler nationality': 'British', 'Accommodation type': 'Hostel', 'Accommodation cost': 55, 'Transportation type': 'Train', 'Transportation cost': 25},
        # user U3 visits D only (single visit)
        {'Trip ID': 'T5', 'User ID': 'UX3', 'Destination': 'D, TC', 'Start date': '2023-05-01', 'End date': '2023-05-03', 'Duration (days)': 2, 'Traveler name': 'Carol', 'Traveler age': 35, 'Traveler gender': 'Female', 'Traveler nationality': 'French', 'Accommodation type': 'Airbnb', 'Accommodation cost': 90, 'Transportation type': 'Bus', 'Transportation cost': 10},
        # user U4 visits E, F
        {'Trip ID': 'T6', 'User ID': 'UX4', 'Destination': 'E, TC', 'Start date': '2023-06-01', 'End date': '2023-06-05', 'Duration (days)': 4, 'Traveler name': 'Dave', 'Traveler age': 45, 'Traveler gender': 'Male', 'Traveler nationality': 'German', 'Accommodation type': 'Hotel', 'Accommodation cost': 200, 'Transportation type': 'Plane', 'Transportation cost': 120},
        {'Trip ID': 'T7', 'User ID': 'UX4', 'Destination': 'F, TC', 'Start date': '2023-07-01', 'End date': '2023-07-04', 'Duration (days)': 3, 'Traveler name': 'Dave', 'Traveler age': 45, 'Traveler gender': 'Male', 'Traveler nationality': 'German', 'Accommodation type': 'Hostel', 'Accommodation cost': 70, 'Transportation type': 'Train', 'Transportation cost': 30},
    ]
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Build processed DF and mappings using the data loader function
    df_proc, maps = load_and_process_data(str(csv_path))
    # Create temp artifacts dir
    artifacts_dir = tmp_path / 'artifacts'
    artifacts_dir.mkdir()

    # Save mappings and config (small toy config)
    cfg = {
        'hidden_channels': 16,
        'out_channels': 8,
        'num_acc': len(maps.get('Accommodation type', {})),
        'num_trans': len(maps.get('Transportation type', {})),
        'num_season': len(maps.get('season', {})),
    }
    with open(artifacts_dir / 'config.json', 'w') as f:
        json.dump(cfg, f)
    with open(artifacts_dir / 'mappings.pkl', 'wb') as f:
        pickle.dump(maps, f)

    # Build graph & save a dummy model with random weights
    g = build_graph(df_proc)
    num_users = len(maps['User'])
    num_dests = len(maps['Destination'])
    model = HGIB_Context_Model(
        hidden_channels=cfg['hidden_channels'],
        out_channels=cfg['out_channels'],
        metadata=g.metadata(),
        num_acc=cfg['num_acc'],
        num_trans=cfg['num_trans'],
        num_season=cfg['num_season'],
        num_users=num_users,
        num_dests=num_dests
    )
    torch.manual_seed(0)
    # Save state
    torch.save(model.state_dict(), str(artifacts_dir / 'hgib_model.pth'))

    # Run compute_ndcg on the toy CSV
    ndcg = compute_ndcg_at_k_for_saved_model(str(artifacts_dir), str(csv_path), k=10)
    print('Toy NDCG@10:', ndcg)
    assert 0.0 <= ndcg <= 1.0
