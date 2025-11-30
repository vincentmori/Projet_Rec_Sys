import os
import json
import tempfile
from Model.plot_metrics import plot_metrics


def test_plot_metrics_creates_file(tmp_path):
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    # create sample metrics
    metrics = [
        {'epoch': 1, 'train_loss': 1.0, 'val_loss': 0.9, 'val_ndcg': 0.1},
        {'epoch': 2, 'train_loss': 0.8, 'val_loss': 0.7, 'val_ndcg': 0.2},
        {'epoch': 3, 'train_loss': 0.5, 'val_loss': 0.6, 'val_ndcg': 0.25},
    ]
    metrics_path = artifacts / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    out = str(artifacts / 'metrics.png')
    plot_metrics(str(artifacts), out)
    assert os.path.exists(out)
