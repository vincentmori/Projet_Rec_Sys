"""
Plot training and validation metrics saved during training.

Usage:
    python -m Model.plot_metrics --artifacts_dir Model/artifacts --output Model/artifacts/metrics.png
"""
import argparse
import json
import os
import matplotlib.pyplot as plt


def plot_metrics(artifacts_dir: str, output_path: str = None):
    metrics_path = os.path.join(artifacts_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found in {artifacts_dir}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    val_ndcg = [m['val_ndcg'] if m.get('val_ndcg') is not None else float('nan') for m in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:blue')
    ax1.plot(epochs, val_loss, label='Val Loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_ndcg, label='Val NDCG@10', color='tab:green')
    ax2.set_ylabel('Val NDCG@10')
    ax2.tick_params(axis='y')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Training Progress: Loss & NDCG@10')
    plt.tight_layout()

    out = output_path or os.path.join(artifacts_dir, 'metrics.png')
    plt.savefig(out)
    print(f"Saved metrics plot to {out}")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from artifacts')
    parser.add_argument('--artifacts_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'artifacts'))
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    plot_metrics(args.artifacts_dir, args.output)


if __name__ == '__main__':
    main()
