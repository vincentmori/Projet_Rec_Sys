# train.py
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
import os
import numpy as np
import json
import pickle

# Imports locaux (Modularité)
from Model.model import HGIB_Context_Model
from Model.data_loader import load_and_process_data, build_graph

# ==========================================
# HYPERPARAMÈTRES (MLE-Ops Responsibility)
# ==========================================
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 32
LEARNING_RATE = 0.0004
EPOCHS = 250
BETA = 0.001
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Seuil de tolérance pour l'arrêt (Delta)
EARLY_STOPPING_DELTA = 0.01  # Si val_loss > min_val_loss + 0.01, on arrête


def train_one_epoch(model, train_data, optimizer):
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    pred, mu, logstd = model(train_data)
    target = train_data['user', 'visits', 'destination'].edge_label

    # 1. Reconstruction Loss (BCE)
    recons_loss = F.binary_cross_entropy_with_logits(pred, target)

    # 2. KL Divergence (Bottleneck)
    kl_loss = 0
    for key in mu.keys():
        kl_loss += -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd[key] - mu[key] ** 2 - logstd[key].exp() ** 2, dim=1)
        )

    # Loss Totale
    loss = recons_loss + (BETA * kl_loss)

    loss.backward()
    optimizer.step()

    return loss.item(), recons_loss.item()


@torch.no_grad()
def evaluate(model, val_data):
    model.eval()
    pred, mu, logstd = model(val_data)
    target = val_data['user', 'visits', 'destination'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return loss.item()


def compute_ndcg_for_split(model, train_data, val_data, k=10):
    """Compute mean NDCG@k using embeddings trained on train_data and evaluation on val_data.

    Excludes train edges from ranking candidate lists.
    """
    model.eval()
    # Encode on train_data (avoid leakage)
    x_dict = train_data.x_dict
    edge_index_dict = train_data.edge_index_dict
    edge_attr_cat_dict = {('user', 'visits', 'destination'): train_data['user', 'visits', 'destination'].edge_attr_cat}
    edge_attr_num_dict = {('user', 'visits', 'destination'): train_data['user', 'visits', 'destination'].edge_attr_num}
    with torch.no_grad():
        mu, _ = model.encode(x_dict, edge_index_dict, edge_attr_cat_dict, edge_attr_num_dict)

    z_user = mu['user'].cpu().numpy()
    z_dest = mu['destination'].cpu().numpy()
    num_dests = z_dest.shape[0]

    # Build train edge set for masking
    ei_train = train_data['user', 'visits', 'destination'].edge_index
    src_train = ei_train[0].cpu().numpy()
    dst_train = ei_train[1].cpu().numpy()
    train_edges = set(zip(src_train.tolist(), dst_train.tolist()))

    # get validation edges (positive only)
    ei_val = val_data['user', 'visits', 'destination'].edge_index
    labels = val_data['user', 'visits', 'destination'].edge_label
    src_val = ei_val[0].cpu().numpy()
    dst_val = ei_val[1].cpu().numpy()
    labels_np = labels.cpu().numpy()
    # Collect positive val edges per user
    val_per_user = {}
    for u, d, lab in zip(src_val.tolist(), dst_val.tolist(), labels_np.tolist()):
        if lab == 1:
            val_per_user.setdefault(u, set()).add(d)

    ndcgs = []
    for uid, test_set in val_per_user.items():
        # compute scores
        scores = z_user[uid].dot(z_dest.T)
        # mask train visited
        mask = np.zeros(num_dests, dtype=bool)
        for dest in range(num_dests):
            if (uid, dest) in train_edges:
                mask[dest] = True
        scores_masked = scores.copy()
        scores_masked[mask] = -np.inf
        ranked = np.argsort(scores_masked)[::-1]
        rel = np.array([1 if idx in test_set else 0 for idx in ranked], dtype=int)
        # compute DCG/IDCG
        discounts = 1.0 / np.log2(np.arange(2, min(len(rel), k) + 2))
        dcg = float(np.sum(rel[:k] * discounts)) if rel.size > 0 else 0.0
        idcg_n = min(len(test_set), k)
        if idcg_n == 0:
            ndcg = 0.0
        else:
            idcg = float(np.sum(1.0 / np.log2(np.arange(2, idcg_n + 2))))
            ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    if len(ndcgs) == 0:
        return 0.0
    return float(np.mean(ndcgs))


def train_main():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- [MLE-Ops] Training Start on {device} ---")

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    df, mappings = load_and_process_data()
    data = build_graph(df)

    # 3. Split Train/Val/Test
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        edge_types=[('user', 'visits', 'destination')],
        rev_edge_types=[('destination', 'rev_visits', 'user')],
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # 4. Initialisation du Modèle
    num_acc = len(mappings['Accommodation type'])
    num_trans = len(mappings['Transportation type'])
    num_season = len(mappings['season'])
    num_users = data['user'].num_nodes
    num_dests = data['destination'].num_nodes

    model = HGIB_Context_Model(
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        metadata=data.metadata(),
        num_acc=num_acc,
        num_trans=num_trans,
        num_season=num_season,
        num_users=num_users,
        num_dests=num_dests
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Boucle d'Entraînement avec Early Stopping
    print(f"--- Début de l'entraînement pour {EPOCHS} époques ---")
    print(f"--- Critère d'arrêt : Val Loss > Min Val Loss + {EARLY_STOPPING_DELTA} ---")

    min_val_loss = float('inf')
    best_model_state = None  # Pour garder le meilleur cerveau en mémoire

    metrics = []
    for epoch in range(1, EPOCHS + 1):
        train_loss, recons_loss = train_one_epoch(model, train_data, optimizer)
        val_loss = evaluate(model, val_data)
        # Compute NDCG@10 using the current model and train/val split
        try:
            val_ndcg = compute_ndcg_for_split(model, train_data, val_data, k=10)
        except Exception as e:
            print(f"[WARN] NDCG computation failed on epoch {epoch}: {e}")
            val_ndcg = float('nan')

        # --- LOGIQUE EARLY STOPPING ---

        # Cas 1 : On trouve un nouveau record (le modèle s'améliore)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model_state = model.state_dict()  # On sauvegarde cet état précieux
            # On pourrait afficher un petit message de "Nouveau record" ici si on veut

        # Cas 2 : Le modèle diverge trop (Critère d'arrêt demandé)
        elif val_loss > min_val_loss + EARLY_STOPPING_DELTA:
            print(f"\n🛑 ARRÊT ANTICIPÉ (Epoch {epoch})")
            print(
                f"   Raison : Val Loss ({val_loss:.4f}) a explosé de +{val_loss - min_val_loss:.4f} par rapport au minimum ({min_val_loss:.4f}).")
            break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Min: {min_val_loss:.4f}) | Val NDCG@10: {val_ndcg:.4f}")

        # Append metrics
        metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_ndcg': val_ndcg
        })

    print("--- Entraînement terminé ---")

    # 6. Sauvegarde des Artefacts
    print("--- Sauvegarde du MEILLEUR modèle et des mappings ---")

    # IMPORTANT : On recharge le meilleur état (celui du minimum) avant de sauvegarder
    # Sinon on sauvegarderait le modèle "cassé" qui vient d'exploser
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Meilleur modèle restauré (Val Loss: {min_val_loss:.4f})")
    else:
        print("⚠️ Attention : Aucun meilleur modèle trouvé (bizarre).")

    model_path = os.path.join(ARTIFACTS_DIR, "hgib_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé : {model_path}")

    mappings_path = os.path.join(ARTIFACTS_DIR, "mappings.pkl")
    with open(mappings_path, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"Mappings sauvegardés : {mappings_path}")

    config = {
        "hidden_channels": HIDDEN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "num_acc": num_acc,
        "num_trans": num_trans,
        "num_season": num_season,
        "num_users": num_users,
        "num_dests": num_dests
    }
    config_path = os.path.join(ARTIFACTS_DIR, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Config sauvegardée : {config_path}")
    # Save metrics to artifacts
    metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics.json')
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics saved: {metrics_path}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")


if __name__ == "__main__":
    train_main()