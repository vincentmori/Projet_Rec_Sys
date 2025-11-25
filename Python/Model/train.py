# train.py
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
import os
import json
import pickle

# Imports locaux (Modularité)
from model import HGIB_Context_Model
from data_loader import load_and_process_data, build_graph

# ==========================================
# HYPERPARAMÈTRES (MLE-Ops Responsibility)
# ==========================================
CSV_FILE = 'Travel_details_ready.csv'
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 32
LEARNING_RATE = 0.0001
EPOCHS = 1000
BETA = 0.001
ARTIFACTS_DIR = "artifacts"

# Seuil de tolérance pour l'arrêt (Delta)
EARLY_STOPPING_DELTA = 0.01  # Si val_loss > min_val_loss + 0.3, on arrête


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


def main():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- [MLE-Ops] Training Start on {device} ---")

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    # 2. Chargement des données
    if not os.path.exists(CSV_FILE):
        print(f"ERREUR: Fichier {CSV_FILE} introuvable.")
        return

    df, mappings = load_and_process_data(CSV_FILE)
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

    for epoch in range(1, EPOCHS + 1):
        train_loss, recons_loss = train_one_epoch(model, train_data, optimizer)
        val_loss = evaluate(model, val_data)

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
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Min: {min_val_loss:.4f})")

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
        "num_season": num_season
    }
    config_path = os.path.join(ARTIFACTS_DIR, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Config sauvegardée : {config_path}")


if __name__ == "__main__":
    main()