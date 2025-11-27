import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Linear


class EdgeBehaviorEncoder(nn.Module):
    """
    (Inchangé) Encode le contexte du voyage (Saison, Coûts...)
    """

    def __init__(self, num_acc, num_trans, num_season, num_numerical, hidden_dim):
        super().__init__()
        emb_dim = hidden_dim // 4
        self.emb_acc = nn.Embedding(num_acc, emb_dim)
        self.emb_trans = nn.Embedding(num_trans, emb_dim)
        self.emb_season = nn.Embedding(num_season, emb_dim)
        self.lin_num = Linear(num_numerical, emb_dim)
        self.out_lin = Linear(emb_dim * 4, hidden_dim)

    def forward(self, edge_attr_cat, edge_attr_num):
        h_acc = self.emb_acc(edge_attr_cat[:, 0])
        h_trans = self.emb_trans(edge_attr_cat[:, 1])
        h_season = self.emb_season(edge_attr_cat[:, 2])
        h_num = self.lin_num(edge_attr_num)
        h = torch.cat([h_acc, h_trans, h_season, h_num], dim=1)
        return self.out_lin(h)


class HGIB_Context_Model(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_acc, num_trans, num_season, num_users, num_dests):
        super().__init__()

        # === CORRECTION MAJEURE : IDENTITÉ DES NOEUDS ===
        # On ajoute des embeddings pour que chaque ID soit unique
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.dest_emb = nn.Embedding(num_dests, hidden_channels)

        # Projection des features existantes (Age, Genre...) pour qu'elles aient la bonne taille
        # User features: 3 colonnes (Genre, Nat, Age) -> vers hidden_channels
        self.lin_user_feat = Linear(3, hidden_channels)
        # Dest features: 1 colonne (le "ones") -> vers hidden_channels
        self.lin_dest_feat = Linear(1, hidden_channels)
        # ================================================

        # 1. Innovation (Edge Encoder)
        self.edge_encoder = EdgeBehaviorEncoder(num_acc, num_trans, num_season, 3, hidden_channels)

        # 2. GNN Layers
        self.conv1_visits = GATConv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=hidden_channels)
        self.conv1_rev = GATConv((-1, -1), hidden_channels, add_self_loops=False)

        self.conv2_visits = GATConv(hidden_channels, hidden_channels, add_self_loops=False, edge_dim=hidden_channels)
        self.conv2_rev = GATConv(hidden_channels, hidden_channels, add_self_loops=False)

        # 3. Bottleneck
        self.lin_mu = Linear(hidden_channels, out_channels)
        self.lin_logstd = Linear(hidden_channels, out_channels)

    def encode(self, x_dict, edge_index_dict, edge_attr_cat_dict, edge_attr_num_dict):
        # === PRÉPARATION DES FEATURES (Identité + Caractéristiques) ===

        # 1. On récupère les IDs (ce sont les indices de 0 à N)
        # On suppose que x_dict['user'] contient les features.
        # On va générer les IDs à la volée : 0, 1, 2... jusqu'à N_users
        device = x_dict['user'].device
        num_users_batch = x_dict['user'].size(0)
        num_dests_batch = x_dict['destination'].size(0)

        user_ids = torch.arange(num_users_batch, device=device)
        dest_ids = torch.arange(num_dests_batch, device=device)

        # 2. On combine : Embedding d'ID + Projection des Features
        # User = Qui je suis (ID) + Comment je suis (Age/Genre)
        h_user = self.user_emb(user_ids) + self.lin_user_feat(x_dict['user'])

        # Dest = Qui je suis (ID) + Features (ici vides)
        h_dest = self.dest_emb(dest_ids) + self.lin_dest_feat(x_dict['destination'])

        # ==============================================================

        # A. Encodage Comportement Arête
        edge_feat = self.edge_encoder(
            edge_attr_cat_dict[('user', 'visits', 'destination')],
            edge_attr_num_dict[('user', 'visits', 'destination')]
        )

        # B. GNN Couche 1
        # Attention : on passe h_user et h_dest maintenant, plus x_dict brut
        out_dest_1 = self.conv1_visits(
            (h_user, h_dest),
            edge_index_dict[('user', 'visits', 'destination')],
            edge_attr=edge_feat
        )
        out_user_1 = self.conv1_rev(
            (h_dest, h_user),
            edge_index_dict[('destination', 'rev_visits', 'user')]
        )

        # Activation
        h_user = out_user_1.relu()
        h_dest = out_dest_1.relu()

        # C. GNN Couche 2
        out_dest_2 = self.conv2_visits(
            (h_user, h_dest),
            edge_index_dict[('user', 'visits', 'destination')],
            edge_attr=edge_feat
        )
        out_user_2 = self.conv2_rev(
            (h_dest, h_user),
            edge_index_dict[('destination', 'rev_visits', 'user')]
        )

        # Mise à jour du dictionnaire pour le Bottleneck
        x_dict_out = {'user': out_user_2.relu(), 'destination': out_dest_2.relu()}

        # D. Bottleneck
        mu = {key: self.lin_mu(x) for key, x in x_dict_out.items()}
        logstd = {key: self.lin_logstd(x) for key, x in x_dict_out.items()}

        return mu, logstd

    # (Le reste : reparameterize, decode, forward... reste inchangé)
    def reparameterize(self, mu, logstd):
        if self.training:
            z_dict = {}
            for key in mu.keys():
                std = torch.exp(logstd[key])
                eps = torch.randn_like(std)
                z_dict[key] = mu[key] + eps * std
            return z_dict
        else:
            return mu

    def decode(self, z_user, z_dest, edge_label_index):
        user_emb = z_user[edge_label_index[0]]
        dest_emb = z_dest[edge_label_index[1]]
        return (user_emb * dest_emb).sum(dim=-1)

    def forward(self, batch):
        cat_dict = {('user', 'visits', 'destination'): batch['user', 'visits', 'destination'].edge_attr_cat}
        num_dict = {('user', 'visits', 'destination'): batch['user', 'visits', 'destination'].edge_attr_num}
        mu, logstd = self.encode(batch.x_dict, batch.edge_index_dict, cat_dict, num_dict)
        z = self.reparameterize(mu, logstd)
        pred = self.decode(z['user'], z['destination'], batch['user', 'visits', 'destination'].edge_label_index)
        return pred, mu, logstd