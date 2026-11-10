"""
model.py
---------
Full model architecture:

  Drug A graph -> GCN encoder -> embedding_a  ┐
                                                ├─> Cross-Attention ─> interaction_vec
  Drug B graph -> GCN encoder -> embedding_b  ┘

  [interaction_vec, kg_embedding_a, kg_embedding_b, organ_embedding]
        -> Toxicity Prediction Head -> risk score (0-1)

Ablation flags let you disable individual components (--ablation flag in
train.py) to produce the ablation table required for the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# ---------------------------------------------------------------------
# 1. Molecular Graph Encoder (per-drug)
# ---------------------------------------------------------------------
class MolecularGCNEncoder(nn.Module):
    """
    3-layer GCN that turns a molecular graph into a fixed-size embedding
    via global mean pooling over atom-level representations.
    """
    def __init__(self, in_channels, hidden_channels=128, out_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        # Pool atom-level features into one vector per molecule
        return global_mean_pool(x, batch)


# ---------------------------------------------------------------------
# 2. Cross-Attention Interaction Layer
# ---------------------------------------------------------------------
class CrossAttentionInteraction(nn.Module):
    """
    Lets Drug A's embedding attend over Drug B's embedding and vice versa,
    then fuses both attended representations. This is what models the
    "do these two drugs interact" signal, rather than just concatenating
    two independent vectors.
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.attn_a_to_b = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_b_to_a = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, emb_a, emb_b):
        # MultiheadAttention expects (batch, seq_len, embed_dim); treat
        # each drug embedding as a sequence of length 1.
        a = emb_a.unsqueeze(1)
        b = emb_b.unsqueeze(1)

        attended_a, _ = self.attn_a_to_b(a, b, b)   # A attends to B
        attended_b, _ = self.attn_b_to_a(b, a, a)   # B attends to A

        attended_a = attended_a.squeeze(1)
        attended_b = attended_b.squeeze(1)

        fused = torch.cat([attended_a, attended_b], dim=-1)
        return self.fuse(fused)  # interaction vector


# ---------------------------------------------------------------------
# 3. Full Model
# ---------------------------------------------------------------------
class DrugPairToxicityModel(nn.Module):
    def __init__(self, atom_feat_dim, kg_embed_dim=64, num_organs=5,
                 hidden_dim=128, use_kg=True, use_attention=True,
                 use_organ_embed=True):
        """
        Ablation switches:
          use_kg          -> if False, KG embeddings are zeroed out
          use_attention   -> if False, interaction vec = simple concat
                              instead of cross-attention
          use_organ_embed -> if False, organ is ignored (single-organ model)
        """
        super().__init__()
        self.use_kg = use_kg
        self.use_attention = use_attention
        self.use_organ_embed = use_organ_embed

        self.encoder = MolecularGCNEncoder(atom_feat_dim, hidden_dim, hidden_dim)

        if use_attention:
            self.interaction = CrossAttentionInteraction(hidden_dim)
            interaction_out_dim = hidden_dim
        else:
            # Ablation: naive concatenation instead of learned interaction
            self.interaction = None
            interaction_out_dim = hidden_dim * 2

        self.organ_embedding = nn.Embedding(num_organs, 16) if use_organ_embed else None

        final_input_dim = interaction_out_dim
        if use_kg:
            final_input_dim += kg_embed_dim * 2   # kg_a + kg_b
        if use_organ_embed:
            final_input_dim += 16

        self.toxicity_head = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, graph_a, graph_b, kg_emb_a, kg_emb_b, organ_idx):
        emb_a = self.encoder(graph_a.x, graph_a.edge_index, graph_a.batch)
        emb_b = self.encoder(graph_b.x, graph_b.edge_index, graph_b.batch)

        if self.use_attention:
            interaction_vec = self.interaction(emb_a, emb_b)
        else:
            interaction_vec = torch.cat([emb_a, emb_b], dim=-1)

        components = [interaction_vec]

        if self.use_kg:
            components.append(kg_emb_a)
            components.append(kg_emb_b)

        if self.use_organ_embed:
            organ_vec = self.organ_embedding(organ_idx)
            components.append(organ_vec)

        final_vec = torch.cat(components, dim=-1)
        logit = self.toxicity_head(final_vec)
        return torch.sigmoid(logit).squeeze(-1)   # risk score in [0, 1]
