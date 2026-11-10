"""
kg_utils.py
------------
Loads a biomedical knowledge graph (Hetionet or PrimeKG edge list) and
computes fixed-size node embeddings via Node2Vec. These embeddings give
the model biological context (shared enzymes/pathways/targets) beyond
what pure molecular structure can provide.

Usage:
    python kg_utils.py --kg data/raw/hetionet-v1.0-edges.sif \
                        --output data/processed/kg_embeddings.pt
"""

import argparse
import networkx as nx
import torch
from node2vec import Node2Vec


def load_hetionet_edges(path: str) -> nx.Graph:
    """
    Hetionet .sif format: source \t metaedge \t target
    Builds an undirected graph; metaedge type is discarded here for
    simplicity, but could be kept as an edge attribute for a
    relation-aware GNN as a future-work extension.
    """
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            source, metaedge, target = parts
            G.add_edge(source, target, relation=metaedge)
    return G


def compute_node_embeddings(G: nx.Graph, dimensions=64, walk_length=30,
                             num_walks=100, workers=4):
    """
    Runs Node2Vec over the KG. Returns a dict: {node_id: embedding_tensor}
    """
    node2vec = Node2Vec(
        G, dimensions=dimensions, walk_length=walk_length,
        num_walks=num_walks, workers=workers, quiet=False
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = {}
    for node in G.nodes():
        try:
            embeddings[node] = torch.tensor(model.wv[node], dtype=torch.float)
        except KeyError:
            # Node didn't appear in any walk (rare, disconnected node)
            embeddings[node] = torch.zeros(dimensions)

    return embeddings


def get_drug_kg_embedding(drug_id: str, embeddings: dict, dim=64):
    """
    Looks up a drug's KG embedding by ID. Returns a zero vector if the
    drug isn't present in the KG (common — not every drug is annotated).
    Document this fallback explicitly as a limitation in the paper.
    """
    return embeddings.get(drug_id, torch.zeros(dim))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg', required=True, help='Path to KG edge file (.sif)')
    parser.add_argument('--output', required=True, help='Path to save embeddings .pt file')
    parser.add_argument('--dimensions', type=int, default=64)
    args = parser.parse_args()

    print("Loading knowledge graph...")
    G = load_hetionet_edges(args.kg)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Computing Node2Vec embeddings (this can take a while on large KGs)...")
    embeddings = compute_node_embeddings(G, dimensions=args.dimensions)

    torch.save(embeddings, args.output)
    print(f"Saved {len(embeddings)} node embeddings to {args.output}")
