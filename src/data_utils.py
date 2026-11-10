"""
data_utils.py
--------------
Converts drug SMILES strings into molecular graphs (atoms = nodes,
bonds = edges) using RDKit, and caches them as PyTorch tensors for
fast loading during training.

Usage:
    python data_utils.py --input data/raw/twosides_liver_filtered.csv \
                          --output data/processed/pairs.pt
"""

import argparse
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops

# ---------------------------------------------------------------------
# Atom feature vocabulary — kept small and interpretable on purpose.
# Each atom becomes a one-hot-ish feature vector built from these fields.
# ---------------------------------------------------------------------
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'UNK']
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]


def one_hot(value, allowed):
    """Standard one-hot encoding with an UNK fallback bucket."""
    if value not in allowed:
        value = allowed[-1]
    return [1.0 if value == a else 0.0 for a in allowed]


def atom_features(atom):
    """
    Build a feature vector per atom:
    [element one-hot] + [hybridization one-hot] + [degree, charge, aromatic, H-count]
    """
    features = []
    features += one_hot(atom.GetSymbol(), ATOM_LIST)
    features += one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)
    features += [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        1.0 if atom.GetIsAromatic() else 0.0,
        atom.GetTotalNumHs(),
    ]
    return features


def smiles_to_graph(smiles: str) -> Data:
    """
    Parses a SMILES string into a PyG Data object.
    Returns None if the SMILES is invalid (caller should skip these rows).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge index (bonds), stored both directions for an undirected graph
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = adj.nonzero()
    edge_index = torch.tensor(
        [edge_index[0].tolist(), edge_index[1].tolist()], dtype=torch.long
    )

    return Data(x=x, edge_index=edge_index, smiles=smiles)


def build_pair_dataset(csv_path: str):
    """
    Expects a CSV with columns: drug_a_smiles, drug_b_smiles, organ, label
    Returns a list of dicts: {graph_a, graph_b, organ, label}
    Rows with unparseable SMILES are dropped and counted.
    """
    df = pd.read_csv(csv_path)
    records = []
    skipped = 0

    for _, row in df.iterrows():
        graph_a = smiles_to_graph(row['drug_a_smiles'])
        graph_b = smiles_to_graph(row['drug_b_smiles'])

        if graph_a is None or graph_b is None:
            skipped += 1
            continue

        records.append({
            'graph_a': graph_a,
            'graph_b': graph_b,
            'drug_a_id': row.get('drug_a_id', row['drug_a_smiles']),
            'drug_b_id': row.get('drug_b_id', row['drug_b_smiles']),
            'organ': row['organ'],
            'label': float(row['label']),
        })

    print(f"Parsed {len(records)} drug pairs, skipped {skipped} invalid SMILES.")
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output', required=True, help='Path to save processed .pt file')
    args = parser.parse_args()

    dataset = build_pair_dataset(args.input)
    torch.save(dataset, args.output)
    print(f"Saved {len(dataset)} processed pairs to {args.output}")
