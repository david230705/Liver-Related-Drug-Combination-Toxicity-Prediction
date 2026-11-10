"""
dataset.py
-----------
PyTorch Dataset / collate function wrapping the processed drug-pair
records produced by data_utils.py, joined with KG embeddings from
kg_utils.py.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

ORGAN_TO_IDX = {'liver': 0, 'heart': 1, 'kidney': 2, 'lung': 3, 'brain': 4}


class DrugPairDataset(Dataset):
    def __init__(self, records, kg_embeddings, kg_dim=64):
        """
        records: list of dicts from data_utils.build_pair_dataset()
        kg_embeddings: dict from kg_utils.compute_node_embeddings()
        """
        self.records = records
        self.kg_embeddings = kg_embeddings
        self.kg_dim = kg_dim

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        kg_a = self.kg_embeddings.get(str(rec['drug_a_id']), torch.zeros(self.kg_dim))
        kg_b = self.kg_embeddings.get(str(rec['drug_b_id']), torch.zeros(self.kg_dim))

        organ_idx = ORGAN_TO_IDX.get(rec['organ'], 0)

        return {
            'graph_a': rec['graph_a'],
            'graph_b': rec['graph_b'],
            'kg_a': kg_a,
            'kg_b': kg_b,
            'organ_idx': organ_idx,
            'label': rec['label'],
        }


def collate_fn(batch):
    """
    Custom collate: batches molecular graphs with PyG's Batch object,
    stacks everything else as normal tensors.
    """
    graphs_a = Batch.from_data_list([item['graph_a'] for item in batch])
    graphs_b = Batch.from_data_list([item['graph_b'] for item in batch])
    kg_a = torch.stack([item['kg_a'] for item in batch])
    kg_b = torch.stack([item['kg_b'] for item in batch])
    organ_idx = torch.tensor([item['organ_idx'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

    return graphs_a, graphs_b, kg_a, kg_b, organ_idx, labels
