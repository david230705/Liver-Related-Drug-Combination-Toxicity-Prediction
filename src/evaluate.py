"""
evaluate.py
------------
Loads every checkpoint in a directory (full model + each ablation
variant), evaluates them on the same held-out test set, and produces a
comparison table (AUC-ROC, F1, Precision, Recall) ready to paste into
the paper.

Usage:
    python evaluate.py --checkpoints outputs/checkpoints/ \
                        --data data/processed/pairs.pt \
                        --kg data/processed/kg_embeddings.pt \
                        --output outputs/results/comparison_table.csv
"""

import argparse
import glob
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from dataset import DrugPairDataset, collate_fn
from model import DrugPairToxicityModel


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DrugPairToxicityModel(
        atom_feat_dim=ckpt['atom_feat_dim'], **ckpt['config']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['test_indices']


@torch.no_grad()
def get_predictions(model, loader, device):
    all_preds, all_labels = [], []
    for graphs_a, graphs_b, kg_a, kg_b, organ_idx, labels in loader:
        graphs_a, graphs_b = graphs_a.to(device), graphs_b.to(device)
        kg_a, kg_b = kg_a.to(device), kg_b.to(device)
        organ_idx = organ_idx.to(device)

        preds = model(graphs_a, graphs_b, kg_a, kg_b, organ_idx)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def compute_metrics(preds, labels, threshold=0.5):
    binary_preds = [1 if p >= threshold else 0 for p in preds]
    return {
        'AUC-ROC': roc_auc_score(labels, preds),
        'F1': f1_score(labels, binary_preds),
        'Precision': precision_score(labels, binary_preds),
        'Recall': recall_score(labels, binary_preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', required=True, help='Directory containing .pt checkpoints')
    parser.add_argument('--data', required=True)
    parser.add_argument('--kg', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    records = torch.load(args.data)
    kg_embeddings = torch.load(args.kg)
    dataset = DrugPairDataset(records, kg_embeddings)

    results = []
    checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoints, '*.pt')))

    for ckpt_path in checkpoint_paths:
        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"Evaluating {name}...")

        model, test_indices = load_model(ckpt_path, device)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=32, collate_fn=collate_fn)

        preds, labels = get_predictions(model, test_loader, device)
        metrics = compute_metrics(preds, labels)
        metrics['model'] = name
        results.append(metrics)

        print(f"  {metrics}")

    df = pd.DataFrame(results)[['model', 'AUC-ROC', 'F1', 'Precision', 'Recall']]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved comparison table to {args.output}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
