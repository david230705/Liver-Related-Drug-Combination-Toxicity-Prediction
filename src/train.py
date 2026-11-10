"""
train.py
---------
Training loop with ablation switches so you can produce the ablation
table required for the paper (full model vs. no-KG vs. no-attention vs.
no-organ-embedding).

Usage:
    python train.py --data data/processed/pairs.pt \
                     --kg data/processed/kg_embeddings.pt \
                     --organ liver --epochs 50 --batch_size 32 \
                     --ablation full \
                     --output outputs/checkpoints/model_full.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import DrugPairDataset, collate_fn
from model import DrugPairToxicityModel

ABLATION_CONFIGS = {
    'full':            dict(use_kg=True,  use_attention=True,  use_organ_embed=True),
    'no_kg':           dict(use_kg=False, use_attention=True,  use_organ_embed=True),
    'no_attention':    dict(use_kg=True,  use_attention=False, use_organ_embed=True),
    'no_organ_embed':  dict(use_kg=True,  use_attention=True,  use_organ_embed=False),
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for graphs_a, graphs_b, kg_a, kg_b, organ_idx, labels in tqdm(loader, desc='train', leave=False):
        graphs_a, graphs_b = graphs_a.to(device), graphs_b.to(device)
        kg_a, kg_b = kg_a.to(device), kg_b.to(device)
        organ_idx, labels = organ_idx.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(graphs_a, graphs_b, kg_a, kg_b, organ_idx)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for graphs_a, graphs_b, kg_a, kg_b, organ_idx, labels in loader:
        graphs_a, graphs_b = graphs_a.to(device), graphs_b.to(device)
        kg_a, kg_b = kg_a.to(device), kg_b.to(device)
        organ_idx, labels = organ_idx.to(device), labels.to(device)

        preds = model(graphs_a, graphs_b, kg_a, kg_b, organ_idx)
        loss = criterion(preds, labels)
        total_loss += loss.item() * labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader.dataset), all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--kg', required=True)
    parser.add_argument('--organ', default='liver')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ablation', choices=list(ABLATION_CONFIGS.keys()), default='full')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    records = torch.load(args.data)
    kg_embeddings = torch.load(args.kg)

    dataset = DrugPairDataset(records, kg_embeddings)

    # 80/10/10 train/val/test split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)

    atom_feat_dim = records[0]['graph_a'].x.shape[1]
    config = ABLATION_CONFIGS[args.ablation]

    model = DrugPairToxicityModel(atom_feat_dim=atom_feat_dim, **config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'atom_feat_dim': atom_feat_dim,
                'test_indices': test_set.indices,
            }, args.output)
            print(f"  -> saved new best checkpoint to {args.output}")

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == '__main__':
    main()
