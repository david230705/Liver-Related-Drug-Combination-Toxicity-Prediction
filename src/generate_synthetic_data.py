"""
generate_synthetic_data.py
----------------------------
Creates a small toy dataset (real, valid SMILES strings for common drugs,
paired up randomly with synthetic labels) so you can run the ENTIRE
pipeline end-to-end and confirm nothing is broken before touching real
TWOSIDES/PrimeKG data. Always run this first.

Usage:
    python generate_synthetic_data.py
"""

import random
import pandas as pd
import os

# A handful of real, valid drug SMILES (well-known drugs) for realistic testing
SAMPLE_DRUGS = {
    'aspirin':      'CC(=O)OC1=CC=CC=C1C(=O)O',
    'ibuprofen':    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'paracetamol':  'CC(=O)NC1=CC=C(O)C=C1',
    'caffeine':     'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'warfarin':     'CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O',
    'simvastatin':  'CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12',
    'metformin':    'CN(C)C(=N)NC(=N)N',
    'omeprazole':   'CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC',
    'atorvastatin': 'CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4',
    'amiodarone':   'CCCCC1=C(C2=CC=CC=C2O1)C(=O)C3=CC(=C(C(=C3)I)OCCN(CC)CC)I',
}

ORGANS = ['liver', 'heart', 'kidney']


def generate(n_pairs=200, output_path='data/processed/synthetic_pairs.csv'):
    drug_names = list(SAMPLE_DRUGS.keys())
    rows = []

    for _ in range(n_pairs):
        a, b = random.sample(drug_names, 2)
        organ = random.choice(ORGANS)
        # Synthetic label: just random for pipeline-testing purposes only.
        # Real labels must come from TWOSIDES/FAERS — see README Section 2.
        label = random.choice([0, 1])

        rows.append({
            'drug_a_id': a,
            'drug_b_id': b,
            'drug_a_smiles': SAMPLE_DRUGS[a],
            'drug_b_smiles': SAMPLE_DRUGS[b],
            'organ': organ,
            'label': label,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic drug-pair rows -> {output_path}")
    print("Reminder: these labels are RANDOM and only for testing that the "
          "pipeline runs end-to-end. Replace with real TWOSIDES data before "
          "training a model you intend to report results from.")


if __name__ == '__main__':
    generate()
