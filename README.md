# Drug Combination Organ-Specific Toxicity Prediction

## Overview
An AI-based system that predicts potential liver-related toxicity risks associated with two-drug combinations. It combines molecular structure analysis and biological knowledge to estimate toxicity probability.

## Technologies Used
- Python
- PyTorch and PyTorch Geometric
- RDKit
- Graph Convolutional Network (GCN)
- Cross-Attention
- Hetionet Knowledge Graph
- FastAPI
- HTML, CSS and JavaScript

## Methodology
1. Obtain drug combination data from the TWOSIDES dataset.
2. Retrieve molecular structures from PubChem.
3. Convert molecular structures into graphs using RDKit.
4. Extract molecular features using GCN.
5. Incorporate biological information using Hetionet embeddings.
6. Combine features using cross-attention and an MLP.
7. Predict liver-related toxicity probability.

## Results
| Evaluation | AUC-ROC |
|---|---:|
| Random Test | 0.7821 |
| Cold-Drug Test | 0.6111 |

## Web Application
A FastAPI-based web application allows users to enter two medicines and view the predicted liver-related toxicity probability.

## Disclaimer
This project is intended for research and early-stage screening purposes only. It is not a substitute for medical advice or clinical evaluation.

## Sample Output:
<img width="925" height="710" alt="Screenshot 2026-09-22 223620" src="https://github.com/user-attachments/assets/280f50e4-d18a-46df-a5a0-8c7e1ffee6b4" />
<img width="912" height="825" alt="Screenshot 2026-09-22 223644" src="https://github.com/user-attachments/assets/d8addb2c-0ecd-4c50-8cef-001137645fca" />

