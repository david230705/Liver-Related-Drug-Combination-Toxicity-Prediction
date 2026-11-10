import os
# ============================================================
# DRUG TOXICITY PREDICTION - FASTAPI BACKEND
# ============================================================

import sys
import torch
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torch_geometric.data import Batch

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_PATH = "/content/drive/MyDrive/Drug-Toxicity-Project"
SRC_PATH = PROJECT_PATH + "/src"

sys.path.append(SRC_PATH)

# ------------------------------------------------------------
# IMPORT PROJECT MODULES
# ------------------------------------------------------------

from data_utils import smiles_to_graph
from model import DrugPairToxicityModel

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ------------------------------------------------------------
# LOAD DRUG SMILES
# ------------------------------------------------------------

SMILES_FILE = (
    PROJECT_PATH +
    "/data/processed/validated_drug_smiles.csv"
)

smiles_df = pd.read_csv(SMILES_FILE)

smiles_df["drug_id"] = smiles_df["drug_id"].astype(str)

smiles_dict = dict(
    zip(
        smiles_df["drug_id"],
        smiles_df["smiles"]
    )
)

print("Loaded drug SMILES:", len(smiles_dict))

# ------------------------------------------------------------
# LOAD KG EMBEDDINGS
# ------------------------------------------------------------

KG_FILE = (
    PROJECT_PATH +
    "/data/processed/kg_embeddings_rxnorm.pt"
)

kg_embeddings = torch.load(
    KG_FILE,
    map_location="cpu",
    weights_only=False
)

print("Loaded KG embeddings:", len(kg_embeddings))

# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

CHECKPOINT = (
    PROJECT_PATH +
    "/outputs/checkpoints/full_model_cold_best.pt"
)

model = DrugPairToxicityModel(
    atom_feat_dim=19,
    kg_embed_dim=64,
    num_organs=5,
    hidden_dim=128,
    use_kg=True,
    use_attention=True,
    use_organ_embed=True
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------
# FASTAPI
# ------------------------------------------------------------

app = FastAPI(
    title="Drug Combination Toxicity Prediction API"
)

# ------------------------------------------------------------
# REQUEST FORMAT
# ------------------------------------------------------------

class DrugRequest(BaseModel):

    drug_a: str
    drug_b: str


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_toxicity(drug_a_id, drug_b_id):

    drug_a_id = str(drug_a_id)
    drug_b_id = str(drug_b_id)

    # Check Drug A
    if drug_a_id not in smiles_dict:

        raise ValueError(
            f"Drug A ID {drug_a_id} was not found."
        )

    # Check Drug B
    if drug_b_id not in smiles_dict:

        raise ValueError(
            f"Drug B ID {drug_b_id} was not found."
        )

    # --------------------------------------------------------
    # GET SMILES
    # --------------------------------------------------------

    smiles_a = smiles_dict[drug_a_id]
    smiles_b = smiles_dict[drug_b_id]

    # --------------------------------------------------------
    # CREATE MOLECULAR GRAPHS
    # --------------------------------------------------------

    graph_a = smiles_to_graph(smiles_a)
    graph_b = smiles_to_graph(smiles_b)

    # Convert to batches
    graph_a = Batch.from_data_list(
        [graph_a]
    ).to(device)

    graph_b = Batch.from_data_list(
        [graph_b]
    ).to(device)

    # --------------------------------------------------------
    # KG EMBEDDINGS
    # --------------------------------------------------------

    kg_a = kg_embeddings.get(
        drug_a_id,
        torch.zeros(64)
    )

    kg_b = kg_embeddings.get(
        drug_b_id,
        torch.zeros(64)
    )

    kg_a = kg_a.unsqueeze(0).to(device)
    kg_b = kg_b.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # ORGAN
    # --------------------------------------------------------

    # Liver = 0
    organ_idx = torch.tensor(
        [0],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        probability = model(
            graph_a,
            graph_b,
            kg_a,
            kg_b,
            organ_idx
        )

    probability = float(
        probability.item()
    )

    return probability


# ------------------------------------------------------------
# PREDICTION API
# ------------------------------------------------------------

@app.post("/predict")
def predict(request: DrugRequest):

    try:

        probability = predict_toxicity(
            request.drug_a,
            request.drug_b
        )

        percentage = round(
            probability * 100,
            2
        )

        if probability >= 0.5:

            risk = "Higher predicted risk"

        else:

            risk = "Lower predicted risk"

        return {

            "drug_a": request.drug_a,

            "drug_b": request.drug_b,

            "toxicity_probability": percentage,

            "risk": risk
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------



# Drug name mapping
DRUG_NAMES_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "processed",
    "rxnorm_drug_names.csv"
)

drug_names_df = pd.read_csv(DRUG_NAMES_PATH)

drug_names = dict(
    zip(
        drug_names_df["drug_id"].astype(str),
        drug_names_df["drug_name"].astype(str)
    )
)

print("Loaded drug names:", len(drug_names))


@app.get("/drugs-data")
def get_drugs():
    return [
        {
            "id": drug_id,
            "name": drug_names.get(
                str(drug_id),
                "Name unavailable"
            )
        }
        for drug_id in sorted(
            smiles_dict.keys(),
            key=lambda x: int(x)
        )
    ]

@app.get("/drugs-data")
def get_drugs():
    return [
        {
            "id": drug_id,
            "name": drug_names.get(drug_id, "Name unavailable")
        }
        for drug_id in sorted(smiles_dict.keys(), key=lambda x: int(x))
    ]


@app.get("/")
def home():
    return FileResponse(
        "/content/drive/MyDrive/Drug-Toxicity-Project/app/index.html"
    )


@app.get("/drugs")
def drugs_page():
    return FileResponse(
        "/content/drive/MyDrive/Drug-Toxicity-Project/app/drugs.html"
    )


@app.get("/drugs-data")
def drugs_data():
    return list(smiles_dict.keys())
