from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from typing import Annotated
from fastapi.responses import JSONResponse
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import pickle
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

app = FastAPI()




with open('model_solubility.pkl', 'rb') as f:
    solubility_model = pickle.load(f)

with open('drug_likeness.pkl', 'rb') as f:
    drug_model = pickle.load(f)

with open('drug_scaler.pkl', 'rb') as f:
    drug_scaler = pickle.load(f)

tox_model = None




class ToxicityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=1908, output_dim=24, dropout=0.1798):
        super(ToxicityNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x





def morgan_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr




def report_toxicity_pytorch(smiles, model, threshold=0.5):
    fp = morgan_fingerprint(smiles)
    if fp is None:
        return "Invalid SMILES string."

    x = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]
        preds = (probs >= threshold).astype(int)

    tasks = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    assay_info = {
        "NR-AR": "Possible interference with androgen receptors (hormone disruption).",
        "NR-AR-LBD": "Binding to androgen receptor ligand domain (affects hormone signaling).",
        "NR-AhR": "Activation of aryl hydrocarbon receptor (potential carcinogen response).",
        "NR-Aromatase": "Affects estrogen synthesis (endocrine disruption).",
        "NR-ER": "Interaction with estrogen receptor (hormone disruption).",
        "NR-ER-LBD": "Binding to estrogen receptor ligand domain.",
        "NR-PPAR-gamma": "Affects metabolism and inflammation pathways.",
        "SR-ARE": "Activates antioxidant response (oxidative stress).",
        "SR-ATAD5": "DNA damage repair pathway activation (genotoxic stress).",
        "SR-HSE": "Heat shock protein activation (cellular stress response).",
        "SR-MMP": "Mitochondrial toxicity (energy metabolism disruption).",
        "SR-p53": "p53 protein activation (DNA damage and apoptosis)."
    }

    report_lines = [f"Toxicity Report for molecule: {smiles}", "-" * 40]
    any_toxic = False

    for i, task in enumerate(tasks):
        prob = probs[i]
        pred = preds[i]
        if pred == 1:
            any_toxic = True
            report_lines.append(f" [{task}] Toxicity Likely (Probability: {prob:.2f})")
            report_lines.append(f"    ↳ {assay_info.get(task, 'No description available.')}")
    if not any_toxic:
        report_lines.append(" No significant toxicity predicted.")

    return "\n".join(report_lines)




@app.on_event("startup")
def load_toxicity_model():
    global tox_model
    state_dict = torch.load("tox21_model_state.pt", map_location="cpu")

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key == "layers.0.weight":
            new_state_dict["fc1.weight"] = value
        elif key == "layers.0.bias":
            new_state_dict["fc1.bias"] = value
        elif key == "output_layer.weight":
            new_state_dict["fc2.weight"] = value
        elif key == "output_layer.bias":
            new_state_dict["fc2.bias"] = value

    model = ToxicityNet(input_dim=1024, hidden_dim=1908, output_dim=24, dropout=0.1798)
    model.load_state_dict(new_state_dict)
    model.eval()

    tox_model = model
    print("Toxicity model loaded successfully!")




def get_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RingCount': Descriptors.RingCount(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'Fsp3': Descriptors.FractionCSP3(mol)
    }

def feature_extract(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'Invalid Smiles'
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "BalabanJ": Descriptors.BalabanJ(mol)
    }

class Molecular_var(BaseModel):
    smiles: Annotated[str, Field(..., description='SMILES of the molecule', example='CCO')]

    @validator('smiles')
    def valid_smiles(cls, v):
        mol = Chem.MolFromSmiles(v)
        if mol is None:
            raise ValueError('Invalid SMILES string')
        return v



@app.get("/")
def root():
    return {"message": "Molecular predictor API is running"}

@app.post('/predictSolubility', tags=['predict'])
def predict_solubility(smiles: Molecular_var):
    details = get_descriptors(smiles.smiles)
    if details is None:
        return JSONResponse(status_code=400, content={'error': 'Invalid SMILES'})

    params = [
        details['MolWt'], details['LogP'], details['NumHDonors'],
        details['NumHAcceptors'], details['TPSA'], details['RingCount'],
        details['NumRotatableBonds'], details['Fsp3']
    ]

    value = solubility_model.predict([params])

    return JSONResponse(status_code=200, content={
        'smiles': smiles.smiles,
        'predicted_solubility': round(float(value[0]), 2),
        'descriptors': details
    })



@app.post('/predictTox', tags=['predict'])
def predict_toxicity(smiles: Molecular_var):
    if tox_model is None:
        return JSONResponse(status_code=500, content={'error': 'Toxicity model not loaded'})

    report = report_toxicity_pytorch(smiles.smiles, tox_model)
    return JSONResponse(status_code=200, content={'toxicity_report': report})

@app.post('/predictdrug', tags=['predict'])
def predict_drug(smiles: Molecular_var):
    desc = feature_extract(smiles.smiles)

    param = [desc['MolWt'], desc['LogP'], desc['TPSA'], desc['NumHDonors'], desc['NumHAcceptors'],
             desc['NumRotatableBonds'], desc['RingCount'], desc['HeavyAtomCount'],
             desc['FractionCSP3'], desc['BalabanJ']]

    scaled = drug_scaler.transform([param])
    prediction = drug_model.predict_proba(scaled)

    return JSONResponse(status_code=200, content={
        'Drug Class': int(np.argmax(prediction)),           
        'Drug Probability': float(prediction[0][1])        
    })
