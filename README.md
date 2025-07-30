# 🧪 Molecular Property Predictor & Drug-Likeness Optimizer

A **web-based chemistry tool** that allows researchers and students to input **SMILES strings** of molecules and receive predictions on:
- **Solubility** 🌊
- **Toxicity** ☣️
- **Drug-likeness** 💊

The tool also provides:
✅ **2D & 3D visualizations** using **RDKit**  
✅ **Downloadable reports & structures (SDF)**  

Built with:
- **FastAPI** for backend (ML inference, structure optimization)  
- **Streamlit** for frontend UI  
- **RDKit** for cheminformatics  
- **DeepChem** for loading datasets and model training  
- **ML models** (LightGBM, RandomForest, DeepChem’s MultitaskClassifier, PyTorch Neural Networks) trained on open datasets like **ESOL**, **Tox21**, and a custom dataset containing their SMILES and binary activity labels (0 = inactive, 1 = active).

---

## 🚀 Features
- 🔬 **Input SMILES** and get predictions for solubility, toxicity, and drug-likeness.  
- ⚗ **2D Visualization** using RDKit & **3D Structure Optimization** with MMFF.  
- 📦 **Download Reports** (TXT) & optimized structures (SDF).  

---

## 🛠 Tech Stack
- **Backend:** FastAPI (Python)  
- **Frontend:** Streamlit  
- **ML Models:** LightGBM, RandomForest, DeepChem’s MultitaskClassifier, PyTorch Neural Networks  
- **Cheminformatics:** RDKit  
- **Reinforcement Learning Module:** DeepChem   
- **Datasets:** ESOL, Tox21 (from MoleculeNet), custom labeled dataset  

---

