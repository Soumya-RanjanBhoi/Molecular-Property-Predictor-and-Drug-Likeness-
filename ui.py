import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import requests
import pandas as pd
from io import StringIO, BytesIO
import py3Dmol

API_URL = "http://127.0.0.1:8000"   

st.set_page_config(page_title="Molecule Analyzer", layout="wide")

st.title("🧪 Molecular Property Predictor")

smiles = st.text_input("Enter SMILES string:", value="CCO", help="Type a valid SMILES (e.g., CCO for ethanol)")

def optimize_structure(smiles):
    """Generate an optimized 3D structure using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "Invalid SMILES"
    
    mol = Chem.AddHs(mol) 
    params = AllChem.ETKDGv3()
    AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol) 
        return mol, "Optimization successful "
    except:
        return mol, "Optimization failed, but 3D coords generated "

def render_3d(mol):
    """Render molecule in an interactive 3D viewer using py3Dmol."""
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    return viewer

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.image(Draw.MolToImage(mol, size=(300, 300)), caption="2D Structure")

        tab1, tab2 = st.tabs(["🔄 Structure Optimization", "📊 Predictions"])

        with tab1:
            if st.button("Optimize Structure"):
                with st.spinner("Optimizing structure..."):
                    optimized_mol, msg = optimize_structure(smiles)
                    if optimized_mol:
                        st.success(msg)
                        st.image(Draw.MolToImage(optimized_mol, size=(300, 300)), caption="Optimized Structure")

                        st.subheader("🌀 Interactive 3D View")
                        viewer = render_3d(optimized_mol)
                        viewer_html = viewer._make_html()
                        st.components.v1.html(viewer_html, height=450)

                    
                        sdf_buffer = StringIO()
                        writer = Chem.SDWriter(sdf_buffer)
                        writer.write(optimized_mol)
                        writer.close()

                
                        sdf_bytes = sdf_buffer.getvalue().encode("utf-8")

                        st.download_button("📥 Download Optimized Structure (SDF)",
                                           data=sdf_bytes,
                                           file_name="optimized_structure.sdf",
                                           mime="chemical/x-mdl-sdfile")

        with tab2:
            if st.button("Run Predictions"):
                data = {"smiles": smiles}
                with st.spinner("Predicting..."):
                    sol = requests.post(f"{API_URL}/predictSolubility", json=data).json()
                    drug = requests.post(f"{API_URL}/predictdrug", json=data).json()
                    tox = requests.post(f"{API_URL}/predictTox", json=data).json()

                st.subheader("🔹 Predicted Solubility")
                st.write(f"**Solubility:** {sol['predicted_solubility']}")
                st.write("**Descriptors:**")
                st.dataframe(pd.DataFrame(sol["descriptors"].items(), columns=["Feature", "Value"]))

                st.subheader("🔹 Drug-likeness")
                st.write(f"**Class:** {'Likely Drug' if drug['Drug Class'] == 1 else 'Non-drug'}")
                st.write(f"**Probability:** {drug['Drug Probability']:.3f}")

                st.subheader("🔹 Toxicity Report")
                st.code(tox["toxicity_report"])


                st.subheader("⬇ Download Report")
                report_text = f"""SMILES: {smiles}\n----------------------------------------\nSolubility: {sol['predicted_solubility']}\n\nDrug-likeness:\n - Class: {'Likely Drug' if drug['Drug Class'] == 1 else 'Non-drug'}\n - Probability: {drug['Drug Probability']:.3f}\n\nToxicity Report:\n{tox['toxicity_report']}\n"""

                buf = BytesIO()
                buf.write(report_text.encode())
                buf.seek(0)
                st.download_button("📄 Download TXT Report", data=buf, file_name="molecule_report.txt", mime="text/plain")
    else:
        st.error("❌ Invalid SMILES string. Please check your input.")