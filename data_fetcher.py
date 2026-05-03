"""
data_fetcher.py — Fetch drug data from ChEMBL and PubChem
"""

import requests
import time
import json
import os
import pandas as pd
from tqdm import tqdm
from config import DATA_DIR, CHEMBL_BASE_URL, PUBCHEM_BASE_URL


# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_chembl_bioactivities(target_chembl_id: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch IC50 / Ki binding affinity records for a given ChEMBL target.

    Args:
        target_chembl_id: e.g. 'CHEMBL2095' (EGFR)
        limit: maximum records

    Returns:
        DataFrame with columns: molecule_chembl_id, smiles, pchembl_value
    """
    url = (
        f"{CHEMBL_BASE_URL}/activity.json"
        f"?target_chembl_id={target_chembl_id}"
        f"&standard_type__in=IC50,Ki,Kd"
        f"&pchembl_value__isnull=false"
        f"&limit={limit}"
        f"&format=json"
    )
    print(f"  → Fetching activities for {target_chembl_id} …")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    records = []
    for act in data.get("activities", []):
        records.append({
            "molecule_chembl_id": act.get("molecule_chembl_id"),
            "smiles"            : act.get("canonical_smiles"),
            "pchembl_value"     : float(act.get("pchembl_value", 0)),
            "target_id"         : target_chembl_id,
            "assay_type"        : act.get("standard_type"),
        })
    df = pd.DataFrame(records).dropna(subset=["smiles", "pchembl_value"])
    print(f"     Retrieved {len(df)} records.")
    return df


def fetch_multiple_targets(target_ids: list[str], limit_per_target: int = 200) -> pd.DataFrame:
    """Fetch activities for a list of target IDs and combine."""
    frames = []
    for tid in tqdm(target_ids, desc="Fetching ChEMBL targets"):
        try:
            frames.append(fetch_chembl_bioactivities(tid, limit=limit_per_target))
            time.sleep(0.5)          # polite delay
        except Exception as e:
            print(f"  ⚠  Failed for {tid}: {e}")
    if not frames:
        raise RuntimeError("No data fetched from ChEMBL.")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PubChem helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_smiles_from_pubchem(drug_name: str) -> str | None:
    """
    Resolve a drug name to its canonical SMILES via PubChem.

    Args:
        drug_name: e.g. 'Aspirin'

    Returns:
        SMILES string or None if not found
    """
    url = f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(drug_name)}/property/CanonicalSMILES/JSON"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            props = resp.json()["PropertyTable"]["Properties"]
            return props[0]["CanonicalSMILES"]
    except Exception:
        pass
    return None


def fetch_drug_info_pubchem(drug_name: str) -> dict:
    """Fetch CID, molecular weight, and SMILES from PubChem."""
    url = (
        f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(drug_name)}"
        f"/property/CanonicalSMILES,MolecularWeight,IUPACName/JSON"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            props = resp.json()["PropertyTable"]["Properties"][0]
            return {
                "name"  : drug_name,
                "cid"   : props.get("CID"),
                "smiles": props.get("CanonicalSMILES"),
                "mw"    : props.get("MolecularWeight"),
                "iupac" : props.get("IUPACName"),
            }
    except Exception:
        pass
    return {"name": drug_name, "smiles": None}


# ─────────────────────────────────────────────────────────────────────────────
# Demo dataset builder (no API key required)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_DRUGS = [
    # (name, SMILES, known_disease, pchembl_value)
    ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O",                     "Inflammation", 5.2),
    ("Ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                "Inflammation", 5.8),
    ("Methotrexate",  "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)N[C@@H](CCC(=O)O)C(=O)O", "Cancer", 8.1),
    ("Erlotinib",     "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1","Cancer",       9.2),
    ("Imatinib",      "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1","Cancer",  9.0),
    ("Metformin",     "CN(C)C(=N)NC(=N)N",                          "Diabetes",     5.0),
    ("Glipizide",     "Cc1cnc(NS(=O)(=O)c2ccc(NCCC3CCCC3)cc2)s1", "Diabetes",     7.3),
    ("Atorvastatin",  "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", "Hyperlipidemia", 8.5),
    ("Simvastatin",   "CCC(C)(C)C(=O)O[C@H]1CC(CC2C1C1CC(O)C2(C)CC1)C",          "Hyperlipidemia", 8.0),
    ("Amlodipine",    "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl",        "Hypertension",   7.1),
    ("Lisinopril",    "OC(=O)[C@@H](N)CCCCN[C@@H](CC1CCCC2CCCCC12)C(=O)N1CCC[C@H]1C(=O)O","Hypertension", 7.9),
    ("Fluoxetine",    "CNCCC(c1ccccc1)Oc1ccc(cc1)C(F)(F)F",       "Depression",   7.5),
    ("Sertraline",    "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",        "Depression",   7.8),
    ("Sildenafil",    "CCCC1=NN(C)C(=C1)C1=C(OCC)C(=NC=N1)c1cc(ccc1S(=O)(=O)N1CCN(C)CC1)C(=O)N","ED", 9.5),
    ("Thalidomide",   "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",      "Cancer",       6.3),
    ("Chloroquine",   "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",         "Malaria",      7.2),
    ("Hydroxychloroquine","CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",    "Malaria",      7.0),
    ("Azithromycin",  "CCC1OC(=O)[C@H](C)[C@@H](O[C@@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@H]2[C@@H]([C@H]([C@@](C)(O)CO2)N(C)C)O)[C@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@@]1(C)O","Infection", 6.5),
    ("Tamoxifen",     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1","Cancer",     8.3),
    ("Celecoxib",     "Cc1ccc(-c2cc(NC(=O)...)n(-n2)...)cc1",      "Inflammation", 7.8),
]


def build_demo_dataset() -> pd.DataFrame:
    """
    Build a self-contained demo dataset so the project runs without API keys.
    In production you replace this with fetch_multiple_targets().
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, "demo_dataset.csv")

    rows = []
    for name, smiles, disease, pvalue in DEMO_DRUGS:
        rows.append({
            "drug_name"     : name,
            "smiles"        : smiles,
            "disease"       : disease,
            "pchembl_value" : pvalue,   # -log10(IC50 in molar); higher = better binding
            "target_id"     : f"DEMO_{disease.upper()[:3]}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✅  Demo dataset saved → {save_path}  ({len(df)} drugs)")
    return df


if __name__ == "__main__":
    df = build_demo_dataset()
    print(df.to_string())
