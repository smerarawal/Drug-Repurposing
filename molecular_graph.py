"""
molecular_graph.py — Convert SMILES → molecular graph for GNN

Each molecule becomes:
  • x          : node features  [N_atoms  × NUM_ATOM_FEATURES]
  • edge_index  : COO adjacency  [2 × N_bonds*2]
  • edge_attr   : bond features  [N_bonds*2 × 4]
  • y           : binding affinity label (pChEMBL value)
"""

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("⚠  RDKit not installed. Molecular graph features will be dummy vectors.")

from config import ATOM_TYPES, NUM_ATOM_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Atom-level feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def one_hot(val, choices: list) -> list[int]:
    """One-hot encode val against choices; last bin = 'other'."""
    vec = [0] * len(choices)
    idx = choices.index(val) if val in choices else len(choices) - 1
    vec[idx] = 1
    return vec


def atom_features(atom) -> list[float]:
    """
    Build a feature vector for a single RDKit atom.
    Returns a 23-dimensional float list.
    """
    sym    = atom.GetSymbol()
    degree = atom.GetDegree()
    charge = atom.GetFormalCharge()

    feat = []
    feat += one_hot(sym, ATOM_TYPES)                          # 10-dim atom type
    feat += one_hot(min(degree, 5), [0, 1, 2, 3, 4, 5])      #  6-dim degree
    feat += one_hot(max(-2, min(charge, 2)),
                    [-2, -1, 0, 1, 2])                         #  5-dim charge
    feat += [int(atom.GetIsAromatic())]                        #  1-dim aromatic
    feat += [int(atom.IsInRing())]                             #  1-dim in_ring
    # Total: 10+6+5+1+1 = 23
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Bond-level feature extractor
# ─────────────────────────────────────────────────────────────────────────────

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

def bond_features(bond) -> list[float]:
    """4-dimensional bond feature vector."""
    bt   = str(bond.GetBondTypeAsDouble())
    bmap = {"1.0": 0, "2.0": 1, "3.0": 2, "1.5": 3}
    idx  = bmap.get(bt, 0)
    vec  = [0, 0, 0, 0]
    vec[idx] = 1
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Main converter
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_graph(smiles: str, label: float = 0.0) -> Data | None:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: canonical SMILES string
        label : pChEMBL binding affinity value (regression target)

    Returns:
        torch_geometric.data.Data or None if SMILES is invalid
    """
    if not RDKIT_OK:
        return _dummy_graph(label)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Node features ────────────────────────────────────────────────────────
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)   # [N, 23]

    # ── Edge index + edge features ───────────────────────────────────────────
    src_list, dst_list, edge_feat_list = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf    = bond_features(bond)
        # Undirected → add both directions
        src_list += [i, j]
        dst_list += [j, i]
        edge_feat_list += [bf, bf]

    if len(src_list) == 0:
        # Single atom molecule
        src_list = dst_list = [0, 0]
        edge_feat_list = [[1,0,0,0], [1,0,0,0]]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)   # [2, E]
    edge_attr  = torch.tensor(edge_feat_list, dtype=torch.float)         # [E, 4]
    y          = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _dummy_graph(label: float) -> Data:
    """Fallback when RDKit is not available."""
    x          = torch.randn(5, NUM_ATOM_FEATURES)
    edge_index = torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]], dtype=torch.long)
    edge_attr  = torch.zeros(8, 4)
    y          = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ─────────────────────────────────────────────────────────────────────────────
# Morgan fingerprint (for Tanimoto similarity)
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Compute Morgan fingerprint from SMILES.

    Returns:
        RDKit ExplicitBitVect, or None
    """
    if not RDKIT_OK:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(fp1, fp2) -> float:
    """Compute Tanimoto (Jaccard) similarity between two Morgan fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_similarity_matrix(smiles_list: list[str]) -> np.ndarray:
    """
    Compute pairwise Tanimoto similarity matrix.

    Returns:
        numpy array [N × N]
    """
    fps = [smiles_to_fingerprint(s) for s in smiles_list]
    n   = len(fps)
    mat = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            sim       = tanimoto_similarity(fps[i], fps[j])
            mat[i][j] = sim
            mat[j][i] = sim
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset from DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_dataset(df) -> list[Data]:
    """
    Convert a pandas DataFrame with 'smiles' and 'pchembl_value' columns
    into a list of PyG Data objects.
    """
    graphs = []
    skipped = 0
    for _, row in df.iterrows():
        g = smiles_to_graph(str(row["smiles"]), float(row["pchembl_value"]))
        if g is not None:
            g.drug_name = row.get("drug_name", "unknown")
            g.disease   = row.get("disease", "unknown")
            graphs.append(g)
        else:
            skipped += 1
    print(f"  Graphs built: {len(graphs)}  |  Skipped (invalid SMILES): {skipped}")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_smiles = [
        ("Aspirin",   "CC(=O)Oc1ccccc1C(=O)O",       5.2),
        ("Erlotinib", "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1", 9.2),
        ("Metformin", "CN(C)C(=N)NC(=N)N",            5.0),
    ]
    print("── Molecular Graph Tests ─────────────────")
    for name, smi, val in test_smiles:
        g = smiles_to_graph(smi, val)
        if g:
            print(f"  {name}: {g.num_nodes} atoms, {g.num_edges} directed edges, "
                  f"x.shape={list(g.x.shape)}, y={g.y.item():.1f}")

    print("\n── Tanimoto similarity: Aspirin vs Ibuprofen ─")
    fp1 = smiles_to_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
    fp2 = smiles_to_fingerprint("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    print(f"  Similarity: {tanimoto_similarity(fp1, fp2):.3f}")
