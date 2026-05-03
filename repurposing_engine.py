"""
repurposing_engine.py — Core drug repurposing logic

Strategy:
  1. For each known drug-disease pair, compute pairwise chemical similarity.
  2. For drugs above the Tanimoto threshold, predict their binding affinity
     using the trained GNN.
  3. Rank candidates by combined score = 0.5 × similarity + 0.5 × norm_affinity.
  4. Return top-K repurposing suggestions with explanations.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import torch

try:
    from torch_geometric.data import DataLoader as PyGLoader
    PYG_OK = True
except ImportError:
    PYG_OK = False

from molecular_graph import (
    compute_similarity_matrix,
    smiles_to_graph,
    smiles_to_fingerprint,
    tanimoto_similarity,
)
from config import TANIMOTO_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RepurposingCandidate:
    candidate_drug   : str
    candidate_smiles : str
    source_drug      : str
    source_disease   : str
    similarity_score : float
    predicted_affinity: float
    combined_score   : float
    reasoning        : str


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class RepurposingEngine:
    """
    Drug repurposing engine combining chemical similarity with GNN predictions.

    Usage:
        engine = RepurposingEngine(model, device="cpu")
        engine.load_database(df)          # df has: drug_name, smiles, disease, pchembl_value
        candidates = engine.suggest(top_k=10)
    """

    def __init__(self, model=None, device: str = "cpu"):
        """
        Args:
            model  : trained DrugGNN (or None for similarity-only mode)
            device : 'cpu' or 'cuda'
        """
        self.model     = model
        self.device    = device
        self.db        = None          # pd.DataFrame
        self.sim_matrix = None         # pairwise Tanimoto similarity
        self.fingerprints = []

        if model is not None:
            model.eval()
            model.to(device)

    # ── Load reference database ──────────────────────────────────────────────

    def load_database(self, df: pd.DataFrame):
        """
        Load known drug-disease associations.

        Required columns: drug_name, smiles, disease, pchembl_value
        """
        self.db = df.reset_index(drop=True).copy()
        print(f"  Database loaded: {len(self.db)} entries, "
              f"{self.db['disease'].nunique()} diseases, "
              f"{self.db['drug_name'].nunique()} unique drugs.")

        print("  Computing pairwise Tanimoto similarity matrix …")
        self.sim_matrix   = compute_similarity_matrix(self.db["smiles"].tolist())
        self.fingerprints = [smiles_to_fingerprint(s) for s in self.db["smiles"]]
        print(f"  Similarity matrix shape: {self.sim_matrix.shape}")

    # ── Predict affinity for a SMILES ────────────────────────────────────────

    def predict_affinity(self, smiles: str) -> float:
        """
        Use GNN to predict pChEMBL affinity for a SMILES.
        Returns 0.0 if model is unavailable.
        """
        if self.model is None:
            return 0.0
        g = smiles_to_graph(smiles, label=0.0)
        if g is None:
            return 0.0
        g = g.to(self.device)

        # PyG needs a batch vector
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = self.model(g)
        return float(pred.item())

    # ── Core repurposing logic ────────────────────────────────────────────────

    def suggest(self, top_k: int = 10,
                threshold: float = TANIMOTO_THRESHOLD) -> list[RepurposingCandidate]:
        """
        Generate repurposing candidates.

        Logic:
          Drug B → Disease X   IF
            ∃ Drug A known for Disease X,
            Tanimoto(A, B) ≥ threshold,
            AND GNN_affinity(B) is high.

        Args:
            top_k    : number of candidates to return
            threshold: minimum Tanimoto similarity

        Returns:
            List of RepurposingCandidate objects sorted by combined_score desc.
        """
        if self.db is None:
            raise RuntimeError("Call load_database() first.")

        candidates = []
        n = len(self.db)

        print(f"\n  Scanning {n}×{n} similarity matrix (threshold ≥ {threshold}) …")
        for i in range(n):
            drug_i    = self.db.loc[i, "drug_name"]
            disease_i = self.db.loc[i, "disease"]
            smiles_i  = self.db.loc[i, "smiles"]

            for j in range(n):
                if i == j:
                    continue

                sim = self.sim_matrix[i, j]
                if sim < threshold:
                    continue

                drug_j   = self.db.loc[j, "drug_name"]
                disease_j = self.db.loc[j, "disease"]
                smiles_j  = self.db.loc[j, "smiles"]

                # Skip if already known to treat the same disease
                if disease_j == disease_i:
                    continue

                # GNN affinity
                aff = self.predict_affinity(smiles_j)

                # Normalize affinity (pChEMBL typically 4–12)
                norm_aff = max(0.0, min(1.0, (aff - 4.0) / 8.0))

                # Combined score
                combined = 0.5 * sim + 0.5 * norm_aff

                reasoning = (
                    f"Chemically similar to {drug_i} (Tanimoto={sim:.2f}), "
                    f"which treats {disease_i}. "
                    f"GNN predicts pChEMBL={aff:.2f} → possible activity "
                    f"against {disease_i} target."
                )

                candidates.append(RepurposingCandidate(
                    candidate_drug    = drug_j,
                    candidate_smiles  = smiles_j,
                    source_drug       = drug_i,
                    source_disease    = disease_i,
                    similarity_score  = round(sim, 3),
                    predicted_affinity= round(aff, 3),
                    combined_score    = round(combined, 3),
                    reasoning         = reasoning,
                ))

        # Deduplicate (keep highest-scoring for each drug-disease pair)
        seen: dict[tuple, RepurposingCandidate] = {}
        for c in candidates:
            key = (c.candidate_drug, c.source_disease)
            if key not in seen or c.combined_score > seen[key].combined_score:
                seen[key] = c

        results = sorted(seen.values(), key=lambda x: x.combined_score, reverse=True)
        print(f"  Found {len(results)} unique repurposing candidates.")
        return results[:top_k]

    # ── Query new drug ────────────────────────────────────────────────────────

    def query_new_drug(self, drug_name: str, smiles: str, top_k: int = 5
                       ) -> list[RepurposingCandidate]:
        """
        Find repurposing opportunities for a completely new drug (not in DB).

        Computes similarity against all DB drugs and uses GNN for affinity.
        """
        fp_new = smiles_to_fingerprint(smiles)
        aff    = self.predict_affinity(smiles)
        norm_aff = max(0.0, min(1.0, (aff - 4.0) / 8.0))

        candidates = []
        for i, row in self.db.iterrows():
            sim = tanimoto_similarity(fp_new, self.fingerprints[i])
            if sim < TANIMOTO_THRESHOLD:
                continue
            combined = 0.5 * sim + 0.5 * norm_aff
            candidates.append(RepurposingCandidate(
                candidate_drug    = drug_name,
                candidate_smiles  = smiles,
                source_drug       = row["drug_name"],
                source_disease    = row["disease"],
                similarity_score  = round(sim, 3),
                predicted_affinity= round(aff, 3),
                combined_score    = round(combined, 3),
                reasoning         = (
                    f"New drug similar to {row['drug_name']} "
                    f"(Tanimoto={sim:.2f}), known for {row['disease']}. "
                    f"Predicted pChEMBL={aff:.2f}."
                ),
            ))

        return sorted(candidates, key=lambda x: x.combined_score, reverse=True)[:top_k]

    # ── Format results ────────────────────────────────────────────────────────

    @staticmethod
    def to_dataframe(candidates: list[RepurposingCandidate]) -> pd.DataFrame:
        return pd.DataFrame([vars(c) for c in candidates])

    @staticmethod
    def print_report(candidates: list[RepurposingCandidate]):
        if not candidates:
            print("  No candidates found.")
            return
        print(f"\n{'═'*70}")
        print(f"  DRUG REPURPOSING REPORT   ({len(candidates)} candidates)")
        print(f"{'═'*70}")
        for rank, c in enumerate(candidates, 1):
            print(f"\n  #{rank}  {c.candidate_drug}  →  {c.source_disease}")
            print(f"       Score     : {c.combined_score:.3f}  "
                  f"(similarity={c.similarity_score:.3f}, "
                  f"affinity={c.predicted_affinity:.3f})")
            print(f"       Reasoning : {c.reasoning}")
        print(f"\n{'═'*70}\n")
