"""
evaluate.py — Evaluation metrics and visualizations

Covers:
  1. Regression metrics (RMSE, MAE, R²) for GNN binding affinity
  2. Repurposing validation (literature match heuristic)
  3. Training curve plots
  4. Similarity distribution plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_regression_metrics(y_true: list, y_pred: list) -> dict:
    """Compute RMSE, MAE, R²."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    return metrics


def print_metrics(metrics: dict, split: str = "Test"):
    print(f"\n  ── {split} Metrics ────────────────────────")
    for k, v in metrics.items():
        print(f"     {k:6s} : {v:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Repurposing validation heuristic
# ─────────────────────────────────────────────────────────────────────────────

# Known repurposed drug–disease pairs (validated in literature)
KNOWN_REPURPOSING = {
    ("Thalidomide", "Cancer")        : "Used for multiple myeloma (FDA approved)",
    ("Sildenafil",  "Hypertension")  : "Pulmonary arterial hypertension treatment",
    ("Metformin",   "Cancer")        : "Studied extensively for cancer prevention",
    ("Chloroquine", "Cancer")        : "Autophagy inhibitor in clinical trials",
    ("Tamoxifen",   "Inflammation")  : "Anti-inflammatory effects studied",
    ("Aspirin",     "Cancer")        : "Colorectal cancer prevention (epidemiological)",
}

def validate_repurposing_candidates(candidates) -> pd.DataFrame:
    """
    Check predicted candidates against known literature pairs.
    Returns DataFrame with a 'literature_validated' column.
    """
    rows = []
    for c in candidates:
        key       = (c.candidate_drug, c.source_disease)
        validated = key in KNOWN_REPURPOSING
        evidence  = KNOWN_REPURPOSING.get(key, "No direct literature match found")
        rows.append({
            "drug"            : c.candidate_drug,
            "disease"         : c.source_disease,
            "combined_score"  : c.combined_score,
            "similarity"      : c.similarity_score,
            "affinity_pred"   : c.predicted_affinity,
            "lit_validated"   : validated,
            "evidence"        : evidence,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "train"    : "#4C9BE8",
    "val"      : "#E8884C",
    "scatter"  : "#3D9970",
    "hist"     : "#9B59B6",
    "correct"  : "#27AE60",
    "wrong"    : "#E74C3C",
}


def plot_training_curves(history: dict, save_path: str = None):
    """Plot train vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=COLORS["train"], lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=COLORS["val"],   lw=2, label="Val Loss",
            linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("GNN Training Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def plot_predicted_vs_actual(y_true, y_pred, metrics: dict, save_path: str = None):
    """Scatter plot of predicted vs actual pChEMBL values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, color=COLORS["scatter"], edgecolors="white", s=60)

    lims = [min(min(y_true), min(y_pred)) - 0.5,
            max(max(y_true), max(y_pred)) + 0.5]
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual pChEMBL", fontsize=12)
    ax.set_ylabel("Predicted pChEMBL", fontsize=12)
    ax.set_title("Predicted vs Actual Binding Affinity", fontsize=14, fontweight="bold")

    txt = "\n".join([f"R² = {metrics['R2']:.3f}",
                     f"RMSE = {metrics['RMSE']:.3f}",
                     f"MAE  = {metrics['MAE']:.3f}"])
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "pred_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def plot_similarity_heatmap(sim_matrix: np.ndarray, labels: list, save_path: str = None):
    """Tanimoto similarity heatmap."""
    n    = min(len(labels), 20)   # cap at 20 for readability
    mat  = sim_matrix[:n, :n]
    lbls = labels[:n]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Tanimoto Similarity")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(lbls, fontsize=8)
    ax.set_title("Drug Chemical Similarity Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "similarity_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def plot_repurposing_candidates(df_candidates: pd.DataFrame, save_path: str = None):
    """Bar chart of top repurposing candidates colored by validation."""
    if df_candidates.empty:
        return
    top = df_candidates.head(15).copy()
    top["label"] = top["drug"] + " → " + top["disease"]
    colors = [COLORS["correct"] if v else COLORS["wrong"]
              for v in top["lit_validated"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top)), top["combined_score"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=9)
    ax.set_xlabel("Combined Score", fontsize=12)
    ax.set_title("Top Repurposing Candidates", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["correct"], label="Literature validated"),
        Patch(facecolor=COLORS["wrong"],   label="No literature match"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "repurposing_candidates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def save_results_json(candidates, metrics: dict, path: str = None):
    """Save all results to JSON."""
    path = path or os.path.join(OUTPUT_DIR, "results.json")
    out  = {
        "metrics"   : metrics,
        "candidates": [vars(c) for c in candidates],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results JSON → {path}")
