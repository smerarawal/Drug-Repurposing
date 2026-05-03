"""
main.py — End-to-end drug repurposing pipeline

Run modes:
  python main.py                   →  demo mode (no API keys needed)
  python main.py --mode real       →  fetch from ChEMBL + use OpenAI
  python main.py --query "Aspirin" →  query a specific drug
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split

try:
    from torch_geometric.loader import DataLoader as PyGLoader
    PYG_OK = True
except ImportError:
    from torch.utils.data import DataLoader as PyGLoader
    PYG_OK = False

from config import (
    GNN_CONFIG, RANDOM_SEED, OUTPUT_DIR, DATA_DIR, MODEL_DIR
)
from data_fetcher   import build_demo_dataset, fetch_multiple_targets
from llm_extractor  import LLMExtractor, SAMPLE_TEXTS
from molecular_graph import build_graph_dataset, compute_similarity_matrix
from gnn_model       import DrugGNN, GNNTrainer
from repurposing_engine import RepurposingEngine
from evaluate import (
    compute_regression_metrics, print_metrics,
    plot_training_curves, plot_predicted_vs_actual,
    plot_similarity_heatmap, plot_repurposing_candidates,
    validate_repurposing_candidates, save_results_json,
)

# ── seed everything ──────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'═'*60}")
print(f"  Drug Repurposing Pipeline")
print(f"  Device : {DEVICE.upper()}")
print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — LLM knowledge extraction (demo)
# ─────────────────────────────────────────────────────────────────────────────

def step_llm_extraction(mode: str = "demo"):
    print("STEP 1 ── LLM Knowledge Extraction")
    print("─"*50)
    extractor = LLMExtractor(mode=mode)
    triples   = extractor.extract_batch(SAMPLE_TEXTS)
    print("\n  Sample extracted triples:")
    for text, triple in zip(SAMPLE_TEXTS, triples):
        print(f"    Text   : {text[:60]}…" if len(text) > 60 else f"    Text   : {text}")
        print(f"    Drug   : {triple['drug']}  |  Disease: {triple['disease']}  "
              f"|  Outcome: {triple['outcome']}")
        print()
    return triples


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load / fetch data
# ─────────────────────────────────────────────────────────────────────────────

def step_load_data(mode: str = "demo"):
    print("\nSTEP 2 ── Loading Data")
    print("─"*50)

    if mode == "real":
        # EGFR, BCR-ABL, COX-2, HMGCR (statin target)
        TARGET_IDS = ["CHEMBL2095", "CHEMBL1862", "CHEMBL230", "CHEMBL402"]
        print(f"  Fetching ChEMBL data for targets: {TARGET_IDS}")
        df = fetch_multiple_targets(TARGET_IDS, limit_per_target=150)
    else:
        df = build_demo_dataset()

    print(f"  Dataset shape: {df.shape}")
    print(f"  Diseases: {sorted(df['disease'].unique())}")
    print(f"  pChEMBL range: {df['pchembl_value'].min():.1f} – "
          f"{df['pchembl_value'].max():.1f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build molecular graphs
# ─────────────────────────────────────────────────────────────────────────────

def step_build_graphs(df):
    print("\nSTEP 3 ── Building Molecular Graphs")
    print("─"*50)
    graphs = build_graph_dataset(df)

    if graphs:
        g = graphs[0]
        print(f"  Example graph  → nodes={g.num_nodes}, edges={g.num_edges}, "
              f"x.shape={list(g.x.shape)}, y={g.y.item():.2f}")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Train GNN
# ─────────────────────────────────────────────────────────────────────────────

def step_train_gnn(graphs: list, epochs: int = None) -> tuple:
    print("\nSTEP 4 ── Training GNN")
    print("─"*50)

    if len(graphs) < 5:
        print("  ⚠  Too few graphs for training. Skipping.")
        return None, None

    epochs = epochs or GNN_CONFIG["epochs"]

    # Train / val / test split (70 / 15 / 15)
    idx = list(range(len(graphs)))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED)
    vl_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, random_state=RANDOM_SEED)

    tr_data = [graphs[i] for i in tr_idx]
    vl_data = [graphs[i] for i in vl_idx]
    te_data = [graphs[i] for i in te_idx]

    print(f"  Split  train={len(tr_data)}  val={len(vl_data)}  test={len(te_data)}")

    bs = min(GNN_CONFIG["batch_size"], len(tr_data))

    if PYG_OK:
        tr_loader = PyGLoader(tr_data, batch_size=bs, shuffle=True)
        vl_loader = PyGLoader(vl_data, batch_size=bs)
        te_loader = PyGLoader(te_data, batch_size=bs)
    else:
        # Fallback: process one graph at a time
        from utils.fallback_loader import SingleGraphLoader
        tr_loader = SingleGraphLoader(tr_data)
        vl_loader = SingleGraphLoader(vl_data)
        te_loader = SingleGraphLoader(te_data)

    # Model
    in_ch = graphs[0].x.shape[1]
    model = DrugGNN(in_channels=in_ch)
    print(f"  Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = GNNTrainer(model, device=DEVICE)
    history = trainer.train(tr_loader, vl_loader, epochs=epochs)

    # Test evaluation
    _, preds, targets = trainer.evaluate(te_loader)
    metrics = compute_regression_metrics(targets, preds)
    print_metrics(metrics, split="Test")

    # Save model
    model_path = os.path.join(MODEL_DIR, "drug_gnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved → {model_path}")

    # Plots
    plot_training_curves(history)
    plot_predicted_vs_actual(targets, preds, metrics)

    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Similarity analysis
# ─────────────────────────────────────────────────────────────────────────────

def step_similarity_analysis(df):
    print("\nSTEP 5 ── Chemical Similarity Analysis")
    print("─"*50)
    sim_matrix = compute_similarity_matrix(df["smiles"].tolist())
    labels     = df["drug_name"].tolist()
    plot_similarity_heatmap(sim_matrix, labels)
    return sim_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Drug repurposing
# ─────────────────────────────────────────────────────────────────────────────

def step_repurposing(df, model, top_k: int = 15):
    print("\nSTEP 6 ── Drug Repurposing")
    print("─"*50)

    engine = RepurposingEngine(model=model, device=DEVICE)
    engine.load_database(df)
    candidates = engine.suggest(top_k=top_k)
    engine.print_report(candidates)

    # Validate
    df_val = validate_repurposing_candidates(candidates)
    n_validated = df_val["lit_validated"].sum()
    print(f"  Literature-validated hits: {n_validated} / {len(df_val)}")

    # Save
    df_val.to_csv(os.path.join(OUTPUT_DIR, "repurposing_results.csv"), index=False)
    plot_repurposing_candidates(df_val)

    return candidates, engine


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Query a specific drug
# ─────────────────────────────────────────────────────────────────────────────

def step_query_drug(engine: RepurposingEngine, drug_name: str, smiles: str):
    print(f"\nSTEP 7 ── Querying drug: {drug_name}")
    print("─"*50)
    results = engine.query_new_drug(drug_name, smiles, top_k=5)
    if results:
        engine.print_report(results)
    else:
        print("  No similar known drugs found above threshold.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Drug Repurposing Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["demo", "real"], default="demo",
        help="demo = offline/no keys; real = ChEMBL + OpenAI"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50 in demo, 100 in real)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Optional: drug name to query repurposing for"
    )
    parser.add_argument(
        "--query_smiles", type=str,
        default="CC(=O)Oc1ccccc1C(=O)O",   # Aspirin default
        help="SMILES for query drug"
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Number of repurposing candidates to show"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Step 1: LLM extraction ──
    step_llm_extraction(mode=args.mode)

    # ── Step 2: Data ──
    df = step_load_data(mode=args.mode)

    # ── Step 3: Graphs ──
    graphs = step_build_graphs(df)

    # ── Step 4: GNN ──
    model, metrics = step_train_gnn(graphs, epochs=args.epochs)

    # ── Step 5: Similarity ──
    step_similarity_analysis(df)

    # ── Step 6: Repurposing ──
    candidates, engine = step_repurposing(df, model, top_k=args.top_k)

    # ── Step 7: Optional query ──
    if args.query:
        step_query_drug(engine, args.query, args.query_smiles)

    # ── Save ──
    save_results_json(candidates, metrics or {})

    print(f"\n{'═'*60}")
    print(f"  ✅  Pipeline complete! Outputs saved to: {OUTPUT_DIR}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
