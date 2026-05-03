# 🧬 ML-Based Drug Repurposing Pipeline

A complete, research-grade system for predicting drug-disease associations by combining:
- **LLMs** (knowledge extraction from biomedical text)
- **RDKit** (molecular graph construction)
- **Graph Neural Networks** (binding affinity prediction)
- **Tanimoto similarity** (chemical fingerprint matching)

---

## 📁 Project Structure

```
drug_repurposing/
├── main.py                  ← Entry point (run this)
├── config.py                ← All hyperparameters & paths
├── data_fetcher.py          ← ChEMBL / PubChem / demo data
├── llm_extractor.py         ← LangChain knowledge extraction
├── molecular_graph.py       ← SMILES → PyG graph conversion
├── gnn_model.py             ← DrugGNN + training loop
├── repurposing_engine.py    ← Core repurposing logic
├── evaluate.py              ← Metrics + all visualizations
├── requirements.txt         ← Dependencies
├── utils/
│   └── fallback_loader.py   ← Fallback DataLoader
├── data/                    ← Auto-generated datasets
├── models/                  ← Saved GNN weights
└── outputs/                 ← All plots + results JSON
```

---

## ⚙️ Setup & Installation

### 1. Create virtual environment

```bash
# Python 3.10+ required
python -m venv venv

# Activate
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows
```

### 2. Install PyTorch first (check your CUDA version)

```bash
# CPU only
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch Geometric

```bash
# After torch is installed:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

pip install torch-geometric
```

### 4. Install RDKit

```bash
# Best method — via conda
conda install -c conda-forge rdkit

# OR via pip (newer rdkit package)
pip install rdkit
```

### 5. Install remaining packages

```bash
pip install -r requirements.txt
```

---

## 🔑 Optional: API Keys

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

**Without this key** → the system runs in **demo mode** (rule-based extraction, no API calls).
**With this key** → LLM extraction uses GPT-3.5-turbo.

---

## 🚀 Running the Pipeline

### Demo Mode (No API keys required — runs fully offline)

```bash
cd drug_repurposing
python main.py
```

**What this does:**
1. Runs rule-based LLM extraction on 5 sample texts
2. Loads built-in demo dataset (19 drugs, 6 diseases)
3. Builds molecular graphs for each drug
4. Trains GNN for 50 epochs
5. Computes pairwise Tanimoto similarity
6. Generates repurposing candidates
7. Saves 4 plots + results JSON

**Expected runtime:** ~2–5 minutes on CPU

---

### Real Mode (Fetches from ChEMBL + OpenAI)

```bash
python main.py --mode real --epochs 100
```

**What changes:**
- Data fetched from ChEMBL API (EGFR, BCR-ABL, COX-2, HMGCR targets)
- LLM extraction uses OpenAI GPT-3.5-turbo
- More training data → better model

---

### Query a Specific Drug

```bash
# Query with drug name + SMILES
python main.py --query "Hydroxychloroquine" \
               --query_smiles "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12"
```

**Output:** List of diseases this drug might treat based on chemical similarity.

---

### Custom Training Parameters

```bash
# Longer training with more candidates
python main.py --epochs 200 --top_k 20
```

---

## 📊 Outputs

After running, check the `outputs/` folder:

| File | Description |
|------|-------------|
| `training_curves.png` | Train vs validation MSE loss |
| `pred_vs_actual.png` | GNN predictions scatter plot |
| `similarity_heatmap.png` | Tanimoto pairwise similarity matrix |
| `repurposing_candidates.png` | Top candidates bar chart |
| `repurposing_results.csv` | Full candidate table |
| `results.json` | All metrics + candidates |

---

## 🔬 Understanding the Output

### Repurposing Candidate Example

```
#1  Hydroxychloroquine → Cancer
    Score     : 0.712  (similarity=0.823, affinity=6.1)
    Reasoning : Chemically similar to Chloroquine (Tanimoto=0.823),
                which treats Malaria. GNN predicts pChEMBL=6.1 →
                possible activity against Malaria target.
```

**Score Interpretation:**
- `similarity_score` → Tanimoto fingerprint overlap (0–1)
- `predicted_affinity` → GNN-predicted pChEMBL value (higher = stronger binding)
- `combined_score` → 0.5 × similarity + 0.5 × normalized_affinity (0–1)

**pChEMBL scale:**
```
< 5.0  =  weak     (IC50 > 10 µM)
5–7    =  moderate (IC50 1–100 nM range)
> 7    =  strong   (IC50 < 100 nM)
> 9    =  very strong
```

---

## 🧠 Model Architecture

```
Input: Drug molecule (SMILES)
  ↓
Atom Features (23-dim per atom)
  ↓
GIN Layer 1 (128 hidden) + BatchNorm + ReLU + Dropout(0.2)
  ↓
GIN Layer 2 (128 hidden) + BatchNorm + ReLU + Dropout(0.2)
  ↓
GIN Layer 3 (128 hidden) + BatchNorm + ReLU + Dropout(0.2)
  ↓
GIN Layer 4 (128 hidden) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Global Mean Pool + Global Sum Pool → concat (256-dim)
  ↓
Dense(128) → ReLU → Dropout
Dense(64)  → ReLU
Dense(1)   → Predicted pChEMBL
```

---

## 🔧 Modifying the System

### Add a new drug to the database

Edit `data_fetcher.py` → `DEMO_DRUGS` list:

```python
DEMO_DRUGS = [
    ...
    ("YourDrug", "YOUR_SMILES_HERE", "YourDisease", 7.5),
]
```

### Adjust similarity threshold

In `config.py`:

```python
TANIMOTO_THRESHOLD = 0.4   # lower = more candidates, noisier
                            # higher = fewer, more reliable
```

### Change GNN hyperparameters

In `config.py`:

```python
GNN_CONFIG = {
    "hidden_channels"  : 256,   # bigger = more capacity
    "num_layers"       : 6,     # deeper = captures longer paths
    "dropout"          : 0.3,
    "learning_rate"    : 5e-4,
    "batch_size"       : 64,
    "epochs"           : 200,
    "patience"         : 20,
}
```

---

## 🐛 Troubleshooting

### "No module named rdkit"
```bash
conda install -c conda-forge rdkit
# OR
pip install rdkit-pypi
```

### "No module named torch_geometric"
```bash
pip install torch-geometric
# If scatter/sparse fail:
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### "Too few graphs for training"
Your SMILES strings may be invalid. Run:
```bash
python molecular_graph.py   # tests 3 molecules
```

### Training loss is NaN
- Reduce learning rate: `"learning_rate": 1e-4`
- Add gradient clipping (already included, but ensure PyG is installed)

---

## 📚 Key Concepts for Viva / Presentation

| Question | Answer |
|----------|--------|
| Why GNN over CNN? | Molecules are graphs; CNN requires fixed grid input. GNN works on arbitrary graph topology. |
| Why GIN (not GCN)? | GIN has maximum expressiveness among MPNNs — provably as powerful as the Weisfeiler-Lehman graph isomorphism test. |
| What does pChEMBL mean? | −log₁₀(IC50). Higher = stronger binding. 9 = nanomolar potency. |
| Why Tanimoto? | Standard chemical similarity metric. Used in drug discovery for 30+ years. Threshold ≥0.4 is industry standard. |
| Limitation? | Chemical similarity ≠ same biological effect. Two drugs can look similar but hit different proteins. |
| Extension? | Add protein structure via AlphaFold2 embeddings + drug-protein interaction model. |

---

## 📖 References

1. Gilmer et al. (2017) — Neural Message Passing for Quantum Chemistry
2. Xu et al. (2019) — How Powerful are Graph Neural Networks? (GIN paper)
3. Corsello et al. (2020) — Drug Repurposing Hub, Nature Chemical Biology
4. Mendez et al. (2019) — ChEMBL: towards direct deposition of bioassay data
5. Landrum (2006) — RDKit: Open-Source Cheminformatics
