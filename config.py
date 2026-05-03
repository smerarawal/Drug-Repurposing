"""
config.py — Central configuration for the drug repurposing pipeline
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

# ─── Data Sources ────────────────────────────────────────────────────────────
CHEMBL_BASE_URL   = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE_URL  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# ─── GNN Hyperparameters ─────────────────────────────────────────────────────
GNN_CONFIG = {
    "hidden_channels"  : 128,
    "num_layers"       : 4,
    "dropout"          : 0.2,
    "learning_rate"    : 1e-3,
    "batch_size"       : 32,
    "epochs"           : 100,
    "patience"         : 15,          # early stopping
}

# ─── Atom Feature Dimensions ─────────────────────────────────────────────────
# Atom types tracked (one-hot encoded)
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Other']
NUM_ATOM_FEATURES = (
    len(ATOM_TYPES)   # atom type
    + 6               # degree (0-5)
    + 5               # formal charge bins
    + 2               # aromaticity, in_ring
)                     # = 23 total

# ─── Similarity threshold ────────────────────────────────────────────────────
TANIMOTO_THRESHOLD = 0.4   # minimum chemical similarity for repurposing

# ─── LLM Settings ────────────────────────────────────────────────────────────
LLM_MODEL       = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 500

# ─── Seed ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
