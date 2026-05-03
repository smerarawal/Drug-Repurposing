"""
llm_extractor.py — Extract structured drug/disease/target triples from biomedical text.

Works in TWO modes:
  • REAL  : uses OpenAI via LangChain (needs OPENAI_API_KEY)
  • DEMO  : uses rule-based extraction so the project runs offline/free
"""

import re
import json
from typing import Optional

# ── LangChain (optional) ──────────────────────────────────────────────────
try:
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# Extraction schema
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_RESULT = {
    "drug"   : str,   # drug / compound name
    "target" : str,   # protein / enzyme / receptor
    "disease": str,   # disease / condition
    "outcome": str,   # inhibition / activation / binding etc.
    "confidence": float,   # 0.0 – 1.0 (LLM estimate)
}

# ─────────────────────────────────────────────────────────────────────────────
# System prompt for LLM
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a biomedical information extraction expert.
Given a piece of clinical trial or research text, extract the following fields as JSON:
{
  "drug":       "<name of drug or compound>",
  "target":     "<protein, receptor, or enzyme the drug acts on>",
  "disease":    "<disease or medical condition being studied>",
  "outcome":    "<inhibition | activation | binding | unknown>",
  "confidence": <0.0 to 1.0>
}

Rules:
- Return ONLY valid JSON. No markdown, no explanation.
- If a field is unknown, use "unknown".
- confidence reflects how sure you are about the extraction (not the science).
"""

# ─────────────────────────────────────────────────────────────────────────────
# LLM Extractor class
# ─────────────────────────────────────────────────────────────────────────────

class LLMExtractor:
    """
    Extracts drug–target–disease triples from free text.

    Usage:
        extractor = LLMExtractor(mode="demo")
        result    = extractor.extract("Aspirin inhibits COX-2 in arthritis patients.")
        # → {"drug": "Aspirin", "target": "COX-2", "disease": "arthritis", ...}
    """

    def __init__(self, mode: str = "demo"):
        """
        Args:
            mode: "real" → use OpenAI,  "demo" → rule-based (no API key needed)
        """
        self.mode = mode
        self.llm  = None

        if mode == "real":
            if not LANGCHAIN_OK:
                print("⚠  LangChain not installed. Falling back to demo mode.")
                self.mode = "demo"
            elif OPENAI_API_KEY == "your-openai-key-here":
                print("⚠  No OpenAI key set. Falling back to demo mode.")
                self.mode = "demo"
            else:
                self.llm = ChatOpenAI(
                    model       = LLM_MODEL,
                    temperature = LLM_TEMPERATURE,
                    max_tokens  = LLM_MAX_TOKENS,
                    openai_api_key = OPENAI_API_KEY,
                )
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    ("human",  "{text}"),
                ])

        print(f"LLMExtractor initialised in [{self.mode}] mode.")

    # ── public API ───────────────────────────────────────────────────────────

    def extract(self, text: str) -> dict:
        """Extract a single triple from text."""
        if self.mode == "real" and self.llm is not None:
            return self._extract_llm(text)
        return self._extract_rules(text)

    def extract_batch(self, texts: list[str]) -> list[dict]:
        """Extract from a list of text passages."""
        return [self.extract(t) for t in texts]

    # ── LLM path ─────────────────────────────────────────────────────────────

    def _extract_llm(self, text: str) -> dict:
        chain    = self.prompt | self.llm
        response = chain.invoke({"text": text})
        raw      = response.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"  ⚠  JSON parse error. Raw response:\n{raw}")
            return self._fallback(text)

    # ── Rule-based path (demo / offline) ────────────────────────────────────

    # Pattern tables
    _DRUG_PATTERNS = [
        r'\b(aspirin|ibuprofen|methotrexate|erlotinib|imatinib|metformin|'
        r'atorvastatin|simvastatin|amlodipine|lisinopril|fluoxetine|sertraline|'
        r'sildenafil|thalidomide|chloroquine|hydroxychloroquine|tamoxifen|celecoxib)\b',
    ]
    _TARGET_PATTERNS = [
        r'\b(COX-?[12]|EGFR|BCR-ABL|HER2|VEGFR|mTOR|PARP|CDK[0-9]+|'
        r'ALK|BRAF|MEK|ERK|PI3K|AKT|p53|PD-?[L1])\b',
    ]
    _DISEASE_PATTERNS = [
        r'\b(cancer|tumor|carcinoma|arthritis|diabetes|hypertension|depression|'
        r'inflammation|infection|malaria|HIV|Alzheimer|Parkinson|lupus|'
        r'hyperlipidemia|asthma|COPD)\b',
    ]
    _OUTCOME_PATTERNS = {
        "inhibition": r'\b(inhibit|block|suppress|downregulate|reduce)\w*\b',
        "activation": r'\b(activat|stimulat|upregulat|induce)\w*\b',
        "binding"   : r'\b(bind|attach|interact|target)\w*\b',
    }

    def _extract_rules(self, text: str) -> dict:
        tl = text.lower()

        drug    = self._find_first(self._DRUG_PATTERNS,   tl) or "unknown"
        target  = self._find_first(self._TARGET_PATTERNS, text) or "unknown"  # keep case for targets
        disease = self._find_first(self._DISEASE_PATTERNS, tl) or "unknown"

        outcome = "unknown"
        for label, pat in self._OUTCOME_PATTERNS.items():
            if re.search(pat, tl):
                outcome = label
                break

        return {
            "drug"      : drug.title() if drug != "unknown" else drug,
            "target"    : target,
            "disease"   : disease.title() if disease != "unknown" else disease,
            "outcome"   : outcome,
            "confidence": 0.6 if "unknown" not in [drug, disease] else 0.3,
        }

    @staticmethod
    def _find_first(patterns: list[str], text: str) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(0)
        return None

    @staticmethod
    def _fallback(text: str) -> dict:
        return {
            "drug": "unknown", "target": "unknown",
            "disease": "unknown", "outcome": "unknown",
            "confidence": 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Erlotinib inhibits EGFR signalling in non-small cell lung cancer patients.",
    "Imatinib was found to block BCR-ABL kinase activity in chronic myeloid leukemia.",
    "Chloroquine reduces viral replication and shows activity against malaria parasites.",
    "Metformin activates AMPK and reduces blood glucose in type-2 diabetes.",
    "Tamoxifen blocks estrogen receptor in breast cancer tissue.",
]

if __name__ == "__main__":
    extractor = LLMExtractor(mode="demo")
    print("\n" + "─"*60)
    print("EXTRACTED TRIPLES")
    print("─"*60)
    for text in SAMPLE_TEXTS:
        result = extractor.extract(text)
        print(f"\nText  : {text}")
        print(f"Result: {json.dumps(result, indent=2)}")
