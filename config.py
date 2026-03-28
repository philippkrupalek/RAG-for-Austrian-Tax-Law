"""
config.py - configuration

Models:
-) DeepSeek-V3: 685B MoE (37B active),  better reasoning
-) Llama-3.1-8B: smaller model
-) GPT-4o-mini: LLM-as-Judge (Evaluation only)
Setups (Ablation):
-) S1: Baseline (no RAG)
-) S2: Hybrid Retrieval (Dense + BM25)
-) S3: + Cross-Encoder Reranking
-) S4: Full RAG (+ Query Rewrite + 2-Pass Backfill)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum



# model configurations

@dataclass
class ModelConfig:
    """Configuration for an LLM"""
    name: str                # Display name
    api_key_env: str           # Environment variable for API key
    base_url: str              # API base URL
    model_string: str          # Model identifier for API calls
    temperature: float = 0.1
    max_tokens: int = 3000
    default_api_key: str = ""  # Fallback if env var not set
    
    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env) or self.default_api_key
        if not key:
            raise ValueError(f"{self.api_key_env} not set!")
        return key


# RAG models
DEEPSEEK_V3 = ModelConfig(
    name="DeepSeek-V3",
    api_key_env="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com",
    model_string="deepseek-chat",
    temperature=0.0,
    max_tokens=3000,
    default_api_key="sk-a923644681204e02b7326e97b7f96da6",
)

LLAMA_8B = ModelConfig(
    name="Llama-3.1-8B",
    api_key_env="GROQ_API_KEY",
    base_url="https://api.groq.com/openai/v1",
    model_string="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=3000,
    default_api_key="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",
)

# Judge Model (Evaluation only)
GPT4O_MINI = ModelConfig(
    name="GPT-4o-mini",
    api_key_env="OPENAI_API_KEY",
    base_url="https://api.openai.com/v1",
    model_string="gpt-4o-mini",
    temperature=0.0,
    max_tokens=1000,
    default_api_key="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",
)


# experiment setup

class SetupID(str, Enum):
    S1_BASELINE = "S1"
    S2_HYBRID = "S2"
    S3_RERANK = "S3"
    S4_FULL = "S4"


@dataclass
class ExperimentSetup:
    """Defines which RAG components are active"""
    setup_id: SetupID
    description: str
    use_retrieval: bool = False
    use_bm25: bool = False          # S2:  BM25 hybrid
    use_reranking: bool = False     # S3:  + Cross-Encoder
    use_query_rewrite: bool = False # S4:  + LLM Rewrite
    use_2pass: bool = False         # S4:   + Backfill


SETUPS = {
    SetupID.S1_BASELINE: ExperimentSetup(
        setup_id=SetupID.S1_BASELINE,
        description="Baseline (no RAG)",
    ),
    SetupID.S2_HYBRID: ExperimentSetup(
        setup_id=SetupID.S2_HYBRID,
        description="Hybrid Retrieval (Dense + BM25)",
        use_retrieval=True,
        use_bm25=True,
    ),
    SetupID.S3_RERANK: ExperimentSetup(
        setup_id=SetupID.S3_RERANK,
        description="+ Cross-Encoder Reranking",
        use_retrieval=True,
        use_bm25=True,
        use_reranking=True,
    ),
    SetupID.S4_FULL: ExperimentSetup(
        setup_id=SetupID.S4_FULL,
        description="Full RAG (+ QR + Backfill)",
        use_retrieval=True,
        use_bm25=True,
        use_reranking=True,
        use_query_rewrite=True,
        use_2pass=True,
    ),
}

# Both models run all 4 setups
DEEPSEEK_SETUPS = [
    SetupID.S1_BASELINE, SetupID.S2_HYBRID,
    SetupID.S3_RERANK, SetupID.S4_FULL,
]

LLAMA_SETUPS = [
    SetupID.S1_BASELINE, SetupID.S2_HYBRID,
    SetupID.S3_RERANK, SetupID.S4_FULL,
]

# paths

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Source files
USTG_RTF_PATH = SCRIPT_DIR / "UStG1994_rtf.rtf"
ANHANG_RTF_PATH = SCRIPT_DIR / "anhang_ustg.rtf"
USTR_XML_PATH = SCRIPT_DIR / "UStR2000_html.xml"

# Golden dataset
GOLDEN_DATASET_PATH = SCRIPT_DIR / "golden_dataset.json"

# Output
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"