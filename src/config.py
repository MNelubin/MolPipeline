"""Configuration for the chemist-agent system."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORDB_DIR = DATA_DIR / "vectordb"

# LLM via OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")
LLM_TEMPERATURE = 0.1

# LangSmith tracing
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com"
)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "hackaton")

# PubChem
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
PUBCHEM_RATE_LIMIT_DELAY = 0.25  # seconds between requests

# RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_REACTIONS = "reactions"
CHROMA_COLLECTION_PROCEDURES = "procedures"
CHROMA_COLLECTION_TECHNIQUES = "techniques"
RAG_TOP_K = 5

# IBM RXN for Chemistry
RXN_API_KEY = os.getenv("RXN_API_KEY", "")
RXN_BASE_URL = os.getenv("RXN_BASE_URL", "https://rxn.res.ibm.com")
RXN_PROJECT_NAME = "chemist-agent"

# SOCKS proxy (for geo-blocked services like IBM RXN)
SOCKS_PROXY = os.getenv("SOCKS_PROXY", "")

# ASKCOS (self-hosted)
ASKCOS_BASE_URL = os.getenv("ASKCOS_BASE_URL", "http://localhost:9100")

# Retrosynthesis
MAX_RETRO_DEPTH = 5  # max steps in retrosynthesis tree
RETRO_TOP_N = 5  # top-N predictions per step

# Reagent price categories (placeholder dict, will be expanded)
CHEAP_REAGENTS = {
    "water", "ethanol", "methanol", "acetone", "sodium hydroxide",
    "hydrochloric acid", "sulfuric acid", "acetic acid", "acetic anhydride",
    "sodium chloride", "sodium bicarbonate", "magnesium sulfate",
    "calcium chloride", "diethyl ether", "dichloromethane", "toluene",
    "hexane", "ethyl acetate", "tetrahydrofuran", "dimethyl sulfoxide",
}

EXPENSIVE_REAGENTS = {
    "palladium", "platinum", "rhodium", "ruthenium", "iridium",
    "grubbs catalyst", "pd/c", "pd(pph3)4", "pd(dba)2",
}
