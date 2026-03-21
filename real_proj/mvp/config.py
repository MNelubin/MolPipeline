"""Configuration for MVP pipeline. All secrets come from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
MVP_DIR = Path(__file__).parent
DATA_DIR = MVP_DIR / "data"

# LLM via OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")
LLM_TEMPERATURE = 0.1

# LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_PROJECT", "hackaton")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)

# PubChem
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
