"""Configuration for MVP pipeline. All secrets come from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a sane default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# Paths
MVP_DIR = Path(__file__).parent
DATA_DIR = MVP_DIR / "data"

# LLM via OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")
LLM_TEMPERATURE = 0.1

# SOCKS proxy (v2ray) for geo-blocked providers (OpenAI, etc.)
SOCKS_PROXY = os.getenv("SOCKS_PROXY", "")

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

# ASKCOS (self-hosted retrosynthesis)
ASKCOS_BASE_URL = os.getenv("ASKCOS_BASE_URL", "http://localhost:9100")

# Retrosynthesis source flags
RETRO_ENABLE_ORD = _get_bool_env("RETRO_ENABLE_ORD", True)
RETRO_ENABLE_WEB = _get_bool_env("RETRO_ENABLE_WEB", True)
RETRO_ENABLE_RETRO_MODEL = _get_bool_env("RETRO_ENABLE_RETRO_MODEL", True)
RETRO_ENABLE_AIZYNTH = _get_bool_env("RETRO_ENABLE_AIZYNTH", False)
RETRO_ENABLE_RETROCAST = _get_bool_env("RETRO_ENABLE_RETROCAST", False)
RETRO_ORD_AUTHORITATIVE = _get_bool_env("RETRO_ORD_AUTHORITATIVE", True)

# Optional future endpoints for additive retrosynthesis sources
AIZYNTH_BASE_URL = os.getenv("AIZYNTH_BASE_URL", "")
RETROCAST_BASE_URL = os.getenv("RETROCAST_BASE_URL", "")


def _make_httpx_client(timeout: float = 120.0):
    """Build an httpx client with SOCKS proxy if SOCKS_PROXY env var is set."""
    import httpx

    proxy = SOCKS_PROXY or None
    if proxy:
        transport = httpx.HTTPTransport(proxy=proxy)
        return httpx.Client(transport=transport, timeout=timeout)
    return httpx.Client(timeout=timeout)


def make_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Create a ChatOpenAI instance routing through SOCKS proxy when available."""
    from langchain_openai import ChatOpenAI

    kwargs: dict = dict(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if SOCKS_PROXY:
        kwargs["http_client"] = _make_httpx_client()
    return ChatOpenAI(**kwargs)


# Accessor functions expected by tools/research.py and services/research_llm.py
def get_openai_api_key() -> str:
    return OPENROUTER_API_KEY


def get_openai_base_url() -> str:
    return OPENROUTER_BASE_URL


def get_llm_model() -> str:
    return LLM_MODEL


# Alias used by services/research_llm.py
def get_openai_model() -> str:
    return LLM_MODEL
