"""LLM factory — creates ChatOpenAI instances configured for OpenRouter + LangSmith."""

import os

from langchain_openai import ChatOpenAI

from src.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

# Set LangSmith env vars (LangChain reads these automatically)
os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT


def get_llm(
    temperature: float | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance routed through OpenRouter.

    OpenRouter is compatible with the OpenAI API, so we just set
    base_url and api_key accordingly.
    """
    return ChatOpenAI(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/chemist-agent",
            "X-Title": "ChemAssistant",
        },
    )
