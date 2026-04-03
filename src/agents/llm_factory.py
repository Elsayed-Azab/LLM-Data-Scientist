"""LLM factory — create chat model instances for any supported provider.

Supported providers:
  - openai:  gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o3-mini, ...
  - anthropic: claude-sonnet-4-20250514, claude-opus-4-20250514, claude-haiku-4-5-20251001, ...

Usage:
    from src.agents.llm_factory import create_llm
    llm = create_llm("claude-sonnet-4-20250514")          # auto-detects Anthropic
    llm = create_llm("gpt-4o")                      # auto-detects OpenAI
    llm = create_llm("gpt-4o", provider="openai")   # explicit provider
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

# Provider prefixes for auto-detection
_ANTHROPIC_PREFIXES = ("claude",)
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")


def detect_provider(model: str) -> str:
    """Infer the provider from the model name."""
    model_lower = model.lower()
    if any(model_lower.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(model_lower.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    # Default to openai for unknown models
    return "openai"


def create_llm(
    model: str = "gpt-4o",
    provider: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    **kwargs,
) -> BaseChatModel:
    """Create a LangChain chat model for the given model/provider.

    Args:
        model: Model name (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        provider: "openai" or "anthropic". Auto-detected from model name if omitted.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        **kwargs: Extra kwargs passed to the model constructor.
    """
    resolved_provider = provider or detect_provider(model)

    if resolved_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5,
            **kwargs,
        )
    elif resolved_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider: {resolved_provider!r}. Use 'openai' or 'anthropic'.")


def list_providers() -> list[str]:
    return ["openai", "anthropic"]
