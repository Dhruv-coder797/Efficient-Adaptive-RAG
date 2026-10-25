import os
from importlib import import_module
from functools import lru_cache
from typing import Any, Type

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from providers.base import LLMProvider, ProviderConfigurationError

load_dotenv()

DEFAULT_LLM_PROVIDER = "gemini"

_PROVIDER_REGISTRY = {
    "gemini": "providers.gemini:GeminiProvider",
    "openai": "providers.openai:OpenAIProvider",
    "ollama": "providers.ollama:OllamaProvider",
    "openrouter": "providers.openrouter:OpenRouterProvider",
}


def _selected_provider_name() -> str:
    return (os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()


def _get_provider() -> LLMProvider:
    provider_name = _selected_provider_name()

    try:
        provider_path = _PROVIDER_REGISTRY[provider_name]
    except KeyError as exc:
        supported_providers = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ProviderConfigurationError(
            f"Unsupported LLM_PROVIDER '{provider_name}'. "
            f"Supported providers: {supported_providers}."
        ) from None

    module_name, class_name = provider_path.split(":", maxsplit=1)
    module = import_module(module_name)
    provider_class = getattr(module, class_name)
    provider = provider_class()
    if not isinstance(provider, LLMProvider):
        raise ProviderConfigurationError(
            f"Registered provider '{provider_name}' must extend LLMProvider."
        )
    return provider


@lru_cache(maxsize=None)
def get_llm(**overrides: Any) -> BaseChatModel:
    """
    Get a standard LLM instance from the configured provider.
    
    The LLM is cached after first creation, so multiple calls return
    the same instance (unless overrides differ).
    
    Args:
        **overrides: Configuration overrides to pass to the provider
        
    Returns:
        BaseChatModel: A LangChain chat model instance
        
    Example:
        llm = get_llm()
        llm = get_llm(model="gpt-4")  # Different instance with override
    """
    return _get_provider().create_llm(**overrides)


def get_structured_llm(schema: Type[BaseModel], **overrides: Any) -> BaseChatModel:
    """
    Get a structured output LLM from the configured provider.
    
    This function abstracts provider-specific structured output implementation.
    The returned LLM is configured to produce output matching the given schema.
    
    Each provider handles structured output differently:
    - Gemini, OpenAI: Use native tool calling via with_structured_output()
    - Ollama, OpenRouter: Attempt tool calling, fallback to unstructured if unavailable
    
    Args:
        schema: Pydantic BaseModel class defining the output structure
        **overrides: Configuration overrides to pass to the provider
        
    Returns:
        BaseChatModel: A LangChain chat model configured for structured output
        
    Example:
        from pydantic import BaseModel, Field
        
        class MyOutput(BaseModel):
            answer: str = Field(description="The answer")
        
        llm = get_structured_llm(MyOutput)
        
    Note:
        For providers that don't support tool_choice, the fallback may return
        an unstructured LLM. In such cases, a separate JSON parsing step
        should follow in the chain.
    """
    return _get_provider().create_structured_llm(schema, **overrides)

