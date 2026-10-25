import os
from typing import Any, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from providers.base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider implementation.
    
    OpenRouter is an API aggregator that provides access to many LLMs
    via a unified OpenAI-compatible interface. Supports structured output
    depending on the underlying model's capabilities.
    
    Some models on OpenRouter may lack tool_choice support, so fallback
    handling is recommended.
    """
    
    name = "openrouter"

    def create_llm(self, **overrides: Any) -> BaseChatModel:
        """Create a standard OpenRouter LLM instance."""
        config = {
            "model": self.get_model("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "temperature": self.get_temperature(0.7),
            "base_url": os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            "api_key": self.require_env("OPENROUTER_API_KEY"),
        }
        config.update(overrides)
        return ChatOpenAI(**config)

    def create_structured_llm(self, schema: Type[BaseModel], **overrides: Any) -> BaseChatModel:
        """
        Create an OpenRouter LLM with structured output support.
        
        OpenRouter supports structured output via tool calling for many models.
        However, not all models or model versions on OpenRouter support tool_choice.
        
        This implementation attempts to use structured output but falls back
        gracefully to an unstructured LLM if the feature is unavailable.
        
        Args:
            schema: Pydantic model defining the output structure
            **overrides: Configuration overrides
            
        Returns:
            BaseChatModel: OpenRouter LLM configured for structured output if available
        """
        config = {
            "model": self.get_model("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "temperature": self.get_temperature(0.7),
            "base_url": os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            "api_key": self.require_env("OPENROUTER_API_KEY"),
        }
        config.update(overrides)
        llm = ChatOpenAI(**config)
        try:
            return llm.with_structured_output(schema)
        except Exception:
            # Fallback for models that don't support tool_choice
            return llm
