import os
from typing import Any, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """
    Ollama provider implementation.
    
    Supports structured output via tool calling for models that support it.
    Falls back gracefully if structured output is not available.
    
    Ollama is a local LLM runtime using OpenAI-compatible APIs.
    """
    
    name = "ollama"

    def create_llm(self, **overrides: Any) -> BaseChatModel:
        """Create a standard Ollama LLM instance."""
        config = {
            "model": self.get_model("OLLAMA_MODEL", "llama3.1"),
            "temperature": self.get_temperature(0.7),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        }
        config.update(overrides)
        return ChatOpenAI(**config)

    def create_structured_llm(self, schema: Type[BaseModel], **overrides: Any) -> BaseChatModel:
        """
        Create an Ollama LLM with structured output support.
        
        Ollama uses OpenAI-compatible APIs. Support for structured output
        depends on the underlying model's capabilities. Most modern models
        (llama, mistral) support it, but some older models may not.
        
        Args:
            schema: Pydantic model defining the output structure
            **overrides: Configuration overrides
            
        Returns:
            BaseChatModel: Ollama LLM configured for structured output
        """
        config = {
            "model": self.get_model("OLLAMA_MODEL", "llama3.1"),
            "temperature": self.get_temperature(0.7),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        }
        config.update(overrides)
        llm = ChatOpenAI(**config)
        try:
            return llm.with_structured_output(schema)
        except Exception:
            # If structured output fails, return the base LLM
            # Chains using this will need to handle the fallback
            return llm
