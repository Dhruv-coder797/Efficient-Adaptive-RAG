import os
from abc import ABC, abstractmethod
from typing import Any, Type

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel


class ProviderConfigurationError(ValueError):
    """Raised when LLM provider configuration is invalid."""


class LLMProvider(ABC):
    """
    Base class for LLM provider implementations.
    
    Providers handle:
    - Creating LLM instances with provider-specific configuration
    - Generating structured outputs in a provider-agnostic way
    
    All configuration is sourced from environment variables, allowing
    easy switching between providers without code changes.
    """
    
    name: str

    @abstractmethod
    def create_llm(self, **overrides: Any) -> BaseChatModel:
        """
        Create a chat model instance for this provider.
        
        Args:
            **overrides: Configuration overrides for the LLM
            
        Returns:
            BaseChatModel: A LangChain chat model instance
        """

    @abstractmethod
    def create_structured_llm(self, schema: Type[BaseModel], **overrides: Any) -> BaseChatModel:
        """
        Create a structured output LLM for this provider.
        
        This method handles provider-specific structured output implementation:
        - Providers with native tool calling (Gemini, OpenAI): Use with_structured_output()
        - Providers without tool calling: Use prompt-based JSON with Pydantic parsing
        
        Args:
            schema: Pydantic model class defining the output structure
            **overrides: Configuration overrides for the LLM
            
        Returns:
            BaseChatModel: An LLM configured to return structured output matching schema
            
        Note:
            The returned LLM should be used in a chain with a parser that validates
            against the schema. For providers using tool calling, the LLM handles
            parsing internally. For others, a JSON parser should follow in the chain.
        """

    def get_model(self, env_name: str, default: str) -> str:
        """Get model name from environment or default."""
        return os.getenv(env_name) or os.getenv("LLM_MODEL") or default

    def get_api_key(self, env_name: str) -> str | None:
        """Get API key from environment."""
        return os.getenv(env_name) or os.getenv("LLM_API_KEY")

    def get_temperature(self, default: float) -> float:
        """Get temperature from environment with validation."""
        raw_temperature = os.getenv("LLM_TEMPERATURE")
        if raw_temperature is None:
            return default

        try:
            return float(raw_temperature)
        except ValueError as exc:
            raise ProviderConfigurationError(
                f"Invalid LLM_TEMPERATURE '{raw_temperature}'. Expected a number."
            ) from exc

    def require_env(self, env_name: str) -> str:
        """Require an environment variable to be set."""
        value = self.get_api_key(env_name)
        if not value:
            raise ProviderConfigurationError(
                f"Missing required environment variable {env_name} "
                f"or LLM_API_KEY "
                f"for LLM provider '{self.name}'."
            )
        return value
