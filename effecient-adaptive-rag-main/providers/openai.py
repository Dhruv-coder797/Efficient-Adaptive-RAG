import os
from typing import Any, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.
    
    Supports structured output via native tool calling when using models
    that support tool_choice (e.g., GPT-4o, GPT-4 Turbo).
    
    Compatible with OpenAI-compatible APIs via OPENAI_BASE_URL.
    """
    
    name = "openai"

    def create_llm(self, **overrides: Any) -> BaseChatModel:
        """Create a standard OpenAI LLM instance."""
        config: dict[str, Any] = {
            "model": self.get_model("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": self.get_temperature(0.7),
            "api_key": self.require_env("OPENAI_API_KEY"),
        }

        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        if base_url:
            config["base_url"] = base_url

        config.update(overrides)
        return ChatOpenAI(**config)

    def create_structured_llm(self, schema: Type[BaseModel], **overrides: Any) -> BaseChatModel:
        """
        Create an OpenAI LLM with structured output support.
        
        OpenAI supports structured output via tool calling and JSON schema.
        This implementation uses LangChain's with_structured_output() which
        uses OpenAI's native tool calling capabilities.
        
        Args:
            schema: Pydantic model defining the output structure
            **overrides: Configuration overrides
            
        Returns:
            BaseChatModel: OpenAI LLM configured for structured output
        """
        config: dict[str, Any] = {
            "model": self.get_model("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": self.get_temperature(0.7),
            "api_key": self.require_env("OPENAI_API_KEY"),
        }

        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        if base_url:
            config["base_url"] = base_url

        config.update(overrides)
        llm = ChatOpenAI(**config)
        return llm.with_structured_output(schema)
