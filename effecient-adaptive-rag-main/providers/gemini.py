from typing import Any, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider implementation.
    
    Supports structured output via native tool calling (Gemini's function calling API).
    The with_structured_output() method is used directly since Gemini has robust
    support for structured responses.
    """
    
    name = "gemini"

    def create_llm(self, **overrides: Any) -> BaseChatModel:
        """Create a standard Gemini LLM instance."""
        config = {
            "model": self.get_model("GEMINI_MODEL", "gemini-flash-latest"),
            "temperature": self.get_temperature(0.7),
        }
        config.update(overrides)
        return ChatGoogleGenerativeAI(**config)

    def create_structured_llm(self, schema: Type[BaseModel], **overrides: Any) -> BaseChatModel:
        """
        Create a Gemini LLM with structured output support.
        
        Gemini natively supports structured output via tool calling.
        This implementation uses LangChain's with_structured_output() directly.
        
        Args:
            schema: Pydantic model defining the output structure
            **overrides: Configuration overrides
            
        Returns:
            BaseChatModel: Gemini LLM configured for structured output
        """
        config = {
            "model": self.get_model("GEMINI_MODEL", "gemini-flash-latest"),
            "temperature": self.get_temperature(0.7),
        }
        config.update(overrides)
        llm = ChatGoogleGenerativeAI(**config)
        return llm.with_structured_output(schema)
