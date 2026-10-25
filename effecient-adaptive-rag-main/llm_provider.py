from typing import Any, Type

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from providers import get_llm as _get_llm
from providers.factory import get_structured_llm as _get_structured_llm

load_dotenv()


def get_llm(**overrides: Any) -> BaseChatModel:
    """
    Get a standard LLM instance.
    
    This is the main entry point for obtaining LLM instances throughout the system.
    The LLM is sourced from the configured provider (Gemini, OpenAI, Ollama, etc.)
    
    Args:
        **overrides: Configuration overrides to pass to the provider
        
    Returns:
        BaseChatModel: A LangChain chat model instance
    """
    return _get_llm(**overrides)


def get_structured_llm(schema: Type[BaseModel], **overrides: Any) -> BaseChatModel:
    """
    Get a structured output LLM instance.
    
    This is the main entry point for chains that need structured output.
    The provider handles implementation details (tool calling, JSON parsing, etc.)
    
    This abstraction enables:
    - Provider-agnostic chains (no provider-specific code in chains)
    - Easy provider switching without chain modifications
    - Provider-specific optimizations for structured output
    
    Args:
        schema: Pydantic BaseModel class defining the output structure
        **overrides: Configuration overrides to pass to the provider
        
    Returns:
        BaseChatModel: An LLM configured to produce structured output matching schema
        
    Example:
        from pydantic import BaseModel, Field
        
        class Decision(BaseModel):
            route: str = Field(description="The selected route")
        
        # This works with any provider (Gemini, OpenAI, etc.)
        structured_llm = get_structured_llm(Decision)
    """
    return _get_structured_llm(schema, **overrides)

