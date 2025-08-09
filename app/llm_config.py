"""Central LLM configuration for ShoppingAssistant.

This module defines which LLMs are used for different tasks throughout the application.
No fallback logic - fails fast if the specified LLM is not reachable.
"""

from typing import Optional
import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM usage across different tasks."""
    
    # Default model for all tasks (can be overridden per task)
    default_model: str = "gpt-4o-mini"
    
    # Task-specific models
    chat_model: str = "gpt-4o-mini"  # DSPy chat/RAG
    eval_model: str = "gpt-4o-mini"  # RAGAS evaluation
    
    # API configuration
    api_base: Optional[str] = None  # e.g., "http://localhost:4000/v1" for LiteLLM proxy
    api_key: Optional[str] = None
    
    # Temperature settings per task
    chat_temperature: float = 0.7
    eval_temperature: float = 0.0  # Evaluation should be deterministic
    
    def __post_init__(self):
        """Load from environment variables if not set."""
        if self.api_base is None:
            self.api_base = os.getenv("OPENAI_API_BASE")
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
    
    def validate(self) -> None:
        """Validate configuration and fail fast if invalid."""
        if not self.api_key:
            raise ValueError(
                "No API key configured. Set OPENAI_API_KEY environment variable or pass api_key to LLMConfig"
            )
        
        # Check if we're using OpenAI directly or a proxy
        if self.api_base:
            print(f"Using LLM proxy at: {self.api_base}")
        else:
            print("Using OpenAI API directly")
    
    def get_dspy_lm(self, task: str = "chat"):
        """Get configured DSPy LM instance for a specific task."""
        import dspy
        
        self.validate()
        
        model = self.chat_model if task == "chat" else self.eval_model
        temperature = self.chat_temperature if task == "chat" else self.eval_temperature
        
        # Configure API base if using proxy
        if self.api_base:
            # DSPy expects the model string to include the provider
            # When using proxy, we typically use "openai/" prefix
            lm = dspy.LM(
                f"openai/{model}",
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=temperature
            )
        else:
            lm = dspy.LM(
                f"openai/{model}",
                api_key=self.api_key,
                temperature=temperature
            )
        
        return lm
    
    def configure_ragas(self) -> None:
        """Configure RAGAS to use the specified LLM for evaluation."""
        import os
        
        self.validate()
        
        # RAGAS uses environment variables for LLM configuration
        os.environ["OPENAI_API_KEY"] = self.api_key or ""
        if self.api_base:
            os.environ["OPENAI_API_BASE"] = self.api_base
        
        # Set the model for RAGAS
        os.environ["RAGAS_LLM_MODEL"] = f"openai/{self.eval_model}"
        
        print(f"Configured RAGAS with model: {self.eval_model}")


# Global singleton instance
_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """Get the global LLM configuration instance."""
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config


def set_llm_config(config: LLMConfig) -> None:
    """Set the global LLM configuration instance."""
    global _config
    _config = config