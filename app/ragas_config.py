"""RAGAS configuration with GPT-5 support."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

def get_ragas_llm(model_name: Optional[str] = None) -> LangchainLLMWrapper:
    """Get configured LLM for RAGAS that works with GPT-5."""
    
    if model_name is None:
        # Try to get from environment or use GPT-5-mini as default
        model_name = os.getenv("RAGAS_LLM_MODEL", "gpt-5-mini")
        
        # Remove openai/ prefix if present
        if model_name.startswith("openai/"):
            model_name = model_name[7:]
    
    # For GPT-5, we need to use temperature=1.0
    # The ragas_gpt5_fix module handles this at the litellm level
    if "gpt-5" in model_name.lower():
        llm = ChatOpenAI(
            model=model_name,
            temperature=1.0,  # GPT-5 only supports 1.0
            max_retries=3,
            request_timeout=60
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Default for other models
            max_retries=3,
            request_timeout=60
        )
    
    # Wrap for RAGAS
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Get configured embeddings for RAGAS."""
    
    # Use OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Or "text-embedding-ada-002"
    )
    
    return LangchainEmbeddingsWrapper(embeddings)


def configure_ragas_metrics(metrics_list):
    """Configure a list of RAGAS metrics with the correct LLM."""
    
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()
    
    configured_metrics = []
    for metric_class in metrics_list:
        # Create instance of the metric
        metric = metric_class()
        
        # Set LLM if the metric has that attribute
        if hasattr(metric, 'llm'):
            metric.llm = llm
        
        # Set embeddings if the metric has that attribute
        if hasattr(metric, 'embeddings'):
            metric.embeddings = embeddings
            
        configured_metrics.append(metric)
    
    return configured_metrics