"""Simple fix for GPT-5 temperature issue in RAGAS.

RAGAS internally uses temperature values like 1e-8 (nearly 0) which GPT-5 doesn't support.
This module provides a simple solution.
"""

import os
import logging

# Configure litellm to handle GPT-5 temperature restrictions
import litellm

# Enable automatic parameter fixing for GPT-5
litellm.drop_params = True

# Also set up a callback to fix temperature issues
original_completion = litellm.completion


def gpt5_safe_completion(*args, **kwargs):
    """Wrapper that fixes temperature for GPT-5 models."""
    
    model = kwargs.get('model', '')
    
    # Check if it's a GPT-5 model
    if 'gpt-5' in str(model).lower():
        # Force temperature to 1.0
        if 'temperature' in kwargs:
            original_temp = kwargs['temperature']
            if original_temp != 1.0:
                logging.debug(f"Fixing GPT-5 temperature: {original_temp} -> 1.0")
                kwargs['temperature'] = 1.0
    
    return original_completion(*args, **kwargs)


# Apply the monkey patch
litellm.completion = gpt5_safe_completion


def setup_gpt5_compatibility():
    """Setup GPT-5 compatibility for RAGAS and other libraries."""
    
    # Set environment variable to use GPT-5
    if not os.getenv("RAGAS_LLM_MODEL"):
        os.environ["RAGAS_LLM_MODEL"] = "gpt-5-mini"
    
    # Enable litellm parameter dropping
    litellm.drop_params = True
    
    # Log that we've set up compatibility
    logging.info("GPT-5 compatibility mode enabled")
    
    return True