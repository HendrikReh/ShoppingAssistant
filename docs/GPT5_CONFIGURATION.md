# GPT-5 Model Configuration

## Overview

The ShoppingAssistant is configured to use GPT-5-main-mini by default. GPT-5 models have specific requirements that differ from GPT-4 models.

## Key Requirements

### Temperature Restriction
- **GPT-5 models only support `temperature=1.0`**
- Any other temperature value will cause an error
- The system automatically enforces this restriction

### Parameter Handling
- GPT-5 models don't support all OpenAI parameters
- `litellm.drop_params = True` is set automatically to drop unsupported parameters
- This prevents errors from unsupported parameters

## Configuration

### Default Settings (app/llm_config.py)
```python
# Models
default_model: str = "gpt-5-main-mini"
chat_model: str = "gpt-5-main-mini"
eval_model: str = "gpt-5-main-mini"

# Temperature (forced to 1.0 for GPT-5)
chat_temperature: float = 1.0
eval_temperature: float = 1.0
```

### Automatic Adjustments
When a GPT-5 model is detected:
1. Temperature is forced to 1.0
2. `litellm.drop_params` is enabled
3. Environment variables are set appropriately for RAGAS

## Usage

### Basic Usage
```bash
# Set your API key
export OPENAI_API_KEY="your-key"

# Run commands - GPT-5 settings are applied automatically
uv run python -m app.cli chat --question "What are good wireless earbuds?"
```

### Evaluation
```bash
# Search evaluation
uv run python -m app.cli eval-search \
    --dataset eval/datasets/search_eval.jsonl \
    --top-k 20 --variants bm25,vec,rrf,rrf_ce

# Chat evaluation
uv run python -m app.cli eval-chat \
    --dataset eval/datasets/chat_eval.jsonl \
    --top-k 8
```

## Switching Models

To use GPT-4 or other models, modify `app/llm_config.py`:
```python
# For GPT-4
default_model: str = "gpt-4o-mini"
chat_temperature: float = 0.7  # Can use any temperature
eval_temperature: float = 0.0  # For deterministic evaluation
```

## Troubleshooting

### Temperature Error
**Error**: `gpt-5 models don't support temperature=0.7`
**Solution**: Already handled automatically - temperature is forced to 1.0

### Unsupported Parameters
**Error**: `UnsupportedParamsError`
**Solution**: Already handled - `litellm.drop_params=True` drops unsupported params

### JSON/Structured Output Issues
**Error**: `Both structured output format and JSON mode failed`
**Solution**: GPT-5 models support structured outputs but with different constraints than GPT-4

## References
- [GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)
- GPT-5 only supports temperature=1.0
- Some features from GPT-4 may not be available in GPT-5-main-mini