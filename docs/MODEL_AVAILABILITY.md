# Model Availability Guide

## Overview
This document provides information about model availability and known limitations when using the Zen MCP Server with different providers.

## Current Status (as of 2025-08-31)

### Working Models via OpenRouter

The following models are confirmed working through OpenRouter API:

#### Anthropic Models
- ✅ `anthropic/claude-opus-4.1` - 200K context
- ✅ `anthropic/claude-sonnet-4.1` - 200K context  
- ✅ `anthropic/claude-3.5-haiku` - 200K context

#### OpenAI Models
- ✅ `openai/o3` - 200K context
- ✅ `openai/o3-mini` - 200K context
- ✅ `openai/o3-mini-high` - 200K context
- ✅ `openai/o3-pro` - 200K context
- ✅ `openai/o4-mini` - 200K context

#### Other Models
- ✅ `deepseek/deepseek-r1-0528` - 65K context
- ✅ `meta-llama/llama-3-70b` - 8K context
- ✅ `mistralai/mistral-large-2411` - 128K context
- ✅ `perplexity/llama-3-sonar-large-32k-online` - 32K context (with web search)
- ✅ `llama3.2` - 128K context (local model)

### Models with Known Issues

#### Google Gemini Models via OpenRouter
- ⚠️ `google/gemini-2.5-flash` - Returns 404 error "No allowed providers are available"
- ⚠️ `google/gemini-2.5-pro` - May have similar availability issues

**Reason**: These models may not be available in all regions or may require specific OpenRouter subscription tiers.

**Workaround**: Use alternative models with similar capabilities:
- Instead of `google/gemini-2.5-flash`, use:
  - `anthropic/claude-3.5-haiku` for fast responses
  - `openai/o3-mini` for balanced performance
- Instead of `google/gemini-2.5-pro`, use:
  - `anthropic/claude-sonnet-4.1` for high quality
  - `openai/o3-pro` for advanced reasoning

## Provider Configuration Requirements

### OpenRouter (Currently Configured)
- **Status**: ✅ Active
- **Required**: `OPENROUTER_API_KEY` environment variable
- **Models**: 15 available (see list above)
- **Notes**: Primary provider, most models work reliably

### Google Gemini (Direct API)
- **Status**: ❌ Not configured
- **Required**: `GEMINI_API_KEY` environment variable
- **Notes**: Would provide direct access to Gemini models without OpenRouter limitations

### OpenAI (Direct API)
- **Status**: ❌ Not configured
- **Required**: `OPENAI_API_KEY` environment variable
- **Notes**: Direct access to GPT models and o-series models

### Other Providers
- **X.AI**: Requires `XAI_API_KEY`
- **DIAL**: Requires `DIAL_API_KEY`
- **Custom/Local**: Requires `CUSTOM_API_URL` (e.g., for Ollama)

## Troubleshooting Model Errors

### Error: "No allowed providers are available for the selected model"
**Cause**: The model is not available through your current OpenRouter configuration.

**Solutions**:
1. Check if the model requires a specific subscription tier on OpenRouter
2. Try an alternative model from the working models list
3. Configure the provider's direct API (e.g., set up `GEMINI_API_KEY` for Gemini models)

### Error: "404 - Model not found"
**Cause**: Model name may be incorrect or deprecated.

**Solutions**:
1. Verify the exact model name from the `listmodels` tool
2. Use model aliases when available
3. Check OpenRouter documentation for current model names

## Recommended Models by Use Case

### Fast Responses
- `anthropic/claude-3.5-haiku` - Best balance of speed and quality
- `openai/o3-mini` - Good for simple tasks
- `llama3.2` (local) - No API limits, requires local setup

### High Quality/Complex Tasks
- `anthropic/claude-opus-4.1` - Most capable model
- `anthropic/claude-sonnet-4.1` - Excellent reasoning
- `openai/o3-pro` - Advanced problem-solving

### Code Analysis & Generation
- `anthropic/claude-sonnet-4.1` - Best for code understanding
- `deepseek/deepseek-r1-0528` - Specialized for coding
- `openai/o3` - Good general-purpose coding

### Web Search & Current Information
- `perplexity/llama-3-sonar-large-32k-online` - Built-in web search capability

### Budget-Conscious Usage
- `llama3.2` (local) - Free after setup
- `meta-llama/llama-3-70b` - Lower cost per token
- `anthropic/claude-3.5-haiku` - Efficient token usage

## Best Practices

1. **Always test model availability** before deploying to production
2. **Have fallback models** configured for critical workflows
3. **Monitor API responses** for availability changes
4. **Use the `listmodels` tool** to verify current availability
5. **Consider local models** (Ollama) for unlimited usage without API constraints

## Updates and Changes

Model availability can change based on:
- OpenRouter service updates
- Regional restrictions
- Subscription tier changes
- Provider API changes

Last verified: 2025-08-31
Server version: 5.11.0

For the most current information, run:
```bash
python -c "from tools.listmodels import ListModelsTool; import asyncio; tool = ListModelsTool(); print(asyncio.run(tool.execute({})))"
```