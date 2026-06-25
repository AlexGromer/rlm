"""DeepSeek provider — OpenAI-compatible endpoint."""
from __future__ import annotations

from .openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"
    default_model = "deepseek-chat"
    env_key = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com"
