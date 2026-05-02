"""
Groq Client
-----------
Uses Groq API for fast, free LLM inference.
Replaced Google AI Studio due to billing requirements.

Set GROQ_API_KEY in Cloud Run environment variables.
"""

import os
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=API_KEY)
    return _client


async def generate(prompt: str, temperature: float = 0.2) -> str:
    """
    Send a prompt to Groq and return the text response.

    Args:
        prompt:      The full prompt string
        temperature: Controls randomness. Keep low for clinical text.

    Returns:
        The model's text response as a string
    """
    client = _get_client()
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()
