# app/models/vllm_client.py
"""
Client for the vLLM model.
It will call the adapter model if use_lora is True, otherwise it will call the base model.
"""

import httpx
import sys
from app.models.prompt import SYSTEM_PROMPT

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"  # Base model
LORA_ADAPTER_NAME = "acronym-lora"  # LoRA adapter name

async def call_vllm(user_query: str, use_lora: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    # Choose model based on use_lora flag
    model_name = LORA_ADAPTER_NAME if use_lora else BASE_MODEL_NAME
    
    payload = {
        "model": model_name,  # This is the key change
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 400
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(VLLM_API_URL, json=payload)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error - vLLM {'LoRA' if use_lora else 'Base'}]: {e}"

