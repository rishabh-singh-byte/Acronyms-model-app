# acronyms_service.py
# -------------------
# This service extracts acronyms from a query, builds a structured prompt,
# and dispatches the prompt to selected models (vLLM base, LoRA, OpenAI GPT).

import json
import re
from typing import Dict, List

from app.models.vllm_client import call_vllm
from app.models.openai_client import call_openai

# -------------------------------
# Load acronym dictionary
# -------------------------------

ACRONYM_FILE = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/app/acronyms_list_cleaned.json"

with open(ACRONYM_FILE, "r") as f:
    ACRONYMS = json.load(f)

# -------------------------------
# Acronym Extraction
# -------------------------------

def extract_acronyms(query: str) -> Dict[str, List[str]]:
    """
    Extracts all acronyms from the query that match keys in the acronym dictionary.
    Acronyms are assumed to be uppercase with 2+ letters (e.g. 'AI', 'ML').
    """
    found = {}
    # Find all words that are 2 or more letters long (case-insensitive)
    words = re.findall(r'\b[a-zA-Z]{1,}\b', query)
    for word in words:
        if word in ACRONYMS:
            found[word] = ACRONYMS[word]
    return found

# -------------------------------
# Build Query Format for Model
# -------------------------------

def build_structured_prompt(query: str, found_acronyms: Dict[str, List[str]]) -> str:
    """
    Formats the user query for model input:
    Example:
    query: "What does ML team do?", candidate acronyms: "(ML: Machine Learning, Machine Learning Engineer)"
    """
    candidate_strs = [
        f"({acro}: {', '.join(expansions)})"
        for acro, expansions in found_acronyms.items()
    ]
    candidate_section = " ".join(candidate_strs)

    return f'query: "{query}", candidate acronyms: "{candidate_section}"'

# -------------------------------
# Main Aggregator Function
# -------------------------------

async def get_all_model_responses(
    query: str,
    use_qwen_base: bool = True,
    use_qwen_lora: bool = True,
    use_openai_gpt: bool = True
) -> Dict:
    """
    Calls selected models (Qwen base, Qwen LoRA, OpenAI) with a formatted prompt.
    Returns their responses along with the matched acronyms.
    """
    found_acronyms = extract_acronyms(query)

    # No acronyms found case
    if not found_acronyms:
        return {
            "query": query,
            "acronyms_found": {},
            "results": {
                "qwen_base": "No known acronyms found." if use_qwen_base else None,
                "qwen_lora": "No known acronyms found." if use_qwen_lora else None,
                "openai_gpt": "No known acronyms found." if use_openai_gpt else None
            }
        }

    # Build the structured prompt
    user_query = build_structured_prompt(query, found_acronyms)

    # Collect model results
    results = {}

    if use_qwen_base:
        base_resp = await call_vllm(user_query, use_lora=False)
        try:
            results["qwen_base"] = json.loads(base_resp)
        except Exception:
            results["qwen_base"] = base_resp

    if use_qwen_lora:
        lora_resp = await call_vllm(user_query, use_lora=True)
        try:
            results["qwen_lora"] = json.loads(lora_resp)
        except Exception:
            results["qwen_lora"] = lora_resp

    if use_openai_gpt:
        openai_resp = await call_openai(user_query)
        try:
            results["openai_gpt"] = json.loads(openai_resp)
        except Exception:
            results["openai_gpt"] = openai_resp

    return {
        "query": query,
        "acronyms_found": found_acronyms,
        "results": results
    }
