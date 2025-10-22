import json
import random
from typing import Dict, Any, List

from app.models.vllm_client import call_vllm
from app.models.openai_client import call_openai


# -------------------------------
# Configuration
# -------------------------------

DATA_FILE = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/app/golden_data_20k.json"


# -------------------------------
# Load JSON Dataset
# -------------------------------

with open(DATA_FILE, "r") as f:
    DATASET = json.load(f)


# -------------------------------
# Helper Function — Sample Random Entries
# -------------------------------

def sample_queries(n: int) -> List[Dict[str, Any]]:
    """Return n random samples from the dataset."""
    return random.sample(DATASET, min(n, len(DATASET)))


# -------------------------------
# Main Function — Call All Models
# -------------------------------

async def get_all_model_responses_random(
    n: int = 5,
    use_qwen_base: bool = True,
    use_qwen_lora: bool = True,
    use_openai_gpt: bool = True
) -> Dict[str, Any]:
    """
    Randomly sample n queries from JSON file and send to Qwen base, Qwen LoRA, and OpenAI GPT models.
    Returns model outputs in structured format.
    """

    samples = sample_queries(n)
    all_results = []

    for item in samples:
        query = item.get("Query", "")
        candidate_acronyms = item.get("Candidate_Acronyms", "")
        formatted_query = f'query: "{query}", candidate acronyms: "{candidate_acronyms}"'

        result_entry = {
            "query": query,
            "candidate_acronyms": candidate_acronyms,
            "results": {}
        }

        # --- Qwen base ---
        if use_qwen_base:
            base_resp = await call_vllm(formatted_query, use_lora=False)
            try:
                result_entry["results"]["qwen_base"] = json.loads(base_resp)
            except Exception:
                result_entry["results"]["qwen_base"] = base_resp

        # --- Qwen LoRA ---
        if use_qwen_lora:
            lora_resp = await call_vllm(formatted_query, use_lora=True)
            try:
                result_entry["results"]["qwen_lora"] = json.loads(lora_resp)
            except Exception:
                result_entry["results"]["qwen_lora"] = lora_resp

        # --- OpenAI GPT ---
        if use_openai_gpt:
            openai_resp = await call_openai(formatted_query)
            try:
                result_entry["results"]["openai_gpt"] = json.loads(openai_resp)
            except Exception:
                result_entry["results"]["openai_gpt"] = openai_resp

        all_results.append(result_entry)

    return {"total_samples": len(all_results), "data": all_results}
