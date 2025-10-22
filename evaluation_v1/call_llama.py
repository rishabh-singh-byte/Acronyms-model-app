#app/evaluation_v1/test_output_llama.py
"""
Calling the llama model using the vLLM API and evaluating the results on 20k samples.
Saving the results to an excel file.
"""

import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
import re
import httpx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from app.models.prompt import SYSTEM_PROMPT

SYSTEM_PROMPT = """You are a precise assistant tasked with selecting only the **most relevant acronym expansions** from a given list, based strictly on the user's query.

Instructions:
- Only include expansions that are clearly and directly related to the query's context.
- If multiple meanings are relevant, include all of them.
- If no acronym is relevant, return an empty dictionary: `{}`.
- Acronyms must appear in the query to be considered.
- Preserve the acronym casing as it appears in the query.
- Output must be a valid **JSON dictionary**, having multiple key and value pairs:
  - Keys: acronyms found in the query.
  - Values: lists of relevant expansions.

Output Format:
```json
{
  "ACRONYM1": ["Relevant Expansion 1", "Relevant Expansion 2",...],
  "ACRONYM2": ["Relevant Expansion 1", "Relevant Expansion 2",...],
}
```

Examples:
###
query: "Who leads the AI team", candidate acronyms: " (AI: artificial intelligence, Artificial Intelligence, Action Items)"
###
{"AI": ["artificial intelligence"]}
###
query: "who is the current cpo", candidate acronyms: " (cpo: Chief People Officer, Chief Product and Customer Officer, Chief Product Officer)"
###
{"cpo": ["Chief People Officer", "Chief Product Officer"]}
###
query: "update the okr", candidate acronyms: " (okr: Objectives and Key Results, Office of Knowledge Research)"
###
{"okr": ["Objectives and Key Results"]}
###
query: "Send the AI and OKR reports", candidate acronyms: " (AI: artificial intelligence, Action Items) (OKR: Objectives and Key Results, Office of Knowledge Research)"
###
{"AI": ["artificial intelligence"], "OKR": ["Objectives and Key Results"]}
###
query: "Is the CFO and CPO attending the leadership meeting?", candidate acronyms: " (CFO: Chief Financial Officer, Chief Future Officer) (CPO: Chief People Officer, Chief Product Officer)"
###
{"CFO": ["Chief Financial Officer"], \n"CPO": ["Chief People Officer"]}
###
query: "can you help me with this", candidate acronyms: " (can: Canada) (you: Young Outstanding Undergraduates)"
###
{}
###
"""

def parse_raw_prompt(raw_prompt_string):
    parts = [part.strip() for part in raw_prompt_string.split("###") if part.strip()]

    messages = []

    # First part is the system message with instructions
    if len(parts) > 0:
        messages.append({"role": "system", "content": parts[0]})

    # Then process each example pair
    for i in range(1, len(parts), 2):
        user_example = parts[i]
        assistant_response = parts[i + 1] if i + 1 < len(parts) else ""
        messages.append({"role": "user", "content": user_example})
        if assistant_response:
            messages.append({"role": "assistant", "content": assistant_response})

    return messages


SYSTEM_PROMPT = parse_raw_prompt(SYSTEM_PROMPT)

# Set up base model and adapter
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_NAME = "acronym-lora"

# Provide your system prompt from wherever it's defined

# ------------------------------------------------------------
# ✅ Inference function — always uses LoRA
# ------------------------------------------------------------
def extract_json(text: str) -> str:
    """
    Extract and return the first valid JSON object (dict) from text.
    Returns '{}' if nothing valid is found.
    """
    try:
        # Try direct parse first (model might return clean JSON)
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass

    # Fallback: search for the first plausible dict-looking substring
    try:
        matches = re.findall(r"\{.*?\}", text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"⚠️ JSON fallback parsing failed: {e}")

    return "{}"

async def call_vllm(user_query: str) -> str:
    messages = SYSTEM_PROMPT + [{"role": "user", "content": user_query}]


    payload = {
        "model": LORA_ADAPTER_NAME,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.9
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(VLLM_API_URL, json=payload)
            res.raise_for_status()

            response_json = res.json()
            # print("\n🔵 Raw response from vLLM server:")
            # print(json.dumps(response_json.get("choices")[0].get("message").get("content"), indent=2, ensure_ascii=False))

            raw_output = response_json["choices"][0]["message"]["content"]
            return extract_json(raw_output)

    except Exception as e:
        return f"[Error - vLLM LoRA]: {e}"


# ------------------------------------------------------------
# ✅ Data formatting
# ------------------------------------------------------------
def load_json(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def construct_user_query(entry: Dict[str, Any]) -> str:
    query = entry["query"]
    candidate_acronyms = entry.get("candidate_acronyms", {})

    acronyms_text = "\n".join(
        f"{acronym}: {', '.join(expansions)}"
        for acronym, expansions in candidate_acronyms.items()
    )

    full_query = f"Query: {query}\nCandidate Acronyms:\n{acronyms_text}"
    return full_query


# ------------------------------------------------------------
# ✅ Processing logic
# ------------------------------------------------------------
async def process_entry(entry: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    async with semaphore:
        user_query = construct_user_query(entry)
        response = await call_vllm(user_query)

        return {
            "query": entry["query"],
            "candidate_acronyms": entry["candidate_acronyms"],
            "expected_output": entry["output"],
            "llama_lora_response": response
        }

async def process_entries(data: List[Dict[str, Any]], concurrency_limit: int = 20) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [
        process_entry(entry, semaphore)
        for entry in data
    ]
    return await tqdm.gather(*tasks, desc="Processing queries")


# ------------------------------------------------------------
# ✅ Output
# ------------------------------------------------------------
def save_to_excel(results: List[Dict[str, Any]], output_path: str):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)


# ------------------------------------------------------------
# ✅ Main entry point
# ------------------------------------------------------------
async def main():
    input_path = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/Notebooks/sampled_20000_queries.json"
    output_path = "llama1B_results_20_lora.xlsx"

    json_data = load_json(input_path)
    # json_data = json_data[:5]
    results = await process_entries(json_data, concurrency_limit=20)
    save_to_excel(results, output_path)
    print(f"✅ Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
