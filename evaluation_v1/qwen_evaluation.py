#app/evaluation_v1/qwen_evaluation.py
"""
Calling the Qwen-base model using the vLLM API and evaluating the results on 20k samples.
Saving the results to an excel file.
"""

import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm.asyncio import tqdm

import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.models.vllm_client import call_vllm


def load_json(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def construct_user_query(entry: Dict[str, Any]) -> str:
    query = entry["query"]
    candidate_acronyms = entry.get("candidate_acronyms", {})

    # Build a formatted string to send to vLLM
    acronyms_text = "\n".join(
        f"{acronym}: {', '.join(expansions)}"
        for acronym, expansions in candidate_acronyms.items()
    )

    full_query = f"Query: {query}\nCandidate Acronyms:\n{acronyms_text}"
    return full_query


async def process_entry(entry: Dict[str, Any], semaphore: asyncio.Semaphore, use_lora: bool = False) -> Dict[str, Any]:
    async with semaphore:
        user_query = construct_user_query(entry)
        response = await call_vllm(user_query, use_lora=use_lora)

        return {
            "query": entry["query"],
            "candidate_acronyms": entry["candidate_acronyms"],
            "expected_output": entry["output"],
            "qwen_lora_response": response
        }


async def process_entries(data: List[Dict[str, Any]], use_lora: bool = False, concurrency_limit: int = 20) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [
        process_entry(entry, semaphore, use_lora=use_lora)
        for entry in data
    ]
    return await tqdm.gather(*tasks, desc="Processing queries")


def save_to_excel(results: List[Dict[str, Any]], output_path: str):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)


async def main():
    input_path = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/Notebooks/sampled_20000_queries.json"  # Change this to your actual input file path
    output_path = "base_results_20000.xlsx"

    json_data = load_json(input_path)
    results = await process_entries(json_data, use_lora=False, concurrency_limit=20)  # Set to False to skip LoRA
    save_to_excel(results, output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
