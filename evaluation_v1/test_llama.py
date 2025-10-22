#app/evaluation_v1/test_llama.py
"""
Test the evaluation of the llama model response based on the user query and the output expected from the model.
Saved the results to a JSON file.
"""

import requests
import json
import openai
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm_asyncio
import sys
import ast



url = "http://localhost:8000/v1"  # ✅ Use chat endpoint
api_key = "sk-1234"
model_name = "meta-llama/Llama-2-13b-chat-hf"

client = AsyncOpenAI(
    api_key=api_key,
    base_url=url,
)

SYSTEM_PROMPT = """
You are a strict and unbiased evaluator tasked with comparing two model outputs in response to a specific user query.
You are provided with:
    - A query (the user’s input).
    - Two model outputs (Model 1 and Model 2), each generated in response to the query.
    - The outputs consist of acronym expansions that are intended to fit appropriately within the context of the query.
Your Task:
    - Determine which of the two model outputs better satisfies the query.
    - Output must be a valid **JSON dictionary** in every case and in format specified, dont include any other text or explanation.
Evaluation Guidelines:
    - Evaluate each output independently, focusing solely on how well it fulfills the query's intent.
    - Assess for correctness, completeness, clarity, relevance, and how well the acronym expansions align with the meaning and context of the query.
    - Only select a "Better" output if one model clearly and completely addresses the query more effectively than the other.
    - If both outputs are equally good, equally flawed, ambiguous, or incorrect, mark the result as a "Tie".
    - When in doubt or if the superiority of one output over the other is not absolutely clear, default to "Tie".
    - More respopnse doesnt mean better in every case, only relevant response based on the query is considered.
    - If both models produces more response and cant decide which one is better, mark the result as a "Tie".

Output Format (JSON only):
```json
{
  "judgment": "Model 1" | "Model 2" | "Tie",
  "explanation": "..."  // A brief justification for the judgment. Required in all cases.
}
```
"""

SEMAPHORE = 10

def safe_parse_dict(value):
    """Safely parse a string to a dictionary if needed"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except:
            return {}
    return {}

import json, re

def extract_json(text: str):
    """
    Extracts the first valid JSON object from the input text that contains both
    'judgment' and 'explanation' keys. Returns the parsed JSON as a dict.
    Returns None if no valid match is found.
    """
    if not text:
        return None

    # Find all possible JSON objects in the text (non-greedy)
    candidates = re.findall(r'\{.*?\}', text, re.DOTALL)

    for candidate in candidates:
        candidate = candidate.strip().strip("`")  # Remove markdown backticks if present
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "judgment" in parsed and "explanation" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue  # Try next block if parsing fails

    return None



async def evaluate_single_entry(entry, semaphore):
    async with semaphore:
        query = entry.get("Query", "")
        # print(model2_output)    
        model1_output = safe_parse_dict(entry.get("model_1", {})) # treat model_1 as model_2
        model2_output = safe_parse_dict(entry.get("model_2", {})) # treat model_2 as model_1
        # print(model1_output)

        candidate_acronyms = list(set(model1_output.keys()) | set(model2_output.keys()))

        user_prompt = f"""
Query:
{query}

Model 1 Output:
{json.dumps(model1_output, indent=2)}

Model 2 Output:
{json.dumps(model2_output, indent=2)}

Based on the query and the outputs, evaluate which model performed better and respond using the JSON format specified.
"""

        try:
            response = await client.chat.completions.create(
                model=model_name,  
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            content = response.choices[0].message.content
            result = extract_json(content)
            if result is None:
                print("⚠️ Failed to parse valid JSON from model output:")
                print(content)
            # else:
                # print("✅ Parsed JSON:", result)
                        
            # Fixed: Use correct keys that match the system prompt
            better_model = result.get("judgment", "invalid")
            justification = result.get("explanation", "No explanation provided")

        except json.JSONDecodeError as e:
            better_model = "invalid"
            justification = f"Invalid JSON response: {content if 'content' in locals() else 'No content'}"
        except Exception as e:
            better_model = "error"
            justification = str(e)

        return {
            "query": query,
            "model_1_gpt": model1_output,
            "model_2_qwen_ft": model2_output,
            "candidate_acronyms": candidate_acronyms,
            "better_model": better_model,
            "justification": justification
        }


# === 🔁 Main Execution ===
async def main():
    input_path = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/app/evaluation_v1/mismatched_outputs.json"
    output_path = "mismatched_evaluation_results_llama.json"

    with open(input_path, 'r') as f:
        data = json.load(f)
    # data = data[:10]
    
    semaphore = asyncio.Semaphore(SEMAPHORE)

    print(f"🚀 Starting evaluation of {len(data)} entries with concurrency={SEMAPHORE}")

    # Run all evaluations asynchronously
    tasks = [evaluate_single_entry(entry, semaphore) for entry in data]
    results = await tqdm_asyncio.gather(*tasks)

    # Save the results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary - Fixed: Use correct case for summary keys
    summary = {"Model 1": 0, "Model 2": 0, "Tie": 0, "invalid": 0, "error": 0}
    for r in results:
        key = r["better_model"]
        summary[key] = summary.get(key, 0) + 1

    print("\n📊 Evaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n✅ Results saved to: {output_path}")


# Run the async main
if __name__ == "__main__":
    asyncio.run(main())