# app/routes/run_inference.py
"""
Routes for the inference of the models.
By default it will call all the models (Qwen base, Qwen LoRA, and OpenAI GPT) with the random query generation.
"""

from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter    

# from app.models.acronyms_service import get_all_model_responses
from app.services.input_query import get_all_model_responses_random

class QueryRequest(BaseModel):
    n: int = 5
    use_qwen_base: Optional[bool] = True
    use_qwen_lora: Optional[bool] = True
    use_openai_gpt: Optional[bool] = True

router = APIRouter()

@router.post("/generate")
async def generate(request: QueryRequest):
    return await get_all_model_responses_random(
        n=request.n,
        use_qwen_base=request.use_qwen_base,
        use_qwen_lora=request.use_qwen_lora,
        use_openai_gpt=request.use_openai_gpt
    )