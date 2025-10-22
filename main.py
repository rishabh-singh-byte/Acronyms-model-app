from fastapi import FastAPI
from app.routes.run_inference import router as inference_router

app = FastAPI(
    title="Acronym Explanation API",
    description="An API to extract acronyms and call multiple LLMs (vLLM base, LoRA, OpenAI) for expanded understanding.",
    version="1.0.0",
    port=8090
)

# Root health check
@app.get("/")
async def root():
    return {"message": "Acronym Explanation API is up and running!"}

# Register the inference route
app.include_router(inference_router, prefix="/inference", tags=["Inference"])
