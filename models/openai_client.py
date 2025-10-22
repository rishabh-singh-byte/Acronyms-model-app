# app/models/openai_client.py
"""
Client for the OpenAI model.
"""

from openai import AsyncAzureOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION
from app.models.prompt import SYSTEM_PROMPT

# Azure OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"

async def call_openai(user_query: str) -> str:
    """
    Call Azure OpenAI API with the given user query.
    
    Args:
        user_query (str): The user query to send to the model
        
    Returns:
        str: The model's response content or error message
    """
    try:
        client = AsyncAzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )

        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error - OpenAI]: {e}"