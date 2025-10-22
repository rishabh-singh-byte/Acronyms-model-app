# Acronym Explanation API

A FastAPI-based application that extracts acronyms from user queries and provides contextually relevant expansions using multiple AI models (Qwen base, Qwen LoRA, and OpenAI GPT).

## Overview

This application helps users understand acronyms in their queries by:
- Extracting acronyms from input text
- Finding candidate expansions from a curated dictionary
- Using multiple AI models to select the most contextually relevant expansions
- Providing results through both API endpoints and interactive Streamlit interfaces

## Project Structure

```
app/
├── main.py                     # FastAPI application entry point
├── instruction.txt             # Setup and running instructions
├── acronyms_list_cleaned.json  # Curated acronym dictionary
├── golden_data_20k.json        # Dataset for random query generation
├── models/                     # AI model clients
│   ├── openai_client.py        # Azure OpenAI integration
│   ├── vllm_client.py          # vLLM model integration
│   └── prompt.py               # System prompts and parsing
├── routes/                     # API endpoints
│   └── run_inference.py        # Inference routes
├── services/                   # Business logic
│   ├── acronyms_service.py     # Acronym extraction and processing
│   └── input_query.py          # Random query generation service
├── streamlit/                  # Web interfaces
│   ├── app.py                  # Basic Streamlit interface
│   ├── app1.py                 # Interface with API integration
│   └── app3.py                 # Standalone interface
└── evaluation_v1/              # Model evaluation scripts
```

## Features

### Core Functionality
- **Acronym Extraction**: It identifies acronyms in user queries
- **Contextual Selection**: Uses AI models to select relevant expansions based on query context
- **Multi-Model Support**: Integrates Qwen base, Qwen LoRA, and OpenAI GPT models
- **Flexible Input**: Supports both specific queries and random query generation

### AI Models
- **Qwen Base**: Base Qwen3-4B-Instruct model via vLLM
- **Qwen LoRA**: Fine-tuned LoRA adapter for acronym expansion
- **OpenAI GPT**: Azure OpenAI GPT-4o-mini for comparison

## Quick Start

### Prerequisites
- Python 3.8+
- vLLM server running on `localhost:8000`
- Azure OpenAI API credentials
- Required dependencies (see requirements.txt in project root)

### Configuration
Update the following configuration files:
- Azure OpenAI credentials in `config.py` (project root)
- vLLM server URL in model clients (default: `http://localhost:8000`)

### Running the Application

#### Option 1: API + Streamlit (Recommended)
```bash
# Start FastAPI server
uvicorn app.main:app --reload --port 8090

# Run Streamlit interface (in another terminal)
streamlit run app/streamlit/app1.py
```

#### Option 2: Query-specific Interface(with user inout query)
```bash
# Start FastAPI server
uvicorn app.main:app --reload --port 8090

# Run basic Streamlit interface
streamlit run app/streamlit/app.py
```

#### Option 3: Standalone Interface
```bash
# Run standalone Streamlit app (no API server needed)
streamlit run app/streamlit/app3.py
```

## API Endpoints

### Health Check
```
GET /
```
Returns API status

### Generate Responses
```
POST /inference/generate
```

**Request Body:**
```json
{
  "n": 5,                    // Number of random queries to generate
  "use_qwen_base": true,     // Enable Qwen base model
  "use_qwen_lora": true,     // Enable Qwen LoRA model  
  "use_openai_gpt": true     // Enable OpenAI GPT model
}
```

**Response:**
```json
{
  "query": "What does ML team do?",
  "acronyms_found": {
    "ML": ["Machine Learning", "Machine Learning Engineer"]
  },
  "results": {
    "qwen_base": {"ML": ["Machine Learning"]},
    "qwen_lora": {"ML": ["Machine Learning"]},
    "openai_gpt": {"ML": ["Machine Learning"]}
  }
}
```

## Usage Examples

### Direct Query Processing
```python
from app.services.acronyms_service import get_all_model_responses

# Process a specific query
result = await get_all_model_responses(
    query="What does AI team do?",
    use_qwen_base=True,
    use_qwen_lora=True,
    use_openai_gpt=True
)
```

### Random Query Generation
```python
from app.services.input_query import get_all_model_responses_random

# Generate and process random queries
results = await get_all_model_responses_random(
    n=5,  # Number of queries
    use_qwen_base=True,
    use_qwen_lora=True,
    use_openai_gpt=True
)
```

## Data Files

- **`acronyms_list_cleaned.json`**: Curated dictionary of acronyms and their possible expansions
- **`golden_data_20k.json`**: Dataset of 20,000 queries for random sampling and evaluation

## Model Configuration

### vLLM Models
- **Base Model**: `Qwen/Qwen3-4B-Instruct-2507-FP8`
- **LoRA Adapter**: `acronym-lora`
- **Server**: `http://localhost:8000/v1/chat/completions`

### OpenAI Configuration
- **Model**: `gpt-4o-mini`
- **Endpoint**: Azure OpenAI
- **Temperature**: 0.0 for consistent results

## Evaluation

The `evaluation_v1/` directory contains scripts for:
- Model performance comparison
- Output analysis and validation
- Mismatch detection between models

## Troubleshooting

### Common Issues
1. **vLLM Server Not Running**: Ensure vLLM server is running on `localhost:8000`
2. **Azure OpenAI Errors**: Check API credentials.
3. **File Path Errors**: Update absolute paths in configuration files
4. **Port Conflicts**: Change port in uvicorn command if 8090 is occupied

### Logs and Debugging
- Check FastAPI logs for API errors
- Streamlit console shows interface errors
- Model errors are returned in response with `[Error - ModelName]` prefix

## Development

### Adding New Models
1. Create client in `models/` directory
2. Update `acronyms_service.py` to include new model
3. Add configuration options to API endpoints

### Modifying Prompts
- Update `SYSTEM_PROMPT` in `models/prompt.py`
- Test with different models to ensure compatibility

## License

This project is part of an AI search and retrieval pipeline proof of concept.
