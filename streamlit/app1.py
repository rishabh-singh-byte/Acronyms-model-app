# app/streamlit/app1.py
"""
Streamlit app for the result of random query generation.
This script will sample random queries from the dataset, 
build a structured prompt for the model, and dispatch the prompt to selected models (vLLM base, LoRA, OpenAI GPT).
Return the results in a structured format.
"""

import streamlit as st
import requests

# ---------------------------
# Configurations
# ---------------------------

API_URL = "http://localhost:8090/inference/generate"

st.set_page_config(
    page_title="Acronym Expansion UI",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h2 style='text-align: center;'>🤖 Acronym Expansion Assistant</h2>",
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar for Model Options
# ---------------------------

st.sidebar.title("🔧 Model Settings")
use_qwen_base = st.sidebar.checkbox("Use Qwen Base", value=True)
use_qwen_lora = st.sidebar.checkbox("Use Qwen LoRA Adapter", value=True)
use_openai_gpt = st.sidebar.checkbox("Use OpenAI GPT", value=True)

st.sidebar.markdown("---")
n_samples = st.sidebar.number_input(
    "Number of Random Samples", 
    min_value=1, 
    max_value=100, 
    value=3, 
    step=1
)

# ---------------------------
# Main Section
# ---------------------------

st.markdown("### 🎲 Generate Random Acronym Queries from Dataset")

if st.button("🚀 Run Evaluation", use_container_width=True):
    with st.spinner("Fetching random samples and generating model outputs... ⏳"):
        try:
            response = requests.post(
                API_URL,
                json={
                    "n": n_samples,
                    "use_qwen_base": use_qwen_base,
                    "use_qwen_lora": use_qwen_lora,
                    "use_openai_gpt": use_openai_gpt
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                total = data.get("total_samples", 0)
                st.success(f"✅ Received results for {total} random queries.")

                all_results = data.get("data", [])

                # ---------------------------
                # Display Each Query Result
                # ---------------------------
                for idx, item in enumerate(all_results, start=1):
                    with st.expander(f"📘 Sample {idx}: {item.get('query', '')}"):
                        st.markdown(f"**🔍 Query:** {item.get('query', '')}")
                        st.markdown(f"**🧩 Candidate Acronyms:** {item.get('candidate_acronyms', '')}")

                        st.markdown("---")
                        st.markdown("#### 🧠 Model Responses")
                        results = item.get("results", {})

                        for model_name, output in results.items():
                            st.markdown(f"**{model_name.replace('_', ' ').title()}**")

                            if isinstance(output, dict):
                                for k, v in output.items():
                                    st.markdown(f"- **{k}**: {', '.join(v)}")
                            else:
                                try:
                                    parsed = eval(output) if isinstance(output, str) else output
                                    if isinstance(parsed, dict):
                                        for k, v in parsed.items():
                                            st.markdown(f"- **{k}**: {', '.join(v)}")
                                    else:
                                        st.text(parsed)
                                except Exception:
                                    st.text(output)

                        st.markdown("---")

            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.text(response.text)

        except requests.exceptions.RequestException as e:
            st.error("🔌 Could not connect to the API.")
            st.exception(e)

# ---------------------------
# Footer
# ---------------------------

st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9em;'>Built using Streamlit</p>",
    unsafe_allow_html=True
)
