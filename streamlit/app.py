# app/streamlit/app.py
"""
Streamlit app for the acronym expansion assistant.
This script will take the user query only and extract the candidate acronyms from the dictionary of saved acronyms.
Build a structured prompt for the model.
Dispatch the prompt to selected models (vLLM base, LoRA, OpenAI GPT).
Return the results in a structured format.
"""

import streamlit as st
import requests

# ---------------------------
# Configurations
# ---------------------------

API_URL = "http://localhost:8090/inference/generate"

st.set_page_config(
    page_title="Acronym Explanation UI",
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

# ---------------------------
# Main Input Section
# ---------------------------

st.markdown("### 🔍 Ask your query:")
query = st.text_area("Type your question here...", placeholder="e.g. Who manages the AI team?")

# ---------------------------
# Submission
# ---------------------------

if st.button("🧠 Generate Explanation", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "use_qwen_base": use_qwen_base,
                        "use_qwen_lora": use_qwen_lora,
                        "use_openai_gpt": use_openai_gpt
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    st.success("✅ Got the results!")

                    # ---------------------------
                    # Acronyms Found (Clean Format)
                    # ---------------------------
                    st.markdown("### 🧩 Acronyms Extracted")
                    if data["acronyms_found"]:
                        for acronym, expansions in data["acronyms_found"].items():
                            st.markdown(f"- **{acronym}**: {', '.join(expansions)}")
                    else:
                        st.info("No acronyms found in the query.")

                    # ---------------------------
                    # Results (Plain Box Format)
                    # ---------------------------
                    st.markdown("### 🧪 Model Responses")
                    results = data.get("results", {})
                    for model_name, output in results.items():
                        st.markdown(f"**🧠 {model_name.replace('_', ' ').title()}**")
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
