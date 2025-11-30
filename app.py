# app.py
import streamlit as st
import google.generativeai as genai
from typing import Any

st.set_page_config(page_title="Medical RAG (simple)", layout="wide")

# Use the short model name (recommended)
MODEL_NAME = "gemini-1.5-flash"

st.sidebar.header("Settings / API")
api_key = st.sidebar.text_input(
    "Gemini API Key (or set via Secrets)", type="password", key="api_key_input"
)

# Keep configured flag in session_state so reruns remember it
if "gemini_configured" not in st.session_state:
    st.session_state["gemini_configured"] = False

if api_key:
    try:
        # configure the genai client with the provided key
        genai.configure(api_key=api_key)
        st.session_state["gemini_configured"] = True
        st.sidebar.success("Gemini configured ✔")
    except Exception as e:
        st.session_state["gemini_configured"] = False
        st.sidebar.error(f"Configuration error: {e}")
else:
    if not st.session_state["gemini_configured"]:
        st.sidebar.warning("No API key provided — set it in Secrets or paste here")
    else:
        st.sidebar.info("Gemini previously configured in this session.")

st.title("Medical RAG — Simple")
st.markdown("Ask a medical question (informational only).")

question = st.text_input("Enter your question:", "")
num_sources = st.slider("Sources to show (fake list)", 1, 5, 3)


def _extract_text_from_response(resp: Any) -> str:
    """Try many common response shapes returned by different genai versions."""
    if resp is None:
        return ""
    # common attribute names
    for attr in ("text", "output_text", "content", "answer"):
        if hasattr(resp, attr):
            val = getattr(resp, attr)
            if isinstance(val, str):
                return val
            # sometimes nested objects
            if hasattr(val, "text"):
                return getattr(val, "text")
    # dict-like responses
    if isinstance(resp, dict):
        for key in ("output", "output_text", "text", "content"):
            if key in resp and isinstance(resp[key], str):
                return resp[key]
        # sometimes the text is deeper
        if "candidates" in resp and isinstance(resp["candidates"], list) and resp["candidates"]:
            cand = resp["candidates"][0]
            if isinstance(cand, dict) and "content" in cand:
                return cand["content"]
    # fallback to str()
    return str(resp)


def ask_gemini(prompt: str, model: str = MODEL_NAME) -> str:
    """Attempt several genai invocation styles; returns text or raises RuntimeError."""
    if not st.session_state.get("gemini_configured", False):
        raise RuntimeError("Gemini API key not configured. Provide key in sidebar or Secrets.")

    last_err = None

    # Pattern A: top-level helper (some versions)
    try:
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=prompt)
            return _extract_text_from_response(resp)
    except Exception as e:
        last_err = e

    # Pattern B: GenerativeModel + generate_content (common pattern)
    try:
        if hasattr(genai, "GenerativeModel"):
            m = genai.GenerativeModel(model)
            # Some versions accept a single string, others require a dict/kwargs
            try:
                resp = m.generate_content(prompt)
            except TypeError:
                resp = m.generate_content({"prompt": prompt})
            return _extract_text_from_response(resp)
    except Exception as e:
        last_err = e

    # Pattern C: generic generate (older/newer variants)
    try:
        if hasattr(genai, "generate"):
            resp = genai.generate(model=model, input=prompt)
            return _extract_text_from_response(resp)
    except Exception as e:
        last_err = e

    # Pattern D: other possible wrapper names
    try:
        if hasattr(genai, "chat"):
            resp = genai.chat(model=model, messages=[{"role": "user", "content": prompt}])
            return _extract_text_from_response(resp)
    except Exception as e:
        last_err = e

    raise RuntimeError(f"All genai calls failed. Last error: {last_err}")


if st.button("Find Answer"):
    if question.strip() == "":
        st.warning("Please type a question.")
    else:
        if not st.session_state.get("gemini_configured", False):
            st.warning("Provide Gemini API key in the sidebar (or set via Streamlit Secrets).")
        else:
            with st.spinner("Contacting Gemini..."):
                prompt = (
                    "You are a helpful, evidence-based medical assistant. Answer concisely and include "
                    "a short disclaimer that this is not medical advice.\n\nQuestion:\n"
                    + question
                )
                try:
                    answer = ask_gemini(prompt)
                except Exception as e:
                    st.error(f"Error calling Gemini: {e}")
                else:
                    st.subheader("Answer")
                    st.write(answer)
                    st.subheader("Sources (example list)")
                    example_sources = ["PubMed", "Mayo Clinic", "WHO", "CDC", "NIH"]
                    for i, s in enumerate(example_sources[:num_sources], start=1):
                        st.write(f"{i}. {s}")

st.caption("Disclaimer: This is informational only — consult healthcare professionals.")
