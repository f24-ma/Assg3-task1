# app.py
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Medical RAG (simple)", layout="wide")

MODEL_NAME = "models/gemini-1.5-flash"  # recommended free-tier model

st.sidebar.header("Settings / API")
# DON'T hardcode your API key here. Put it in Streamlit secrets or paste it in the sidebar temporarily.
api_key = st.sidebar.text_input("Gemini API Key (or set via Secrets)", type="password", key="api_key_input")

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("Gemini configured ✔")
    except Exception as e:
        st.sidebar.error(f"Configuration error: {e}")
else:
    st.sidebar.warning("No API key provided — set it in Secrets or paste here")

st.title("Medical RAG — Simple")
st.markdown("Ask a medical question (informational only).")

question = st.text_input("Enter your question:", "")

num_sources = st.slider("Sources to show (fake list)", 1, 5, 3)

def ask_gemini(prompt, model=MODEL_NAME):
    """Try a couple of possible genai call styles depending on installed library version."""
    if not genai.api_key:
        raise RuntimeError("Gemini API key not configured (genai.configure).")
    # Try modern / common patterns. Wrap in try/except and return the first success.
    last_err = None
    try:
        # pattern A: generate_text (some library versions)
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=prompt)
            # try to extract text
            if hasattr(resp, "text"):
                return resp.text
            if isinstance(resp, dict):
                return resp.get("output", str(resp))
    except Exception as e:
        last_err = e

    try:
        # pattern B: generate_content using GenerativeModel()
        if hasattr(genai, "GenerativeModel"):
            m = genai.GenerativeModel(model)
            resp = m.generate_content(prompt)
            if hasattr(resp, "text"):
                return resp.text
            if isinstance(resp, dict):
                return resp.get("output", str(resp))
    except Exception as e:
        last_err = e

    try:
        # pattern C: generic generate (older/newer variants)
        if hasattr(genai, "generate"):
            resp = genai.generate(model=model, input=prompt)
            if hasattr(resp, "output_text"):
                return resp.output_text
            if isinstance(resp, dict):
                # try common keys
                return resp.get("output", resp.get("text", str(resp)))
    except Exception as e:
        last_err = e

    # If nothing worked, raise the last captured exception so UI shows it.
    raise RuntimeError(f"All genai calls failed. Last error: {last_err}")

if st.button("Find Answer"):
    if question.strip() == "":
        st.warning("Please type a question.")
    else:
        if not api_key:
            st.warning("Provide Gemini API key in the sidebar (or set via Streamlit Secrets).")
        else:
            with st.spinner("Contacting Gemini..."):
                prompt = (
                    "You are a helpful, evidence-based medical assistant. Answer concisely and include "
                    "a short disclaimer that this is not medical advice.\n\nQuestion:\n" + question
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
