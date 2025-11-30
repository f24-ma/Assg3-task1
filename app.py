# app.py
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Medical RAG (Simple)", layout="wide")

# Use the OLD model name fully compatible with your installed API (v1beta)
MODEL_NAME = "models/text-bison-001"

st.sidebar.header("Settings / API")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Configure OLD API
if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("Gemini configured ✔")
    except Exception as e:
        st.sidebar.error(f"Configuration error: {e}")
else:
    st.sidebar.warning("Enter API key to continue.")

st.title("Medical RAG — Simple (Legacy API)")
question = st.text_input("Enter your medical question:")
num_sources = st.slider("Sources to show (fake list)", 1, 5, 3)


def ask_gemini(prompt: str) -> str:
    """Uses legacy v1beta API format — fully compatible with your device."""
    try:
        response = genai.generate_text(
            model=MODEL_NAME,
            prompt=prompt,
            temperature=0.2,
            max_output_tokens=256
        )
        return response.generations[0].text
    except Exception as e:
        raise RuntimeError(str(e))


if st.button("Find Answer"):
    if not api_key:
        st.warning("Please provide API key.")
    elif question.strip() == "":
        st.warning("Enter a question.")
    else:
        with st.spinner("Thinking..."):
            prompt = (
                "You are a helpful medical assistant. "
                "Answer briefly and include a disclaimer that this is not medical advice.\n\n"
                f"Question: {question}"
            )
            try:
                answer = ask_gemini(prompt)
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources (example list)")
                fake_sources = ["PubMed", "WHO", "CDC", "NIH", "Mayo Clinic"]
                for i in range(num_sources):
                    st.write(f"{i+1}. {fake_sources[i]}")
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")

st.caption("Disclaimer: Informational only — consult healthcare professionals.")
