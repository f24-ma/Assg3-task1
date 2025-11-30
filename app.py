# app.py
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Medical RAG (Simple)", layout="wide")

# The ONLY correct model name for the new API:
MODEL_NAME = "gemini-1.5-flash"

# Sidebar
st.sidebar.header("Settings / API")
api_key = st.sidebar.text_input(
    "Gemini API Key (or set via Secrets)",
    type="password"
)

# Configure API (correct way)
if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("Gemini configured ✔")
    except Exception as e:
        st.sidebar.error(f"API Configuration failed: {e}")
else:
    st.sidebar.warning("Enter API Key to continue.")

# UI
st.title("Medical RAG — Simple")
question = st.text_input("Enter your medical question:")
num_sources = st.slider("Sources to show (fake list)", 1, 5, 3)


def ask_gemini(prompt: str) -> str:
    """The ONLY correct call for new Gemini API."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Display exact Gemini error
        raise RuntimeError(str(e))


if st.button("Find Answer"):
    if not api_key:
        st.warning("Please add API key first!")
    elif question.strip() == "":
        st.warning("Ask a question first!")
    else:
        with st.spinner("Thinking..."):
            prompt = (
                "You are a helpful medical assistant. "
                "Answer concisely and include a short disclaimer that this is not medical advice.\n\n"
                f"Question: {question}"
            )
            try:
                answer = ask_gemini(prompt)
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources (example list)")
                sources = ["PubMed", "WHO", "CDC", "NIH", "Mayo Clinic"]
                for i in range(num_sources):
                    st.write(f"{i+1}. {sources[i]}")

            except Exception as e:
                st.error(f"Error calling Gemini: {e}")

st.caption("Disclaimer: This is informational only — consult healthcare professionals.")
