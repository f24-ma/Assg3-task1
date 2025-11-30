import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Medical RAG (Simple)", layout="wide")

st.sidebar.header("Settings / API")
api_key = st.sidebar.text_input("AI Studio API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    st.sidebar.success("Configured ✔")
else:
    st.sidebar.warning("Enter AI Studio API key to continue.")

MODEL_NAME = "gemini-1.5-flash"

st.title("Medical RAG — Simple (AI Studio Version)")
question = st.text_input("Enter your medical question:")

def ask_gemini(q):
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(q)
    return response.text

if st.button("Find Answer"):
    if not api_key:
        st.warning("Please enter your AI Studio key.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = ask_gemini(question)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(str(e))

st.caption("Informational only — not medical advice.")
