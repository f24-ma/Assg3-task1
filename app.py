import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Medical RAG - Simple", layout="wide")

st.sidebar.header("System Status")

# Put your API key here or type in sidebar (do NOT commit real keys in github)
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

api_key = st.sidebar.text_input("Enter Gemini / GenAI API Key", type="password", value=st.session_state.api_key)
st.session_state.api_key = api_key

# Choose a model name that is commonly available for text gen.
# If this returns 404, you'll need to replace this with a model your account can access.
model_name = "models/text-bison-001"

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("API key set (not validated).")
    except Exception as e:
        st.sidebar.error(f"Configure error: {e}")

st.title("Medical RAG — Very Simple (fixed)")

question = st.text_input("Enter your question:", value="")
# Fix: use named args for slider
num_sources = st.slider("Sources:", min_value=1, max_value=10, value=4)

if st.button("Find Answer"):
    if not question:
        st.warning("Please enter a question.")
    elif not api_key:
        st.warning("Please enter your Gemini/GenAI API key in the sidebar.")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Simple prompt
                prompt = f"You are a medical assistant. Answer concisely and cite sources.\n\nQuestion: {question}"
                # Use a simple generate call. If your library uses different name, change this.
                resp = genai.generate_text(model=model_name, input=prompt)
                
                # resp may differ by library: check attributes
                text = getattr(resp, "text", None) or (resp.get("output") if isinstance(resp, dict) else None)
                if text is None:
                    # fallback try to print whole response
                    st.write("Raw response object:")
                    st.write(resp)
                else:
                    st.subheader("Answer")
                    st.write(text)
                
                st.subheader("Sources (example)")
                example_sources = ["PubMed", "WHO", "Mayo Clinic", "CDC", "NIH"]
                for s in example_sources[:num_sources]:
                    st.write(f
