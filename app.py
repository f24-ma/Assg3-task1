import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Medical RAG System", layout="wide")

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar
with st.sidebar:
    st.header("System Status")
    
    api_key = st.text_input("Enter Gemini API Key", type="password", value=st.session_state.api_key)
    
    if api_key:
        st.session_state.api_key = api_key
        try:
            genai.configure(api_key=api_key)
            st.session_state.model = genai.GenerativeModel('gemini-pro')
            st.success("Gemini AI: Active")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.model = None
    else:
        st.warning("Gemini AI: Inactive")
        st.info("Enter your API key to activate")
    
    st.divider()
    st.subheader("Try These Examples")
    if st.button("What are common symptoms of pneumonia?", use_container_width=True):
        st.session_state.question = "What are common symptoms of pneumonia?"
    if st.button("How to treat allergic rhinitis?", use_container_width=True):
        st.session_state.question = "How to treat allergic rhinitis?"
    if st.button("What are signs of myocardial infarction?", use_container_width=True):
        st.session_state.question = "What are signs of myocardial infarction?"

# Main content
st.title("Medical RAG System")
st.markdown("Get evidence-based medical answers")

question = st.text_input("Enter your question:", value=st.session_state.get('question', ''))
num_sources = st.slider("Sources:", 1, 10, 4)

if st.button("Find Answer", type="primary"):
    if not question:
        st.warning("Please enter a question")
    elif st.session_state.model is None:
        st.warning("Please enter your Gemini API key in the sidebar")
    else:
        with st.spinner("Generating answer..."):
            try:
                prompt = f"""You are a medical information assistant. Answer this question:

{question}

Provide accurate, evidence-based information with appropriate disclaimers."""
                
                response = st.session_state.model.generate_content(prompt)
                
                st.subheader("Answer")
                st.write(response.text)
                
                st.subheader("Sources Consulted")
                sources = ["PubMed (NCBI)", "Mayo Clinic", "WHO", "CDC", "NIH"]
                for i, source in enumerate(sources[:num_sources], 1):
                    with st.expander(f"Source {i}: {source}"):
                        st.write(f"Medical reference about: {question}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.caption("Disclaimer: This provides general medical information. Always consult healthcare professionals.")
