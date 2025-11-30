import streamlit as st
import google.generativeai as genai
import os
from typing import List, Dict, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Medical RAG System",
    layout="wide"
)

# ============================================================================
# SIDEBAR - API KEY CONFIGURATION
# ============================================================================

st.sidebar.title("Settings")
st.sidebar.markdown("API Key")

api_key = st.sidebar.text_input(
    "Google AI Studio API Key:",
    type="password"
)

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("Configured ✔")
    except Exception as e:
        st.sidebar.error(f"Configuration failed: {str(e)}")
        st.stop()
else:
    st.sidebar.warning("Enter API key to continue")
    st.info("Enter your Google AI Studio API key in the sidebar")
    st.stop()

# ============================================================================
# MODEL DISCOVERY
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Selection")

@st.cache_data(ttl=3600)
def discover_models(_api_key: str) -> List[str]:
    """Discover available models"""
    try:
        models = genai.list_models()
        model_names = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_names.append(model.name)
        return sorted(model_names)
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        return []

with st.sidebar.expander("Available Models", expanded=False):
    if st.button("Refresh"):
        st.cache_data.clear()
    
    available_models = discover_models(api_key)
    
    if available_models:
        st.success(f"Found {len(available_models)} models")
        st.json(available_models)
    else:
        st.error("No models found")

# ============================================================================
# MODEL SELECTION
# ============================================================================

def auto_select_best_model(models: List[str]) -> str:
    """Auto-select the best available text generation model"""
    priority_models = [
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-exp",
        "models/gemini-pro-latest",
        "models/gemini-flash-latest",
    ]
    
    for preferred in priority_models:
        if preferred in models:
            return preferred
    
    # Fallback: return first gemini model
    for model in models:
        if "gemini" in model.lower() and "embedding" not in model.lower():
            return model
    
    return models[0] if models else "models/gemini-2.5-flash"

st.sidebar.markdown("**Selected Model:**")

if available_models:
    auto_selected = auto_select_best_model(available_models)
    st.sidebar.info(f"{auto_selected}")
    
    selected_model = st.sidebar.selectbox(
        "Choose model:",
        options=available_models,
        index=available_models.index(auto_selected) if auto_selected in available_models else 0
    )
else:
    st.sidebar.error("No models available")
    st.stop()

# ============================================================================
# MEDICAL KNOWLEDGE BASE (Simple RAG)
# ============================================================================

MEDICAL_KNOWLEDGE = {
    "diabetes": """
    Diabetes Mellitus is a chronic metabolic disorder characterized by high blood glucose levels.
    
    Types:
    - Type 1: Autoimmune destruction of pancreatic beta cells
    - Type 2: Insulin resistance and relative insulin deficiency
    - Gestational: Occurs during pregnancy
    
    Symptoms: Polyuria, polydipsia, polyphagia, weight loss, fatigue
    Diagnosis: Fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, OGTT ≥200 mg/dL
    Treatment: Lifestyle modification, oral hypoglycemics, insulin therapy
    Complications: Retinopathy, nephropathy, neuropathy, cardiovascular disease
    """,
    
    "hypertension": """
    Hypertension (High Blood Pressure) is a chronic condition where blood pressure is consistently elevated.
    
    Classification:
    - Normal: <120/80 mmHg
    - Elevated: 120-129/<80 mmHg
    - Stage 1: 130-139/80-89 mmHg
    - Stage 2: ≥140/90 mmHg
    
    Risk Factors: Age, obesity, smoking, high sodium intake, stress, genetics
    Symptoms: Often asymptomatic (silent killer), headache, dizziness, blurred vision
    Diagnosis: Multiple BP readings on different occasions
    Treatment: Lifestyle changes, ACE inhibitors, ARBs, beta-blockers, diuretics, calcium channel blockers
    Complications: Stroke, heart attack, heart failure, kidney disease
    """,
    
    "asthma": """
    Asthma is a chronic inflammatory disease of the airways causing reversible airway obstruction.
    
    Pathophysiology: Airway inflammation, bronchial hyperresponsiveness, mucus hypersecretion
    Triggers: Allergens, cold air, exercise, infections, stress, pollution
    
    Symptoms: Wheezing, shortness of breath, chest tightness, coughing (worse at night)
    Diagnosis: Spirometry (FEV1/FVC ratio), peak flow monitoring, bronchodilator response
    
    Treatment:
    - Quick relief: Short-acting beta-agonists (albuterol)
    - Long-term control: Inhaled corticosteroids, long-acting beta-agonists, leukotriene modifiers
    - Severe: Biologics (omalizumab, mepolizumab)
    
    Action Plan: Identify triggers, monitor peak flow, use medications correctly, emergency plan
    """,
    
    "covid19": """
    COVID-19 (Coronavirus Disease 2019) is caused by SARS-CoV-2 virus.
    
    Transmission: Respiratory droplets, aerosols, contact with contaminated surfaces
    Incubation: 2-14 days (average 5 days)
    
    Symptoms:
    - Common: Fever, cough, fatigue, loss of taste/smell, sore throat
    - Severe: Difficulty breathing, chest pain, confusion, bluish lips
    
    Diagnosis: RT-PCR, rapid antigen test, antibody test
    
    Treatment:
    - Mild: Supportive care, rest, hydration, fever reducers
    - Moderate: Antivirals (Paxlovid), monoclonal antibodies
    - Severe: Oxygen therapy, remdesivir, dexamethasone, ICU care
    
    Prevention: Vaccination, masks, hand hygiene, social distancing, ventilation
    Complications: Long COVID, ARDS, multiorgan failure, death
    """,
    
    "pneumonia": """
    Pneumonia is an infection of the lung parenchyma causing inflammation of air sacs.
    
    Types:
    - Community-acquired (CAP): Most common, Streptococcus pneumoniae
    - Hospital-acquired (HAP): Develops 48+ hours after admission
    - Ventilator-associated (VAP): Occurs in mechanically ventilated patients
    - Aspiration: Inhalation of gastric/oropharyngeal contents
    
    Causes: Bacteria (most common), viruses, fungi, atypical organisms
    
    Symptoms: Fever, productive cough, chest pain, shortness of breath, fatigue
    Physical Exam: Crackles, bronchial breath sounds, dullness to percussion
    
    Diagnosis: Chest X-ray, CT scan, sputum culture, blood cultures, CBC
    
    Treatment:
    - Empiric antibiotics: Macrolides, fluoroquinolones, beta-lactams
    - Supportive: Oxygen, IV fluids, fever control
    - Severe: ICU admission, mechanical ventilation
    
    Prevention: Pneumococcal vaccine, influenza vaccine, smoking cessation
    """,
}

def retrieve_relevant_context(query: str) -> str:
    """Simple keyword-based retrieval from knowledge base"""
    query_lower = query.lower()
    relevant_docs = []
    
    # Check for keyword matches
    for condition, info in MEDICAL_KNOWLEDGE.items():
        if condition in query_lower or any(word in query_lower for word in condition.split()):
            relevant_docs.append(f"=== {condition.upper()} ===\n{info}")
    
    # If no specific match, return all knowledge (fallback)
    if not relevant_docs:
        for condition, info in MEDICAL_KNOWLEDGE.items():
            relevant_docs.append(f"=== {condition.upper()} ===\n{info}")
    
    return "\n\n".join(relevant_docs)

# ============================================================================
# QUERY GENERATION FUNCTION
# ============================================================================

def generate_response(model_name: str, question: str, max_tokens: int = 2048) -> str:
    """Generate response using Gemini model with RAG context"""
    try:
        # Clean model name (remove "models/" prefix if present)
        clean_model_name = model_name.replace("models/", "")
        
        # Retrieve relevant context
        context = retrieve_relevant_context(question)
        
        # Create enhanced prompt with RAG context
        enhanced_prompt = f"""You are a medical AI assistant. Use the following medical knowledge to answer the question accurately and professionally.

MEDICAL KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

Please provide a clear, accurate, and professional medical response based on the knowledge above. If the question is outside the provided knowledge, use your general medical knowledge but indicate that clearly.

IMPORTANT: This is for educational purposes only. Always advise consulting healthcare professionals for actual medical decisions."""

        # Initialize model
        model = genai.GenerativeModel(clean_model_name)
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.7,
        )
        
        # Generate response
        response = model.generate_content(
            enhanced_prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return f"❌ Model not found. Please select a different model from the dropdown. Error: {error_msg}"
        elif "quota" in error_msg.lower():
            return f"❌ API quota exceeded. Please check your Google AI Studio quota. Error: {error_msg}"
        elif "api key" in error_msg.lower():
            return f"❌ API key error. Please verify your API key is correct. Error: {error_msg}"
        else:
            return f"❌ Error generating response: {error_msg}"

# ============================================================================
# MAIN APP UI
# ============================================================================

st.title("Medical RAG System")
st.markdown("AI-Powered Medical Information Retrieval")

st.info("Available Topics: Diabetes, Hypertension, Asthma, COVID-19, Pneumonia")

# Quick test section
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Test")

test_question = st.sidebar.text_input(
    "Question:",
    value="What are the symptoms of diabetes?"
)

max_tokens = st.sidebar.slider(
    "Max tokens",
    min_value=256,
    max_value=8192,
    value=2048,
    step=256
)

if st.sidebar.button("Generate", type="primary"):
    if not test_question.strip():
        st.sidebar.error("Enter a question")
    else:
        with st.spinner("Generating..."):
            response = generate_response(selected_model, test_question, max_tokens)
            
            st.markdown("---")
            st.markdown("### Response:")
            st.markdown(response)
            
            with st.expander("Retrieved Context"):
                context = retrieve_relevant_context(test_question)
                st.text(context)

# Main chat interface
st.markdown("---")
st.markdown("### Chat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(selected_model, prompt, max_tokens)
            st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Instructions:
1. Enter API key
2. Select model (auto-selected)
3. Ask medical questions
4. Get AI responses with context

**Note:** Educational purposes only.
""")

st.sidebar.markdown("---")
st.sidebar.caption("Streamlit & Google Gemini")
