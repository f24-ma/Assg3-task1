# app.py
import streamlit as st
import google.generativeai as genai
from typing import List, Any

st.set_page_config(page_title="Medical RAG — Diagnostic", layout="wide")
st.title("Medical RAG — diagnose model access (fix 404)")

st.sidebar.header("Settings / API")
api_key = st.sidebar.text_input("Gemini API Key (or set via Secrets)", type="password")

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("Gemini configured ✔")
    except Exception as e:
        st.sidebar.error(f"Configuration failed: {e}")
        st.stop()
else:
    st.sidebar.warning("Provide API key (or set via Streamlit secrets) to continue.")
    st.stop()

# Helpful info / instructions
st.markdown(
    "This page will list **models visible to your API key**. If you see none, "
    "your key likely lacks permission, the Generative AI API is not enabled, or billing is off."
)

def extract_models_from_response(resp: Any) -> List[str]:
    """Normalize different shapes returned by various genai versions."""
    if resp is None:
        return []
    # dict/list shapes
    if isinstance(resp, dict):
        # common shapes: {"models": [...]}
        if "models" in resp and isinstance(resp["models"], list):
            # models may be dicts with name field
            out = []
            for m in resp["models"]:
                if isinstance(m, str):
                    out.append(m)
                elif isinstance(m, dict) and "name" in m:
                    out.append(m["name"])
            return out
    # some libs return an object with .models or .models property
    if hasattr(resp, "models"):
        mlist = getattr(resp, "models")
        out = []
        try:
            for m in mlist:
                if isinstance(m, str):
                    out.append(m)
                elif hasattr(m, "name"):
                    out.append(getattr(m, "name"))
                elif isinstance(m, dict) and "name" in m:
                    out.append(m["name"])
        except Exception:
            pass
        return out
    # fallback: if resp is list of strings
    if isinstance(resp, list):
        return [str(x) for x in resp]
    return []

def list_models_safe() -> (List[str], str):
    """
    Try multiple list-models call styles and return (models, message).
    """
    attempts = []
    # Attempt A: genai.list_models()
    try:
        resp = genai.list_models()
        models = extract_models_from_response(resp)
        if models:
            return models, "Listed via genai.list_models()"
        attempts.append(("list_models", resp))
    except Exception as e:
        attempts.append(("list_models_exc", str(e)))

    # Attempt B: genai.Model.list() or genai.Model.list_models()
    try:
        Model = getattr(genai, "Model", None)
        if Model is not None and hasattr(Model, "list"):
            resp = Model.list()
            models = extract_models_from_response(resp)
            if models:
                return models, "Listed via genai.Model.list()"
            attempts.append(("Model.list", resp))
        elif Model is not None and hasattr(Model, "list_models"):
            resp = Model.list_models()
            models = extract_models_from_response(resp)
            if models:
                return models, "Listed via genai.Model.list_models()"
            attempts.append(("Model.list_models", resp))
    except Exception as e:
        attempts.append(("Model_list_exc", str(e)))

    # Attempt C: some libs expose get_models or listAvailableModels
    for name in ("get_models", "listAvailableModels", "models"):
        try:
            fn = getattr(genai, name, None)
            if fn is not None:
                resp = fn()
                models = extract_models_from_response(resp)
                if models:
                    return models, f"Listed via genai.{name}()"
                attempts.append((name, resp))
        except Exception as e:
            attempts.append((name + "_exc", str(e)))

    # No models found — return attempts summary for debug
    msg = "No models discovered. Attempts:\n" + "\n".join(f"{a}: {b}" for a, b in attempts[:10])
    return [], msg

models, message = list_models_safe()
st.subheader("Model discovery")
if models:
    st.success(f"Found {len(models)} model(s) — {message}")
    st.write(models)
else:
    st.error("No models found for this API key.")
    st.code(message)

# Provide fallback candidate list (legacy names) so user can try them
fallback_candidates = [
    "models/text-bison-001",
    "models/chat-bison-001",
    "text-bison-001",
    "chat-bison-001",
    "models/text-bison-001:2023-11-01"  # older variant
]

# Combine discovered with fallback (unique)
combined = list(dict.fromkeys(models + fallback_candidates))

st.subheader("Choose model to use")
selected_model = st.selectbox("Select a model", combined, index=0 if combined else None)

st.subheader("Test the selected model (run a short query)")
question = st.text_input("Enter a quick test question:", value="What are common symptoms of pneumonia?")
max_tokens = st.slider("Max tokens (response max)", 64, 1024, 256)

def try_call_generate_text(model_name: str, prompt: str) -> str:
    """
    Call either legacy generate_text or new GenerativeModel depending on availability.
    Return text or raise an exception with server message.
    """
    # Try legacy v1beta generate_text (many users on older clients)
    try:
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model_name, prompt=prompt, max_output_tokens=max_tokens, temperature=0.2)
            # many legacy responses have resp.generations[0].text
            if hasattr(resp, "generations"):
                gens = getattr(resp, "generations")
                if isinstance(gens, list) and gens:
                    first = gens[0]
                    # first may be object with .text
                    if hasattr(first, "text"):
                        return first.text
                    elif isinstance(first, dict) and "text" in first:
                        return first["text"]
            # fallback to dict-like shapes
            if isinstance(resp, dict):
                for key in ("output", "text", "content"):
                    if key in resp:
                        return resp[key] if isinstance(resp[key], str) else str(resp[key])
            # try .text attribute
            if hasattr(resp, "text"):
                return getattr(resp, "text")
            return str(resp)
    except Exception as e:
        # bubble up exception for debugging
        raise

    # Try new GenerativeModel API
    try:
        if hasattr(genai, "GenerativeModel"):
            m = genai.GenerativeModel(model_name)
            # some libs accept string, some need dict prompt
            try:
                r = m.generate_content(prompt)
            except TypeError:
                r = m.generate_content({"prompt": prompt})
            if hasattr(r, "text"):
                return r.text
            if isinstance(r, dict):
                for key in ("output", "content", "text"):
                    if key in r:
                        return r[key]
            return str(r)
    except Exception as e:
        raise

    raise RuntimeError("No supported generate method found in the installed genai client.")

if st.button("Run test query"):
    if not selected_model:
        st.warning("Select a model first.")
    else:
        with st.spinner("Calling model..."):
            try:
                out = try_call_generate_text(selected_model, question)
            except Exception as e:
                st.error("Error calling model:")
                st.exception(e)
                st.info(
                    "If you see 404 or 'not found', that model name is not available to your API key. "
                    "Try choosing another model from the dropdown or check your API key / project permissions."
                )
            else:
                st.subheader("Model output")
                st.write(out)

st.caption("If you get “404 requested entity not found”, the model you selected is not reachable with this API key. Check project, billing, and API enablement.")
