# app.py
import streamlit as st
import google.generativeai as genai
from typing import Any, List

st.set_page_config(page_title="Medical RAG — debug & run", layout="wide")
st.title("Medical RAG — pick a working model (debug)")

st.sidebar.header("Settings / API")
api_key = st.sidebar.text_input("AI Studio / API Key (do NOT share)", type="password")

if not api_key:
    st.sidebar.warning("Enter your API key to list models and run.")
    st.stop()

# configure client
try:
    genai.configure(api_key=api_key)
    st.sidebar.success("Configured ✔")
except Exception as e:
    st.sidebar.error(f"Configure failed: {e}")
    st.stop()

###########################
# Utilities
###########################
def normalize_models(resp: Any) -> List[str]:
    """Return a list of model name strings from different response shapes."""
    if resp is None:
        return []
    # dict with "models"
    if isinstance(resp, dict) and "models" in resp:
        out = []
        for m in resp["models"]:
            if isinstance(m, str):
                out.append(m)
            elif isinstance(m, dict) and "name" in m:
                out.append(m["name"])
        return out
    # object with .models
    if hasattr(resp, "models"):
        out = []
        for m in getattr(resp, "models"):
            if isinstance(m, str):
                out.append(m)
            elif hasattr(m, "name"):
                out.append(getattr(m, "name"))
            elif isinstance(m, dict) and "name" in m:
                out.append(m["name"])
        return out
    # list of strings
    if isinstance(resp, list):
        return [str(x) for x in resp]
    # fallback single item
    try:
        return [str(resp)]
    except Exception:
        return []

def list_models_try() -> (List[str], str):
    """Try multiple list-models entry points and return (models, msg)."""
    attempts = []
    # try genai.list_models()
    try:
        resp = genai.list_models()
        models = normalize_models(resp)
        if models:
            return models, "Listed with genai.list_models()"
        attempts.append(("list_models", str(resp)))
    except Exception as e:
        attempts.append(("list_models_exc", str(e)))

    # try Model.list or model listing helpers
    try:
        Model = getattr(genai, "Model", None)
        if Model is not None:
            if hasattr(Model, "list"):
                resp = Model.list()
                models = normalize_models(resp)
                if models:
                    return models, "Listed with genai.Model.list()"
                attempts.append(("Model.list", str(resp)))
            elif hasattr(Model, "list_models"):
                resp = Model.list_models()
                models = normalize_models(resp)
                if models:
                    return models, "Listed with genai.Model.list_models()"
                attempts.append(("Model.list_models", str(resp)))
    except Exception as e:
        attempts.append(("Model_list_exc", str(e)))

    # last try some other names
    for name in ("get_models", "listAvailableModels", "models"):
        try:
            fn = getattr(genai, name, None)
            if fn is not None:
                resp = fn()
                models = normalize_models(resp)
                if models:
                    return models, f"Listed with genai.{name}()"
                attempts.append((name, str(resp)))
        except Exception as e:
            attempts.append((name + "_exc", str(e)))

    msg = "No models discovered. Attempts:\n" + "\n".join(f"{a}: {b}" for a, b in attempts[:8])
    return [], msg

def extract_text(resp: Any) -> str:
    """Try common response shapes to extract readable text."""
    if resp is None:
        return ""
    # attribute text-like names
    for attr in ("text", "output_text", "content", "result", "answer"):
        if hasattr(resp, attr):
            v = getattr(resp, attr)
            if isinstance(v, str):
                return v
            if hasattr(v, "text"):
                return getattr(v, "text")
    # legacy: resp.generations[0].text
    if hasattr(resp, "generations"):
        gens = getattr(resp, "generations")
        if isinstance(gens, (list, tuple)) and gens:
            first = gens[0]
            if hasattr(first, "text"):
                return getattr(first, "text")
            if isinstance(first, dict) and "text" in first:
                return first["text"]
    # dict
    if isinstance(resp, dict):
        for k in ("output", "text", "content"):
            if k in resp:
                return resp[k] if isinstance(resp[k], str) else str(resp[k])
        if "candidates" in resp and isinstance(resp["candidates"], list) and resp["candidates"]:
            c = resp["candidates"][0]
            if isinstance(c, dict) and "content" in c:
                return c["content"]
    return str(resp)

###########################
# Page: list models
###########################
st.header("1) Discover models your key can access")
models, msg = list_models_try()
if models:
    st.success(f"Found {len(models)} model(s) — {msg}")
    st.write(models)
else:
    st.error("No models found for this key.")
    st.code(msg)

# fallback candidates to let user try if list empty
candidates = models + [
    "models/gemini-1.5-flash",    # new API style (some clients)
    "gemini-1.5-flash",           # plain name (some clients)
    "models/text-bison-001",      # v1beta legacy
    "text-bison-001",             # legacy plain
    "models/chat-bison-001",
]
# unique
seen = []
for x in candidates:
    if x not in seen:
        seen.append(x)
candidates = seen

st.header("2) Choose model to use (pick one shown above if available)")
selected = st.selectbox("Select model", candidates, index=0 if candidates else None)

st.header("3) Quick test query")
question = st.text_input("Enter question to test:", value="What are common symptoms of pneumonia?")
max_tokens = st.slider("Max tokens", 64, 512, 256)

def try_generate(model_name: str, prompt: str) -> str:
    """Try common generation calls and return text or raise last exception."""
    last_exc = None

    # Helper to try model name shapes: string, "models/...", {"name": ...}
    model_variants = [model_name]
    if not str(model_name).startswith("models/"):
        model_variants.append(f"models/{model_name}")
    # also try dict form
    model_variants.append({"name": model_name})
    if not str(model_name).startswith("models/"):
        model_variants.append({"name": f"models/{model_name}"})

    # Try legacy generate_text (v1beta)
    try:
        if hasattr(genai, "generate_text"):
            for m in model_variants:
                try:
                    resp = genai.generate_text(model=m, prompt=prompt, max_output_tokens=max_tokens, temperature=0.2)
                    return extract_text(resp)
                except Exception as e:
                    last_exc = e
    except Exception as e:
        last_exc = e

    # Try new GenerativeModel API
    try:
        if hasattr(genai, "GenerativeModel"):
            for m in model_variants:
                try:
                    # if m is dict, pass m (some libs accept dict)
                    if isinstance(m, dict):
                        gm = genai.GenerativeModel(m)  # try building with dict
                    else:
                        gm = genai.GenerativeModel(m)
                    # call generate_content with string or dict
                    try:
                        r = gm.generate_content(prompt)
                    except TypeError:
                        r = gm.generate_content({"prompt": prompt, "max_output_tokens": max_tokens})
                    return extract_text(r)
                except Exception as e:
                    last_exc = e
    except Exception as e:
        last_exc = e

    # Try generic genai.generate
    try:
        if hasattr(genai, "generate"):
            for m in model_variants:
                try:
                    resp = genai.generate(model=m, input=prompt, max_output_tokens=max_tokens)
                    return extract_text(resp)
                except Exception as e:
                    last_exc = e
    except Exception as e:
        last_exc = e

    raise RuntimeError(f"All attempts failed. Last error: {last_exc}")

if st.button("Run test query"):
    if not selected:
        st.warning("Select a model first.")
    else:
        with st.spinner("Calling model..."):
            try:
                out = try_generate(selected, question)
            except Exception as e:
                st.error("Error calling model (full server error below):")
                st.exception(e)
                st.info(
                    "If you see 404 model not found: pick a different model from the dropdown that appeared above, "
                    "or confirm you used the AI Studio API key (not a Cloud key), or use a Google Cloud project with billing enabled."
                )
            else:
                st.subheader("Model output")
                st.write(out)

st.caption("Do NOT paste your API key anywhere public. If no models appear, your key lacks model access; create an AI Studio key or use a Google Cloud key with billing enabled.")
