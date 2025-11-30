# app.py (fixed: handles generator responses, stable error messages, auto-selects first listed model)
import streamlit as st
import google.generativeai as genai
from typing import Any, List, Iterable

st.set_page_config(page_title="Medical RAG — debug & run (fixed)", layout="wide")
st.title("Medical RAG — pick a working model (fixed)")

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
def to_list_safe(obj: Any) -> List[Any]:
    """Return a concrete list for lists, iterables, generator objects, or single items."""
    # Already a list
    if isinstance(obj, list):
        return obj
    # Generator or iterable (but not str)
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict)):
        try:
            return list(obj)
        except Exception:
            pass
    # Single dict or object -> return as single-element list
    return [obj]

def normalize_models(resp: Any) -> List[str]:
    """Return a list of model name strings from different response shapes."""
    if resp is None:
        return []
    # If resp is a generator/iterable produce list
    items = to_list_safe(resp)

    out = []
    for m in items:
        # string item
        if isinstance(m, str):
            out.append(m)
        # object with name attribute
        elif hasattr(m, "name"):
            try:
                out.append(getattr(m, "name"))
            except Exception:
                out.append(str(m))
        # dict with 'name'
        elif isinstance(m, dict) and "name" in m:
            out.append(str(m["name"]))
        # dict or other -> try to stringify useful fields
        else:
            out.append(str(m))
    # final cleaning: unique preserving order
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen

def list_models_try() -> (List[str], str):
    """Try multiple list-models entry points and return (models, msg)."""
    attempts = []
    # try genai.list_models()
    try:
        resp = genai.list_models()
        models = normalize_models(resp)
        if models:
            return models, "Listed with genai.list_models()"
        attempts.append(("list_models", repr(resp)))
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
                attempts.append(("Model.list", repr(resp)))
            elif hasattr(Model, "list_models"):
                resp = Model.list_models()
                models = normalize_models(resp)
                if models:
                    return models, "Listed with genai.Model.list_models()"
                attempts.append(("Model.list_models", repr(resp)))
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
                attempts.append((name, repr(resp)))
        except Exception as e:
            attempts.append((name + "_exc", str(e)))

    msg = "No models discovered. Attempts:\n" + "\n".join(f"{a}: {b}" for a, b in attempts[:10])
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
        try:
            gens = getattr(resp, "generations")
            if isinstance(gens, (list, tuple)) and gens:
                first = gens[0]
                if hasattr(first, "text"):
                    return getattr(first, "text")
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
        except Exception:
            pass
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

# If we discovered one or more, auto-select the first listed model to save you clicking
if models:
    default_index = 0
else:
    default_index = 0

# provide sensible candidates (discovered models first)
candidates = models + [
    "models/gemini-1.5-flash",
    "gemini-1.5-flash",
    "models/text-bison-001",
    "text-bison-001",
    "models/chat-bison-001",
]

# unique preserve order
seen = []
for x in candidates:
    if x not in seen:
        seen.append(x)
candidates = seen

st.header("2) Choose model to use (pick one shown above if available)")
selected = st.selectbox("Select model", candidates, index=default_index if candidates else None)

st.header("3) Quick test query")
question = st.text_input("Enter question to test:", value="What are common symptoms of pneumonia?")
max_tokens = st.slider("Max tokens", 64, 512, 256)

def try_generate(model_name: str, prompt: str) -> str:
    """Try common generation calls and return text or raise last exception (stringified)."""
    last_exc = None

    # Helper to create model name variants (strings and dicts)
    model_variants = []
    # ensure model_name as string
    try:
        mstr = str(model_name)
    except Exception:
        mstr = model_name

    model_variants.append(mstr)
    if not mstr.startswith("models/"):
        model_variants.append(f"models/{mstr}")
    # dict form variants (string wrapped in dict)
    model_variants.append({"name": mstr})
    if not mstr.startswith("models/"):
        model_variants.append({"name": f"models/{mstr}"})

    # Try legacy generate_text (v1beta)
    try:
        if hasattr(genai, "generate_text"):
            for m in model_variants:
                try:
                    # ensure we pass a proper model param (string or dict as accepted)
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
                    # For dict variant, try passing dict to constructor if the lib supports it
                    gm = genai.GenerativeModel(m)
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

    # Always raise stringified error to avoid type-concat issues
    raise RuntimeError("All attempts failed. Last error: " + (repr(last_exc) if last_exc is not None else "None"))

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
                    "or confirm you used the AI Studio API key (not a Cloud key)."
                )
            else:
                st.subheader("Model output")
                st.write(out)

st.caption("Do NOT paste your API key anywhere public. If no models appear, your key lacks model access; create an AI Studio key or use a Google Cloud key with billing enabled.")
