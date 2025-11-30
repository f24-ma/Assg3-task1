# app.py — final fixed version (auto-select text model, safe error handling)
import streamlit as st
import google.generativeai as genai
from typing import Any, List, Iterable

st.set_page_config(page_title="Medical RAG — final fixed", layout="wide")
st.title("Medical RAG — final fixed (auto-select text model)")

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
# Helpers
###########################
def to_list_safe(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    # generator/iterable but not string/dict
    try:
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict)):
            return list(obj)
    except Exception:
        pass
    return [obj]

def normalize_models(raw: Any) -> List[str]:
    items = to_list_safe(raw)
    out = []
    for m in items:
        if isinstance(m, str):
            out.append(m)
        elif isinstance(m, dict) and "name" in m:
            out.append(str(m["name"]))
        elif hasattr(m, "name"):
            try:
                out.append(getattr(m, "name"))
            except Exception:
                out.append(str(m))
        else:
            out.append(str(m))
    # unique while preserving order
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen

def pick_text_model(models: List[str]) -> str:
    """
    Prefer Gemini text models. Avoid embeddings/images.
    Priority:
      1) models/gemini-*-flash or models/gemini-flash-latest or models/gemini-pro-latest
      2) any models/gemini-* that aren't embedding/imagen/veo
      3) text-bison legacy models
      4) first non-embedding model
    """
    if not models:
        return None
    lower = [m.lower() for m in models]

    # helper to test if model is embedding or image
    def is_embedding(m):
        return "embed" in m or "embedding" in m or "gecko" in m

    def is_image(m):
        return "imagen" in m or "veo" in m or "image" in m or "vision" in m

    # 1) best picks
    for i, m in enumerate(lower):
        if ("gemini" in m and ("flash" in m or "pro" in m or "latest" in m)) and not (is_embedding(m) or is_image(m)):
            return models[i]
    # 2) any gemini non-image, non-embed
    for i, m in enumerate(lower):
        if "gemini" in m and not (is_embedding(m) or is_image(m)):
            return models[i]
    # 3) legacy text-bison
    for i, m in enumerate(lower):
        if "text-bison" in m or "chat-bison" in m:
            return models[i]
    # 4) any non-embed, non-image
    for i, m in enumerate(lower):
        if not (is_embedding(m) or is_image(m)):
            return models[i]
    # fallback first model
    return models[0]

def safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        try:
            import reprlib
            return reprlib.repr(x)
        except Exception:
            return "<unstringifiable error>"

def extract_text(resp: Any) -> str:
    if resp is None:
        return ""
    for attr in ("text", "output_text", "content", "result", "answer"):
        if hasattr(resp, attr):
            v = getattr(resp, attr)
            if isinstance(v, str):
                return v
            if hasattr(v, "text"):
                return getattr(v, "text")
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
    if isinstance(resp, dict):
        for k in ("output", "text", "content"):
            if k in resp:
                return resp[k] if isinstance(resp[k], str) else safe_str(resp[k])
        if "candidates" in resp and isinstance(resp["candidates"], list) and resp["candidates"]:
            c = resp["candidates"][0]
            if isinstance(c, dict) and "content" in c:
                return c["content"]
    return safe_str(resp)

###########################
# Discover models
###########################
st.header("1) Discover models your key can access")
try:
    raw = genai.list_models()
    models = normalize_models(raw)
    if models:
        st.success(f"Found {len(models)} model(s)")
        st.write(models)
    else:
        st.error("No models found for this key.")
        st.code("genai.list_models() returned empty result.")
        st.stop()
except Exception as e:
    st.error("Failed to list models.")
    st.exception(e)
    st.stop()

# Auto-select a suitable text model
chosen_model = pick_text_model(models)
st.header("2) Selected model (auto-chosen best text model)")
st.write("Auto-chosen model:", chosen_model)
# Also allow manual override (dropdown)
selected = st.selectbox("Or choose another model manually", models, index=models.index(chosen_model) if chosen_model in models else 0)

###########################
# Test query
###########################
st.header("3) Quick test query")
question = st.text_input("Enter question to test:", value="What are common symptoms of pneumonia?")
max_tokens = st.slider("Max tokens", 64, 512, 256)

def try_generate(model_name: str, prompt: str) -> str:
    last_exc = None

    # build model variants
    model_variants = []
    try:
        mstr = str(model_name)
    except Exception:
        mstr = model_name
    model_variants.append(mstr)
    if not mstr.startswith("models/"):
        model_variants.append(f"models/{mstr}")
    model_variants.append({"name": mstr})
    if not mstr.startswith("models/"):
        model_variants.append({"name": f"models/{mstr}"})

    # Try new GenerativeModel first (preferred)
    try:
        if hasattr(genai, "GenerativeModel"):
            for m in model_variants:
                try:
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

    # Try legacy generate_text
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

    # Try generic generate
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

    raise RuntimeError("All attempts failed. Last error: " + safe_str(last_exc))

if st.button("Run test query"):
    if not selected:
        st.warning("Select a model first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Calling model..."):
            try:
                out = try_generate(selected, question)
            except Exception as e:
                st.error("Error calling model (full server error below):")
                st.exception(e)
                st.info("If you see a 404 model not found: choose a different model from the dropdown (pick a gemini-* flash/pro/latest model).")
            else:
                st.subheader("Model output")
                st.write(out)

st.caption("Do NOT paste your API key anywhere public. Use AI Studio key for no-billing access, or Google Cloud key with billing if you prefer Cloud.")
