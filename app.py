import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import html

# ---------------------------
# Configuration: Hugging Face Model ID
# ---------------------------
# We now use a model ID from the Hugging Face Hub
MODEL_ID = "BISHAL2301/summarizer"

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="Text Summarizer", initial_sidebar_state="expanded")

# ---------------------------
# Session defaults
# ---------------------------
st.session_state.setdefault("article", "")
st.session_state.setdefault("summary", "")
st.session_state.setdefault("model_loaded", False)
st.session_state.setdefault("tokenizer", None)
st.session_state.setdefault("model", None)
st.session_state.setdefault("device", None)
st.session_state.setdefault("generating", False)
st.session_state.setdefault("error", None)
st.session_state.setdefault("mode", "Paragraph")
st.session_state.setdefault("summary_length", 100)
st.session_state.setdefault("custom_bullets", 5)

# ---------------------------
# Utilities
# ---------------------------
def split_into_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    # Split by sentences, preserving delimiters, and handle newlines
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

# ---------------------------
# Load model (cached) from Hugging Face
# ---------------------------
@st.cache_resource
def load_model_cached(model_id: str):
    """Loads and caches the model/tokenizer from Hugging Face Hub."""
    # local_files_only is removed to allow downloading
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Initialize model loading
if not st.session_state["model_loaded"]:
    with st.spinner(f"Loading model '{MODEL_ID}'..."):
        try:
            # The check for a local directory is removed
            tokenizer, model, device = load_model_cached(MODEL_ID)
            st.session_state.update({
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
                "model_loaded": True
            })
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.session_state["model_loaded"] = False

# ---------------------------
# Callbacks
# ---------------------------
def start_summarization():
    """Callback to start the summarization process."""
    if not st.session_state.article.strip():
        st.warning("Please enter some text to summarize.")
        return
    st.session_state.generating = True
    st.session_state.summary = ""
    st.session_state.error = None

def clear_all():
    """Clears all inputs and outputs."""
    st.session_state.article = ""
    st.session_state.summary = ""
    st.session_state.generating = False
    st.session_state.error = None

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
:root {
    --primary-color: #0ea36b;
    --background-color: #F8F9FA;
    --card-background-color: #FFFFFF;
    --text-color: #31333F;
    --subtle-text-color: #64748B;
    --border-color: #DEE2E6;
}
body { background-color: var(--background-color); color: var(--text-color); }
@keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}
.loading-text { color: var(--subtle-text-color); font-weight: bold; animation: pulse 1.5s ease-in-out infinite; }
.stButton > button { background-color: var(--primary-color) !important; border: 1px solid var(--primary-color) !important; color: white !important; }
.stButton > button:hover { opacity: 0.8; }
.header { text-align:center; margin-bottom:14px; }
.header h1 { color:var(--primary-color); margin:0; }
.top-bar { background: var(--card-background-color); padding: 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid var(--border-color); box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.stTextArea textarea { background-color: #FFFFFF; color: var(--text-color); border: 1px solid var(--border-color); }
.text-area { border: 1px solid var(--border-color); padding: 12px; border-radius: 8px; background: var(--card-background-color); color: var(--text-color); min-height: 200px; display: flex; align-items: center; justify-content: center; }
.text-area-content { width: 100%; text-align: left; }
.footer-text { text-align:center; margin-top:12px; color: var(--subtle-text-color); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# UI Header
# ---------------------------
st.markdown("<div class='header'><h1>Text Summarizer</h1></div>", unsafe_allow_html=True)

# Top controls
st.markdown('<div class="top-bar">', unsafe_allow_html=True)
c1, c2 = st.columns([2, 3])
with c1:
    st.radio("Mode:", ["Paragraph", "Bullet Points", "Custom"], key="mode", horizontal=True)
with c2:
    st.slider("Summary Length", 50, 300, 100, key="summary_length")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Main columns
# ---------------------------
left_col, right_col = st.columns([3, 2])

with left_col:
    st.text_area("Enter your text to summarize", key="article", height=320, placeholder="Paste your text here...")

    if st.session_state.mode == "Custom":
        st.number_input(
            "Number of bullets", min_value=1, max_value=20, key="custom_bullets"
        )

    col1, col2 = st.columns(2)
    with col1:
        st.button("🚀 Summarize", on_click=start_summarization, use_container_width=True)
    with col2:
        st.button("🧹 Clear", on_click=clear_all, use_container_width=True)

# --- This is the main logic block that runs when generating ---
if st.session_state.generating:
    try:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model
        device = st.session_state.device
        max_len = st.session_state.summary_length
        mode = st.session_state.mode

        prompt = "summarize: " + st.session_state.article
        if mode == "Bullet Points":
            prompt += " in bullet points"
        elif mode == "Custom":
            prompt += f" in exactly {st.session_state.custom_bullets} concise bullet points"

        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_len,
                min_length=max(20, max_len // 4), # Ensure min_length is reasonable
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.session_state.summary = summary

    except Exception as e:
        st.session_state.error = f"Error during generation: {e}"
        st.session_state.summary = ""
    finally:
        # After generation (or error), turn off the generating flag
        st.session_state.generating = False
        st.rerun() # Rerun to update the display in the right column


with right_col:
    st.subheader("Summary Output")
    summary_placeholder = st.empty()

    # Display spinner, summary, error, or initial message
    if st.session_state.generating:
        summary_placeholder.markdown(
            "<div class='text-area'><div class='loading-text'>Generating summary...</div></div>",
            unsafe_allow_html=True
        )
    elif st.session_state.error:
        summary_placeholder.markdown(
            f"<div class='text-area'><div class='text-area-content' style='color:#D32F2F; text-align:center;'>⚠️ {html.escape(st.session_state.error)}</div></div>",
            unsafe_allow_html=True
        )
    elif st.session_state.summary:
        summary = st.session_state.summary
        mode = st.session_state.mode
        if mode in ["Bullet Points", "Custom"]:
            sentences = split_into_sentences(summary)
            if mode == "Custom":
                sentences = sentences[:st.session_state.custom_bullets]
            list_items = "".join(f"<li>{html.escape(s)}</li>" for s in sentences if s.strip())
            summary_placeholder.markdown(
                f"<div class='text-area'><div class='text-area-content'><ul>{list_items}</ul></div></div>",
                unsafe_allow_html=True
            )
        else: # Paragraph mode
            safe_html = "<div style='white-space:pre-wrap'>" + html.escape(summary) + "</div>"
            summary_placeholder.markdown(
                f"<div class='text-area'><div class='text-area-content'>{safe_html}</div></div>",
                unsafe_allow_html=True
            )
    else:
        summary_placeholder.markdown(
            "<div class='text-area'><div class='text-area-content' style='color:var(--subtle-text-color); text-align:center;'>Your generated summary will appear here.</div></div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("<div class='footer-text'>Made with ❤️ using Streamlit & Transformers</div>", unsafe_allow_html=True)
