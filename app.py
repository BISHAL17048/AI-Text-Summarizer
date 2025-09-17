import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="📝 Text Summarizer — Bishal Ray",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Load Model & Tokenizer
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    MODEL_NAME = "BISHAL2301/summarizer"  # Replace with your Hugging Face model repo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown(
    """
    <style>
    :root{
        --bg-gradient: linear-gradient(135deg, #f0f7ff 0%, #ffffff 50%, #f7fbf9 100%);
        --card-bg: #ffffff;
        --accent: #0b74de;
    }
    .stApp {
        background: var(--bg-gradient);
        padding-top: 20px;
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .header {
        text-align: center;
        padding: 18px 10px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .title {
        font-size: 34px;
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 6px;
    }
    .tagline {
        color: #556b7a;
        font-size: 14px;
        margin-bottom: 2px;
    }
    .credit {
        color: #7a8b93;
        font-size: 12px;
    }
    .card {
        background: var(--card-bg);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06);
        margin-bottom: 20px;
    }
    .summary-text {
        font-size: 16px;
        line-height: 1.5;
        color: #102a43;
        white-space: pre-wrap;
    }
    .small {
        font-size: 13px;
        color: #6b7a86;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="header"><div class="title">📝 Text Summarizer</div>'
    '<div class="tagline">General-purpose summarizer — fine-tuned LLM</div>'
    '<div class="credit">LLM model by Bishal Ray</div></div>',
    unsafe_allow_html=True
)

# ---------------------------
# User Input
# ---------------------------
article = st.text_area(
    "Enter your text to summarize:",
    height=250,
    placeholder="Paste your article, essay, or text here..."
)

# ---------------------------
# Summarization
# ---------------------------
if st.button("Summarize"):
    if article.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing... 🧠"):
            inputs = tokenizer.encode(article, return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = model.generate(
                inputs,
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        st.markdown(f'<div class="card"><div class="summary-text">{summary}</div></div>', unsafe_allow_html=True)



