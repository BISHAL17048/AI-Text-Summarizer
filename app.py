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
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """
    Loads and caches the tokenizer and the model from Hugging Face.
    """
    MODEL_NAME = "BISHAL2301/summarizer"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

tokenizer, model, device = load_model()

# ---------------------------
# Custom CSS for QuillBot-like UI
# ---------------------------
st.markdown(
    """
    <style>
    /* General Styles */
    .stApp {
        background-color: #F0F2F6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 24px;
        color: #007A5A; /* QuillBot green */
        font-weight: 700;
        padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] .stButton>button {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
        background-color: transparent;
        color: #333;
        border: none;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 16px;
        margin-bottom: 8px;
        text-align: left;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #F0F8F5;
        color: #007A5A;
    }
    [data-testid="stSidebar"] .stButton>button.active {
        background-color: #E6F2EF;
        color: #007A5A;
        font-weight: 600;
    }

    /* Main Content Area */
    .main-container {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
    }

    /* Controls Styling (Modes and Length) */
    .controls-container {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 10px;
    }
    .stRadio [role="radiogroup"] {
        gap: 5px;
    }
    .stRadio label {
        background-color: #F0F2F6;
        padding: 8px 20px;
        border-radius: 20px;
        border: 1px solid transparent;
        cursor: pointer;
        transition: all 0.2s;
    }
    .stRadio input[type="radio"]:checked + div {
        background-color: #E6F2EF;
        color: #007A5A;
        font-weight: 600;
        border: 1px solid #007A5A;
    }
    
    /* Output Box */
    .summary-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        margin-top: 1rem;
        min-height: 150px;
    }
    .summary-text {
        color: #333;
        font-size: 16px;
        line-height: 1.6;
    }
    .summary-text ul {
        padding-left: 20px;
    }
    .summary-text li {
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar UI
# ---------------------------
with st.sidebar:
    st.markdown("<h2>QuillBot</h2>", unsafe_allow_html=True)
    st.button(" paraphraser Paraphraser")
    st.button(" grammer Grammar Checker")
    st.button(" ai AI Detector")
    st.button(" plagiarism Plagiarism Checker")
    st.button(" ai-chat AI Chat")
    st.button(" translate Translator")
    # Using a CSS class to highlight the current page
    st.markdown('<button class="active">📝 Summarizer</button>', unsafe_allow_html=True)
    st.button(" citation Citation Generator")


# ---------------------------
# Main Page UI
# ---------------------------
st.title("Summarizer")

# Wrap main content in a container for styling
with st.container(border=True): #border=True is a new Streamlit feature for a simple border
    
    # --- Controls for Mode and Length ---
    col1, col2 = st.columns([1, 1])
    with col1:
        mode = st.radio(
            "Modes:",
            ["Paragraph", "Bullet Points"],
            horizontal=True,
        )
    with col2:
        # Map friendly names to actual length values
        length_options = {"Short": 80, "Medium": 150, "Long": 250}
        selected_length = st.select_slider(
            "Summary Length:",
            options=list(length_options.keys()),
            value="Medium"
        )
        max_len = length_options[selected_length]
        min_len = max_len // 3 # Ensure min length is proportional to max

    # --- User Input ---
    article = st.text_area(
        "Enter your text and press \"Summarize.\"",
        height=280,
        placeholder="Paste your article, essay, or text here..."
    )
    
    # --- Summarize Button ---
    summarize_button = st.button("Summarize", type="primary", use_container_width=True)
    
    # --- Output Area ---
    summary_placeholder = st.empty()


# ---------------------------
# Summarization Logic
# ---------------------------
if summarize_button:
    if not article.strip():
        st.warning("Please enter some text to summarize.")
    elif not model or not tokenizer:
        st.error("The summarization model is not available. Please try again later.")
    else:
        # Put the spinner and the result in the placeholder
        with summary_placeholder.container():
            with st.spinner("🧠 Generating Summary... Please wait."):
                
                # Prepare the prompt based on the selected mode
                prompt = article
                if mode == "Bullet Points":
                    prompt = f"Summarize the following text in clear and concise bullet points:\n\n{article}"

                try:
                    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
                    
                    summary_ids = model.generate(
                        inputs,
                        max_length=max_len,
                        min_length=min_len,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True
                    )
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                    # Display the final summary in the styled card
                    st.markdown(
                        f'<div class="summary-card"><div class="summary-text">{summary}</div></div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")



