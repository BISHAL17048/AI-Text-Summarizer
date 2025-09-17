import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

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
    MODEL_NAME = "BISHAL2301/summarizer" # Replace with your Hugging Face model repo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ---------------------------
# Helper Functions
# ---------------------------
def convert_to_bullet_points(text):
    """Convert paragraph text to bullet points"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    bullet_points = []
    for sentence in sentences:
        if len(sentence) > 10:  # Filter out very short fragments
            bullet_points.append(f"• {sentence}")
    
    return '\n'.join(bullet_points)

def count_words(text):
    """Count words in text"""
    return len(text.split()) if text.strip() else 0

def count_sentences(text):
    """Count sentences in text"""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #f8f9fa;
        font-family: "Segoe UI", Roboto, sans-serif;
    }
    
    .main-container {
        max-width: 1100px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .summarizer-card {
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        overflow: hidden;
        border: 1px solid #e8eaed;
    }
    
    .modes-header {
        padding: 20px 24px 0px 24px;
        border-bottom: none;
    }
    
    .modes-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .modes-section {
        display: flex;
        align-items: center;
        gap: 0px;
    }
    
    .modes-label {
        font-size: 14px;
        color: #5f6368;
        font-weight: 500;
        margin-right: 16px;
    }
    
    .mode-tab {
        padding: 8px 16px;
        background: none;
        border: none;
        color: #5f6368;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
        margin-right: 24px;
    }
    
    .mode-tab.active {
        color: #1a73e8;
        border-bottom-color: #1a73e8;
    }
    
    .mode-tab:hover {
        color: #1a73e8;
    }
    
    .length-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .length-label {
        font-size: 14px;
        color: #5f6368;
        font-weight: 500;
    }
    
    .length-slider-container {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .length-option {
        font-size: 12px;
        color: #5f6368;
    }
    
    .input-section {
        padding: 20px 24px;
        border-bottom: 1px solid #f1f3f4;
        position: relative;
    }
    
    .text-input-area {
        width: 100%;
        min-height: 200px;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 16px;
        font-size: 14px;
        line-height: 1.5;
        resize: vertical;
        font-family: inherit;
        background: #fafbfc;
        color: #202124;
    }
    
    .text-input-area:focus {
        outline: none;
        border-color: #1a73e8;
        background: white;
    }
    
    .text-input-area::placeholder {
        color: #9aa0a6;
    }
    
    .paste-text-button {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #e8f0fe;
        border: 2px dashed #1a73e8;
        border-radius: 8px;
        padding: 20px 24px;
        color: #1a73e8;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .paste-text-button:hover {
        background: #d2e3fc;
    }
    
    .paste-icon {
        width: 16px;
        height: 16px;
        fill: currentColor;
    }
    
    .bottom-section {
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .upload-doc {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #5f6368;
        font-size: 14px;
        cursor: pointer;
        padding: 8px 12px;
        border-radius: 6px;
        transition: background 0.2s ease;
    }
    
    .upload-doc:hover {
        background: #f8f9fa;
    }
    
    .summarize-button {
        background: #137333;
        color: white;
        border: none;
        border-radius: 24px;
        padding: 12px 32px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .summarize-button:hover {
        background: #0f5720;
        box-shadow: 0 2px 8px rgba(19, 115, 51, 0.3);
    }
    
    .summarize-button:disabled {
        background: #dadce0;
        cursor: not-allowed;
    }
    
    .stats {
        font-size: 14px;
        color: #5f6368;
    }
    
    .stat-number {
        color: #202124;
        font-weight: 500;
    }
    
    .summary-box {
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e8eaed;
        min-height: 300px;
        margin-top: 24px;
        position: relative;
    }
    
    .summary-content {
        padding: 24px;
        min-height: 250px;
        position: relative;
    }
    
    .summary-text {
        font-size: 16px;
        line-height: 1.6;
        color: #202124;
        white-space: pre-wrap;
    }
    
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 250px;
        text-align: center;
    }
    
    .empty-icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.4;
    }
    
    .empty-text {
        color: #9aa0a6;
        font-size: 16px;
        margin-bottom: 8px;
    }
    
    .empty-subtext {
        color: #9aa0a6;
        font-size: 14px;
    }
    
    .spinner-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 250px;
        gap: 16px;
    }
    
    .spinner {
        width: 32px;
        height: 32px;
        border: 3px solid #f1f3f4;
        border-top: 3px solid #1a73e8;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner-text {
        color: #5f6368;
        font-size: 14px;
        font-weight: 500;
    }
    
    .summary-stats {
        padding: 16px 24px;
        border-top: 1px solid #f1f3f4;
        font-size: 14px;
        color: #5f6368;
    }
    
    .hidden {
        display: none !important;
    }
    
    /* Hide Streamlit elements */
    .stTextArea > label {
        display: none;
    }
    
    .stSelectbox > label {
        display: none;
    }
    
    .stButton > button {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Initialize session state
# ---------------------------
if 'summary_mode' not in st.session_state:
    st.session_state.summary_mode = 'Paragraph'
if 'summary_length' not in st.session_state:
    st.session_state.summary_length = 'Short'
if 'generated_summary' not in st.session_state:
    st.session_state.generated_summary = ""
if 'is_summarizing' not in st.session_state:
    st.session_state.is_summarizing = False
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# ---------------------------
# Main Container
# ---------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---------------------------
# Main Summarizer Card
# ---------------------------
st.markdown('<div class="summarizer-card">', unsafe_allow_html=True)

# Modes Header
st.markdown('<div class="modes-header">', unsafe_allow_html=True)
st.markdown(
    '<div class="modes-row">'
    '<div class="modes-section">'
    '<span class="modes-label">Modes:</span>'
    f'<button class="mode-tab {"active" if st.session_state.summary_mode == "Paragraph" else ""}" onclick="setMode(\'Paragraph\')">Paragraph</button>'
    f'<button class="mode-tab {"active" if st.session_state.summary_mode == "Bullet Points" else ""}" onclick="setMode(\'Bullet Points\')">Bullet Points</button>'
    f'<button class="mode-tab {"active" if st.session_state.summary_mode == "Custom" else ""}" onclick="setMode(\'Custom\')">Custom</button>'
    '</div>'
    '<div class="length-section">'
    '<span class="length-label">Summary Length:</span>'
    '<div class="length-slider-container">'
    '<span class="length-option">Short</span>'
    '<input type="range" min="1" max="3" value="1" style="margin: 0 8px;">'
    '<span class="length-option">Long</span>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# Hidden Streamlit controls for functionality
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Paragraph", key="para_btn"):
        st.session_state.summary_mode = 'Paragraph'
        st.rerun()

with col2:
    if st.button("Bullet Points", key="bullet_btn"):
        st.session_state.summary_mode = 'Bullet Points'
        st.rerun()

with col3:
    if st.button("Custom", key="custom_btn"):
        st.session_state.summary_mode = 'Custom'
        st.rerun()

# Length selector (hidden)
length = st.selectbox("Length", ["Short", "Medium", "Long"], 
                     index=["Short", "Medium", "Long"].index(st.session_state.summary_length),
                     key="length_select")
st.session_state.summary_length = length

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Text input
article = st.text_area("", 
                      height=200,
                      placeholder="Enter or paste your text and press \"Summarize.\"",
                      key="text_input",
                      value=st.session_state.input_text)

# Show paste button when text area is empty
if not article.strip():
    st.markdown(
        '<div class="paste-text-button" onclick="document.querySelector(\'textarea\').focus()">'
        '<svg class="paste-icon" viewBox="0 0 24 24">'
        '<path d="M19,3H14.82C14.4,1.84 13.3,1 12,1C10.7,1 9.6,1.84 9.18,3H5A2,2 0 0,0 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V5A2,2 0 0,0 19,3M12,3A1,1 0 0,1 13,4A1,1 0 0,1 12,5A1,1 0 0,1 11,4A1,1 0 0,1 12,3"/>'
        '</svg>'
        'Paste Text'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Bottom Section
st.markdown('<div class="bottom-section">', unsafe_allow_html=True)

# Upload Doc button
st.markdown(
    '<div class="upload-doc">'
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="#5f6368">'
    '<path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>'
    '</svg>'
    'Upload Doc'
    '</div>',
    unsafe_allow_html=True
)

# Stats and Summarize button
input_word_count = count_words(article)
input_sentence_count = count_sentences(article)

st.markdown(
    f'<div style="display: flex; align-items: center; gap: 24px;">'
    f'<button class="summarize-button" onclick="summarize()" {"disabled" if not article.strip() else ""}>Summarize</button>'
    f'<div class="stats"><span class="stat-number">{input_sentence_count}</span> sentences • <span class="stat-number">{input_word_count}</span> words</div>'
    f'</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Hidden summarize button for functionality
if st.button("Hidden Summarize", key="hidden_summarize"):
    if article.strip():
        st.session_state.input_text = article
        st.session_state.is_summarizing = True
        st.rerun()

# ---------------------------
# Summary Box
# ---------------------------
st.markdown('<div class="summary-box">', unsafe_allow_html=True)
st.markdown('<div class="summary-content">', unsafe_allow_html=True)

if st.session_state.is_summarizing:
    # Show spinner
    st.markdown(
        '<div class="spinner-container">'
        '<div class="spinner"></div>'
        '<div class="spinner-text">Generating summary...</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Perform summarization
    try:
        # Adjust max_length based on selected length
        length_mapping = {
            "Short": {"max_length": 100, "min_length": 25},
            "Medium": {"max_length": 200, "min_length": 50},
            "Long": {"max_length": 300, "min_length": 75}
        }
        
        length_params = length_mapping[st.session_state.summary_length]
        
        inputs = tokenizer.encode(st.session_state.input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        summary_ids = model.generate(
            inputs,
            max_length=length_params["max_length"],
            min_length=length_params["min_length"],
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Format summary based on mode
        if st.session_state.summary_mode == 'Bullet Points':
            summary = convert_to_bullet_points(summary)
        
        st.session_state.generated_summary = summary
        st.session_state.is_summarizing = False
        st.rerun()
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.session_state.is_summarizing = False

elif st.session_state.generated_summary:
    # Display generated summary
    st.markdown(f'<div class="summary-text">{st.session_state.generated_summary}</div>', unsafe_allow_html=True)

else:
    # Empty state
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">📄</div>'
        '<div class="empty-text">Your summary will appear here</div>'
        '<div class="empty-subtext">Enter text above and click "Summarize" to get started</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Summary stats
if st.session_state.generated_summary:
    summary_word_count = count_words(st.session_state.generated_summary)
    summary_sentence_count = count_sentences(st.session_state.generated_summary)
    
    st.markdown(
        f'<div class="summary-stats">'
        f'<span class="stat-number">{summary_sentence_count}</span> sentences • '
        f'<span class="stat-number">{summary_word_count}</span> words • '
        f'Mode: <span class="stat-number">{st.session_state.summary_mode}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# JavaScript for interactivity
st.markdown(
    """
    <script>
    function summarize() {
        // Trigger the hidden summarize button
        const buttons = document.querySelectorAll('button');
        const summarizeBtn = Array.from(buttons).find(btn => btn.textContent === 'Hidden Summarize');
        if (summarizeBtn) {
            summarizeBtn.click();
        }
    }
    
    function setMode(mode) {
        // Trigger the appropriate mode button
        const buttons = document.querySelectorAll('button');
        let targetBtn;
        if (mode === 'Paragraph') {
            targetBtn = Array.from(buttons).find(btn => btn.textContent === 'Paragraph' && btn.style.display !== 'none');
        } else if (mode === 'Bullet Points') {
            targetBtn = Array.from(buttons).find(btn => btn.textContent === 'Bullet Points' && btn.style.display !== 'none');
        } else if (mode === 'Custom') {
            targetBtn = Array.from(buttons).find(btn => btn.textContent === 'Custom' && btn.style.display !== 'none');
        }
        if (targetBtn) {
            targetBtn.click();
        }
    }
    </script>
    """,
    unsafe_allow_html=True
)



