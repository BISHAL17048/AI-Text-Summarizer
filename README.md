# 📝 AI-Powered Text Summarizer

Live demo: **[https://summary-analysis-1-ktnobq3ehutvc46tyiaqxl.streamlit.app](https://summary-analysis-1-ktnobq3ehutvc46tyiaqxl.streamlit.app)**  

🤗Hugging Face Model: **[https://huggingface.co/BISHAL2301/summarizer](https://huggingface.co/BISHAL2301/summarizer)**



> A simple AI-powered web application that generates concise summaries from long text inputs.
> Built with **Streamlit** and a **And My Trained model from Hugging Face**.
> Users can choose summary format (paragraph, bullet points, or custom length) and adjust output size.

---

## 🎥 Demo

Open the live application here: [https://summary-analysis-1-ktnobq3ehutvc46tyiaqxl.streamlit.app](https://summary-analysis-1-ktnobq3ehutvc46tyiaqxl.streamlit.app)

---
## 🎥 Dataset for Fine Tuning:

Dataset: [https://www.kaggle.com/datasets/bishalray2401/summary-analysis-dataset](https://www.kaggle.com/datasets/bishalray2401/summary-analysis-dataset)

---

## ✨ Features

* Enter long text and receive a summarized version in seconds.
* Choose **paragraph**, **bullet points**, or **custom bullet points**.
* Adjustable **summary length** using a slider.
* **Loading spinner** indicates when the model is processing.
* Clean, responsive UI powered by Streamlit.
* Efficient model caching with `st.cache_resource`.

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend/ML Model:** Hugging Face Transformer, PyTorch
* **Data Handling:** NumPy, Pandas
* **Utilities:** SentencePiece, Hugging Face Hub
* **Deployment:** Streamlit Cloud

---

## 🚀 Run Locally

1.  **Clone the repo:**
    ```bash
    git clone [https://huggingface.co/spaces/BISHAL2301/summarizer](https://huggingface.co/spaces/BISHAL2301/summarizer)
    cd summarizer
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate

    # Activate on macOS / Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    Visit **http://localhost:8501** in your browser to use the app locally.

---

## 📁 Project Structure

```
.
├── app.py              # Streamlit web app
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```
---

## 🐍 Example: app.py

```python
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import html

MODEL_NAME = "t5-small"

st.set_page_config(layout="wide", page_title="AI-Powered Text Summarizer")

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def generate_summary(text, mode="Paragraph", num_points=5, max_len=120, min_len=30):
    text = "summarize: " + text.strip()
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_len, min_length=min_len,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
    decoded = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    decoded = html.unescape(re.sub(r"\s+", " ", decoded))

    if mode == "Paragraph":
        return decoded
    elif mode == "Bullet Points":
        return "\n".join([f"• {s.strip()}" for s in re.split(r'(?<=[.!?]) +', decoded) if s.strip()])
    elif mode == "Custom":
        sentences = re.split(r'(?<=[.!?]) +', decoded)
        return "\n".join([f"• {s.strip()}" for s in sentences[:num_points] if s.strip()])
    return decoded

st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Select Summary Mode:", ["Paragraph", "Bullet Points", "Custom"])
num_points = st.sidebar.slider("Number of bullet points:", 2, 10, 5) if mode == "Custom" else 5
summary_length = st.sidebar.slider("Summary length:", 30, 300, 120, step=10)

st.title("📝 AI-Powered Text Summarizer")
article = st.text_area("Enter your text to summarize", height=300)

if st.button("Summarize"):
    if article.strip():
        with st.spinner("⏳ Generating summary..."):
            summary = generate_summary(article, mode=mode, num_points=num_points, max_len=summary_length)
        st.subheader("📌 Summary Output:")
        st.text_area("Summary", summary, height=250)
    else:
        st.warning("⚠️ Please enter some text before summarizing.")
```
---

## requirements.txt

```
streamlit
torch
transformers
datasets
sentencepiece
accelerate
evaluate
matplotlib
seaborn
tensorflow
pandas
numpy
huggingface_hub

```
## 🔌 API Usage

You can also use the summarizer as an API endpoint if you run the app with Flask or FastAPI.

### Example Flask server (api.py):

```python
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re, html

app = Flask(__name__)

MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_summary(text, max_len=120, min_len=30):
    text = "summarize: " + text.strip()
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_len, min_length=min_len,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
    decoded = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return html.unescape(re.sub(r"\s+", " ", decoded))

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    summary = generate_summary(text)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
```
---
### Example API Request (cURL)

```bash
curl -X POST http://127.0.0.1:5000/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long input article goes here..."}'
```
---
### Example API Response

```json
{
  "summary": "This is the generated summary of the provided text."
}
```
---
## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request with a clear description of the change.

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/your-change`)  
3. Commit your changes  
4. Push to your branch and open a PR
---

## 📜 License
MIT License
---

## 📬 Contact

For questions or feedback, contact **Bishal Ray**. 
