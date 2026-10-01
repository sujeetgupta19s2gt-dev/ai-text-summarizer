# AI Text Summarizer 🚀

An end-to-end Natural Language Processing (NLP) web application that fine-tunes a **T5 Transformer model** (`t5-small`) using Hugging Face on dialogue datasets (SAMSum) and serves it through a lightweight **FastAPI** backend with an interactive HTML/CSS/JS frontend[cite: 1, 2].

---

## 📌 Project Overview
Text summarization condenses one or more texts into shorter summaries for enhanced information extraction. This project implements a Text-to-Text Transfer Transformer (an encoder-decoder model) to process conversation logs and extract key dialogue points. It covers the complete machine learning and deployment lifecycle:
1. **Data Preprocessing & Random Sampling**: Cleaning datasets via regular expressions and selecting random subsets for efficient training.
2. **Model Tokenization & Fine-Tuning**: Tokenizing inputs and target summaries using `t5-small`, configuring training arguments, and training via the Hugging Face `Trainer` API.
3. **Model Saving & Inference Pipeline**: Saving model weights and testing core logic through cleaning, tokenization, generation via token IDs, and decoding back into text summaries.
4. **Backend API Development**: Using FastAPI as a Python-based web framework to expose endpoints, manage server requests, and communicate with the model.
5. **Frontend UI Integration**: Client-side development using HTML, CSS, and JavaScript connected to backend API routes via a lightweight Uvicorn server[cite: 1, 2].

---

## 🛠️ Tech Stack & Libraries Used
* **Python**: Core programming language for data science and web components
* **Hugging Face Transformers**: `T5Tokenizer`, `T5ForConditionalGeneration`, `Trainer`, and `TrainingArguments`
* **PyTorch**: Deep learning backend framework with automated hardware accelerator detection (`cuda`, `mps`, or `cpu`)
* **Pandas**: Dataset manipulation and CSV loading (`samsum-train.csv`, `samsum-validation.csv`)
* **Re (Regex)**: Text cleaning utility for removing line breaks (`\r\n`), extra spaces (`\s+`), and HTML tags (`<.*?>`)[cite: 1]
* **FastAPI**: Modern, high-performance Python web framework for building APIs
* **Uvicorn**: Lightweight ASGI web server implementation
* **Jinja2**: Template engine for rendering HTML responses[cite: 2]
* **HTML5, CSS3, JavaScript**: Responsive client-side interface for user interaction[cite: 1]

---

## 📂 Project Structure
```text
AI-Text-Summarizer/
│
├── text_summarizer.ipynb     # Jupyter Notebook covering data pipeline, fine-tuning, and model saving
├── app.py                    # FastAPI application server handling routing and backend summarization logic
├── index.html                # Frontend client-side UI (HTML, CSS, asynchronous JavaScript fetch)
├── saved_summary_model/      # Directory containing the saved fine-tuned T5 model weights and tokenizer
└── README.md                 # Comprehensive project documentation


🚀 How to Run Locally

Prerequisites
Ensure you have Python installed, then install the required dependencies:
pip install fastapi uvicorn transformers torch pandas jinja2 pydantic

Step 1: Clone the Repository
git clone https://github.com/your-username/ai-text-summarizer.git
cd ai-text-summarizer

Step 2: Prepare Model Artifacts
Ensure that your fine-tuned model directory (saved_summary_model/) is located in the root folder alongside app.py and index.html.

Step 3: Run the FastAPI Application Server
Start the lightweight Uvicorn server: uvicorn app:app --reload

Step 4: Access the Web App
Open your web browser and go to: http://127.0.0.1:8000

