# Dual-AI CV Studio

Streamlit CV tailoring with two hosted models and no model downloads:

1. `Qwen/Qwen2.5-72B-Instruct` through Hugging Face extracts structured fields, classifies skills, analyzes JD keywords, and flags quality issues.
2. OpenAI `gpt-4o` performs semantic matching, ATS rewriting, bullet and summary generation, and the final language-quality pass.

Users can edit every section, inspect transparent ATS keyword guidance, and download an editable Word CV.

## Run

Python 3.9+ is recommended; compatible pins are included for Python 3.7.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholders with `OPENAI_API_KEY` and `HF_TOKEN`. Credentials are read server-side and are never shown in the application UI. Never commit the real secrets file.

OpenAI API usage is separate from a ChatGPT subscription and may be billed. Hugging Face provides small monthly experimentation credits subject to change; this is not unlimited free infrastructure.

Supported uploads: DOCX, text-based PDF, and TXT. Scanned PDFs require OCR. Always verify AI output for factual accuracy.
