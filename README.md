# Sentiment Analysis — Streamlit App (Gemini)

A lightweight Streamlit app for classifying **movie-review sentiment** using Google’s Gemini (**`gemini-2.5-flash-lite`**) with two modes: **Strict** and **Lenient**.

- **Single text**: paste a review and get label + confidence + short rationale.  
- **Small CSV**: you can upload a tiny CSV (≈ **10 rows max** recommended on the free tier).  
- **Bigger jobs**: use the **CLI batch** flow (`batch.py`) with **your own API key**.

---

## 1) Quickstart — Streamlit UI

### Install
```bash
git clone https://github.com/mayank-sethia1/Sentiment-Analysis-streamlit-app.git
cd Sentiment-Analysis-streamlit-app
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure your Gemini key
Use **GEMINI_API_KEY** (required).

macOS/Linux:
```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

Windows (PowerShell):
```powershell
setx GEMINI_API_KEY "YOUR_KEY_HERE"
# close & reopen terminal after setx
```

### Run
```bash
streamlit run streamlit_app.py
```

**Notes**
- The UI supports a **single text** box and a **small CSV** upload. Keep CSVs to ~10 rows on the free tier; for larger datasets, use the CLI below.

---

## 2) Batch classification via CLI (`batch.py`)

For larger files, run the classifier from the command line.

**Usage**
```bash
python batch.py <strict|lenient> <in_csv> [out_csv] [rpm]
```

**Arguments**
- `mode` (required): `strict` or `lenient`
- `in_csv` (required): path to input CSV of reviews (the first column is treated as the review text)
- `out_csv` (optional): output path (default: `predictions.csv`)
- `rpm` (optional): requests per minute throttle (default: `30.0`)

**Environment**
Export **your own** Gemini key before running:
```bash
# macOS/Linux
export GEMINI_API_KEY="YOUR_KEY_HERE"

# Windows PowerShell
setx GEMINI_API_KEY "YOUR_KEY_HERE"
```

**Examples**
```bash
# Strict mode, default output name, default RPM
python batch.py strict data.csv

# Lenient mode, custom output name, custom RPM
python batch.py lenient reviews.csv my_preds.csv 20
```

**What you get**
- A CSV of predictions written to `out_csv`
- Console summary including overall accuracy and label counts (actual vs predicted)

---

## 3) Prompting / model logic (`sentiment_llm.py`)

- **Model**: `gemini-2.5-flash-lite`
- **Task**: Classify a **movie review** into one of **Positive / Negative / Neutral**, and produce a brief explanation.
- **Modes**: 
  - **Strict** — more conservative about calling borderline positives/negatives  
  - **Lenient** — more forgiving when cues suggest sentiment but are mild  
- **Cues considered** (high level): polarity words, negation patterns (e.g., “not bad”), contrast/sarcasm cues. The prompt primes the model to justify its label briefly.

Both the Streamlit app and `batch.py` call into `sentiment_llm.py`, so UI and CLI use the same logic.

---

## 4) Tips

- Keep Streamlit uploads small on the free tier; switch to CLI for bigger CSVs.
- Tune `rpm` if you hit rate limits on batch runs.
