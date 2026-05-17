# RAG Intelligence

A secure, role-aware RAG system that answers natural-language queries over heterogeneous enterprise data (text docs, CSVs, JSON logs) with strict RBAC, citations, and audit logging.

## Architecture

```
User query + role
      │
      ▼
FastAPI /ask
      │
      ├─ Embed query (all-MiniLM-L6-v2)
      ├─ FAISS search → top 15 candidates
      ├─ RBAC filter (pre-LLM, no leakage)
      ├─ Build context from allowed chunks
      ├─ Groq LLM (llama-3.3-70b-versatile)
      └─ Append to audit.log
```

## Data Sources

| File | Sensitivity | Allowed Roles |
|------|------------|---------------|
| `hr_policy.txt` | public | admin, hr, engineer, finance, intern |
| `eng_runbook.txt` | internal | admin, engineer |
| `finance_q3.txt` | confidential | admin, finance |
| `employees.csv` | confidential | admin, hr |
| `incidents.csv` | internal | admin, engineer |
| `audit.json` | restricted | admin |

## Setup

```bash
pip install -r requirements.txt
python ingest.py          # builds store.faiss + chunks.pkl
```

## Run

```bash
# Terminal 1 — API server
set GROQ_API_KEY=your_key_here
uvicorn app:app --reload

# Terminal 2 — UI
streamlit run ui.py
```

Open `http://localhost:8501` in your browser.

## API

```
POST /ask
{"question": "What is the leave policy?", "role": "intern"}

→ {"answer": "...", "citations": ["hr_policy.txt"], "denied_sources": 9, "chunks_used": 5}
```

## RBAC Design

- **Pre-LLM enforcement** — forbidden chunks are filtered before the prompt is built; they never reach the model
- **Generic refusal** — when no chunks are accessible, returns "No accessible information for your role"
- **Audit trail** — every query logged to `audit.log` with timestamp, role, sources used, and denial count

## Key Points

1. RBAC enforced pre-LLM — zero information leakage path
2. Unified retrieval over 3 formats — text, CSV rows, JSON entries all embedded with metadata
3. Citations forced via prompt + chunk tagging — every answer traceable to source
4. Grounded generation — temperature 0.1, explicit no-outside-knowledge instruction
