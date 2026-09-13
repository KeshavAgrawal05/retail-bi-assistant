# Retail BI Assistant

A full-stack business intelligence assistant that answers plain-English questions about retail sales data — grounded entirely in real, computed numbers, never invented by the AI.

> "Which region had the biggest revenue drop recently?" → *"The West region declined 49.0%, falling from $150,521.65 to $76,756.96 between Sep-Oct and Nov-Dec 2025..."*

## Why this project

Most student ML/AI projects are either a standalone model (no product around it) or a chatbot with no real data grounding. This project combines both: a real pandas-based analytics engine, a RAG context layer, and an LLM that can only speak using numbers it actually computed — never hallucinated.

## Architecture
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ React Frontend │ ───▶ │ FastAPI Backend │ ───▶ │ Gemini 3.6 │
│ (index.html) │ │ (main.py) │ │ Flash (LLM) │
└─────────────────┘ └──────────────────┘ └────────┬────────┘
│ │
▼ │ decides which
┌──────────────────┐ │ tool to call
│ analytics.py │◀───────────────┘
│ (ground truth: │
│ real pandas math) │
└──────────────────┘
▲
┌──────────────────┐
│ vectorstore.py │
│ (RAG context via │
│ ChromaDB, local) │
└──────────────────┘

**The core design principle:** the LLM is never allowed to calculate a number itself. Every figure comes from `analytics.py`'s pandas functions. The LLM's only job is to decide which function to call (via Gemini's native function calling) and explain the result in plain English.

## Features

- **Grounded Q&A** — ask questions like "why is Furniture underperforming?" and get answers backed by real computed statistics, not guesses
- **9 analytics functions** covering category performance, regional trends, profitability, seasonality, segment analysis, and year-over-year comparison
- **RAG context layer** — local (free, no API cost) semantic retrieval of business glossary/context to help ground vague questions
- **Live dashboard** — 4 charts populate directly from the analytics API, with zero LLM cost or latency
- **Graceful degradation** — if the AI's API quota is exhausted, the app falls back to a clear message instead of crashing, and the dashboard keeps working independently

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React (via CDN, no build step), hand-rolled SVG charts |
| Backend | FastAPI, Python |
| Analytics | pandas |
| RAG / vector store | ChromaDB (local embeddings, no API cost) |
| LLM | Google Gemini (function calling / tool use) |
| Deployment | Docker |

## Local setup

```bash
# 1. Clone and enter the project
git clone https://github.com/KeshavAgrawal05/retail-bi-assistant.git
cd retail-bi-assistant

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key (free at aistudio.google.com)
# Create a .env file in the project root containing:
# GEMINI_API_KEY=your_key_here

# 5. Start the backend
python backend/main.py

# 6. Open the frontend
# Just double-click frontend/index.html in File Explorer
```

Visit `http://localhost:8000/docs` for the API, and open `frontend/index.html` in a browser for the dashboard.

## Known limitations

- **Gemini free tier is rate-limited** (5 req/min, limited daily quota) — the app handles this gracefully with a fallback message, but heavy testing can temporarily exhaust it
- **RAG retrieval isn't perfect** — the local embedding model (all-MiniLM-L6-v2) sometimes retrieves adjacent-but-not-ideal context chunks for ambiguous questions; capped at 2 chunks to reduce noise
- **No SKU-level profitability** — the assistant correctly says so rather than guessing, when asked about individual product losses

## Possible next steps

- Add a local LLM (Ollama) fallback for unlimited free usage when the API quota is exhausted
- Deploy the backend (Docker image included) to Render/Railway for a live public demo link
- Add authentication for multi-user deployment
- Expand to SKU-level data for more granular profitability analysis
