# Retail BI Assistant

A full-stack business intelligence assistant that answers plain-English questions about retail sales data — grounded entirely in real, computed numbers, never invented by the AI. Backed by two interchangeable LLMs (cloud + local) with automatic failover.

> "Which region had the biggest revenue drop recently?" → *"The West region declined 49.0%, falling from $150,521.65 to $76,756.96 between Sep-Oct and Nov-Dec 2025..."*

## Why this project

Most student ML/AI projects are either a standalone model (no product around it) or a chatbot with no real data grounding. This project combines both: a real pandas-based analytics engine, a RAG context layer, and an LLM that can only speak using numbers it actually computed — never hallucinated. It also handles a real-world constraint most student projects ignore entirely: what happens when your API quota runs out.

## Architecture
┌─────────────────┐
│ React Frontend │ ───▶ │ FastAPI Backend │ ───▶ │ Gemini 3.6 │
│ (index.html) │ │ (main.py) │ │ Flash (primary) │
└─────────────────┘ └────────┬─────────┘ └────────┬────────┘
│ │
│ falls back if quota │ decides which
│ exceeded │ tool to call
▼ │
┌──────────────────┐ │
│ Qwen3 8B via │ │
│ Ollama (local, │ │
│ unlimited fallback) │ │
└────────┬─────────┘ │
│ │
▼ ▼
┌──────────────────────────────────────┐
│ analytics.py (ground truth: │
│ real pandas math, never guessed) │
└──────────────────────────────────────┘
▲
┌──────────────────┐
│ vectorstore.py │
│ (RAG context via │
│ ChromaDB, local) │
└──────────────────┘

**The core design principle:** neither LLM is ever allowed to calculate a number itself. Every figure comes from `analytics.py`'s pandas functions. Each model's only job is to decide which function to call (via native function/tool calling) and explain the result in plain English.

## Features

- **Grounded Q&A** — ask questions like "why is Furniture underperforming?" and get answers backed by real computed statistics, not guesses
- **Dual-LLM resilience** — Gemini (cloud, higher quality) is primary; if its free-tier quota is exhausted, the app automatically falls back to Qwen3 8B running locally via Ollama — no API key, no internet, no limit
- **9 analytics functions** covering category performance, regional trends, profitability, seasonality, segment analysis, and year-over-year comparison
- **RAG context layer** — local (free, no API cost) semantic retrieval of business glossary/context to help ground vague questions
- **Live dashboard** — 4 charts populate directly from the analytics API, with zero LLM cost or latency
- **Graceful degradation throughout** — quota limits, rate limits, and missing data are all handled with clear messaging instead of crashes

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React (via CDN, no build step), hand-rolled SVG charts |
| Backend | FastAPI, Python |
| Analytics | pandas |
| RAG / vector store | ChromaDB (local embeddings, no API cost) |
| LLM (primary) | Google Gemini (function calling / tool use) |
| LLM (fallback) | Qwen3 8B via Ollama (local, unlimited, free) |
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

# 5. (Optional but recommended) Set up the local fallback
# Install Ollama from ollama.com, then:
ollama pull qwen3:8b

# 6. Start the backend
python backend/main.py

# 7. Open the frontend
# Just double-click frontend/index.html in File Explorer
```

Visit `http://localhost:8000/docs` for the API, and open `frontend/index.html` in a browser for the dashboard.

## Known limitations

- **Gemini free tier is rate-limited** (5 req/min, limited daily quota) — handled automatically via fallback to the local Qwen3 model, so the app stays usable either way
- **Local fallback is slower** — Qwen3 8B on a laptop CPU takes noticeably longer per answer than Gemini's cloud inference
- **RAG retrieval isn't perfect** — the local embedding model (all-MiniLM-L6-v2) sometimes retrieves adjacent-but-not-ideal context chunks for ambiguous questions; capped at 2 chunks to reduce noise
- **No SKU-level profitability** — the assistant correctly says so rather than guessing, when asked about individual product losses

## Possible next steps

- Deploy the backend (Docker image included) to Render/Railway for a live public demo link
- Add authentication for multi-user deployment
- Expand to SKU-level data for more granular profitability analysis
- Add a model-selector toggle in the UI so users can manually choose Gemini vs. local