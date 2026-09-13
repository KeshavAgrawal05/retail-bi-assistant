"""
main.py — FastAPI backend exposing our assistant + analytics as a REST API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

import analytics
from assistant import ask as ask_assistant
from ollama_assistant import ask_local

app = FastAPI(title="Retail BI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Retail BI Assistant API is running"}


@app.post("/ask")
def ask_question(req: QuestionRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask_assistant(req.question)
        if result.get("quota_exceeded"):
            print("Gemini quota exceeded, falling back to local Ollama model...")
            result = ask_local(req.question)
        return result
    except Exception as e:
        print("=" * 70)
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        print("=" * 70)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/monthly-trend")
def get_monthly_trend(category: str | None = None, region: str | None = None):
    return analytics.monthly_trend(category=category, region=region)


@app.get("/analytics/category-performance")
def get_category_performance(quarter: str | None = None):
    return analytics.category_performance(quarter=quarter)


@app.get("/analytics/region-revenue")
def get_region_revenue(recent_months: int = 2):
    return analytics.region_revenue_change(recent_months=recent_months)


@app.get("/analytics/profitability")
def get_profitability():
    return analytics.profitability_analysis()


@app.get("/analytics/segment-profitability")
def get_segment_profitability():
    return analytics.segment_profitability()


@app.get("/analytics/top-bottom-subcategories")
def get_top_bottom(n: int = 3):
    return analytics.top_bottom_subcategories(n=n)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)