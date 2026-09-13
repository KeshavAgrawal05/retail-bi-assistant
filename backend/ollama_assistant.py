"""
ollama_assistant.py — local, free, unlimited fallback using Qwen3 8B via Ollama.

WHY THIS EXISTS: Gemini's free tier has a daily quota. Rather than the app
just failing once that's exhausted, it falls back to a model running
entirely on your own machine — no API key, no internet required, no limit.

KEY DIFFERENCE FROM assistant.py: Gemini's SDK auto-generates tool schemas
from our Python functions' docstrings/type hints. Ollama's tool-calling
needs the JSON schema written out explicitly, so that's what TOOL_SCHEMAS
below does. The actual function EXECUTION still calls the exact same
analytics.py functions — same ground truth, same guardrails, different
LLM doing the explaining.
"""
import json
import ollama

from analytics import (
    category_performance,
    region_revenue_change,
    monthly_trend,
    fastest_growing_combo,
    profitability_analysis,
    seasonal_pattern,
    yoy_comparison,
    segment_profitability,
    top_bottom_subcategories,
)
from vectorstore import retrieve_context

MODEL_NAME = "qwen3:8b"

# Maps a tool name (string) to the actual Python function to execute.
TOOL_MAP = {
    "category_performance": category_performance,
    "region_revenue_change": region_revenue_change,
    "monthly_trend": monthly_trend,
    "fastest_growing_combo": fastest_growing_combo,
    "profitability_analysis": profitability_analysis,
    "seasonal_pattern": seasonal_pattern,
    "yoy_comparison": yoy_comparison,
    "segment_profitability": segment_profitability,
    "top_bottom_subcategories": top_bottom_subcategories,
}

# Explicit JSON schemas describing each tool to the model.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "category_performance",
        "description": "Get revenue by category for a quarter, compared to that category's own historical average. Use for questions about which category is over/underperforming.",
        "parameters": {"type": "object", "properties": {
            "quarter": {"type": "string", "description": "e.g. '2024Q3'. Omit for the latest quarter."}
        }},
    }},
    {"type": "function", "function": {
        "name": "region_revenue_change",
        "description": "Compares each region's revenue in the most recent months vs the months before, to detect regional revenue drops or growth.",
        "parameters": {"type": "object", "properties": {
            "recent_months": {"type": "integer", "description": "How many recent months to compare. Default 2."}
        }},
    }},
    {"type": "function", "function": {
        "name": "monthly_trend",
        "description": "Get month-by-month revenue trend, optionally filtered by category or region. Use for trend/time-series questions.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "Optional category filter, e.g. 'Technology'."},
            "region": {"type": "string", "description": "Optional region filter, e.g. 'West'."},
        }},
    }},
    {"type": "function", "function": {
        "name": "fastest_growing_combo",
        "description": "Finds the category/region combination with the highest year-over-year growth.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "profitability_analysis",
        "description": "Get profit margins by category and details on loss-making orders. Use for questions about profitability, margin, or losses.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "seasonal_pattern",
        "description": "Detects which calendar month a category peaks in, to answer seasonality questions.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "The category to analyze, e.g. 'Technology'. REQUIRED."}
        }, "required": ["category"]},
    }},
    {"type": "function", "function": {
        "name": "yoy_comparison",
        "description": "Compares revenue month-by-month across different years.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "segment_profitability",
        "description": "Get profit and margin broken down by customer segment (Consumer/Corporate/Home Office).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "top_bottom_subcategories",
        "description": "Get the top and bottom performing sub-categories by revenue.",
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer", "description": "How many top/bottom items to return. Default 3."}
        }},
    }},
]

SYSTEM_INSTRUCTION = """You are a retail business intelligence assistant for operations managers.

STRICT RULES:
1. You MUST call one of the provided tools to get real numbers before answering ANY
   question about sales, revenue, profit, trends, or performance. Never skip this step.
2. NEVER invent, estimate, or guess a number. Every figure in your answer must come
   directly from a tool result.
3. Keep explanations concise: 3-5 sentences, plain business language, no jargon.
4. If the available tools don't fully answer the question, say plainly what data
   you don't have instead of guessing.
"""


def ask_local(question: str) -> dict:
    """Same interface as assistant.ask(), but runs entirely locally via Ollama.
    No API key, no internet, no rate limits — just needs Ollama running."""
    context_chunks = retrieve_context(question, n_results=2)
    context_text = "\n".join(context_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Relevant business context:\n{context_text}\n\nUser question: {question}"},
    ]

    try:
        response = ollama.chat(model=MODEL_NAME, messages=messages, tools=TOOL_SCHEMAS)
    except Exception as e:
        return {
            "question": question,
            "answer": (
                f"Couldn't reach the local Ollama model: {e}. "
                "Make sure Ollama is installed and running (check the system tray), "
                "and that you've run 'ollama pull qwen3:8b' at least once."
            ),
            "error": True,
        }

    message = response["message"]

    # If the model decided to call a tool, execute it for real and send
    # the result back so it can write the final explanation.
    if message.get("tool_calls"):
        messages.append(message)
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"].get("arguments", {})
            if func_name in TOOL_MAP:
                result = TOOL_MAP[func_name](**func_args)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "name": func_name,
                })
        final_response = ollama.chat(model=MODEL_NAME, messages=messages)
        answer_text = final_response["message"]["content"]
    else:
        # Model answered without calling a tool — per our system instruction
        # this shouldn't happen for data questions, but handle it gracefully.
        answer_text = message["content"]

    return {"question": question, "answer": answer_text, "source": "local (Qwen3 8B)"}


if __name__ == "__main__":
    test_questions = [
        "Which region had the biggest revenue drop recently?",
        "Is there a seasonal pattern in Technology sales?",
    ]
    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = ask_local(q)
        print(result["answer"])