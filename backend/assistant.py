"""
assistant.py — ties everything together using Gemini function calling.
"""
import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv

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

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

TOOLS = [
    category_performance,
    region_revenue_change,
    monthly_trend,
    fastest_growing_combo,
    profitability_analysis,
    seasonal_pattern,
    yoy_comparison,
    segment_profitability,
    top_bottom_subcategories,
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
5. When explaining a decline or underperformance, name the specific percentage
   and time period from the tool result — vague answers are not acceptable.
"""


def ask(question, max_retries=3):
    context_chunks = retrieve_context(question, n_results=2)
    context_text = "\n".join(context_chunks)

    config = types.GenerateContentConfig(tools=TOOLS, system_instruction=SYSTEM_INSTRUCTION)
    chat = client.chats.create(model="gemini-3.6-flash", config=config)

    prompt = "Relevant business context:\n" + context_text + "\n\nUser question: " + question

    for attempt in range(max_retries):
        try:
            response = chat.send_message(prompt)
            return {"question": question, "answer": response.text, "source": "Gemini 3.6 Flash"}
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 50
                    print(f"  (rate limit hit — waiting {wait_time}s, retry {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return {
                        "question": question,
                        "answer": (
                            "I've hit today's free-tier API quota, so I can't generate a fresh "
                            "explanation right now. Falling back to the local model..."
                        ),
                        "quota_exceeded": True,
                    }
            else:
                raise
    return {
        "question": question,
        "answer": "Something went wrong while contacting the AI. Please try again.",
        "quota_exceeded": True,
    }


if __name__ == "__main__":
    test_questions = [
        "Which category is underperforming this quarter, and why?",
        "Which region had the biggest revenue drop recently?",
    ]
    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = ask(q)
        print(result["answer"])
        time.sleep(15)