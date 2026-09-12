"""
assistant.py — ties everything together using Gemini function calling.

Uses the CURRENT google-genai SDK (the old google-generativeai package
was fully deprecated/sunset — this is the actively maintained one).

HOW IT WORKS (this is the "agentic" part):
1. User asks a question in plain English.
2. We retrieve relevant business context from vectorstore.py.
3. Gemini reads the question + context, and DECIDES which analytics.py
function(s) it needs to call to answer — using its native function
calling / tool use capability.
4. The SDK executes those functions for real (real pandas math, real
numbers) automatically, behind the scenes.
5. Gemini receives the actual results and writes the explanation.

The critical guardrail: Gemini is NEVER allowed to just answer from its
own knowledge. The system instruction forces it to call a tool first.
This is what keeps it grounded instead of hallucinating numbers.
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
    raise RuntimeError(
        "GEMINI_API_KEY not found. Checklist:\n"
        "  1. Is there a file named exactly '.env' (not '.env.txt') in your PROJECT ROOT "
        "(same folder as requirements.txt, NOT inside backend/)?\n"
        "  2. Does it contain a line exactly like: GEMINI_API_KEY=AIzaSy...\n"
        "     (no quotes, no spaces around the =)\n"
        "  3. Are you running the script from the project root, e.g. "
        "'python backend\\assistant.py' from C:\\Users\\...\\retail-bi-assistant> ?"
    )

client = genai.Client(api_key=api_key)

# Every analytics.py function is handed to Gemini as a callable "tool".
# The SDK reads each function's docstring + type hints to understand
# what it does and when to use it — no manual schema-writing needed.
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


def ask(question: str, max_retries: int = 3) -> dict:
    """Main entry point: ask a natural language business question,
    get back a grounded, tool-backed explanation. Retries automatically
    if the free tier's rate limit (5 requests/minute) is hit."""
    context_chunks = retrieve_context(question, n_results=2)
    context_text = "\n".join(context_chunks)

    config = types.GenerateContentConfig(
        tools=TOOLS,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat = client.chats.create(model="gemini-3.6-flash", config=config)

    prompt = f"""Relevant business context:
{context_text}

User question: {question}"""

    for attempt in range(max_retries):
        try:
            response = chat.send_message(prompt)
            return {"question": question, "answer": response.text}
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 50
                print(f"  (rate limit hit — waiting {wait_time}s, retry {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting.")


if __name__ == "__main__":
    test_questions = [
        "Which category is underperforming this quarter, and why?",
        "Which region had the biggest revenue drop recently?",
        "Is there a seasonal pattern in Technology sales?",
        "Which products or categories are losing money despite high sales?",
    ]
    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = ask(q)
        print(result["answer"])
        time.sleep(15)  # small pause between questions to avoid rate limits
