"""
vectorstore.py — the semantic/RAG layer.

IMPORTANT: this does NOT store raw sales rows. analytics.py already
computes exact numbers from the real data — that's our ground truth.

This layer stores DESCRIPTIONS of the data's structure and business
context (schema, glossary, category definitions). When a user asks a
fuzzy question, we retrieve the most relevant descriptions and hand
them to the LLM alongside the real numbers from analytics.py, so the
LLM's explanation is grounded in both "what the numbers are" AND
"what they mean" — without ever letting it invent either.

Embeddings run locally via ChromaDB's default model (all-MiniLM-L6-v2),
so building/querying this store costs zero API calls. The Gemini API
is only used later, for generating the actual explanation text.
"""
import chromadb
from pathlib import Path

DB_PATH = Path(__file__).parent / "chroma_db"


def build_documents() -> list[dict]:
    """Static knowledge chunks describing our dataset's structure and
    business context. In a real company deployment, this would be
    auto-generated from the actual schema + a data dictionary."""
    return [
        {
            "id": "schema_overview",
            "text": (
                "This retail sales dataset contains orders from 2024-2025 with columns: "
                "OrderID, OrderDate, Category, SubCategory, Region, Segment, Quantity, "
                "Sales (revenue in dollars), and Profit (in dollars). Each row is one order line item."
            ),
        },
        {
            "id": "categories",
            "text": (
                "There are 3 product categories: Technology (Laptops, Monitors, Printers, "
                "Accessories), Furniture (Chairs, Desks, Bookcases, Tables), and Office "
                "Supplies (Paper, Binders, Pens, Storage)."
            ),
        },
        {
            "id": "regions",
            "text": (
                "Sales are tracked across 4 regions: East, West, Central, and South. "
                "Regional performance is compared using recent revenue vs prior period revenue."
            ),
        },
        {
            "id": "segments",
            "text": (
                "Customers fall into 3 segments: Consumer (individual buyers), Corporate "
                "(business accounts), and Home Office (small/home-based businesses)."
            ),
        },
        {
            "id": "glossary_margin",
            "text": (
                "Profit margin percentage is calculated as (Total Profit / Total Sales) * 100. "
                "A category with a low or negative margin is losing money relative to its revenue, "
                "even if its total sales figure looks high."
            ),
        },
        {
            "id": "glossary_underperformance",
            "text": (
                "A category or region is considered 'underperforming' in a given period when its "
                "revenue is meaningfully below its own historical average for that same metric — "
                "not just below other categories. This avoids unfairly comparing a naturally smaller "
                "category to a larger one."
            ),
        },
        {
            "id": "glossary_seasonality",
            "text": (
                "Seasonal patterns are detected by comparing a category's sales across calendar "
                "months, regardless of year, to find recurring peak and low months — useful for "
                "planning inventory and promotions ahead of predictable demand spikes."
            ),
        },
        {
            "id": "business_context",
            "text": (
                "This assistant is designed for retail operations managers who need quick, "
                "trustworthy answers about revenue trends, regional performance, profitability, "
                "and seasonality — without manually digging through spreadsheets. All numeric "
                "claims must be traceable to the analytics functions, never invented."
            ),
        },
    ]


def get_or_create_vectorstore():
    """Creates (or loads, if already built) a persistent local ChromaDB
    collection. Persistent = survives between runs, stored on disk."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name="retail_context")

    # Only populate it if empty (avoids duplicate inserts on re-runs)
    if collection.count() == 0:
        docs = build_documents()
        collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
        )
        print(f"Vector store built with {len(docs)} context documents.")
    else:
        print(f"Vector store already exists with {collection.count()} documents.")

    return collection


def retrieve_context(query: str, n_results: int = 3) -> list[str]:
    """Given a user's natural-language question, returns the most
    relevant context chunks to ground the LLM's explanation."""
    collection = get_or_create_vectorstore()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    # Smoke test: build the store, then try a few realistic queries
    get_or_create_vectorstore()

    test_queries = [
        "Why did Office Supplies underperform?",
        "What does profit margin mean?",
        "Is there a seasonal pattern in Technology sales?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for chunk in retrieve_context(q):
            print(f"  → {chunk[:80]}...")
