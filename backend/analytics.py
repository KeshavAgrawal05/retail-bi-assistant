"""
analytics.py — the "ground truth" compute layer.

RULE: every number the app ever shows the user comes from THIS file.
The LLM is never allowed to calculate anything itself — it only
explains numbers these functions already computed. This is what
keeps the assistant grounded instead of hallucinating.

Each function returns a plain dict with:
- "data": chart-ready data (for the frontend)
- "summary": the key numbers as plain values (for the LLM prompt)
"""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "retail_sales.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["OrderDate"])
    df["Month"] = df["OrderDate"].dt.to_period("M").astype(str)
    df["Quarter"] = df["OrderDate"].dt.to_period("Q").astype(str)
    df["Year"] = df["OrderDate"].dt.year
    return df


# ---------- Q1 & Q3: Category performance / underperformance ----------
def category_performance(quarter: str | None=None) -> dict:
    """Revenue by category, optionally filtered to a specific quarter,
    compared against the category's own average across all quarters."""
    df = load_data()
    by_qtr_cat = df.groupby(["Quarter", "Category"])["Sales"].sum().reset_index()
    avg_by_cat = by_qtr_cat.groupby("Category")["Sales"].mean()

    if quarter is None:
        quarter = df["Quarter"].max()

    current = by_qtr_cat[by_qtr_cat["Quarter"] == quarter].set_index("Category")["Sales"]
    comparison = []
    for cat in avg_by_cat.index:
        cur_val = float(current.get(cat, 0))
        avg_val = float(avg_by_cat[cat])
        pct_vs_avg = round(((cur_val - avg_val) / avg_val) * 100, 1) if avg_val else 0
        comparison.append({
            "category": cat, "quarter_revenue": round(cur_val, 2),
            "avg_quarter_revenue": round(avg_val, 2), "pct_vs_average": pct_vs_avg,
        })
    comparison.sort(key=lambda x: x["pct_vs_average"])

    return {
        "data": comparison,
        "summary": {
            "quarter": quarter,
            "worst_performer": comparison[0]["category"],
            "worst_pct_vs_avg": comparison[0]["pct_vs_average"],
            "best_performer": comparison[-1]["category"],
        },
    }


# ---------- Q2: Regional revenue change ----------
def region_revenue_change(recent_months: int=2) -> dict:
    """Compares each region's revenue in the most recent N months vs the
    N months before that, to detect drops."""
    df = load_data()
    months_sorted = sorted(df["Month"].unique())
    recent = months_sorted[-recent_months:]
    prior = months_sorted[-2 * recent_months:-recent_months]

    recent_rev = df[df["Month"].isin(recent)].groupby("Region")["Sales"].sum()
    prior_rev = df[df["Month"].isin(prior)].groupby("Region")["Sales"].sum()

    results = []
    for region in df["Region"].unique():
        r = float(recent_rev.get(region, 0))
        p = float(prior_rev.get(region, 0))
        pct_change = round(((r - p) / p) * 100, 1) if p else 0
        results.append({
            "region": region, "recent_revenue": round(r, 2),
            "prior_revenue": round(p, 2), "pct_change": pct_change,
        })
    results.sort(key=lambda x: x["pct_change"])

    return {
        "data": results,
        "summary": {
            "recent_period": recent, "prior_period": prior,
            "biggest_decline_region": results[0]["region"],
            "biggest_decline_pct": results[0]["pct_change"],
        },
    }


# ---------- Q3/Q4/Q7: Monthly trend, growth, seasonality (shared engine) ----------
def monthly_trend(category: str | None=None, region: str | None=None) -> dict:
    df = load_data()
    if category:
        df = df[df["Category"] == category]
    if region:
        df = df[df["Region"] == region]

    trend = df.groupby("Month")["Sales"].sum().reset_index().sort_values("Month")
    trend["pct_change"] = trend["Sales"].pct_change().round(3) * 100
    trend = trend.astype(object).where(pd.notnull(trend), None)

    return {
        "data": trend.to_dict(orient="records"),
        "summary": {
            "filter": {"category": category, "region": region},
            "latest_month": trend.iloc[-1]["Month"] if len(trend) else None,
            "latest_revenue": round(float(trend.iloc[-1]["Sales"]), 2) if len(trend) else 0,
            "latest_pct_change": round(float(trend.iloc[-1]["pct_change"]), 1) if len(trend) > 1 and trend.iloc[-1]["pct_change"] is not None else 0,
        },
    }


# ---------- Q4: Fastest growing category/region combo ----------
def fastest_growing_combo() -> dict:
    df = load_data()
    years = sorted(df["Year"].unique())
    if len(years) < 2:
        return {"data": [], "summary": {"note": "Not enough years of data"}}

    by_combo = df.groupby(["Category", "Region", "Year"])["Sales"].sum().reset_index()
    pivot = by_combo.pivot_table(index=["Category", "Region"], columns="Year", values="Sales", fill_value=0)
    pivot["growth_pct"] = ((pivot[years[-1]] - pivot[years[0]]) / pivot[years[0]].replace(0, 1)) * 100
    pivot = pivot.reset_index().sort_values("growth_pct", ascending=False)

    top = pivot.head(5)[["Category", "Region", "growth_pct"]].round(1)
    top["growth_pct"] = top["growth_pct"].astype(float)
    top = top.to_dict(orient="records")
    years = [int(y) for y in years]
    return {
        "data": top,
        "summary": {"fastest_growing": top[0] if top else None, "years_compared": years},
    }


# ---------- Q5 & Q6: Profitability ----------
def profitability_analysis() -> dict:
    df = load_data()
    by_cat = df.groupby("Category").agg(
        total_sales=("Sales", "sum"), total_profit=("Profit", "sum"), orders=("OrderID", "count")
    ).reset_index()
    by_cat["margin_pct"] = round((by_cat["total_profit"] / by_cat["total_sales"]) * 100, 1)

    loss_making = df[df["Profit"] < 0]
    loss_by_cat = loss_making.groupby("Category").agg(
        loss_orders=("OrderID", "count"), total_loss=("Profit", "sum")
    ).reset_index()

    return {
        "data": {
            "by_category": by_cat.round(2).to_dict(orient="records"),
            "loss_making_orders": loss_by_cat.round(2).to_dict(orient="records"),
        },
        "summary": {
            "lowest_margin_category": by_cat.loc[by_cat["margin_pct"].idxmin(), "Category"],
            "total_loss_making_orders": int(len(loss_making)),
            "total_loss_amount": round(float(loss_making["Profit"].sum()), 2),
        },
    }


# ---------- Q7: Seasonal pattern detection ----------
def seasonal_pattern(category: str) -> dict:
    df = load_data()
    df = df[df["Category"] == category]
    df["MonthNum"] = df["OrderDate"].dt.month
    by_month = df.groupby("MonthNum")["Sales"].sum().reset_index()
    peak_month = int(by_month.loc[by_month["Sales"].idxmax(), "MonthNum"])

    return {
        "data": by_month.round(2).to_dict(orient="records"),
        "summary": {"category": category, "peak_month": peak_month},
    }


# ---------- Q8: Year-over-year comparison ----------
def yoy_comparison() -> dict:
    df = load_data()
    df["MonthOnly"] = df["OrderDate"].dt.month
    by_year_month = df.groupby(["Year", "MonthOnly"])["Sales"].sum().reset_index()
    pivot = by_year_month.pivot(index="MonthOnly", columns="Year", values="Sales").fillna(0)

    return {"data": pivot.round(2).reset_index().to_dict(orient="records"), "summary": {"years": list(pivot.columns)}}


# ---------- Q9: Segment profitability ----------
def segment_profitability() -> dict:
    df = load_data()
    by_seg = df.groupby("Segment").agg(
        total_sales=("Sales", "sum"), total_profit=("Profit", "sum")
    ).reset_index()
    by_seg["margin_pct"] = round((by_seg["total_profit"] / by_seg["total_sales"]) * 100, 1)
    by_seg = by_seg.sort_values("total_profit", ascending=False)

    return {
        "data": by_seg.round(2).to_dict(orient="records"),
        "summary": {"most_profitable_segment": by_seg.iloc[0]["Segment"]},
    }


# ---------- Q10: Top/bottom sub-categories ----------
def top_bottom_subcategories(n: int=3) -> dict:
    df = load_data()
    by_sub = df.groupby("SubCategory")["Sales"].sum().sort_values(ascending=False)

    return {
        "data": {
            "top": by_sub.head(n).round(2).to_dict(),
            "bottom": by_sub.tail(n).round(2).to_dict(),
        },
        "summary": {"top_subcategory": by_sub.index[0], "bottom_subcategory": by_sub.index[-1]},
    }


if __name__ == "__main__":
    # Quick smoke test of every function
    import json
    print("=== Category performance (latest quarter) ===")
    print(json.dumps(category_performance()["summary"], indent=2))
    print("\n=== Category performance (2024Q3 — where our real dip is) ===")
    print(json.dumps(category_performance("2024Q3")["summary"], indent=2))
    print("\n=== Region revenue change ===")
    print(json.dumps(region_revenue_change()["summary"], indent=2))
    print("\n=== Fastest growing combo ===")
    print(json.dumps(fastest_growing_combo()["summary"], indent=2))
    print("\n=== Profitability ===")
    print(json.dumps(profitability_analysis()["summary"], indent=2))
    print("\n=== Seasonal pattern (Technology) ===")
    print(json.dumps(seasonal_pattern("Technology")["summary"], indent=2))
    print("\n=== Segment profitability ===")
    print(json.dumps(segment_profitability()["summary"], indent=2))
    print("\n=== Top/bottom sub-categories ===")
    print(json.dumps(top_bottom_subcategories()["summary"], indent=2))
