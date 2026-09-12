"""
Generates a realistic synthetic retail sales dataset.
We bake in a few deliberate patterns so our AI assistant has
real signal to explain later:
  1. "Office Supplies" category underperforms in Q3 (a real dip)
  2. "West" region has a revenue decline in the last 2 months (churn-like pattern)
  3. "Technology" category spikes every November (seasonal pattern - Black Friday)
  4. A few products have negative profit margins (loss-making SKUs)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

CATEGORIES = {
    "Technology": ["Laptops", "Monitors", "Printers", "Accessories"],
    "Furniture": ["Chairs", "Desks", "Bookcases", "Tables"],
    "Office Supplies": ["Paper", "Binders", "Pens", "Storage"],
}
REGIONS = ["East", "West", "Central", "South"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)
date_range_days = (end_date - start_date).days

rows = []
order_id = 1000

for _ in range(6000):
    order_date = start_date + timedelta(days=int(np.random.uniform(0, date_range_days)))
    category = np.random.choice(list(CATEGORIES.keys()), p=[0.35, 0.25, 0.40])
    sub_category = np.random.choice(CATEGORIES[category])
    region = np.random.choice(REGIONS)
    segment = np.random.choice(SEGMENTS)

    base_price = {"Technology": 450, "Furniture": 300, "Office Supplies": 40}[category]
    quantity = np.random.randint(1, 8)

    # --- Pattern 1: Office Supplies underperforms in Q3 (Jul-Sep) ---
    seasonal_multiplier = 1.0
    if category == "Office Supplies" and order_date.month in [7, 8, 9]:
        seasonal_multiplier = 0.55  # deliberate dip

    # --- Pattern 2: West region declines in last 2 months of dataset ---
    if region == "West" and order_date >= datetime(2025, 11, 1):
        seasonal_multiplier *= 0.5

    # --- Pattern 3: Technology spikes every November ---
    if category == "Technology" and order_date.month == 11:
        seasonal_multiplier *= 1.8

    sales = round(base_price * quantity * seasonal_multiplier * np.random.uniform(0.8, 1.2), 2)

    # --- Pattern 4: ~8% of Office Supplies orders are loss-making ---
    if category == "Office Supplies" and np.random.random() < 0.08:
        profit = round(-abs(sales * np.random.uniform(0.05, 0.25)), 2)
    else:
        margin = {"Technology": 0.18, "Furniture": 0.12, "Office Supplies": 0.22}[category]
        profit = round(sales * margin * np.random.uniform(0.6, 1.3), 2)

    rows.append({
        "OrderID": f"ORD-{order_id}",
        "OrderDate": order_date.strftime("%Y-%m-%d"),
        "Category": category,
        "SubCategory": sub_category,
        "Region": region,
        "Segment": segment,
        "Quantity": quantity,
        "Sales": sales,
        "Profit": profit,
    })
    order_id += 1

df = pd.DataFrame(rows).sort_values("OrderDate").reset_index(drop=True)
df.to_csv("/home/claude/retail-bi-assistant/data/retail_sales.csv", index=False)
print(f"Generated {len(df)} rows")
print(df.head())
print("\nDate range:", df["OrderDate"].min(), "to", df["OrderDate"].max())
print("\nCategories:", df["Category"].unique())
print("Regions:", df["Region"].unique())
