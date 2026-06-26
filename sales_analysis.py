"""
Sales Performance Analysis
Ty Sipper — Portfolio Project

Analyzes monthly sales data for a fictional retail business.
Demonstrates: data cleaning, aggregation, trend analysis, and visualization.

Requirements: pip install pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
from datetime import datetime, timedelta

# ── 1. Generate realistic sample data ─────────────────────────────────────────

random.seed(42)

REGIONS     = ["West", "East", "Midwest", "South"]
CATEGORIES  = ["Electronics", "Apparel", "Home Goods", "Office Supplies"]

def make_dataset(n=500):
    start = datetime(2023, 1, 1)
    rows = []
    for _ in range(n):
        region   = random.choice(REGIONS)
        category = random.choice(CATEGORIES)
        date     = start + timedelta(days=random.randint(0, 364))
        units    = random.randint(1, 20)
        price    = round(random.uniform(10, 300), 2)
        # Electronics trend slightly higher; West region slightly higher
        price   *= 1.3 if category == "Electronics" else 1.0
        price   *= 1.1 if region == "West" else 1.0
        revenue  = round(units * price, 2)
        rows.append({
            "date":     date,
            "region":   region,
            "category": category,
            "units":    units,
            "price":    round(price, 2),
            "revenue":  revenue,
        })
    return pd.DataFrame(rows)

# ── 2. Load & clean ────────────────────────────────────────────────────────────

df = make_dataset()
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M")

# Drop any rows with nulls (none here, but good practice)
before = len(df)
df.dropna(inplace=True)
print(f"Loaded {before} records — {before - len(df)} dropped after cleaning.\n")

# ── 3. Summary statistics ──────────────────────────────────────────────────────

total_revenue = df["revenue"].sum()
avg_order     = df["revenue"].mean()
top_category  = df.groupby("category")["revenue"].sum().idxmax()
top_region    = df.groupby("region")["revenue"].sum().idxmax()

print("── Summary ──────────────────────────────────────")
print(f"  Total Revenue    : ${total_revenue:>12,.2f}")
print(f"  Avg Order Value  : ${avg_order:>12,.2f}")
print(f"  Top Category     : {top_category}")
print(f"  Top Region       : {top_region}")
print()

# ── 4. Monthly revenue trend ──────────────────────────────────────────────────

monthly = (
    df.groupby("month")["revenue"]
    .sum()
    .reset_index()
)
monthly["month_str"] = monthly["month"].astype(str)

# ── 5. Revenue by category ────────────────────────────────────────────────────

by_category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)

# ── 6. Region × category heatmap data ────────────────────────────────────────

heatmap_data = (
    df.groupby(["region", "category"])["revenue"]
    .sum()
    .unstack(fill_value=0)
)

# ── 7. Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sales Performance Analysis — 2023", fontsize=16, fontweight="bold", y=1.01)

PALETTE = ["#2563EB", "#3B82F6", "#93C5FD", "#DBEAFE"]

# Chart 1 — Monthly revenue trend
ax1 = axes[0, 0]
ax1.plot(
    monthly["month_str"], monthly["revenue"],
    color="#2563EB", linewidth=2.5, marker="o", markersize=5
)
ax1.fill_between(monthly["month_str"], monthly["revenue"], alpha=0.12, color="#2563EB")
ax1.set_title("Monthly Revenue Trend", fontweight="bold")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue ($)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.tick_params(axis="x", rotation=45)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Chart 2 — Revenue by category (horizontal bar)
ax2 = axes[0, 1]
bars = ax2.barh(by_category.index, by_category.values, color=PALETTE)
ax2.set_title("Revenue by Category", fontweight="bold")
ax2.set_xlabel("Revenue ($)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax2.grid(axis="x", linestyle="--", alpha=0.4)
for bar, val in zip(bars, by_category.values):
    ax2.text(val + 500, bar.get_y() + bar.get_height() / 2,
             f"${val:,.0f}", va="center", fontsize=9)

# Chart 3 — Revenue by region (pie)
ax3 = axes[1, 0]
by_region = df.groupby("region")["revenue"].sum()
ax3.pie(
    by_region.values,
    labels=by_region.index,
    autopct="%1.1f%%",
    colors=PALETTE,
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax3.set_title("Revenue Share by Region", fontweight="bold")

# Chart 4 — Heatmap: region × category
ax4 = axes[1, 1]
im = ax4.imshow(heatmap_data.values, cmap="Blues", aspect="auto")
ax4.set_xticks(range(len(heatmap_data.columns)))
ax4.set_xticklabels(heatmap_data.columns, rotation=30, ha="right")
ax4.set_yticks(range(len(heatmap_data.index)))
ax4.set_yticklabels(heatmap_data.index)
ax4.set_title("Revenue Heatmap: Region × Category", fontweight="bold")
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.values[i, j]
        ax4.text(j, i, f"${val:,.0f}", ha="center", va="center",
                 fontsize=8, color="white" if val > heatmap_data.values.max() * 0.6 else "black")

plt.colorbar(im, ax=ax4, label="Revenue ($)")
plt.tight_layout()
plt.savefig("sales_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to sales_analysis.png")
