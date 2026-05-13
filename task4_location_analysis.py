"""
Task 4: Location-Based Analysis
Objective: Perform a geographical analysis of restaurants in the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")


# 1. LOAD & PREPROCESS

print("=" * 65)
print("  TASK 4: LOCATION-BASED ANALYSIS")
print("=" * 65)

df = pd.read_csv("/mnt/user-data/uploads/Dataset_.csv", encoding="utf-8-sig")
df["Cuisines"].fillna("Unknown", inplace=True)
df["Primary_Cuisine"] = df["Cuisines"].apply(
    lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown"
)

# Remove invalid coordinates
df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()
df = df[df["Aggregate rating"] > 0].copy()
df.reset_index(drop=True, inplace=True)
print(f"\n Valid restaurants with coordinates: {len(df)}")


# 2. CITY-LEVEL STATISTICS
 
city_stats = df.groupby("City").agg(
    Count            = ("Restaurant ID", "count"),
    Avg_Rating       = ("Aggregate rating", "mean"),
    Avg_Price        = ("Price range", "mean"),
    Avg_Cost         = ("Average Cost for two", "mean"),
    Avg_Votes        = ("Votes", "mean"),
    Online_Del_Pct   = ("Has Online delivery",
                        lambda x: (x == "Yes").sum() / len(x) * 100),
    Table_Book_Pct   = ("Has Table booking",
                        lambda x: (x == "Yes").sum() / len(x) * 100),
    Lat              = ("Latitude",  "mean"),
    Lon              = ("Longitude", "mean"),
).reset_index()

city_stats = city_stats.sort_values("Count", ascending=False)
top10_cities = city_stats.head(10).copy()

print(f"\n📊 Total cities: {len(city_stats)}")
print(f"\n🏙️  Top 10 cities by restaurant count:")
print(top10_cities[["City","Count","Avg_Rating","Avg_Price",
                      "Online_Del_Pct"]].to_string(index=False))


# 3. COUNTRY-LEVEL STATISTICS

# Map Country Code → Country name (Zomato dataset country codes)
country_map = {
    1:"India", 14:"Australia", 30:"Brazil", 37:"Canada",
    94:"Indonesia", 148:"New Zealand", 162:"Philippines",
    166:"Qatar", 184:"Singapore", 189:"South Africa",
    191:"Sri Lanka", 208:"Turkey", 214:"UAE",
    215:"United Kingdom", 216:"United States"
}
df["Country"] = df["Country Code"].map(country_map).fillna("Other")

country_stats = df.groupby("Country").agg(
    Count      = ("Restaurant ID", "count"),
    Avg_Rating = ("Aggregate rating", "mean"),
    Avg_Price  = ("Price range", "mean"),
).reset_index().sort_values("Count", ascending=False)

print(f"\n🌍 Restaurants by country:")
print(country_stats.to_string(index=False))


# 4. LOCALITY STATISTICS (India only — most data)
india = df[df["Country"] == "India"].copy()
locality_stats = india.groupby("Locality").agg(
    Count      = ("Restaurant ID", "count"),
    Avg_Rating = ("Aggregate rating", "mean"),
    Avg_Cost   = ("Average Cost for two", "mean"),
).reset_index().sort_values("Count", ascending=False)

print(f"\n Top 10 localities in India:")
print(locality_stats.head(10).to_string(index=False))


# 5. INTERESTING PATTERNS

print("\n" + "=" * 65)
print("  KEY GEOGRAPHICAL INSIGHTS")
print("=" * 65)

highest_rated_city = city_stats.loc[city_stats["Avg_Rating"].idxmax()]
most_restaurants   = city_stats.iloc[0]
most_delivery      = city_stats.loc[city_stats["Online_Del_Pct"].idxmax()]
most_expensive     = city_stats.loc[city_stats["Avg_Cost"].idxmax()]

print(f"\n🏆 Most restaurants   : {most_restaurants['City']} ({int(most_restaurants['Count'])})")
print(f"⭐  Highest avg rating  : {highest_rated_city['City']} ({highest_rated_city['Avg_Rating']:.2f})")
print(f"🚚 Most online delivery: {most_delivery['City']} ({most_delivery['Online_Del_Pct']:.1f}%)")
print(f"💰 Most expensive city : {most_expensive['City']} (avg cost: {most_expensive['Avg_Cost']:.0f})")


# 6. VISUALISATIONS  (7 panels)

BG     = "#F8F9FA"
PALBG  = "#FFFFFF"
COLORS = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
          "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]

fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor(BG)
fig.suptitle("Task 4 — Location-Based Restaurant Analysis",
             fontsize=18, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

# ── Panel 1: Geographic scatter map ──────────────────────────
ax0 = fig.add_subplot(gs[0, :])          # full top row
ax0.set_facecolor("#e8f4f8")

# colour by country
countries = df["Country"].unique()
cmap_c    = plt.cm.get_cmap("tab10", len(countries))
country_color = {c: cmap_c(i) for i, c in enumerate(sorted(countries))}

for country, grp in df.groupby("Country"):
    ax0.scatter(grp["Longitude"], grp["Latitude"],
                s=grp["Aggregate rating"] * 3,
                color=country_color[country],
                alpha=0.45, linewidths=0,
                label=f"{country} ({len(grp)})")

ax0.set_xlabel("Longitude", fontsize=10)
ax0.set_ylabel("Latitude",  fontsize=10)
ax0.set_title("Global Restaurant Distribution\n"
              "(dot size ∝ rating, colour = country)",
              fontsize=12, fontweight="bold")
ax0.legend(loc="lower left", fontsize=7.5, ncol=4,
           framealpha=0.85, markerscale=1.2)
ax0.spines[["top","right"]].set_visible(False)

# annotate top cities
for _, row in top10_cities.head(5).iterrows():
    ax0.annotate(row["City"],
                 xy=(row["Lon"], row["Lat"]),
                 fontsize=7, color="black",
                 xytext=(5, 5), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.2",
                           fc="white", alpha=0.7, lw=0))

# ── Panel 2: Top-10 cities by count ──────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor(PALBG)
bars = ax1.barh(top10_cities["City"][::-1],
                top10_cities["Count"][::-1],
                color=COLORS, edgecolor="white", linewidth=0.7)
ax1.set_xlabel("Number of Restaurants", fontsize=10)
ax1.set_title("Top 10 Cities\nby Restaurant Count", fontsize=11, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, top10_cities["Count"][::-1]):
    ax1.text(val + 8, bar.get_y() + bar.get_height()/2,
             str(int(val)), va="center", fontsize=8.5)

# ── Panel 3: Average rating by top-10 city ───────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor(PALBG)
city_rating = city_stats.nlargest(10, "Avg_Rating")
cmap_rating = plt.cm.RdYlGn
norm_r      = plt.Normalize(city_rating["Avg_Rating"].min(),
                             city_rating["Avg_Rating"].max())
bar_colors  = [cmap_rating(norm_r(v)) for v in city_rating["Avg_Rating"]]
bars2 = ax2.barh(city_rating["City"][::-1],
                  city_rating["Avg_Rating"][::-1],
                  color=bar_colors[::-1], edgecolor="white", linewidth=0.7)
ax2.set_xlabel("Average Rating", fontsize=10)
ax2.set_xlim(0, 5.5)
ax2.set_title("Top 10 Cities\nby Average Rating", fontsize=11, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars2, city_rating["Avg_Rating"][::-1]):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=8.5, fontweight="bold")

# ── Panel 4: Restaurants by country (pie) ────────────────────
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor(PALBG)
top_countries = country_stats.head(8)
others_count  = country_stats.iloc[8:]["Count"].sum()
pie_labels    = list(top_countries["City"] if "City" in top_countries
                     else top_countries["Country"])
pie_labels    = list(top_countries["Country"])
pie_vals      = list(top_countries["Count"])
if others_count > 0:
    pie_labels.append("Others")
    pie_vals.append(others_count)

wedges, texts, autotexts = ax3.pie(
    pie_vals, labels=pie_labels,
    autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
    colors=COLORS[:len(pie_vals)],
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1)
)
for t in texts:      t.set_fontsize(8)
for t in autotexts:  t.set_fontsize(7.5)
ax3.set_title("Restaurant Share\nby Country", fontsize=11, fontweight="bold")

# ── Panel 5: Avg cost for two by top cities ───────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(PALBG)
city_cost = city_stats[city_stats["Count"] >= 20].nlargest(10, "Avg_Cost")
bars3 = ax4.bar(range(len(city_cost)),
                city_cost["Avg_Cost"],
                color=COLORS[:len(city_cost)],
                edgecolor="white", linewidth=0.7, width=0.6)
ax4.set_xticks(range(len(city_cost)))
ax4.set_xticklabels(city_cost["City"], rotation=35, ha="right", fontsize=8)
ax4.set_ylabel("Avg Cost for Two", fontsize=10)
ax4.set_title("Top 10 Most Expensive Cities\n(Avg Cost for Two)", fontsize=11, fontweight="bold")
ax4.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars3, city_cost["Avg_Cost"]):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 50,
             f"{val:.0f}", ha="center", fontsize=7.5, fontweight="bold")

# ── Panel 6: Online delivery % by top cities ─────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(PALBG)
city_del = city_stats[city_stats["Count"] >= 30].sort_values(
    "Online_Del_Pct", ascending=False).head(10)
bar_cols6 = [plt.cm.Blues(0.4 + 0.5 * v/100) for v in city_del["Online_Del_Pct"]]
bars4 = ax5.barh(city_del["City"][::-1],
                  city_del["Online_Del_Pct"][::-1],
                  color=bar_cols6[::-1], edgecolor="white", linewidth=0.7)
ax5.set_xlabel("Online Delivery %", fontsize=10)
ax5.set_xlim(0, 115)
ax5.set_title("Top 10 Cities\nby Online Delivery Rate", fontsize=11, fontweight="bold")
ax5.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars4, city_del["Online_Del_Pct"][::-1]):
    ax5.text(val + 1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=8.5, fontweight="bold")

# ── Panel 7: Rating vs Price range scatter by country ─────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor(PALBG)
for country, grp in df.groupby("Country"):
    ax6.scatter(grp["Price range"] + np.random.uniform(-0.15, 0.15, len(grp)),
                grp["Aggregate rating"],
                alpha=0.18, s=12,
                color=country_color[country],
                label=country)

# overlay mean line per price range
pr_mean = df.groupby("Price range")["Aggregate rating"].mean()
ax6.plot(pr_mean.index, pr_mean.values, "r-o",
         linewidth=2, markersize=7, label="Mean rating", zorder=5)
for x, y in pr_mean.items():
    ax6.text(x + 0.08, y + 0.05, f"{y:.2f}", fontsize=8.5,
             fontweight="bold", color="darkred")

ax6.set_xticks([1, 2, 3, 4])
ax6.set_xticklabels(["Cheap\n(1)", "Moderate\n(2)",
                      "Expensive\n(3)", "Very Exp.\n(4)"], fontsize=8)
ax6.set_ylabel("Aggregate Rating", fontsize=10)
ax6.set_title("Rating vs Price Range\n(all countries)", fontsize=11, fontweight="bold")
ax6.spines[["top","right"]].set_visible(False)
ax6.legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.8)

plt.savefig("/mnt/user-data/outputs/task4_location_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n Chart saved → task4_location_analysis.png")


# 7. FINAL SUMMARY TABLE

print("\n" + "=" * 65)
print("  CITY STATS SUMMARY (Top 15 by count)")
print("=" * 65)
display_cols = ["City","Count","Avg_Rating","Avg_Price",
                "Avg_Cost","Online_Del_Pct","Table_Book_Pct"]
print(city_stats.head(15)[display_cols].round(2).to_string(index=False))

print("\n" + "=" * 65)
print("  COUNTRY STATS SUMMARY")
print("=" * 65)
print(country_stats.round(2).to_string(index=False))

print("\n" + "=" * 65)
print("  NOTABLE PATTERNS")
print("=" * 65)
# Price vs Rating correlation
corr = df[["Price range","Aggregate rating"]].corr().iloc[0,1]
print(f"\n Price–Rating correlation       : {corr:.4f}")
print(f"   → Higher price range → slightly higher ratings")

votes_corr = df[["Votes","Aggregate rating"]].corr().iloc[0,1]
print(f"\n Votes–Rating correlation        : {votes_corr:.4f}")
print(f"   → More votes strongly linked to higher ratings")

india_pct = len(india) / len(df) * 100
print(f"\n India's share of dataset        : {india_pct:.1f}%")
print(f"   → Dataset is heavily India-centric")

# New Delhi dominance
nd = df[df["City"] == "New Delhi"]
print(f"\n New Delhi restaurants           : {len(nd)} ({len(nd)/len(df)*100:.1f}% of total)")
print(f"   Avg rating in New Delhi        : {nd['Aggregate rating'].mean():.2f}")
print(f"   Online delivery in New Delhi   : {(nd['Has Online delivery']=='Yes').mean()*100:.1f}%")
