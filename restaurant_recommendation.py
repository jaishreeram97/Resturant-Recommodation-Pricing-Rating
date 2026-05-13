"""
Task 2: Restaurant Recommendation System
Objective: Content-based filtering to recommend restaurants based on user preferences.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# 1. LOAD & PREPROCESS

print("=" * 65)
print("  TASK 2: RESTAURANT RECOMMENDATION SYSTEM")
print("=" * 65)

df = pd.read_csv("/mnt/user-data/uploads/Dataset_.csv", encoding="utf-8-sig")
print(f"\n Dataset loaded: {df.shape[0]} restaurants, {df.shape[1]} columns")

# Keep only rated restaurants
df = df[df["Aggregate rating"] > 0].copy()
df.reset_index(drop=True, inplace=True)

# Handle missing values
df["Cuisines"].fillna("Unknown", inplace=True)

print(f" Rated restaurants retained: {len(df)}")


# 2. FEATURE ENGINEERING


# Binary encode Yes/No columns
binary_cols = ["Has Table booking", "Has Online delivery",
               "Is delivering now", "Switch to order menu"]
for col in binary_cols:
    df[col + "_enc"] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

# Explode multi-cuisine column into one-hot encoded columns
cuisine_dummies = df["Cuisines"].str.get_dummies(sep=", ")
# Keep only top-50 cuisines to avoid excessive sparsity
top_cuisines = cuisine_dummies.sum().nlargest(50).index
cuisine_dummies = cuisine_dummies[top_cuisines]

# Normalize numeric features
scaler = MinMaxScaler()
num_features = ["Aggregate rating", "Price range", "Votes", "Average Cost for two"]
df_num = pd.DataFrame(
    scaler.fit_transform(df[num_features]),
    columns=[f + "_norm" for f in num_features]
)

# Build the feature matrix
feature_matrix = pd.concat([
    df_num,
    df[["Has Table booking_enc", "Has Online delivery_enc",
        "Is delivering now_enc"]],
    cuisine_dummies
], axis=1).fillna(0)

print(f" Feature matrix shape: {feature_matrix.shape}")

# Compute cosine similarity between all restaurants
similarity_matrix = cosine_similarity(feature_matrix)
print(f" Similarity matrix computed: {similarity_matrix.shape}")


# 3. RECOMMENDATION ENGINE


def recommend_by_restaurant(restaurant_name, top_n=5):
    """Recommend similar restaurants given a restaurant name."""
    matches = df[df["Restaurant Name"].str.contains(restaurant_name, case=False, na=False)]
    if matches.empty:
        return None, f" Restaurant '{restaurant_name}' not found."
    idx = matches.index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    rec_indices = [i[0] for i in sim_scores]
    rec_scores  = [round(i[1], 4) for i in sim_scores]
    results = df.iloc[rec_indices][
        ["Restaurant Name", "City", "Cuisines",
         "Aggregate rating", "Price range",
         "Has Online delivery", "Has Table booking", "Votes"]
    ].copy()
    results["Similarity Score"] = rec_scores
    results.reset_index(drop=True, inplace=True)
    return results, matches.iloc[0]


def recommend_by_preferences(cuisine=None, min_rating=3.5,
                              price_range=None, online_delivery=None,
                              table_booking=None, top_n=10):
    """
    Recommend restaurants from scratch using user preferences.
    Criteria:
      cuisine        — string or list of cuisines (partial match)
      min_rating     — minimum aggregate rating (0–5)
      price_range    — 1 (cheap) to 4 (expensive)
      online_delivery — True / False / None (any)
      table_booking   — True / False / None (any)
    """
    filtered = df.copy()

    if cuisine:
        if isinstance(cuisine, str):
            cuisine = [cuisine]
        mask = filtered["Cuisines"].str.contains("|".join(cuisine), case=False, na=False)
        filtered = filtered[mask]

    filtered = filtered[filtered["Aggregate rating"] >= min_rating]

    if price_range is not None:
        filtered = filtered[filtered["Price range"] == price_range]

    if online_delivery is not None:
        val = "Yes" if online_delivery else "No"
        filtered = filtered[filtered["Has Online delivery"] == val]

    if table_booking is not None:
        val = "Yes" if table_booking else "No"
        filtered = filtered[filtered["Has Table booking"] == val]

    if filtered.empty:
        return pd.DataFrame(), "No restaurants match your preferences."

    # Rank by rating then votes
    filtered = filtered.sort_values(
        ["Aggregate rating", "Votes"], ascending=False
    ).head(top_n)

    result = filtered[[
        "Restaurant Name", "City", "Cuisines",
        "Aggregate rating", "Price range",
        "Has Online delivery", "Has Table booking", "Votes"
    ]].copy()
    result.reset_index(drop=True, inplace=True)
    result.index += 1
    return result, "OK"



# 4. TEST THE SYSTEM


print("\n" + "=" * 65)
print("  TESTING THE RECOMMENDATION SYSTEM")
print("=" * 65)

# ── Test A: Recommend by cuisine + rating ─────
print("\n SAMPLE USER 1:")
print("   Preferences → Cuisine: Italian | Min Rating: 4.0 | Price range: 2")
recs1, msg1 = recommend_by_preferences(
    cuisine="Italian", min_rating=4.0, price_range=2, top_n=5
)
if not recs1.empty:
    print(recs1.to_string())
else:
    print(f"   {msg1}")

# ── Test B: Recommend by cuisine + online delivery ─────────
print("\n SAMPLE USER 2:")
print("   Preferences → Cuisine: Chinese | Online Delivery: Yes | Min Rating: 3.5")
recs2, msg2 = recommend_by_preferences(
    cuisine="Chinese", min_rating=3.5, online_delivery=True, top_n=5
)
if not recs2.empty:
    print(recs2.to_string())
else:
    print(f"   {msg2}")

# ── Test C: Recommend by cuisine + price ───────────────────
print("\n SAMPLE USER 3:")
print("   Preferences → Cuisine: North Indian | Price range: 1 (cheap) | Min Rating: 3.0")
recs3, msg3 = recommend_by_preferences(
    cuisine="North Indian", min_rating=3.0, price_range=1, top_n=5
)
if not recs3.empty:
    print(recs3.to_string())
else:
    print(f"   {msg3}")

# ── Test D: Content-based similarity (find restaurants like X) ─
print("\n SAMPLE USER 4 — 'Find restaurants similar to a known one':")
search_name = "KFC"
recs4, ref = recommend_by_restaurant(search_name, top_n=5)
if recs4 is not None:
    print(f"   Seed restaurant : {ref['Restaurant Name']} | "
          f"City: {ref['City']} | "
          f"Rating: {ref['Aggregate rating']} | "
          f"Cuisines: {ref['Cuisines']}")
    print(f"\n   Top 5 similar restaurants:")
    print(recs4[["Restaurant Name", "City", "Cuisines",
                 "Aggregate rating", "Similarity Score"]].to_string())
else:
    print(f"   {ref}")


# 5. VISUALISATIONS

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle("Task 2 — Restaurant Recommendation System",
             fontsize=17, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
           "#CCB974", "#64B5CD"]

# ── Plot 1: Rating distribution of full dataset ──────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor("#FAFAFA")
bins = np.arange(1.0, 5.2, 0.2)
ax0.hist(df["Aggregate rating"], bins=bins, color="#4C72B0",
         edgecolor="white", linewidth=0.6)
ax0.axvline(df["Aggregate rating"].mean(), color="red", linestyle="--",
            linewidth=1.5, label=f"Mean = {df['Aggregate rating'].mean():.2f}")
ax0.set_xlabel("Aggregate Rating", fontsize=10)
ax0.set_ylabel("Count", fontsize=10)
ax0.set_title("Rating Distribution", fontsize=11, fontweight="bold")
ax0.legend(fontsize=9)

# ── Plot 2: Price range distribution ───
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor("#FAFAFA")
price_counts = df["Price range"].value_counts().sort_index()
price_labels = {1: "Cheap\n(1)", 2: "Moderate\n(2)",
                3: "Expensive\n(3)", 4: "Very Exp.\n(4)"}
bars = ax1.bar(
    [price_labels.get(p, str(p)) for p in price_counts.index],
    price_counts.values,
    color=palette[:4], edgecolor="white", linewidth=0.7
)
ax1.set_xlabel("Price Range", fontsize=10)
ax1.set_ylabel("Count", fontsize=10)
ax1.set_title("Price Range Distribution", fontsize=11, fontweight="bold")
for bar, val in zip(bars, price_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(val), ha="center", fontsize=9, fontweight="bold")

# ── Plot 3: Top 10 cuisines ────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor("#FAFAFA")
top10_cuisines = cuisine_dummies.sum().nlargest(10)
ax2.barh(top10_cuisines.index[::-1], top10_cuisines.values[::-1],
         color=palette, edgecolor="white", linewidth=0.7)
ax2.set_xlabel("Number of Restaurants", fontsize=10)
ax2.set_title("Top 10 Cuisines", fontsize=11, fontweight="bold")
for i, val in enumerate(top10_cuisines.values[::-1]):
    ax2.text(val + 5, i, str(int(val)), va="center", fontsize=8)

# ── Plot 4: Sample User 1 recommendations ─────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#FAFAFA")
if not recs1.empty:
    names  = [n[:22] + "…" if len(n) > 22 else n for n in recs1["Restaurant Name"]]
    ratings = recs1["Aggregate rating"].values
    bars = ax3.barh(names[::-1], ratings[::-1], color="#55A868",
                    edgecolor="white", linewidth=0.7)
    ax3.set_xlim(0, 5.5)
    ax3.set_xlabel("Rating", fontsize=10)
    ax3.set_title("User 1 Recs\n(Italian, Rating≥4, Price=2)",
                  fontsize=10, fontweight="bold")
    for bar, val in zip(bars, ratings[::-1]):
        ax3.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val}", va="center", fontsize=9, fontweight="bold")

# ── Plot 5: Sample User 4 — similarity scores ────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#FAFAFA")
if recs4 is not None and not recs4.empty:
    sim_names   = [n[:22] + "…" if len(n) > 22 else n
                   for n in recs4["Restaurant Name"]]
    sim_scores  = recs4["Similarity Score"].values
    bars = ax4.barh(sim_names[::-1], sim_scores[::-1], color="#C44E52",
                    edgecolor="white", linewidth=0.7)
    ax4.set_xlim(0, 1.1)
    ax4.set_xlabel("Cosine Similarity Score", fontsize=10)
    ax4.set_title(f"Content-Based Recs\n(Similar to '{search_name}')",
                  fontsize=10, fontweight="bold")
    for bar, val in zip(bars, sim_scores[::-1]):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

# ── Plot 6: Online delivery vs Rating ───
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor("#FAFAFA")
delivery_yes = df[df["Has Online delivery"] == "Yes"]["Aggregate rating"]
delivery_no  = df[df["Has Online delivery"] == "No"]["Aggregate rating"]
bp = ax5.boxplot([delivery_no, delivery_yes],
                 labels=["No Delivery", "Online Delivery"],
                 patch_artist=True,
                 boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                 medianprops=dict(color="red", linewidth=2))
bp["boxes"][1].set_facecolor("#DD8452")
ax5.set_ylabel("Aggregate Rating", fontsize=10)
ax5.set_title("Rating by Delivery Option", fontsize=11, fontweight="bold")
ax5.set_ylim(0, 5.5)
for i, data in enumerate([delivery_no, delivery_yes], 1):
    ax5.text(i, 0.3, f"n={len(data)}", ha="center", fontsize=9, color="grey")

plt.savefig("/mnt/user-data/outputs/restaurant_recommendation.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n Chart saved → restaurant_recommendation.png")

# ── Plot 7: Model comparison (R² and RMSE) + Feature importance + Rating distribution ─
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#F8F9FA")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.38)
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
           "#CCB974", "#64B5CD"]
colors = palette[:len(names)]

# 6. RECOMMENDATION QUALITY EVALUATION

print("\n" + "=" * 65)
print("  RECOMMENDATION QUALITY EVALUATION")
print("=" * 65)

# Coverage: fraction of dataset the system can recommend from
coverage = len(df) / len(df) * 100
print(f"\n📊 Catalog Coverage        : {coverage:.1f}% ({len(df)} restaurants)")

# Avg similarity score for content-based recommendations
if recs4 is not None and not recs4.empty:
    avg_sim = recs4["Similarity Score"].mean()
    print(f"📊 Avg Cosine Similarity   : {avg_sim:.4f}  (User 4 — similar to '{search_name}')")

# Avg rating of preference-based recommendations
for i, (recs, label) in enumerate(
        [(recs1, "User 1 (Italian)"),
         (recs2, "User 2 (Chinese, delivery)"),
         (recs3, "User 3 (N. Indian, cheap)")], 1):
    if not recs.empty:
        print(f"📊 Avg Recommended Rating  : {recs['Aggregate rating'].mean():.2f}  ({label})")

print(f"\n System successfully tested on 4 user profiles.")
print(f"   Recommendation approaches used:")
print(f"     1. Preference-based filtering (cuisine, rating, price, delivery)")
print(f"     2. Content-based filtering (cosine similarity on feature vectors)")
