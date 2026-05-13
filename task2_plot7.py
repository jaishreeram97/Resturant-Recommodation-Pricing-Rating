"""
Plot 7 (Task 2): Model Comparison (R² & RMSE) + Feature Importance + Rating Distribution
Uses the recommendation system's feature matrix to run regression models
so all three panels are grounded in Task 2's data pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. LOAD & BUILD FEATURE MATRIX (same pipeline as Task 2) ──────────────────
df = pd.read_csv("/mnt/user-data/uploads/Dataset_.csv", encoding="utf-8-sig")
df = df[df["Aggregate rating"] > 0].copy()
df.reset_index(drop=True, inplace=True)
df["Cuisines"].fillna("Unknown", inplace=True)

binary_cols = ["Has Table booking", "Has Online delivery",
               "Is delivering now", "Switch to order menu"]
for col in binary_cols:
    df[col + "_enc"] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

cuisine_dummies = df["Cuisines"].str.get_dummies(sep=", ")
top_cuisines    = cuisine_dummies.sum().nlargest(50).index
cuisine_dummies = cuisine_dummies[top_cuisines]

scaler      = MinMaxScaler()
num_cols    = ["Price range", "Votes", "Average Cost for two"]
df_num      = pd.DataFrame(
    scaler.fit_transform(df[num_cols]),
    columns=[c + "_norm" for c in num_cols]
)

enc_cols = ["Has Table booking_enc", "Has Online delivery_enc",
            "Is delivering now_enc", "Switch to order menu_enc"]

X = pd.concat([df_num, df[enc_cols], cuisine_dummies], axis=1).fillna(0)
y = df["Aggregate rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. TRAIN MODELS ───────────────────────────────────────────────────────────
models = {
    "Linear\nRegression":   LinearRegression(),
    "Decision\nTree":       DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random\nForest":       RandomForestRegressor(n_estimators=100, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    "Gradient\nBoosting":   GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                       learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "R²":   r2_score(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "model": model,
    }

best_name  = max(results, key=lambda k: results[k]["R²"])
best_model = results[best_name]["model"]

# Feature importance from best model
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_)

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# ── 3. PLOT 7 ─────────────────────────────────────────────────────────────────
PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
BG        = "#F8F9FA"
PANEL_BG  = "#FFFFFF"

fig = plt.figure(figsize=(20, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Plot 7  |  Task 2 — Model Comparison · Feature Importance · Rating Distribution",
    fontsize=14, fontweight="bold", y=1.02
)

gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                       width_ratios=[1.2, 1.2, 1.8, 1.2])

names = list(results.keys())
r2s   = [results[n]["R²"]   for n in names]
rmses = [results[n]["RMSE"] for n in names]

# ── Panel A: R² comparison ────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.set_facecolor(PANEL_BG)
bars = ax0.bar(names, r2s, color=PALETTE, edgecolor="white", linewidth=0.8,
               width=0.55)
ax0.set_ylim(0, max(r2s) * 1.18)
ax0.set_ylabel("R² Score", fontsize=10)
ax0.set_title("Model Comparison — R²", fontsize=11, fontweight="bold", pad=8)
ax0.tick_params(axis="x", labelsize=8)
ax0.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, r2s):
    ax0.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold")

# ── Panel B: RMSE comparison ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.set_facecolor(PANEL_BG)
bars2 = ax1.bar(names, rmses, color=PALETTE, edgecolor="white", linewidth=0.8,
                width=0.55)
ax1.set_ylim(0, max(rmses) * 1.18)
ax1.set_ylabel("RMSE", fontsize=10)
ax1.set_title("Model Comparison — RMSE", fontsize=11, fontweight="bold", pad=8)
ax1.tick_params(axis="x", labelsize=8)
ax1.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars2, rmses):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.003,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold")

# ── Panel C: Feature importance (top 12) ─────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.set_facecolor(PANEL_BG)
top_feats = feat_imp.head(12)

# clean up label names for display
clean_labels = (
    top_feats.index
    .str.replace("_norm", " (norm)", regex=False)
    .str.replace("_enc",  " (enc)",  regex=False)
    .str.replace("_freq", " (freq)", regex=False)
)

bar_colors = ["#4C72B0" if i < 3 else "#8aabb8" for i in range(len(top_feats))]
hbars = ax2.barh(clean_labels[::-1], top_feats.values[::-1],
                 color=bar_colors[::-1], edgecolor="white", linewidth=0.6)
ax2.set_xlabel("Importance Score", fontsize=10)
ax2.set_title(
    f"Top 12 Feature Importances\n({best_name.replace(chr(10),' ')})",
    fontsize=11, fontweight="bold", pad=8
)
ax2.spines[["top","right"]].set_visible(False)
for bar, val in zip(hbars, top_feats.values[::-1]):
    ax2.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=7.5)

# ── Panel D: Rating distribution ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.set_facecolor(PANEL_BG)
bins = np.arange(1.0, 5.25, 0.25)
n, bins_out, patches = ax3.hist(y, bins=bins, edgecolor="white",
                                 linewidth=0.6, color="#4C72B0")

# colour-grade bars by rating value
cmap = plt.cm.RdYlGn
norm = plt.Normalize(bins_out[:-1].min(), bins_out[:-1].max())
for patch, left_edge in zip(patches, bins_out[:-1]):
    patch.set_facecolor(cmap(norm(left_edge)))

mean_r = y.mean()
ax3.axvline(mean_r, color="red", linestyle="--", linewidth=1.5,
            label=f"Mean = {mean_r:.2f}")
ax3.axvline(y.median(), color="navy", linestyle=":", linewidth=1.5,
            label=f"Median = {y.median():.2f}")
ax3.set_xlabel("Aggregate Rating", fontsize=10)
ax3.set_ylabel("Count", fontsize=10)
ax3.set_title("Rating Distribution", fontsize=11, fontweight="bold", pad=8)
ax3.legend(fontsize=8)
ax3.spines[["top","right"]].set_visible(False)

plt.savefig("/mnt/user-data/outputs/task2_plot7.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── 4. PRINT SUMMARY ─────────────────────────────────────────────────────────
print("=" * 55)
print("  PLOT 7 — Task 2 Summary")
print("=" * 55)
summary = pd.DataFrame(
    {n: {"R²": results[n]["R²"], "RMSE": results[n]["RMSE"]} for n in results}
).T.round(4)
print(f"\n{summary.to_string()}")
print(f"\n🏆 Best model : {best_name.replace(chr(10), ' ')}  "
      f"(R² = {results[best_name]['R²']:.4f})")
print(f"\n🔑 Top 5 Features:")
for feat, imp in feat_imp.head(5).items():
    label = feat.replace("_norm","").replace("_enc","").replace("_freq","")
    print(f"   {label:35s} {imp:.4f}")
print("\n✅ Saved → task2_plot7.png")
