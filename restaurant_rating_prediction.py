"""
Task 1: Predict Restaurant Ratings
Objective: Build a ML model to predict the aggregate rating of a restaurant.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. LOAD DATA

print("=" * 60)
print("TASK 1: PREDICT RESTAURANT RATINGS")
print("=" * 60)

df = pd.read_csv("/mnt/user-data/uploads/Dataset_.csv", encoding="utf-8-sig")
print(f"\n Dataset shape: {df.shape}")
print(f"\n Columns:\n{list(df.columns)}")
print(f"\n Target — 'Aggregate rating' stats:\n{df['Aggregate rating'].describe()}")


# 2. PREPROCESSING
print("\n" + "=" * 60)
print("STEP 1: PREPROCESSING")
print("=" * 60)

# Drop leaky / irrelevant columns
drop_cols = [
    "Restaurant ID", "Restaurant Name", "Address",
    "Locality", "Locality Verbose", "Currency",
    "Rating color", "Rating text"         # direct derivatives of target
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Handle missing values
print(f"\n Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df["Cuisines"].fillna("Unknown", inplace=True)
df.dropna(inplace=True)
print(f"\n Missing values after cleaning: {df.isnull().sum().sum()}")

# Filter out rows with 0 rating (unrated restaurants)
df = df[df["Aggregate rating"] > 0].copy()
print(f"\n Shape after removing unrated rows: {df.shape}")

# Encode binary yes/no columns
binary_cols = ["Has Table booking", "Has Online delivery",
               "Is delivering now", "Switch to order menu"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

# Encode high-cardinality categoricals with frequency encoding
for col in ["City", "Cuisines"]:
    freq = df[col].value_counts()
    df[col + "_freq"] = df[col].map(freq)
    df.drop(columns=[col], inplace=True)

# Country Code is already numeric — keep as-is
print(f"\n Feature engineering done. Final shape: {df.shape}")


# 3. TRAIN / TEST SPLIT

X = df.drop(columns=["Aggregate rating"])
y = df["Aggregate rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


# 4. TRAIN MODELS

print("\n" + "=" * 60)
print("STEP 2: TRAINING MODELS")
print("=" * 60)

models = {
    "Linear Regression":        LinearRegression(),
    "Decision Tree":            DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2, "preds": preds, "model": model}
    print(f"\n  {name}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["R²"])
best      = results[best_name]
print(f"\n🏆 Best model: {best_name}  (R² = {best['R²']:.4f})")


# 5. FEATURE IMPORTANCE

feature_names = X.columns.tolist()

if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
else:                                           # Linear Regression
    importances = np.abs(best["model"].coef_)

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(f"\n Top Features ({best_name}):")
print(feat_imp.to_string())


# 6. VISUALISATIONS

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Task 1 — Restaurant Rating Prediction", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# — 6a. Model comparison (R²) —
ax0 = fig.add_subplot(gs[0, 0])
names = list(results.keys())
r2s   = [results[n]["R²"]   for n in names]
rmses = [results[n]["RMSE"] for n in names]
bars = ax0.bar(range(len(names)), r2s, color=colors, edgecolor="white", linewidth=0.8)
ax0.set_xticks(range(len(names)))
ax0.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax0.set_ylabel("R² Score")
ax0.set_title("Model Comparison — R²")
ax0.set_ylim(0, 1.05)
for bar, val in zip(bars, r2s):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# — 6b. Model comparison (RMSE) —
ax1 = fig.add_subplot(gs[0, 1])
bars2 = ax1.bar(range(len(names)), rmses, color=colors, edgecolor="white", linewidth=0.8)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax1.set_ylabel("RMSE")
ax1.set_title("Model Comparison — RMSE")
for bar, val in zip(bars2, rmses):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# — 6c. Actual vs Predicted (best model) —
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(y_test, best["preds"], alpha=0.3, s=10, color="#4C72B0", label="Predictions")
lims = [y_test.min(), y_test.max()]
ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax2.set_xlabel("Actual Rating")
ax2.set_ylabel("Predicted Rating")
ax2.set_title(f"Actual vs Predicted\n({best_name})")
ax2.legend(fontsize=8)
ax2.text(0.05, 0.92, f"R²={best['R²']:.3f}", transform=ax2.transAxes,
         fontsize=9, color="darkgreen", fontweight="bold")

# — 6d. Feature Importance —
ax3 = fig.add_subplot(gs[1, :2])
top_feats = feat_imp.head(10)
ax3.barh(top_feats.index[::-1], top_feats.values[::-1], color="#4C72B0", edgecolor="white")
ax3.set_xlabel("Importance Score")
ax3.set_title(f"Top 10 Feature Importances ({best_name})")
for i, (val, feat) in enumerate(zip(top_feats.values[::-1], top_feats.index[::-1])):
    ax3.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)

# — 6e. Rating distribution —
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(y, bins=20, color="#55A868", edgecolor="white", linewidth=0.5)
ax4.set_xlabel("Aggregate Rating")
ax4.set_ylabel("Count")
ax4.set_title("Rating Distribution")
ax4.axvline(y.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={y.mean():.2f}")
ax4.legend(fontsize=8)

plt.savefig("/mnt/user-data/outputs/restaurant_rating_prediction.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\n Chart saved → restaurant_rating_prediction.png")


# 7. SUMMARY

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(
    {n: {"RMSE": results[n]["RMSE"], "MAE": results[n]["MAE"], "R²": results[n]["R²"]}
     for n in results}
).T.round(4)
print(results_df.to_string())
print(f"\n Winner: {best_name}")
print(f"   R²   = {best['R²']:.4f}  (explains {best['R²']*100:.1f}% of variance)")
print(f"   RMSE = {best['RMSE']:.4f}")
print(f"   MAE  = {best['MAE']:.4f}")
print("\n Most influential features:")
for feat, imp in feat_imp.head(5).items():
    print(f"   {feat:30s} {imp:.4f}")
