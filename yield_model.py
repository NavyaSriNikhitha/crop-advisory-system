# ============================================================
# yield_model.py
# What this file does:
#   1. Loads the Indian crop yield dataset (49999 rows)
#   2. Engineers the target variable: yield = Production / Area
#   3. EDA — explores patterns
#   4. Trains regression models (predicts a NUMBER not a category)
#   5. Saves best model for Tab 5 in app.py
#
# KEY DIFFERENCE FROM PHASE 1 & 2:
#   Phase 1 & 2 = Classification (predict one of N categories)
#   Phase 3     = Regression (predict a continuous number)
#   Metrics change: no accuracy% — we use R², RMSE, MAE instead
# ============================================================

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ── STEP 1: Load dataset ──────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading yield dataset")
print("=" * 60)

df = pd.read_csv("data/yield.csv")
df.columns = df.columns.str.strip()

print(f"Shape    : {df.shape}")
print(f"Columns  : {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))


# ── STEP 2: Engineer target variable ─────────────────────────
# The dataset has Production (total tonnes) and Area (hectares).
# We want yield = how much per hectare — that's what's useful.
# A farmer with 5 hectares wants to know yield per hectare,
# not total production (which depends on how much land they have).
#
# yield_per_hectare = Production / Area

print("\n" + "=" * 60)
print("STEP 2: Feature Engineering — creating yield target")
print("=" * 60)

# Remove rows where Area or Production is 0 or missing
# Division by zero would create infinite values
df = df.dropna(subset=["Production", "Area"])
df = df[df["Area"] > 0]
df = df[df["Production"] > 0]

# Create yield column
df["yield_per_hectare"] = df["Production"] / df["Area"]

# Remove extreme outliers — values beyond 99th percentile
# These are likely data entry errors in the government dataset
p99 = df["yield_per_hectare"].quantile(0.99)
df  = df[df["yield_per_hectare"] <= p99]

print(f"Rows after cleaning  : {df.shape[0]}")
print(f"Yield stats:")
print(df["yield_per_hectare"].describe().round(2))
print(f"\nTop 5 crops by avg yield:")
print(df.groupby("Crop")["yield_per_hectare"].mean()
      .sort_values(ascending=False).head(5).round(2))


# ── STEP 3: EDA ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: EDA")
print("=" * 60)

print(f"\nUnique States   : {df['State_Name'].nunique()}")
print(f"Unique Crops    : {df['Crop'].nunique()}")
print(f"Unique Seasons  : {df['Season'].nunique()}")
print(f"Year range      : {df['Crop_Year'].min()} - {df['Crop_Year'].max()}")
print(f"\nSeasons: {df['Season'].unique().tolist()}")
print(f"\nMissing values:")
print(df.isnull().sum())

os.makedirs("eda_charts", exist_ok=True)

# Chart 1: Top 15 crops by average yield
plt.figure(figsize=(12, 5))
top_crops = (df.groupby("Crop")["yield_per_hectare"]
             .mean().sort_values(ascending=False).head(15))
sns.barplot(x=top_crops.index, y=top_crops.values, palette="viridis")
plt.title("Top 15 Crops by Average Yield (tonnes/hectare)")
plt.xlabel("Crop")
plt.ylabel("Avg Yield (t/ha)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_charts/yield_by_crop.png")
plt.close()
print("\nSaved: eda_charts/yield_by_crop.png")

# Chart 2: Yield by season
plt.figure(figsize=(9, 4))
season_yield = df.groupby("Season")["yield_per_hectare"].mean().sort_values(ascending=False)
sns.barplot(x=season_yield.index, y=season_yield.values, palette="Set2")
plt.title("Average Yield by Season")
plt.xlabel("Season")
plt.ylabel("Avg Yield (t/ha)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("eda_charts/yield_by_season.png")
plt.close()
print("Saved: eda_charts/yield_by_season.png")

# Chart 3: Yield distribution
plt.figure(figsize=(9, 4))
plt.hist(df["yield_per_hectare"], bins=50, color="steelblue", edgecolor="white")
plt.title("Distribution of Yield per Hectare")
plt.xlabel("Yield (tonnes/hectare)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_charts/yield_distribution.png")
plt.close()
print("Saved: eda_charts/yield_distribution.png")


# ── STEP 4: Preprocessing ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Preprocessing")
print("=" * 60)

# 4a. Select features
# We use: Crop, Season, State, Temperature, Humidity,
#         Soil_Moisture, Area, Crop_Year
# We drop District — too many unique values, adds noise
FEATURE_COLS = ["Crop", "Season", "State_Name",
                "Temperature", "Humidity", "Soil_Moisture",
                "Area", "Crop_Year"]
TARGET       = "yield_per_hectare"

# Drop rows with missing feature values
df = df.dropna(subset=FEATURE_COLS)

X = df[FEATURE_COLS].copy()
y = df[TARGET]

print(f"Features : {FEATURE_COLS}")
print(f"X shape  : {X.shape}")
print(f"y shape  : {y.shape}")

# 4b. Encode categorical columns
# Crop, Season, State_Name are text — must convert to numbers
le_crop_yield   = LabelEncoder()
le_season       = LabelEncoder()
le_state        = LabelEncoder()

X["Crop"]       = le_crop_yield.fit_transform(X["Crop"].astype(str))
X["Season"]     = le_season.fit_transform(X["Season"].astype(str))
X["State_Name"] = le_state.fit_transform(X["State_Name"].astype(str))

print(f"\nUnique crops encoded  : {len(le_crop_yield.classes_)}")
print(f"Unique seasons encoded: {len(le_season.classes_)}")
print(f"Unique states encoded : {len(le_state.classes_)}")

# 4c. Scale features
scaler_yield = StandardScaler()
X_scaled     = scaler_yield.fit_transform(X)

# 4d. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)
print(f"\nTraining set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")


# ── STEP 5: Define evaluation function ───────────────────────
# Regression uses different metrics than classification:
#
# R² (R-squared): How much variance in yield the model explains.
#   1.0 = perfect, 0.0 = no better than predicting the mean,
#   negative = worse than mean
#
# RMSE (Root Mean Square Error): Average prediction error in
#   the same unit as yield (tonnes/hectare).
#   Lower = better.
#
# MAE (Mean Absolute Error): Average of absolute errors.
#   More interpretable than RMSE — "on average off by X t/ha"

def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"\n  {name}")
    print(f"  R²   : {r2:.4f}   (1.0 = perfect)")
    print(f"  RMSE : {rmse:.4f}  (avg error in t/ha)")
    print(f"  MAE  : {mae:.4f}  (avg absolute error)")
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# ── STEP 6: Train 4 models ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Training 4 regression models")
print("=" * 60)

results_yield = {}


# ── MODEL 1: Ridge Regression (baseline) ─────────────────────
print("\n[1/4] Ridge Regression (baseline)...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
results_yield["Ridge"] = evaluate("Ridge Regression", y_test, ridge_preds)


# ── MODEL 2: Random Forest Regressor ─────────────────────────
# Same idea as Phase 1 but for regression.
# n_estimators=100 trees, each predicting a yield number.
# Final prediction = average of all 100 tree predictions.
print("\n[2/4] Random Forest Regressor...")
rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)
rf_preds = rf_reg.predict(X_test)
results_yield["Random Forest"] = evaluate("Random Forest", y_test, rf_preds)

# Cross validation for regression uses R² score
rf_cv = cross_val_score(rf_reg, X_scaled, y, cv=3, scoring="r2", n_jobs=-1)
print(f"  CV R² : {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")


# ── MODEL 3: XGBoost Regressor ───────────────────────────────
# Gradient boosting for regression.
# Builds trees sequentially, each correcting previous errors.
print("\n[3/4] XGBoost Regressor...")
xgb_reg = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=0
)
xgb_reg.fit(X_train, y_train)
xgb_preds = xgb_reg.predict(X_test)
results_yield["XGBoost"] = evaluate("XGBoost", y_test, xgb_preds)


# ── MODEL 4: Gradient Boosting Regressor ─────────────────────
print("\n[4/4] Gradient Boosting Regressor...")
gb_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gb_reg.fit(X_train, y_train)
gb_preds = gb_reg.predict(X_test)
results_yield["Gradient Boosting"] = evaluate("Gradient Boosting", y_test, gb_preds)


# ── STEP 7: Compare models ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Model Comparison")
print("=" * 60)
print(f"\n{'Model':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 52)
for name, metrics in sorted(results_yield.items(),
                             key=lambda x: x[1]["R2"], reverse=True):
    marker = " <- BEST" if metrics["R2"] == max(
        m["R2"] for m in results_yield.values()) else ""
    print(f"{name:<25} {metrics['R2']:>8} {metrics['RMSE']:>8} "
          f"{metrics['MAE']:>8}{marker}")

# Save comparison chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_names = ["R2", "RMSE", "MAE"]
titles        = ["R² Score (higher = better)",
                 "RMSE (lower = better)",
                 "MAE (lower = better)"]
colors        = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

for i, (metric, title) in enumerate(zip(metrics_names, titles)):
    vals  = [results_yield[m][metric] for m in results_yield]
    names = list(results_yield.keys())
    axes[i].bar(names, vals, color=colors)
    axes[i].set_title(title, fontsize=11)
    axes[i].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for j, v in enumerate(vals):
        axes[i].text(j, v + 0.001, f"{v:.3f}", ha="center",
                     va="bottom", fontsize=8)
plt.suptitle("Yield Prediction Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("eda_charts/yield_model_comparison.png")
plt.close()
print("\nSaved: eda_charts/yield_model_comparison.png")


# ── STEP 8: Feature Importance ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Feature Importance")
print("=" * 60)

importances   = rf_reg.feature_importances_
importance_df = pd.DataFrame({
    "Feature":    FEATURE_COLS,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nFeature importances:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(9, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance — Yield Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("eda_charts/yield_feature_importance.png")
plt.close()
print("\nSaved: eda_charts/yield_feature_importance.png")


# ── STEP 9: Save model files ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Saving model files")
print("=" * 60)

os.makedirs("models", exist_ok=True)

# Save best model by R² score
best_name = max(results_yield, key=lambda x: results_yield[x]["R2"])
print(f"Best model: {best_name} (R²={results_yield[best_name]['R2']})")

# We save Random Forest regardless (for feature importance in app)
joblib.dump(rf_reg,          "models/yield_model.pkl")
joblib.dump(scaler_yield,    "models/yield_scaler.pkl")
joblib.dump(le_crop_yield,   "models/yield_crop_encoder.pkl")
joblib.dump(le_season,       "models/yield_season_encoder.pkl")
joblib.dump(le_state,        "models/yield_state_encoder.pkl")
joblib.dump(results_yield,   "models/yield_model_results.pkl")

# Save crop list, season list and state list for app dropdowns
joblib.dump(list(le_crop_yield.classes_),  "models/yield_crops.pkl")
joblib.dump(list(le_season.classes_),      "models/yield_seasons.pkl")
joblib.dump(list(le_state.classes_),       "models/yield_states.pkl")

print("Saved: models/yield_model.pkl")
print("Saved: models/yield_scaler.pkl")
print("Saved: models/yield_crop_encoder.pkl")
print("Saved: models/yield_season_encoder.pkl")
print("Saved: models/yield_state_encoder.pkl")
print("Saved: models/yield_model_results.pkl")
print("Saved: models/yield_crops.pkl")
print("Saved: models/yield_seasons.pkl")
print("Saved: models/yield_states.pkl")

print("\n" + "=" * 60)
print("PHASE 3 TRAINING COMPLETE!")
print(f"Best R²        : {results_yield[best_name]['R2']} ({best_name})")
print(f"Saved model    : Random Forest Regressor")
print(f"Crops covered  : {len(le_crop_yield.classes_)}")
print(f"States covered : {len(le_state.classes_)}")
print(f"Seasons        : {list(le_season.classes_)}")
print("Next step      : update app.py with Tab 5")
print("=" * 60)
