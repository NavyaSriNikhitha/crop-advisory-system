# ============================================================
# fertilizer_model.py
# What this file does:
#   1. Loads the fertilizer dataset
#   2. EDA — explores the data
#   3. Encodes categorical columns (Soil type, Crop type)
#   4. Trains 4 models and compares accuracy
#   5. Saves best model for app.py Tab 4
# ============================================================


# ── STEP 1: Imports ───────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ── STEP 2: Load dataset ──────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading fertilizer dataset")
print("=" * 60)

df = pd.read_csv("data/fertilizer.csv")

# Strip whitespace from column names and string values
# Real-world datasets often have hidden spaces — this prevents errors
df.columns  = df.columns.str.strip()
df          = df.map(lambda x: x.strip() if isinstance(x, str) else x)

print(f"Shape    : {df.shape}")
print(f"Columns  : {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())


# ── STEP 3: EDA ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe().round(2))

print("\nFertilizer types and counts:")
print(df["Fertilizer"].value_counts())

print("\nSoil types:")
print(df["Soil"].value_counts())

print("\nCrop types:")
print(df["Crop"].value_counts())

os.makedirs("eda_charts", exist_ok=True)

# Chart 1: Fertilizer distribution
plt.figure(figsize=(12, 5))
df["Fertilizer"].value_counts().plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Fertilizer Distribution in Dataset")
plt.xlabel("Fertilizer")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_charts/fertilizer_distribution.png")
plt.close()
print("\nSaved: eda_charts/fertilizer_distribution.png")

# Chart 2: Average N, P, K per fertilizer
# Shows which fertilizer is recommended when nutrients are low/high
npk_by_fert = df.groupby("Fertilizer")[["N", "P", "K"]].mean().sort_values("N")
fig, ax = plt.subplots(figsize=(12, 6))
npk_by_fert.plot(kind="barh", ax=ax, color=["#2ECC71", "#3498DB", "#E67E22"])
ax.set_title("Average N, P, K by Fertilizer Type")
ax.set_xlabel("Average Value")
plt.tight_layout()
plt.savefig("eda_charts/npk_by_fertilizer.png")
plt.close()
print("Saved: eda_charts/npk_by_fertilizer.png")

# Chart 3: Fertilizer usage per crop type
plt.figure(figsize=(14, 6))
crop_fert = df.groupby(["Crop", "Fertilizer"]).size().unstack(fill_value=0)
crop_fert.plot(kind="bar", figsize=(14, 6), colormap="Set2")
plt.title("Fertilizer Usage Per Crop Type")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("eda_charts/fertilizer_by_crop.png")
plt.close()
print("Saved: eda_charts/fertilizer_by_crop.png")


# ── STEP 4: Preprocessing ─────────────────────────────────────
# This dataset has TWO categorical columns: Soil and Crop
# These must be encoded to numbers before ML models can use them.
# We use LabelEncoder for each one separately.
# We also save each encoder so app.py can decode predictions.

print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

# 4a. Encode categorical input columns
# Soil type: "Clayey"→0, "laterite"→1, "silty clay"→2 etc.
# Crop type: "rice"→0, "Coconut"→1 etc.
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()           # encodes the TARGET (fertilizer name)

df["Soil_encoded"] = le_soil.fit_transform(df["Soil"])
df["Crop_encoded"] = le_crop.fit_transform(df["Crop"])

print(f"\nSoil types encoded: {dict(zip(le_soil.classes_, le_soil.transform(le_soil.classes_)))}")
print(f"\nCrop types encoded: {dict(zip(le_crop.classes_, le_crop.transform(le_crop.classes_)))}")

# 4b. Define features and target
# We use encoded versions of Soil and Crop
FEATURE_COLS = ["Temperature", "Humidity", "Rainfall", "pH", "N", "P", "K",
                "Soil_encoded", "Crop_encoded"]

X = df[FEATURE_COLS]
y = df["Fertilizer"]

print(f"\nFeatures shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"\nFertilizer types: {y.unique().tolist()}")

# 4c. Encode target
y_encoded = le_fert.fit_transform(y)
print(f"\nFertilizer → number mapping:")
for fert, num in zip(le_fert.classes_, range(len(le_fert.classes_))):
    print(f"  {fert} → {num}")

# 4d. Scale features
scaler_fert = StandardScaler()
X_scaled    = scaler_fert.fit_transform(X)

# 4e. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"\nTraining set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")


# ── STEP 5: Train 4 models ────────────────────────────────────
# Why 4 models instead of 5?
# Dataset has 200 rows — SVM and XGBoost can overfit small datasets.
# We use Random Forest, CatBoost, Gradient Boosting, and a baseline.
# This gives a fair comparison without overfitting risk.

print("\n" + "=" * 60)
print("STEP 4: Training 4 models")
print("=" * 60)

results_fert = {}


# ── MODEL 1: Random Forest ────────────────────────────────────
print("\n[1/4] Training Random Forest...")
rf_fert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fert.fit(X_train, y_train)
rf_preds    = rf_fert.predict(X_test)
rf_acc      = accuracy_score(y_test, rf_preds)
results_fert["Random Forest"] = round(rf_acc * 100, 2)

rf_cv = cross_val_score(rf_fert, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy : {rf_acc * 100:.2f}%")
print(f"  CV Score : {rf_cv.mean():.3f} (+/- {rf_cv.std():.3f})")
print("\n  Classification Report:")
print(classification_report(y_test, rf_preds, target_names=le_fert.classes_))


# ── MODEL 2: CatBoost ─────────────────────────────────────────
# Especially relevant here because Soil and Crop are categorical.
# Even though we encoded them, CatBoost handles the patterns better.
print("\n[2/4] Training CatBoost...")
cb_fert = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,        # lower learning rate = more careful = better on small data
    depth=6,
    random_seed=42,
    verbose=0
)
cb_fert.fit(X_train, y_train)
cb_preds    = cb_fert.predict(X_test)
cb_acc      = accuracy_score(y_test, cb_preds)
results_fert["CatBoost"] = round(cb_acc * 100, 2)

cb_cv = cross_val_score(cb_fert, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy : {cb_acc * 100:.2f}%")
print(f"  CV Score : {cb_cv.mean():.3f} (+/- {cb_cv.std():.3f})")


# ── MODEL 3: Gradient Boosting ────────────────────────────────
# scikit-learn's built-in gradient boosting.
# Slower than XGBoost/CatBoost but very reliable on small datasets.
# Good to include for comparison — shows CatBoost vs sklearn boosting.
print("\n[3/4] Training Gradient Boosting...")
gb_fert = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gb_fert.fit(X_train, y_train)
gb_preds    = gb_fert.predict(X_test)
gb_acc      = accuracy_score(y_test, gb_preds)
results_fert["Gradient Boosting"] = round(gb_acc * 100, 2)

gb_cv = cross_val_score(gb_fert, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy : {gb_acc * 100:.2f}%")
print(f"  CV Score : {gb_cv.mean():.3f} (+/- {gb_cv.std():.3f})")


# ── MODEL 4: SVM ──────────────────────────────────────────────
print("\n[4/4] Training SVM...")
svm_fert = SVC(kernel="rbf", C=10, gamma="scale", random_state=42, probability=True)
svm_fert.fit(X_train, y_train)
svm_preds   = svm_fert.predict(X_test)
svm_acc     = accuracy_score(y_test, svm_preds)
results_fert["SVM (RBF)"] = round(svm_acc * 100, 2)

svm_cv = cross_val_score(svm_fert, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy : {svm_acc * 100:.2f}%")
print(f"  CV Score : {svm_cv.mean():.3f} (+/- {svm_cv.std():.3f})")


# ── STEP 6: Compare models ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Model Comparison")
print("=" * 60)
print(f"\n{'Model':<25} {'Accuracy':>10}")
print("-" * 36)
for name, acc in sorted(results_fert.items(), key=lambda x: x[1], reverse=True):
    marker = " <- BEST" if acc == max(results_fert.values()) else ""
    print(f"{name:<25} {acc:>9.2f}%{marker}")

# Save comparison chart
plt.figure(figsize=(9, 5))
names   = list(results_fert.keys())
scores  = list(results_fert.values())
colors  = ["#4CAF50" if s == max(scores) else "#90CAF9" for s in scores]
bars    = plt.bar(names, scores, color=colors, edgecolor="white", linewidth=1.2)
plt.title("Fertilizer Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy (%)")
plt.ylim([min(scores) - 5, 103])
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{score}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_charts/fertilizer_model_comparison.png")
plt.close()
print("\nSaved: eda_charts/fertilizer_model_comparison.png")


# ── STEP 7: Hyperparameter Tuning on Random Forest ───────────
print("\n" + "=" * 60)
print("STEP 6: Hyperparameter Tuning — Random Forest")
print("=" * 60)

param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters : {grid_search.best_params_}")
print(f"Best CV score   : {grid_search.best_score_ * 100:.2f}%")

best_rf_fert  = grid_search.best_estimator_
final_preds   = best_rf_fert.predict(X_test)
final_acc     = accuracy_score(y_test, final_preds)
print(f"Tuned RF Test Accuracy: {final_acc * 100:.2f}%")


# ── STEP 8: Feature Importance ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Feature Importance")
print("=" * 60)

importances   = best_rf_fert.feature_importances_
importance_df = pd.DataFrame({
    "Feature":    FEATURE_COLS,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nFeature importances:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(9, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance — Fertilizer Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("eda_charts/fertilizer_feature_importance.png")
plt.close()
print("\nSaved: eda_charts/fertilizer_feature_importance.png")


# ── STEP 9: Save all model files ──────────────────────────────
# We save 5 objects:
#   1. best_rf_fert  — the trained model
#   2. scaler_fert   — StandardScaler (must use same in app.py)
#   3. le_fert       — LabelEncoder for fertilizer names (target)
#   4. le_soil       — LabelEncoder for soil type (input)
#   5. le_crop       — LabelEncoder for crop type (input)
#   6. results_fert  — all accuracy scores for comparison tab
#
# We need 3 encoders this time (vs 1 in Phase 1) because
# this dataset has categorical INPUTS (soil, crop) in addition
# to a categorical TARGET (fertilizer name).

print("\n" + "=" * 60)
print("STEP 8: Saving model files")
print("=" * 60)

os.makedirs("models", exist_ok=True)

joblib.dump(best_rf_fert,  "models/fertilizer_model.pkl")
joblib.dump(scaler_fert,   "models/fertilizer_scaler.pkl")
joblib.dump(le_fert,       "models/fertilizer_label_encoder.pkl")
joblib.dump(le_soil,       "models/fertilizer_soil_encoder.pkl")
joblib.dump(le_crop,       "models/fertilizer_crop_encoder.pkl")
joblib.dump(results_fert,  "models/fertilizer_model_results.pkl")

print("Saved: models/fertilizer_model.pkl")
print("Saved: models/fertilizer_scaler.pkl")
print("Saved: models/fertilizer_label_encoder.pkl")
print("Saved: models/fertilizer_soil_encoder.pkl")
print("Saved: models/fertilizer_crop_encoder.pkl")
print("Saved: models/fertilizer_model_results.pkl")

print("\n" + "=" * 60)
print("PHASE 2 TRAINING COMPLETE!")
print(f"Best model     : {max(results_fert, key=results_fert.get)} ({max(results_fert.values())}%)")
print(f"Saved model    : Random Forest (tuned) — for predictions + explanations")
print(f"Soil types     : {list(le_soil.classes_)}")
print(f"Crop types     : {list(le_crop.classes_)}")
print(f"Fertilizers    : {list(le_fert.classes_)}")
print("Next step      : update app.py with Tab 4")
print("=" * 60)
