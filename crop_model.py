# ============================================================
# crop_model.py
# What this file does:
#   1. Loads the crop dataset
#   2. Explores the data (EDA)
#   3. Trains 5 models: Ridge, Random Forest, CatBoost,
#      XGBoost, SVM
#   4. Compares their accuracy
#   5. Saves the best model so app.py can use it
# ============================================================


# ── STEP 1: Import libraries ─────────────────────────────────
# Think of imports like opening your toolbox before starting work.
# Each line brings in a specific tool.

import pandas as pd                               # for loading and working with the CSV
import numpy as np                                # for number operations
import joblib                                     # for saving the trained model to disk
import os                                         # for creating folders

from sklearn.model_selection import (
    train_test_split,                             # splits data into train and test sets
    cross_val_score,                              # tests model on multiple splits fairly
    GridSearchCV                                  # tries many settings to find the best one
)
from sklearn.preprocessing import LabelEncoder    # converts crop names to numbers
from sklearn.preprocessing import StandardScaler  # makes all numbers on same scale
from sklearn.linear_model import Ridge            # Model 1: Ridge Regression (baseline)
from sklearn.ensemble import RandomForestClassifier  # Model 2: Random Forest (main model)
from sklearn.svm import SVC                       # Model 4: Support Vector Machine
from sklearn.metrics import (
    accuracy_score,                               # what % of predictions were correct
    classification_report,                        # detailed accuracy per crop
    confusion_matrix                              # shows where model got confused
)
from catboost import CatBoostClassifier           # Model 3: CatBoost
from xgboost import XGBClassifier                 # Model 5: XGBoost

import matplotlib.pyplot as plt                   # for drawing charts
import seaborn as sns                             # for drawing prettier charts
import warnings
warnings.filterwarnings("ignore")                 # hides unimportant warning messages


# ── STEP 2: Load the dataset ──────────────────────────────────
# pd.read_csv() opens the CSV file and loads it into a DataFrame.
# A DataFrame is like an Excel table in Python — rows and columns.

print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv("data/crop_recommendation.csv")

print(f"Shape of dataset : {df.shape}")          # (2200, 8) — rows and columns
print(f"Columns          : {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())


# ── STEP 3: Explore the data (EDA) ───────────────────────────
# EDA = Exploratory Data Analysis
# Before training, you must UNDERSTAND your data.
# This step answers: What does the data look like? Any problems?

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis (EDA)")
print("=" * 60)

# 3a. Check for missing values
# Missing values break ML models — we must check first
print("\nMissing values per column:")
print(df.isnull().sum())                          # should show 0 for all columns

# 3b. Basic statistics — mean, min, max for each column
print("\nBasic statistics:")
print(df.describe().round(2))

# 3c. How many samples per crop?
# Good ML needs balanced data — each crop should have similar rows
print("\nSamples per crop:")
print(df["label"].value_counts())

# 3d. Save EDA charts
os.makedirs("eda_charts", exist_ok=True)

# Chart 1: Distribution of each feature
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Feature Distributions", fontsize=16)
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
for i, feature in enumerate(features):
    row = i // 4
    col = i % 4
    axes[row, col].hist(df[feature], bins=30, color="steelblue", edgecolor="white")
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel("Value")
    axes[row, col].set_ylabel("Count")
plt.tight_layout()
plt.savefig("eda_charts/feature_distributions.png")
plt.close()
print("\nSaved: eda_charts/feature_distributions.png")

# Chart 2: Correlation heatmap
plt.figure(figsize=(10, 7))
numeric_df = df.drop("label", axis=1)
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Features")
plt.tight_layout()
plt.savefig("eda_charts/correlation_heatmap.png")
plt.close()
print("Saved: eda_charts/correlation_heatmap.png")

# Chart 3: Average rainfall per crop
plt.figure(figsize=(14, 6))
rainfall_by_crop = df.groupby("label")["rainfall"].mean().sort_values(ascending=False)
sns.barplot(x=rainfall_by_crop.index, y=rainfall_by_crop.values, palette="Blues_d")
plt.title("Average Rainfall Required Per Crop")
plt.xlabel("Crop")
plt.ylabel("Avg Rainfall (mm)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_charts/rainfall_per_crop.png")
plt.close()
print("Saved: eda_charts/rainfall_per_crop.png")


# ── STEP 4: Prepare data for ML ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Preparing data for ML")
print("=" * 60)

# 4a. Separate features (X) and target (y)
X = df.drop("label", axis=1)                     # inputs: N, P, K, temp, humidity, ph, rainfall
y = df["label"]                                   # output: crop name to predict

print(f"Features (X) shape : {X.shape}")
print(f"Target (y) shape   : {y.shape}")

# 4b. Encode crop names to numbers
# LabelEncoder converts: "rice"->0, "wheat"->1, "maize"->2 etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nCrop → number mapping (first 5):")
for crop, num in zip(le.classes_[:5], range(5)):
    print(f"  {crop} → {num}")

# 4c. Scale features
# Without scaling, rainfall (0-300) unfairly dominates pH (0-14)
# StandardScaler brings everything to mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nAfter scaling — first row: {X_scaled[0].round(3)}")

# 4d. Train / test split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded                            # keeps crop proportions equal in both splits
)
print(f"\nTraining set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")


# ── STEP 5: Train 5 models and compare ───────────────────────
# All 5 models train on the SAME data — fair comparison.

print("\n" + "=" * 60)
print("STEP 4: Training 5 models")
print("=" * 60)

results = {}                                      # stores accuracy of each model


# ── MODEL 1: Ridge Regression ─────────────────────────────────
# Linear model. Fast and simple.
# Purpose: BASELINE — the minimum score every other model must beat.
# Why Ridge over plain Linear Regression?
#   Ridge adds a penalty (alpha) that prevents overfitting on noisy features.
print("\n[1/5] Training Ridge Regression (baseline)...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Ridge predicts continuous numbers — round them to nearest class index
ridge_preds_raw = ridge.predict(X_test)
ridge_preds = np.clip(np.round(ridge_preds_raw).astype(int), 0, len(le.classes_) - 1)
ridge_acc = accuracy_score(y_test, ridge_preds)
results["Ridge Regression"] = round(ridge_acc * 100, 2)

ridge_cv = cross_val_score(ridge, X_scaled, y_encoded, cv=5, scoring="r2")
print(f"  Accuracy  : {ridge_acc * 100:.2f}%")
print(f"  CV R2     : {ridge_cv.mean():.3f} (+/- {ridge_cv.std():.3f})")


# ── MODEL 2: Random Forest ────────────────────────────────────
# Builds 100 decision trees, combines their votes.
# Handles non-linear patterns. Gives feature importance.
# This is your PRIMARY model.
print("\n[2/5] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
results["Random Forest"] = round(rf_acc * 100, 2)

rf_cv = cross_val_score(rf, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy  : {rf_acc * 100:.2f}%")
print(f"  CV Score  : {rf_cv.mean():.3f} (+/- {rf_cv.std():.3f})")

print("\n  Classification Report (Random Forest):")
print(classification_report(y_test, rf_preds, target_names=le.classes_))


# ── MODEL 3: CatBoost ─────────────────────────────────────────
# Gradient boosting — builds trees sequentially, each fixing previous errors.
# Designed specifically for categorical variables. No manual encoding needed.
# Why CatBoost over plain GradientBoosting?
#   CatBoost handles categories natively and trains faster with less tuning.
print("\n[3/5] Training CatBoost...")
cb = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,                            # how fast model learns — lower = more careful
    depth=6,
    random_seed=42,
    verbose=0                                     # silent — no training log spam
)
cb.fit(X_train, y_train)
cb_preds = cb.predict(X_test)
cb_acc = accuracy_score(y_test, cb_preds)
results["CatBoost"] = round(cb_acc * 100, 2)

cb_cv = cross_val_score(cb, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy  : {cb_acc * 100:.2f}%")
print(f"  CV Score  : {cb_cv.mean():.3f} (+/- {cb_cv.std():.3f})")


# ── MODEL 4: XGBoost ──────────────────────────────────────────
# Also gradient boosting — very similar to CatBoost but older and widely used.
# Why add XGBoost alongside CatBoost?
#   XGBoost uses L1/L2 regularisation which sometimes generalises better.
#   CatBoost handles categories better. Comparing them shows which boosting
#   approach wins on THIS specific dataset — that is a genuine ML insight.
print("\n[4/5] Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="mlogloss",                       # multi-class log loss metric
    verbosity=0                                   # silent mode
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
results["XGBoost"] = round(xgb_acc * 100, 2)

xgb_cv = cross_val_score(xgb, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy  : {xgb_acc * 100:.2f}%")
print(f"  CV Score  : {xgb_cv.mean():.3f} (+/- {xgb_cv.std():.3f})")


# ── MODEL 5: SVM (Support Vector Machine) ────────────────────
# Finds the best boundary (hyperplane) separating each crop class.
# Works by maximising the margin between classes.
# Why SVM?
#   SVMs excel when classes are well-separated in feature space.
#   Crop data with distinct N/P/K ranges per crop is ideal for SVM.
#   It also provides a completely different approach from tree-based
#   models — giving a genuinely diverse comparison.
# Why kernel='rbf'?
#   RBF (Radial Basis Function) handles non-linear boundaries.
#   Linear SVM would fail here — crop boundaries are not straight lines.
print("\n[5/5] Training SVM (RBF kernel)...")
svm = SVC(
    kernel="rbf",                                 # non-linear kernel
    C=10,                                         # penalty for misclassification
    gamma="scale",                                # auto-calculates gamma from feature variance
    random_state=42,
    probability=True                              # needed for confidence scores in app
)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
results["SVM (RBF)"] = round(svm_acc * 100, 2)

svm_cv = cross_val_score(svm, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  Accuracy  : {svm_acc * 100:.2f}%")
print(f"  CV Score  : {svm_cv.mean():.3f} (+/- {svm_cv.std():.3f})")


# ── STEP 6: Compare all 5 models ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Model Comparison — all 5 models")
print("=" * 60)
print(f"\n{'Model':<25} {'Accuracy':>10}")
print("-" * 36)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    marker = " <- BEST" if accuracy == max(results.values()) else ""
    print(f"{model_name:<25} {accuracy:>9.2f}%{marker}")

# Save comparison chart — green = best, blue = others
plt.figure(figsize=(10, 5))
model_names = list(results.keys())
accuracies  = list(results.values())
colors = ["#4CAF50" if a == max(accuracies) else "#90CAF9" for a in accuracies]
bars = plt.bar(model_names, accuracies, color=colors, edgecolor="white", linewidth=1.2)
plt.title("5-Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy (%)")
plt.ylim([min(accuracies) - 5, 103])
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{acc}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
    )
plt.tight_layout()
plt.savefig("eda_charts/model_comparison.png")
plt.close()
print("\nSaved: eda_charts/model_comparison.png")


# ── STEP 7: Hyperparameter Tuning ────────────────────────────
# We always tune Random Forest regardless of who "won" above.
# Why? Because RF gives us feature_importances_ which powers
# the explanation feature in the app. Even if SVM scores higher,
# we need RF for explainability.
# GridSearchCV tries every combination of parameters — picks best.

print("\n" + "=" * 60)
print("STEP 6: Hyperparameter Tuning — Random Forest")
print("=" * 60)

param_grid = {
    "n_estimators":      [100, 200],              # number of trees
    "max_depth":         [None, 10, 20],          # how deep each tree grows
    "min_samples_split": [2, 5]                   # minimum rows needed to split a node
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,                                         # 3-fold cross validation
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters : {grid_search.best_params_}")
print(f"Best CV score   : {grid_search.best_score_ * 100:.2f}%")

best_rf     = grid_search.best_estimator_
final_preds = best_rf.predict(X_test)
final_acc   = accuracy_score(y_test, final_preds)
print(f"Tuned RF Test Accuracy: {final_acc * 100:.2f}%")


# ── STEP 8: Feature Importance ────────────────────────────────
# Which input features matter most to the model?
# This powers the "Why this crop?" explanation in the Streamlit app.

print("\n" + "=" * 60)
print("STEP 7: Feature Importance")
print("=" * 60)

feature_names = X.columns.tolist()
importances   = best_rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nFeature importances (higher = more influence on prediction):")
print(importance_df.to_string(index=False))

plt.figure(figsize=(9, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance — what affects crop prediction most?")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("eda_charts/feature_importance.png")
plt.close()
print("\nSaved: eda_charts/feature_importance.png")


# ── STEP 9: Save model, scaler and encoder ────────────────────
# joblib saves Python objects to .pkl files on disk.
# app.py loads these files to make predictions without retraining.
# IMPORTANT: always use the SAME scaler in app.py that was used here.

print("\n" + "=" * 60)
print("STEP 8: Saving model files")
print("=" * 60)

os.makedirs("models", exist_ok=True)

joblib.dump(best_rf, "models/crop_model.pkl")     # tuned Random Forest
joblib.dump(scaler,  "models/scaler.pkl")          # MUST use same scaler in app
joblib.dump(le,      "models/label_encoder.pkl")   # converts number back to crop name
joblib.dump(results, "models/model_results.pkl")   # all 5 accuracy scores for comparison tab

print("Saved: models/crop_model.pkl")
print("Saved: models/scaler.pkl")
print("Saved: models/label_encoder.pkl")
print("Saved: models/model_results.pkl")

print("\n" + "=" * 60)
print("ALL DONE! 5 models trained and compared.")
print(f"Best overall   : {max(results, key=results.get)} ({max(results.values())}%)")
print(f"Saved model    : Random Forest (tuned) — used for predictions + explanations")
print("Next step      : streamlit run app.py")
print("=" * 60)
