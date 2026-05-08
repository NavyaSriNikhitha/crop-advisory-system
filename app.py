# ============================================================
# app.py — complete file with all 6 tabs
# Tab 1 — Predict & Explain     (crop recommendation + SHAP)
# Tab 2 — Insights Dashboard    (EDA charts)
# Tab 3 — Model Comparison      (crop model accuracy)
# Tab 4 — Fertiliser            (fertilizer recommendation)
# Tab 5 — Yield Prediction      (yield per hectare regression)
# Tab 6 — AI Farming Chatbot    (Gemini + RAG powered)
#
# Run with: streamlit run app.py
# Requires: uv add google-genai chromadb sentence-transformers
# Build KB:  python knowledge_base.py  (run once before app)
# ============================================================

import matplotlib
matplotlib.use("Agg")                # fixes threading issue on Windows Python 3.13

# ── Auto-download models on Streamlit Cloud ───────────────────
import os

# Fix double folder issue from gdown download
if os.path.exists("models/models/crop_model.pkl"):
    import shutil
    for f in os.listdir("models/models"):
        shutil.move(f"models/models/{f}", f"models/{f}")
    os.rmdir("models/models")

if not os.path.exists("models/crop_model.pkl"):
    try:
        from setup import download_models
        download_models()
        if os.path.exists("models/models/crop_model.pkl"):
            import shutil
            for f in os.listdir("models/models"):
                shutil.move(f"models/models/{f}", f"models/{f}")
            os.rmdir("models/models")
    except Exception as e:
        pass

# ── Auto-build knowledge base on Streamlit Cloud ─────────────
if not os.path.exists("vector_db"):
    try:
        from setup import build_knowledge_base
        build_knowledge_base()
    except Exception as e:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import json
import re
import warnings
warnings.filterwarnings("ignore")

# New official Gemini SDK — install with: uv add google-genai
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# RAG pipeline — ChromaDB knowledge base
try:
    from rag_pipeline import load_knowledge_base, search_knowledge, \
                             build_rag_prompt, build_general_prompt
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Advisory System",
    page_icon="🌾",
    layout="wide"
)


# ── Load all model files ──────────────────────────────────────
@st.cache_resource
def load_models():
    # ── Crop model (Phase 1) ──
    model   = joblib.load("models/crop_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    results = joblib.load("models/model_results.pkl")

    # ── Fertilizer model (Phase 2) ──
    fert_model    = joblib.load("models/fertilizer_model.pkl")
    fert_scaler   = joblib.load("models/fertilizer_scaler.pkl")
    fert_encoder  = joblib.load("models/fertilizer_label_encoder.pkl")
    fert_soil_enc = joblib.load("models/fertilizer_soil_encoder.pkl")
    fert_crop_enc = joblib.load("models/fertilizer_crop_encoder.pkl")
    fert_results  = joblib.load("models/fertilizer_model_results.pkl")

    # ── Yield model (Phase 3) ──
    yield_model      = joblib.load("models/yield_model.pkl")
    yield_scaler     = joblib.load("models/yield_scaler.pkl")
    yield_crop_enc   = joblib.load("models/yield_crop_encoder.pkl")
    yield_season_enc = joblib.load("models/yield_season_encoder.pkl")
    yield_state_enc  = joblib.load("models/yield_state_encoder.pkl")
    yield_results    = joblib.load("models/yield_model_results.pkl")
    yield_crops      = joblib.load("models/yield_crops.pkl")
    yield_seasons    = joblib.load("models/yield_seasons.pkl")
    yield_states     = joblib.load("models/yield_states.pkl")

    # Handle case where pkl saved a LabelEncoder instead of a list
    from sklearn.preprocessing import LabelEncoder as LE
    if isinstance(yield_crops, LE):
        yield_crops = list(yield_crops.classes_)
    if isinstance(yield_seasons, LE):
        yield_seasons = list(yield_seasons.classes_)
    if isinstance(yield_states, LE):
        yield_states = list(yield_states.classes_)

    return (
        model, scaler, encoder, results,
        fert_model, fert_scaler, fert_encoder, fert_soil_enc, fert_crop_enc, fert_results,
        yield_model, yield_scaler, yield_crop_enc, yield_season_enc,
        yield_state_enc, yield_results, yield_crops, yield_seasons, yield_states
    )

(model, scaler, le, model_results,
 fert_model, fert_scaler, fert_le, fert_soil_le, fert_crop_le, fert_results,
 yield_model, yield_scaler, yield_crop_le, yield_season_le,
 yield_state_le, yield_results, yield_crops, yield_seasons, yield_states) = load_models()


# ── Load datasets ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/crop_recommendation.csv")

df = load_data()

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Load RAG Knowledge Base ───────────────────────────────────
# No cache — must check fresh so it detects vector_db after build
def load_rag():
    if RAG_AVAILABLE:
        collection, loaded = load_knowledge_base()
        return collection, loaded
    return None, False

rag_collection, rag_loaded = load_rag()


# ── Header ────────────────────────────────────────────────────
st.title("🌾 AI-Powered Crop Advisory System")
st.markdown(
    "Enter your soil and weather values to get a **crop recommendation**, "
    "understand **why** that crop was suggested, and explore **data insights**."
)
st.divider()


# ── 6 Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌱 Predict & Explain",
    "📊 Insights Dashboard",
    "🏆 Model Comparison",
    "🧪 Fertiliser Recommendation",
    "📈 Yield Prediction",
    "🤖 AI Farming Chatbot"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT & EXPLAIN
# ════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Enter Soil & Weather Values")
    st.markdown("Move the sliders to match your field conditions.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        N = st.slider("Nitrogen (N)", min_value=0, max_value=140, value=50,
                      help="Nitrogen content in soil (kg/ha)")
    with col2:
        P = st.slider("Phosphorus (P)", min_value=5, max_value=145, value=50,
                      help="Phosphorus content in soil (kg/ha)")
    with col3:
        K = st.slider("Potassium (K)", min_value=5, max_value=205, value=50,
                      help="Potassium content in soil (kg/ha)")
    with col4:
        temperature = st.slider("Temperature (°C)", min_value=8.0, max_value=44.0,
                                value=25.0, step=0.5)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        humidity = st.slider("Humidity (%)", min_value=14.0, max_value=100.0,
                             value=70.0, step=0.5)
    with col6:
        ph = st.slider("Soil pH", min_value=3.5, max_value=10.0,
                       value=6.5, step=0.1)
    with col7:
        rainfall = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0,
                             value=100.0, step=1.0)
    with col8:
        st.write("")

    st.divider()
    predict_btn = st.button("🔍 Predict Crop", type="primary",
                            use_container_width=True, key="crop_btn")

    if predict_btn:

        input_data         = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled       = scaler.transform(input_data)
        prediction_encoded = model.predict(input_scaled)[0]
        predicted_crop     = le.inverse_transform([prediction_encoded])[0]
        probabilities      = model.predict_proba(input_scaled)[0]
        confidence         = probabilities[prediction_encoded] * 100

        st.success(f"### 🌾 Recommended Crop: **{predicted_crop.upper()}**")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Model Confidence", f"{confidence:.1f}%")
        with res_col2:
            top3_idx   = np.argsort(probabilities)[::-1][:3]
            top3_crops = le.inverse_transform(top3_idx)
            top3_probs = probabilities[top3_idx] * 100
            st.markdown("**Top 3 alternatives:**")
            for crop, prob in zip(top3_crops, top3_probs):
                st.markdown(f"- {crop.capitalize()}: {prob:.1f}%")

        st.divider()

        # SHAP explanation
        st.subheader("🔎 Why was this crop recommended?")
        st.markdown(
            "Red = pushed **towards** this crop. Blue = pushed **against** it."
        )

        with st.spinner("Calculating explanation..."):
            explainer     = shap.TreeExplainer(model)
            shap_values   = explainer.shap_values(input_scaled)
            shap_for_pred = shap_values[0, :, prediction_encoded]

            shap_df = pd.DataFrame({
                "Feature": FEATURES,
                "Value":   [N, P, K, temperature, humidity, ph, rainfall],
                "SHAP":    shap_for_pred
            }).sort_values("SHAP", key=abs, ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            colors  = ["#E74C3C" if v > 0 else "#3498DB" for v in shap_df["SHAP"]]
            bars    = ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors)
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title(f"Why {predicted_crop.capitalize()}? — Feature Contributions")
            for bar, val in zip(bars, shap_df["SHAP"]):
                ax.text(val + (0.001 if val >= 0 else -0.001),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center",
                        ha="left" if val >= 0 else "right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Plain English summary
        st.subheader("📝 Plain English Summary")
        top_pos = shap_df[shap_df["SHAP"] > 0].head(3)
        top_neg = shap_df[shap_df["SHAP"] < 0].head(2)

        if len(top_pos) > 0:
            st.markdown(
                f"✅ **{top_pos.iloc[0]['Feature']}** (value: {top_pos.iloc[0]['Value']}) "
                f"is the strongest reason {predicted_crop.capitalize()} is recommended."
            )
        for _, row in top_pos.iloc[1:].iterrows():
            st.markdown(
                f"✅ **{row['Feature']}** (value: {row['Value']}) also supports "
                f"{predicted_crop.capitalize()}."
            )
        for _, row in top_neg.iterrows():
            st.markdown(
                f"⚠️ **{row['Feature']}** (value: {row['Value']}) slightly works "
                f"against {predicted_crop.capitalize()} but other factors outweigh it."
            )

        st.divider()
        st.info(
            f"**About {predicted_crop.capitalize()}:** Based on your inputs — "
            f"N:{N}, P:{P}, K:{K}, Temp:{temperature}°C, Humidity:{humidity}%, "
            f"pH:{ph}, Rainfall:{rainfall}mm — {predicted_crop.capitalize()} is "
            f"the most suitable crop with {confidence:.1f}% model confidence."
        )


# ════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS DASHBOARD
# ════════════════════════════════════════════════════════════
with tab2:

    st.subheader("📊 Data Insights Dashboard")
    st.markdown("Explore patterns in the crop recommendation dataset.")

    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        st.markdown("### 🌧️ Average Rainfall Per Crop")
        st.markdown("Higher rainfall crops need more irrigation or wet climates.")

        rainfall_df = (
            df.groupby("label")["rainfall"]
            .mean()
            .sort_values(ascending=True)
        )

        fig1, ax1 = plt.subplots(figsize=(8, 7))
        ax1.barh(rainfall_df.index, rainfall_df.values,
                 color=sns.color_palette("Blues_d", len(rainfall_df)))
        ax1.set_xlabel("Average Rainfall (mm)")
        ax1.set_title("Average Rainfall Requirement Per Crop")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with ins_col2:
        st.markdown("### 🧪 Average N, P, K Per Crop")
        st.markdown("Nutrient needs differ — helps match fertilizer to crop.")

        npk_df = (
            df.groupby("label")[["N", "P", "K"]]
            .mean()
            .sort_values("N", ascending=True)
        )

        fig2, ax2 = plt.subplots(figsize=(8, 7))
        npk_df.plot(kind="barh", ax=ax2,
                    color=["#2ECC71", "#3498DB", "#E67E22"])
        ax2.set_xlabel("Average Value")
        ax2.set_title("N, P, K by Crop")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with ins_col1:
        st.markdown("### 🌡️ Avg Temperature & Humidity Per Crop")
        st.markdown("Shows the climate conditions each crop prefers.")

        climate_df = (
            df.groupby("label")[["temperature", "humidity"]]
            .mean()
            .sort_values("temperature", ascending=False)
        )

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        climate_df.plot(kind="barh", ax=ax3, color=["#E74C3C", "#1ABC9C"])
        ax3.set_xlabel("Average Value")
        ax3.set_title("Temperature & Humidity by Crop")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.divider()

    st.markdown("### 🔗 Feature Correlation Heatmap")
    st.markdown(
        "Values close to 1 or -1 mean strong relationship. "
        "Close to 0 means no relationship."
    )

    fig4, ax4 = plt.subplots(figsize=(9, 6))
    numeric_df = df.drop("label", axis=1)
    sns.heatmap(
        numeric_df.corr(), annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax4
    )
    ax4.set_title("Correlation Between Soil & Weather Features")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.divider()

    st.markdown("### 🎯 Feature Importance — What affects prediction most?")
    st.markdown(
        "Higher bar = that feature has more influence on which crop is recommended."
    )

    feature_names = FEATURES
    importances   = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    ax5.barh(importance_df["Feature"], importance_df["Importance"],
             color=sns.color_palette("viridis", len(importance_df)))
    ax5.set_xlabel("Importance Score")
    ax5.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    top_feature = importance_df.iloc[-1]["Feature"]
    top_score   = importance_df.iloc[-1]["Importance"]
    st.success(
        f"🏆 **Most important feature: {top_feature}** "
        f"(score: {top_score:.3f}) — "
        f"This single feature has the most influence on which crop is recommended."
    )

    st.divider()

    st.markdown("### 🌿 Dataset — Samples Per Crop")
    st.markdown("Confirms the dataset is balanced — each crop has 100 samples.")

    crop_counts = df["label"].value_counts().sort_values(ascending=True)

    fig6, ax6 = plt.subplots(figsize=(8, 7))
    ax6.barh(crop_counts.index, crop_counts.values,
             color=sns.color_palette("Set2", len(crop_counts)))
    ax6.set_xlabel("Number of samples")
    ax6.set_title("Samples per crop in dataset")
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()


# ════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
with tab3:

    st.subheader("🏆 Model Comparison — All 5 Algorithms")
    st.markdown(
        "We trained 5 different ML algorithms on the same dataset "
        "and compared their accuracy. Here are the results."
    )

    results_df = pd.DataFrame({
        "Model":    list(model_results.keys()),
        "Accuracy": [f"{v}%" for v in model_results.values()]
    }).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    results_df.index = results_df.index + 1
    st.dataframe(results_df, use_container_width=True)

    st.divider()

    st.markdown("### 📊 Accuracy Chart")

    names  = list(model_results.keys())
    scores = list(model_results.values())
    colors = ["#2ECC71" if s == max(scores) else "#85C1E9" for s in scores]

    fig7, ax7 = plt.subplots(figsize=(10, 5))
    bars = ax7.bar(names, scores, color=colors, edgecolor="white", linewidth=1.2)
    ax7.set_ylabel("Accuracy (%)")
    ax7.set_title("5-Model Accuracy Comparison")
    ax7.set_ylim([min(scores) - 5, 103])

    for bar, score in zip(bars, scores):
        ax7.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{score}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

    st.divider()

    st.markdown("### 🤔 Why is Random Forest the saved model?")

    best_model = max(model_results, key=model_results.get)
    best_score = model_results[best_model]
    rf_score   = model_results.get("Random Forest", 0)

    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        st.metric("Best Accuracy", f"{best_score}%", delta=f"{best_model}")
    with mc2:
        st.metric("Random Forest Accuracy", f"{rf_score}%")
    with mc3:
        diff = round(best_score - rf_score, 2)
        st.metric("Difference", f"{diff}%",
                  delta="within margin" if diff < 1 else "notable gap")

    st.info(
        "**Why Random Forest is saved even if another model scores higher:** \n\n"
        "Random Forest provides **feature_importances_** — a built-in way to "
        "see which features drove each prediction. This powers the "
        "'Why this crop?' explanation in Tab 1. \n\n"
        "SVM and XGBoost don't provide this natively. "
        "The accuracy difference between top models is typically under 1%, "
        "but the explainability benefit is significant for farmers."
    )

    st.divider()
    st.markdown("### 📚 What each algorithm does — in plain English")

    algo_data = {
        "Ridge Regression": (
            "A linear model used as a **baseline**. "
            "Low accuracy confirms crop patterns are non-linear."
        ),
        "Random Forest": (
            "100 decision trees voting together. Handles non-linear patterns, "
            "gives feature importance. **Primary model.**"
        ),
        "CatBoost": (
            "Gradient boosting designed for categorical variables. "
            "Builds trees sequentially, each fixing previous mistakes."
        ),
        "XGBoost": (
            "Gradient boosting with L1/L2 regularisation. "
            "Widely used in industry and ML competitions."
        ),
        "SVM (RBF)": (
            "Finds the best boundary separating each crop class. "
            "RBF kernel handles non-linear boundaries."
        ),
    }
    for algo, desc in algo_data.items():
        score = model_results.get(algo, "N/A")
        with st.expander(f"{algo}  —  {score}%"):
            st.markdown(desc)


# ════════════════════════════════════════════════════════════
# TAB 4 — FERTILISER RECOMMENDATION
# ════════════════════════════════════════════════════════════
with tab4:

    st.subheader("🧪 Fertiliser Recommendation")
    st.markdown(
        "Select your soil type and crop, then enter field measurements "
        "to get the most suitable **fertiliser recommendation**."
    )

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        selected_soil = st.selectbox(
            "Soil Type",
            options=list(fert_soil_le.classes_),
            help="Select the type of soil in your field"
        )
    with f_col2:
        selected_crop = st.selectbox(
            "Crop Type",
            options=list(fert_crop_le.classes_),
            help="Select the crop you are growing or planning to grow"
        )

    st.markdown("---")
    st.markdown("**Enter soil and weather measurements:**")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        f_temp = st.slider("Temperature (°C)", 10.0, 45.0, 25.0, 0.5, key="f_temp")
    with fc2:
        f_humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0, 0.5, key="f_hum")
    with fc3:
        f_rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 1.0, key="f_rain")
    with fc4:
        f_ph = st.slider("Soil pH", 3.5, 10.0, 6.5, 0.1, key="f_ph")

    fc5, fc6, fc7, fc8 = st.columns(4)
    with fc5:
        f_N = st.slider("Nitrogen (N)", 0.0, 20.0, 5.0, 0.5, key="f_N")
    with fc6:
        f_P = st.slider("Phosphorus (P)", 0.0, 20.0, 5.0, 0.5, key="f_P")
    with fc7:
        f_K = st.slider("Potassium (K)", 0.0, 20.0, 5.0, 0.5, key="f_K")
    with fc8:
        st.write("")

    st.divider()

    fert_btn = st.button(
        "🧪 Recommend Fertiliser",
        type="primary",
        use_container_width=True,
        key="fert_btn"
    )

    if fert_btn:

        soil_encoded = fert_soil_le.transform([selected_soil])[0]
        crop_encoded = fert_crop_le.transform([selected_crop])[0]

        fert_input = np.array([[
            f_temp, f_humidity, f_rainfall, f_ph,
            f_N, f_P, f_K,
            soil_encoded, crop_encoded
        ]])

        fert_input_scaled = fert_scaler.transform(fert_input)
        fert_pred_encoded = fert_model.predict(fert_input_scaled)[0]
        fert_name         = fert_le.inverse_transform([fert_pred_encoded])[0]
        fert_probs        = fert_model.predict_proba(fert_input_scaled)[0]
        fert_confidence   = fert_probs[fert_pred_encoded] * 100

        st.success(f"### 🧪 Recommended Fertiliser: **{fert_name.upper()}**")

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Model Confidence", f"{fert_confidence:.1f}%")
        with r2:
            top3_idx   = np.argsort(fert_probs)[::-1][:3]
            top3_names = fert_le.inverse_transform(top3_idx)
            top3_probs = fert_probs[top3_idx] * 100
            st.markdown("**Top 3 alternatives:**")
            for fname, fprob in zip(top3_names, top3_probs):
                st.markdown(f"- {fname}: {fprob:.1f}%")

        st.divider()

        st.subheader("🔎 Why this fertiliser?")
        st.markdown(
            "Higher bar = that factor had more influence on this recommendation."
        )

        feat_names  = ["Temperature", "Humidity", "Rainfall",
                       "pH", "N", "P", "K", "Soil Type", "Crop Type"]
        importances = fert_model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature":    feat_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig_f, ax_f = plt.subplots(figsize=(8, 4))
        ax_f.barh(imp_df["Feature"], imp_df["Importance"],
                  color=sns.color_palette("viridis", len(imp_df)))
        ax_f.set_xlabel("Feature Importance Score")
        ax_f.set_title(f"Why {fert_name}? — Feature Contributions")
        plt.tight_layout()
        st.pyplot(fig_f)
        plt.close()

        st.subheader("📝 Plain English Summary")

        top_feat = imp_df.iloc[-1]["Feature"]
        st.markdown(
            f"✅ **{top_feat}** is the most influential factor in recommending "
            f"**{fert_name}** for your field."
        )

        deficient = []
        if f_N < 5:
            deficient.append("Nitrogen (N)")
        if f_P < 5:
            deficient.append("Phosphorus (P)")
        if f_K < 5:
            deficient.append("Potassium (K)")

        if deficient:
            st.markdown(
                f"⚠️ Your soil shows **low levels** of: {', '.join(deficient)}. "
                f"**{fert_name}** helps address this deficiency."
            )
        else:
            st.markdown(
                f"✅ Your soil nutrient levels (N:{f_N}, P:{f_P}, K:{f_K}) "
                f"are adequate. **{fert_name}** will maintain and optimise growth."
            )

        st.info(
            f"**Summary:** For **{selected_crop}** grown in **{selected_soil}** soil "
            f"with temperature {f_temp}°C, humidity {f_humidity}%, "
            f"rainfall {f_rainfall}mm, and pH {f_ph} — "
            f"**{fert_name}** is recommended with {fert_confidence:.1f}% confidence."
        )

        st.divider()

        st.subheader("📊 Fertiliser Model Accuracy Comparison")
        fres_df = pd.DataFrame({
            "Model":    list(fert_results.keys()),
            "Accuracy": [f"{v}%" for v in fert_results.values()]
        }).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        fres_df.index = fres_df.index + 1
        st.dataframe(fres_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — YIELD PREDICTION
# Regression task — predicts a NUMBER (tonnes per hectare)
# Different from Tabs 1 & 4 which predict a category.
# Metrics: R², RMSE, MAE instead of accuracy %
# ════════════════════════════════════════════════════════════
with tab5:

    st.subheader("📈 Crop Yield Prediction")
    st.markdown(
        "Select your crop, state, and season, then enter field measurements "
        "to predict the expected **yield in tonnes per hectare (t/ha)**."
    )

    # ── KEY DIFFERENCE FROM PHASE 1 & 2 ─────────────────────────
    # This is a REGRESSION task — we predict a continuous number.
    # Phase 1 & 2 = Classification (predict one of N categories)
    # Phase 3     = Regression (predict a continuous number like 3.47 t/ha)
    # This is why we use dropdowns (not encoders for output) and
    # show R², RMSE, MAE instead of accuracy %.

    # ── Row 1: Dropdowns for categorical inputs ───────────────
    y_col1, y_col2, y_col3 = st.columns(3)

    with y_col1:
        selected_yield_crop = st.selectbox(
            "🌾 Crop",
            options=sorted(yield_crops),
            help="Select the crop you want to predict yield for"
        )

    with y_col2:
        selected_yield_state = st.selectbox(
            "📍 State",
            options=sorted(yield_states),
            help="Select your state/region"
        )

    with y_col3:
        # Strip whitespace from season names for clean display
        clean_seasons = [s.strip() for s in yield_seasons]
        selected_yield_season = st.selectbox(
            "🗓️ Season",
            options=clean_seasons,
            help="Select the growing season"
        )

    st.markdown("---")
    st.markdown("**Enter field measurements:**")

    # ── Row 2: Numerical inputs ───────────────────────────────
    yc1, yc2, yc3, yc4 = st.columns(4)

    with yc1:
        y_temp = st.slider(
            "Temperature (°C)", 10.0, 45.0, 28.0, 0.5,
            key="y_temp",
            help="Average temperature during growing season"
        )

    with yc2:
        y_humidity = st.slider(
            "Humidity (%)", 10.0, 100.0, 65.0, 0.5,
            key="y_hum",
            help="Average relative humidity"
        )

    with yc3:
        y_moisture = st.slider(
            "Soil Moisture (%)", 0.0, 100.0, 40.0, 0.5,
            key="y_moist",
            help="Soil moisture content percentage"
        )

    with yc4:
        y_area = st.slider(
            "Area (hectares)", 0.5, 500.0, 10.0, 0.5,
            key="y_area",
            help="Total cultivated area in hectares"
        )

    yc5, yc6 = st.columns([1, 3])

    with yc5:
        y_year = st.number_input(
            "Crop Year",
            min_value=1997, max_value=2030,
            value=2024,
            step=1,
            key="y_year",
            help="The year of cultivation"
        )

    st.divider()

    yield_btn = st.button(
        "📈 Predict Yield",
        type="primary",
        use_container_width=True,
        key="yield_btn"
    )

    if yield_btn:

        # ── Step 1: Encode categorical inputs ─────────────────
        # Use the saved encoders from yield_model.py
        # These MUST match exactly what was used during training
        try:
            y_crop_enc  = yield_crop_le.transform([selected_yield_crop])[0]
        except ValueError:
            st.error(f"Crop '{selected_yield_crop}' not found in trained encoder. "
                     f"Please retrain the model with this crop.")
            st.stop()

        try:
            y_state_enc = yield_state_le.transform([selected_yield_state])[0]
        except ValueError:
            st.error(f"State '{selected_yield_state}' not in training data. "
                     f"Please choose another state.")
            st.stop()

        # Match season — the encoder may have spaces, clean before matching
        matched_season = selected_yield_season
        for s in yield_seasons:
            if s.strip() == selected_yield_season.strip():
                matched_season = s
                break

        try:
            y_season_enc = yield_season_le.transform([matched_season])[0]
        except ValueError:
            st.error(f"Season '{matched_season}' not in training data.")
            st.stop()

        # ── Step 2: Build input array ─────────────────────────
        # MUST match training column order exactly:
        # Crop, Season, State_Name, Temperature, Humidity,
        # Soil_Moisture, Area, Crop_Year
        yield_input = np.array([[
            y_crop_enc,
            y_season_enc,
            y_state_enc,
            y_temp,
            y_humidity,
            y_moisture,
            y_area,
            y_year
        ]])

        # ── Step 3: Scale using yield scaler ──────────────────
        yield_input_scaled = yield_scaler.transform(yield_input)

        # ── Step 4: Predict ───────────────────────────────────
        # This returns a float — e.g. 3.47 (tonnes per hectare)
        predicted_yield = yield_model.predict(yield_input_scaled)[0]

        # Ensure prediction is non-negative
        predicted_yield = max(0.0, predicted_yield)

        # Calculate total production estimate
        total_production = predicted_yield * y_area

        # ── Step 5: Show results ──────────────────────────────
        st.success(
            f"### 📈 Predicted Yield: **{predicted_yield:.2f} tonnes per hectare**"
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                label="Yield per Hectare",
                value=f"{predicted_yield:.2f} t/ha",
                help="Predicted yield in tonnes per hectare"
            )

        with r2:
            st.metric(
                label="Estimated Total Production",
                value=f"{total_production:.1f} tonnes",
                help=f"Yield × Area = {predicted_yield:.2f} × {y_area} ha"
            )

        with r3:
            # Yield quality rating
            if predicted_yield >= 5.0:
                rating = "🌟 Excellent"
                rating_color = "green"
            elif predicted_yield >= 2.5:
                rating = "✅ Good"
                rating_color = "blue"
            elif predicted_yield >= 1.0:
                rating = "⚠️ Average"
                rating_color = "orange"
            else:
                rating = "❌ Low"
                rating_color = "red"

            st.metric(
                label="Yield Rating",
                value=rating,
                help="Based on typical Indian crop yield benchmarks"
            )

        st.divider()

        # ── Feature Importance chart ──────────────────────────
        st.subheader("🔎 What influenced this yield prediction?")
        st.markdown(
            "Higher bar = that factor had more influence on the predicted yield."
        )

        yield_feat_names  = ["Crop", "Season", "State",
                              "Temperature", "Humidity",
                              "Soil Moisture", "Area", "Crop Year"]
        yield_importances = yield_model.feature_importances_

        yield_imp_df = pd.DataFrame({
            "Feature":    yield_feat_names,
            "Importance": yield_importances
        }).sort_values("Importance", ascending=True)

        fig_y, ax_y = plt.subplots(figsize=(9, 5))
        colors_y = sns.color_palette("YlOrRd", len(yield_imp_df))
        ax_y.barh(yield_imp_df["Feature"], yield_imp_df["Importance"],
                  color=colors_y)
        ax_y.set_xlabel("Feature Importance Score")
        ax_y.set_title(
            f"Yield Prediction — Feature Importance\n"
            f"({selected_yield_crop.capitalize()} | {selected_yield_state} | {selected_yield_season})"
        )
        for i, (val, name) in enumerate(zip(yield_imp_df["Importance"],
                                             yield_imp_df["Feature"])):
            ax_y.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_y)
        plt.close()

        # ── Plain English Summary ─────────────────────────────
        st.subheader("📝 Plain English Summary")

        top_yield_feat = yield_imp_df.iloc[-1]["Feature"]

        st.markdown(
            f"✅ **{top_yield_feat}** is the most influential factor in this yield prediction."
        )

        # Contextual insights
        if y_temp > 38:
            st.markdown(
                "🌡️ **High temperature** (above 38°C) may stress the crop and "
                "reduce actual yield — consider irrigation or shade netting."
            )
        elif y_temp < 15:
            st.markdown(
                "❄️ **Low temperature** (below 15°C) can slow crop growth. "
                "Consider season adjustment or protected farming."
            )

        if y_humidity < 30:
            st.markdown(
                "💧 **Low humidity** may cause water stress — ensure adequate irrigation."
            )

        if y_moisture < 20:
            st.markdown(
                "🌱 **Low soil moisture** (below 20%) — drip irrigation is recommended "
                "to maintain optimal soil conditions."
            )

        st.info(
            f"**Summary:** **{selected_yield_crop.capitalize()}** grown in "
            f"**{selected_yield_state}** during **{selected_yield_season.strip()}** season "
            f"({y_year}) on **{y_area} hectares** with temperature {y_temp}°C, "
            f"humidity {y_humidity}%, and soil moisture {y_moisture}% — "
            f"predicted yield is **{predicted_yield:.2f} t/ha**, "
            f"estimated total production **{total_production:.1f} tonnes**."
        )

        st.divider()

        # ── Model Performance — Regression Metrics ────────────
        st.subheader("📊 Yield Model Performance")
        st.markdown(
            "Yield prediction uses **regression metrics** — not accuracy %. "
            "Because we're predicting a continuous number, not a category."
        )

        # Metric explanations
        me1, me2, me3 = st.columns(3)
        with me1:
            st.info(
                "**R² Score (R-squared)**\n\n"
                "How much variance in yield the model explains.\n\n"
                "1.0 = perfect | 0.0 = no better than mean\n\n"
                "Higher is better."
            )
        with me2:
            st.info(
                "**RMSE (Root Mean Square Error)**\n\n"
                "Average prediction error in tonnes/hectare.\n\n"
                "Same unit as the prediction.\n\n"
                "Lower is better."
            )
        with me3:
            st.info(
                "**MAE (Mean Absolute Error)**\n\n"
                "Average of absolute errors.\n\n"
                "'On average, prediction is off by X t/ha'\n\n"
                "Lower is better."
            )

        # ── Model comparison table ────────────────────────────
        st.markdown("### All Regression Models Compared")

        yield_res_rows = []
        for model_name, metrics in yield_results.items():
            yield_res_rows.append({
                "Model":  model_name,
                "R²":     metrics["R2"],
                "RMSE":   metrics["RMSE"],
                "MAE":    metrics["MAE"]
            })

        yield_res_df = (
            pd.DataFrame(yield_res_rows)
            .sort_values("R²", ascending=False)
            .reset_index(drop=True)
        )
        yield_res_df.index = yield_res_df.index + 1

        st.dataframe(yield_res_df, use_container_width=True)

        # ── R² bar chart ──────────────────────────────────────
        st.markdown("### 📊 R² Score Comparison (higher = better)")

        y_model_names = yield_res_df["Model"].tolist()
        y_r2_scores   = yield_res_df["R²"].tolist()
        y_colors      = ["#2ECC71" if s == max(y_r2_scores) else "#85C1E9"
                         for s in y_r2_scores]

        fig_yr, ax_yr = plt.subplots(figsize=(10, 4))
        bars_yr = ax_yr.bar(y_model_names, y_r2_scores,
                            color=y_colors, edgecolor="white", linewidth=1.2)
        ax_yr.set_ylabel("R² Score")
        ax_yr.set_title("Yield Model R² Comparison")
        ax_yr.set_ylim([0, 1.05])

        for bar, score in zip(bars_yr, y_r2_scores):
            ax_yr.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )
        plt.tight_layout()
        st.pyplot(fig_yr)
        plt.close()

        st.divider()

        # ── Algorithm explanations for regression ─────────────
        st.markdown("### 📚 Regression Algorithms — Plain English")

        regression_algos = {
            "Ridge": (
                "Linear regression with regularisation. "
                "Used as a **baseline** — if yield patterns were linear, "
                "Ridge would do well. Poor R² confirms non-linear relationships."
            ),
            "Random Forest": (
                "100 decision trees averaging their yield predictions. "
                "Handles non-linear patterns well. Provides feature importance "
                "that powers the explanation chart above. **Primary model.**"
            ),
            "XGBoost": (
                "Gradient boosting for regression. Builds trees sequentially, "
                "each correcting previous prediction errors. Very effective on "
                "tabular data with mixed feature types."
            ),
            "Gradient Boosting": (
                "scikit-learn's built-in gradient boosting regressor. "
                "More conservative than XGBoost — good for comparison. "
                "Uses a slower learning rate (0.05) to avoid overfitting."
            ),
        }

        for algo, desc in regression_algos.items():
            metrics = yield_results.get(algo, {})
            r2_val  = metrics.get("R2", "N/A")
            with st.expander(f"{algo}  —  R²: {r2_val}"):
                st.markdown(desc)


# ════════════════════════════════════════════════════════════
# TAB 6 — AI FARMING CHATBOT (Gemini-powered)
#
# What this tab does:
#   1. Accepts any farming/crop/agriculture question in plain text
#   2. Detects if the user wants a model prediction (crop/fertilizer/yield)
#   3. If yes → runs the actual ML model and feeds result to Gemini
#   4. Gemini gives a rich, expert farming answer in context
#   5. Full conversation history maintained in session state
#
# HOW INTENT DETECTION WORKS:
#   We look for keywords in the user's message.
#   If "predict crop" / "what crop" → run crop model with default values
#   If "fertilizer" / "fertiliser" → run fertilizer model
#   If "yield" → run yield model
#   Then we inject the model result into the Gemini prompt so the
#   AI can give a context-aware answer that references the prediction.
#
# WHY SESSION STATE?
#   Streamlit reruns the entire script on every interaction.
#   st.session_state persists data across reruns — without it,
#   chat history would be lost every time the user sends a message.
# ════════════════════════════════════════════════════════════

with tab6:

    st.subheader("🤖 AI Farming Chatbot")
    st.markdown(
        "Ask me **anything** about farming, crops, soil, fertilizers, yield, "
        "irrigation, pest control, seasons, or market advice. "
        "I can also **run the prediction models** for you — just ask!"
    )

    # ── Example prompts ───────────────────────────────────────
    with st.expander("💡 Example questions you can ask"):
        st.markdown("""
        - *What crop should I grow with N=80, P=40, K=50, temp=25°C, humidity=70%, pH=6.5, rainfall=120mm?*
        - *What fertilizer is best for rice in clayey soil?*
        - *What is the expected yield for wheat in Punjab during Rabi season?*
        - *How do I control aphids on cotton crops?*
        - *What is drip irrigation and when should I use it?*
        - *Which crops grow best in black soil in Maharashtra?*
        - *What are the best practices for organic farming in India?*
        - *When is the best time to sow maize in Telangana?*
        - *How does soil pH affect crop growth?*
        - *Compare Urea vs DAP fertilizer for wheat.*
        """)

    st.divider()

    # ── API Key input ─────────────────────────────────────────
    # We store the key in session state so the user only types it once.
    # In production, use st.secrets or environment variables instead.
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    if not st.session_state.gemini_api_key:
        api_key_input = st.text_input(
            "🔑 Enter your Gemini API Key to start chatting",
            type="password",
            placeholder="AIza...",
            help="Get a free key at https://aistudio.google.com/app/apikey"
        )
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
            st.rerun()
        st.info(
            "👆 Enter your Gemini API key above to enable the chatbot. "
            "Get a **free** key at [Google AI Studio](https://aistudio.google.com/app/apikey)."
        )
        st.stop()

    # ── Initialise Gemini ─────────────────────────────────────
    if not GEMINI_AVAILABLE:
        st.error(
            "❌ `google-genai` package not installed. "
            "Run: `uv add google-genai` then restart the app."
        )
        st.stop()

    # System prompt injected into every message for old SDK compatibility
    # Old SDK versions (v1beta) don't support system_instruction parameter
    FARMING_SYSTEM_PROMPT = """You are an expert AI agricultural advisor for Indian farmers.

IMPORTANT: You MUST always respond in ENGLISH ONLY. Never use Hindi, Telugu, or any other language. only if the user writes in another language, respond in that language.

You have deep knowledge of:
- Indian crops: rice, wheat, maize, cotton, sugarcane, pulses, oilseeds, vegetables, fruits
- Soil types: alluvial, black, red laterite, sandy, clayey, silty
- Fertilizers: Urea, DAP, MOP, NPK blends, organic manures, micronutrients
- Crop diseases, pests, and integrated pest management (IPM)
- Irrigation methods: drip, sprinkler, flood, furrow
- Seasons: Kharif (June-November), Rabi (November-April), Zaid (April-June)
- States of India and their agro-climatic zones
- Government schemes: PM-KISAN, soil health card, MSP
- Organic farming, sustainable agriculture, precision farming
- Post-harvest storage, market prices, crop insurance

Always:
- Give practical, actionable advice
- Mention specific quantities (e.g., "apply 50 kg Urea per acre")
- Reference Indian farming conditions, seasons, and states where relevant
- Explain in simple language suitable for farmers
- When ML model predictions are shared with you, use them as the basis of your advice
- Format answers clearly with bullet points when listing multiple items

Never give advice outside agriculture/farming topics. Politely redirect off-topic questions.

Now answer the following farmer's question:
"""

    try:
        gemini_client = genai.Client(api_key=st.session_state.gemini_api_key)
        GEMINI_READY  = True
    except Exception as e:
        st.error(f"❌ Could not connect to Gemini: {e}")
        GEMINI_READY = False
        st.stop()

    # ── Session state: chat history ───────────────────────────
    # Each message: {"role": "user"/"assistant", "content": "..."}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Helper: detect what the user wants ───────────────────
    def detect_intent(message: str) -> str:
        """
        Scans the user message for keywords to decide if
        we should run an ML model before calling Gemini.
        Returns: "crop" | "fertilizer" | "yield" | "general"
        """
        msg = message.lower()

        crop_keywords     = ["what crop", "which crop", "predict crop",
                             "suggest crop", "recommend crop", "best crop",
                             "suitable crop", "crop recommendation"]
        fert_keywords     = ["fertilizer", "fertiliser", "fertilisers",
                             "fertilizers", "manure", "npk", "urea", "dap",
                             "recommend fertilizer", "which fertilizer"]
        yield_keywords    = ["yield", "production", "tonnes per hectare",
                             "t/ha", "how much will i get", "expected output",
                             "harvest estimate", "crop output"]

        if any(k in msg for k in crop_keywords):
            return "crop"
        if any(k in msg for k in fert_keywords) and "predict" in msg:
            return "fertilizer"
        if any(k in msg for k in yield_keywords) and "predict" in msg:
            return "yield"
        return "general"

    # ── Helper: extract numbers from user message ─────────────
    def extract_numbers(message: str) -> dict:
        """
        Tries to extract N, P, K, temperature, humidity, pH, rainfall
        from the user's message using simple regex patterns.
        Falls back to sensible Indian farm defaults if not found.
        """
        def find_value(patterns, default):
            for pat in patterns:
                match = re.search(pat, message, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            return default

        return {
            "N":           find_value([r"N\s*[=:]\s*([\d.]+)", r"nitrogen\s*[=:]\s*([\d.]+)"], 60),
            "P":           find_value([r"P\s*[=:]\s*([\d.]+)", r"phosphorus\s*[=:]\s*([\d.]+)"], 40),
            "K":           find_value([r"K\s*[=:]\s*([\d.]+)", r"potassium\s*[=:]\s*([\d.]+)"], 40),
            "temperature": find_value([r"temp\w*\s*[=:]\s*([\d.]+)", r"([\d.]+)\s*°?[cC]"], 25.0),
            "humidity":    find_value([r"humid\w*\s*[=:]\s*([\d.]+)"], 65.0),
            "ph":          find_value([r"ph\s*[=:]\s*([\d.]+)"], 6.5),
            "rainfall":    find_value([r"rainfall\s*[=:]\s*([\d.]+)", r"rain\s*[=:]\s*([\d.]+)"], 100.0),
        }

    # ── Helper: run crop model ────────────────────────────────
    def run_crop_model(message: str) -> str:
        """Runs the crop prediction model and returns a result string."""
        try:
            vals         = extract_numbers(message)
            input_arr    = np.array([[vals["N"], vals["P"], vals["K"],
                                      vals["temperature"], vals["humidity"],
                                      vals["ph"], vals["rainfall"]]])
            input_scaled = scaler.transform(input_arr)
            pred_enc     = model.predict(input_scaled)[0]
            crop_name    = le.inverse_transform([pred_enc])[0]
            probs        = model.predict_proba(input_scaled)[0]
            confidence   = probs[pred_enc] * 100

            top3_idx     = np.argsort(probs)[::-1][:3]
            top3_crops   = le.inverse_transform(top3_idx)
            top3_probs   = probs[top3_idx] * 100
            top3_str     = ", ".join([f"{c} ({p:.1f}%)"
                                      for c, p in zip(top3_crops, top3_probs)])

            return (
                f"[ML MODEL RESULT — Crop Prediction]\n"
                f"Input: N={vals['N']}, P={vals['P']}, K={vals['K']}, "
                f"Temp={vals['temperature']}°C, Humidity={vals['humidity']}%, "
                f"pH={vals['ph']}, Rainfall={vals['rainfall']}mm\n"
                f"Predicted crop: {crop_name.upper()} (confidence: {confidence:.1f}%)\n"
                f"Top 3 alternatives: {top3_str}\n"
                f"Model: Random Forest (99.55% accuracy)\n"
            )
        except Exception as e:
            return f"[Crop model error: {e}. Answering from general knowledge.]"

    # ── Helper: run fertilizer model ──────────────────────────
    def run_fertilizer_model(message: str) -> str:
        """Runs the fertilizer model and returns a result string."""
        try:
            vals = extract_numbers(message)

            # Pick soil and crop from message or use defaults
            msg_lower = message.lower()
            soil_options = list(fert_soil_le.classes_)
            crop_options = list(fert_crop_le.classes_)

            detected_soil = next(
                (s for s in soil_options if s.lower() in msg_lower), soil_options[0]
            )
            detected_crop = next(
                (c for c in crop_options if c.lower() in msg_lower), crop_options[0]
            )

            soil_enc  = fert_soil_le.transform([detected_soil])[0]
            crop_enc  = fert_crop_le.transform([detected_crop])[0]

            f_input   = np.array([[vals["temperature"], vals["humidity"],
                                   vals["rainfall"], vals["ph"],
                                   vals["N"], vals["P"], vals["K"],
                                   soil_enc, crop_enc]])
            f_scaled  = fert_scaler.transform(f_input)
            f_pred    = fert_model.predict(f_scaled)[0]
            fert_name = fert_le.inverse_transform([f_pred])[0]
            f_probs   = fert_model.predict_proba(f_scaled)[0]
            f_conf    = f_probs[f_pred] * 100

            return (
                f"[ML MODEL RESULT — Fertilizer Prediction]\n"
                f"Crop: {detected_crop}, Soil: {detected_soil}\n"
                f"N={vals['N']}, P={vals['P']}, K={vals['K']}, pH={vals['ph']}\n"
                f"Recommended fertilizer: {fert_name.upper()} (confidence: {f_conf:.1f}%)\n"
                f"Model: Random Forest (tuned)\n"
            )
        except Exception as e:
            return f"[Fertilizer model error: {e}. Answering from general knowledge.]"

    # ── Helper: run yield model ───────────────────────────────
    def run_yield_model(message: str) -> str:
        """Runs the yield model and returns a result string."""
        try:
            vals = extract_numbers(message)

            msg_lower = message.lower()

            detected_crop = next(
                (c for c in yield_crops if c.lower() in msg_lower), yield_crops[0]
            )
            detected_state = next(
                (s for s in yield_states if s.lower() in msg_lower), yield_states[0]
            )
            detected_season = next(
                (s for s in yield_seasons if s.strip().lower() in msg_lower),
                yield_seasons[0]
            )

            yc_enc  = yield_crop_le.transform([detected_crop])[0]
            yst_enc = yield_state_le.transform([detected_state])[0]
            yse_enc = yield_season_le.transform([detected_season])[0]

            y_input  = np.array([[yc_enc, yse_enc, yst_enc,
                                   vals["temperature"], vals["humidity"],
                                   40.0,   # default soil moisture
                                   10.0,   # default area
                                   2024]])
            y_scaled = yield_scaler.transform(y_input)
            y_pred   = yield_model.predict(y_scaled)[0]
            y_pred   = max(0.0, y_pred)

            return (
                f"[ML MODEL RESULT — Yield Prediction]\n"
                f"Crop: {detected_crop}, State: {detected_state}, "
                f"Season: {detected_season.strip()}\n"
                f"Temp={vals['temperature']}°C, Humidity={vals['humidity']}%\n"
                f"Predicted yield: {y_pred:.2f} tonnes per hectare\n"
                f"Model: Random Forest Regressor\n"
            )
        except Exception as e:
            return f"[Yield model error: {e}. Answering from general knowledge.]"

    # ── Helper: get Gemini response ───────────────────────────
    def call_gemini(prompt: str) -> str:
        """
        Calls Gemini with automatic retry for 503 (server overload) errors.
        Retries up to 3 times with increasing wait time.
        """
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                # 503 = server overload — retry after waiting
                if "503" in error_str and attempt < max_retries - 1:
                    wait = (attempt + 1) * 3   # 3s, 6s, 9s
                    time.sleep(wait)
                    continue
                raise e  # re-raise if not 503 or out of retries

    def get_gemini_response(user_message: str, model_context: str = "") -> tuple:
        """
        RAG-powered Gemini response with 3-tier priority:
        1. ML model result → grounded in prediction
        2. RAG (ChromaDB) → grounded in knowledge base
        3. General Gemini knowledge → fallback

        Returns: (response_text, mode_used)
        """
        history_text = ""
        for msg in st.session_state.chat_history[:-1]:
            role = "Farmer" if msg["role"] == "user" else "Advisor"
            history_text += f"{role}: {msg['content']}\n\n"

        # ── Priority 1: ML model result ───────────────────────
        if model_context:
            full_prompt = (
                f"{FARMING_SYSTEM_PROMPT}"
                f"{model_context}\n\n"
                f"Conversation so far:\n{history_text}"
                f"Based on the ML model prediction above, answer this farmer's question:\n"
                f"{user_message}\n\nGive detailed farming advice referencing the prediction."
            )
            return call_gemini(full_prompt), "ml_model"

        # ── Priority 2: RAG knowledge base search ─────────────
        if rag_loaded and rag_collection is not None:
            rag_result = search_knowledge(user_message, rag_collection)
            if rag_result["use_rag"]:
                full_prompt = build_rag_prompt(
                    query=user_message,
                    context=rag_result["context"],
                    system_prompt=FARMING_SYSTEM_PROMPT
                )
                return call_gemini(full_prompt), "rag"

        # ── Priority 3: General Gemini fallback ───────────────
        full_prompt = build_general_prompt(
            query=user_message,
            system_prompt=FARMING_SYSTEM_PROMPT,
            history_text=history_text
        )
        return call_gemini(full_prompt), "general"

    # ── Chat UI ───────────────────────────────────────────────
    # Display conversation history
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                "👋 **Hello! I'm your AI Farming Advisor.** Ask me anything about "
                "crops, soil, fertilizers, pests, irrigation, or farming practices. "
                "I can also run the prediction models for you!"
            )
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👨‍🌾"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🌾"):
                        st.markdown(msg["content"])
                        # Show badge indicating which mode was used
                        mode = msg.get("response_mode", "")
                        if mode == "rag":
                            st.caption("📚 **Answer from Knowledge Base** (RAG)")
                        elif mode == "ml_model":
                            st.caption(f"🤖 **ML Model Used:** {msg.get('used_model', '')}")
                        elif mode == "general":
                            st.caption("🌐 **General AI Knowledge**")

    # ── RAG status banner ─────────────────────────────────────
    if rag_loaded:
        st.success(f"📚 Knowledge Base: **Active** — {rag_collection.count()} farming documents loaded")
    else:
        st.warning(
            "📚 Knowledge Base: **Not loaded** — Run `python knowledge_base.py` once to enable RAG. "
            "Chatbot works without it using Gemini general knowledge."
        )

    # ── Chat input ────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask about crops, soil, fertilizers, pests, irrigation...",
        key="chat_input"
    )

    if user_input:

        # 1. Add user message to history
        st.session_state.chat_history.append({
            "role":    "user",
            "content": user_input
        })

        # 2. Detect intent — should we run an ML model first?
        intent       = detect_intent(user_input)
        model_result = ""
        used_model   = None

        if intent == "crop":
            with st.spinner("🌾 Running crop prediction model..."):
                model_result = run_crop_model(user_input)
                used_model   = "Crop Recommendation (Random Forest)"

        elif intent == "fertilizer":
            with st.spinner("🧪 Running fertilizer model..."):
                model_result = run_fertilizer_model(user_input)
                used_model   = "Fertilizer Recommendation (Random Forest)"

        elif intent == "yield":
            with st.spinner("📈 Running yield model..."):
                model_result = run_yield_model(user_input)
                used_model   = "Yield Prediction (Random Forest Regressor)"

        # 3. Call Gemini — RAG pipeline handles mode selection
        with st.spinner("🤖 Getting AI response..."):
            try:
                ai_response, response_mode = get_gemini_response(user_input, model_result)
            except Exception as e:
                ai_response   = (
                    f"⚠️ Gemini API error: {e}\n\n"
                    f"Please check your API key and try again."
                )
                response_mode = "error"

        # 4. Add assistant response to history with mode badge
        st.session_state.chat_history.append({
            "role":          "assistant",
            "content":       ai_response,
            "used_model":    used_model,
            "response_mode": response_mode
        })

        # 5. Rerun to refresh chat display
        st.rerun()

    # ── Sidebar controls ─────────────────────────────────────
    st.divider()

    col_clear, col_export = st.columns(2)

    with col_clear:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with col_export:
        if st.session_state.chat_history:
            # Export conversation as plain text
            export_text = "\n\n".join([
                f"{'You' if m['role'] == 'user' else 'AI Advisor'}: {m['content']}"
                for m in st.session_state.chat_history
            ])
            st.download_button(
                label="💾 Download Conversation",
                data=export_text,
                file_name="farming_chat.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ── API key management ────────────────────────────────────
    st.divider()
    with st.expander("⚙️ Settings — API Key & Model Info"):
        st.markdown(f"**API Key:** `{'*' * 20}{st.session_state.gemini_api_key[-4:]}`")
        st.markdown("**Gemini Model:** `gemini-2.5-flash` (free quota available)")
        st.markdown("**Conversation turns:** " + str(len(st.session_state.chat_history) // 2))

        if st.button("🔄 Reset API Key"):
            st.session_state.gemini_api_key = ""
            st.rerun()

        st.markdown("""
        **How the chatbot uses your ML models:**
        | Keyword in your question | Model triggered |
        |---|---|
        | "what crop", "predict crop", "which crop" | 🌾 Crop Recommendation |
        | "predict fertilizer/fertiliser" | 🧪 Fertilizer Model |
        | "predict yield" | 📈 Yield Model |
        | Everything else | 💬 Gemini general farming knowledge |
        """)
