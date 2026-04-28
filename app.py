# ============================================================
# app.py — complete file with all 4 tabs
# Tab 1 — Predict & Explain     (crop recommendation)
# Tab 2 — Insights Dashboard    (EDA charts)
# Tab 3 — Model Comparison      (5 crop models)
# Tab 4 — Fertiliser            (fertilizer recommendation)
#
# Run with: streamlit run app.py
# ============================================================

import matplotlib
matplotlib.use("Agg")                # fixes threading issue on Windows Python 3.13

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")


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
    return (model, scaler, encoder, results,
            fert_model, fert_scaler, fert_encoder,
            fert_soil_enc, fert_crop_enc, fert_results)

(model, scaler, le, model_results,
 fert_model, fert_scaler, fert_le,
 fert_soil_le, fert_crop_le, fert_results) = load_models()


# ── Load dataset ──────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/crop_recommendation.csv")

df = load_data()

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ── Header ────────────────────────────────────────────────────
st.title("🌾 AI-Powered Crop Advisory System")
st.markdown(
    "Enter your soil and weather values to get a **crop recommendation**, "
    "understand **why** that crop was suggested, and explore **data insights**."
)
st.divider()


# ── 4 Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌱 Predict & Explain",
    "📊 Insights Dashboard",
    "🏆 Model Comparison",
    "🧪 Fertiliser Recommendation"
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

    st.subheader("📊 Data Insights")
    st.markdown("Explore patterns in the crop dataset used to train the model.")

    # Rainfall per crop
    st.markdown("### 🌧️ Average Rainfall Required Per Crop")
    rainfall_by_crop = (df.groupby("label")["rainfall"].mean()
                        .sort_values(ascending=False).reset_index())
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(rainfall_by_crop["label"], rainfall_by_crop["rainfall"],
            color=sns.color_palette("Blues_d", len(rainfall_by_crop)))
    ax1.set_xlabel("Crop")
    ax1.set_ylabel("Avg Rainfall (mm)")
    ax1.set_title("Average Rainfall Required Per Crop")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.divider()

    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.markdown("### 🧪 Avg Soil Nutrients Per Crop (N, P, K)")
        npk_df = (df.groupby("label")[["N", "P", "K"]].mean()
                  .sort_values("N", ascending=False))
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        npk_df.plot(kind="barh", ax=ax2, color=["#2ECC71", "#3498DB", "#E67E22"])
        ax2.set_xlabel("Average Value")
        ax2.set_title("N, P, K by Crop")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with ins_col2:
        st.markdown("### 🌡️ Avg Temperature & Humidity Per Crop")
        climate_df = (df.groupby("label")[["temperature", "humidity"]].mean()
                      .sort_values("temperature", ascending=False))
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        climate_df.plot(kind="barh", ax=ax3, color=["#E74C3C", "#1ABC9C"])
        ax3.set_xlabel("Average Value")
        ax3.set_title("Temperature & Humidity by Crop")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.divider()

    st.markdown("### 🔗 Feature Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    sns.heatmap(df.drop("label", axis=1).corr(), annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax4)
    ax4.set_title("Correlation Between Soil & Weather Features")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.divider()

    st.markdown("### 🎯 Feature Importance — What affects prediction most?")
    importance_df = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    ax5.barh(importance_df["Feature"], importance_df["Importance"],
             color=sns.color_palette("viridis", len(importance_df)))
    ax5.set_xlabel("Importance Score")
    ax5.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    top_feat  = importance_df.iloc[-1]["Feature"]
    top_score = importance_df.iloc[-1]["Importance"]
    st.success(
        f"🏆 **Most important feature: {top_feat}** (score: {top_score:.3f}) — "
        f"This single feature has the most influence on which crop is recommended."
    )

    st.divider()

    st.markdown("### 🌿 Dataset — Samples Per Crop")
    crop_counts = df["label"].value_counts().sort_values(ascending=True)
    fig6, ax6   = plt.subplots(figsize=(8, 7))
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
        "and compared their accuracy."
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
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{score}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
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
        "Random Forest provides **feature_importances_** — a built-in way to see "
        "which features drove each prediction. This powers the 'Why this crop?' "
        "explanation in Tab 1. SVM and XGBoost don't provide this natively. "
        "The accuracy difference between top models is typically under 1%, but "
        "the explainability benefit is significant for farmers who need to "
        "understand and trust the recommendation."
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

    # ── Soil and crop dropdowns ───────────────────────────────
    # selectbox = dropdown — used for categorical options
    # The options come directly from the saved encoder classes
    # so they always match exactly what the model was trained on
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

        # 1. Encode categorical inputs using saved encoders
        soil_encoded = fert_soil_le.transform([selected_soil])[0]
        crop_encoded = fert_crop_le.transform([selected_crop])[0]

        # 2. Build input array — MUST match training column order:
        #    Temperature, Humidity, Rainfall, pH, N, P, K,
        #    Soil_encoded, Crop_encoded
        fert_input = np.array([[
            f_temp, f_humidity, f_rainfall, f_ph,
            f_N, f_P, f_K,
            soil_encoded, crop_encoded
        ]])

        # 3. Scale using fertilizer scaler (NOT the crop scaler)
        fert_input_scaled = fert_scaler.transform(fert_input)

        # 4. Predict
        fert_pred_encoded = fert_model.predict(fert_input_scaled)[0]
        fert_name         = fert_le.inverse_transform([fert_pred_encoded])[0]

        # 5. Confidence scores
        fert_probs      = fert_model.predict_proba(fert_input_scaled)[0]
        fert_confidence = fert_probs[fert_pred_encoded] * 100

        # ── Show result ───────────────────────────────────────
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

        # ── Feature importance chart ──────────────────────────
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

        # ── Plain English explanation ─────────────────────────
        st.subheader("📝 Plain English Summary")

        top_feat = imp_df.iloc[-1]["Feature"]
        st.markdown(
            f"✅ **{top_feat}** is the most influential factor in recommending "
            f"**{fert_name}** for your field."
        )

        # Nutrient deficiency check
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

        # ── Fertilizer model comparison ───────────────────────
        st.subheader("📊 Fertiliser Model Accuracy Comparison")
        fres_df = pd.DataFrame({
            "Model":    list(fert_results.keys()),
            "Accuracy": [f"{v}%" for v in fert_results.values()]
        }).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        fres_df.index = fres_df.index + 1
        st.dataframe(fres_df, use_container_width=True)