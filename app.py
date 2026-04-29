# ============================================================
# app.py — complete file with all 5 tabs
# Tab 1 — Predict & Explain     (crop recommendation)
# Tab 2 — Insights Dashboard    (EDA charts)
# Tab 3 — Model Comparison      (5 crop models)
# Tab 4 — Fertiliser            (fertilizer recommendation)
# Tab 5 — Yield Prediction      (regression — tonnes/hectare)
#
# Run with: streamlit run app.py
# ============================================================

import matplotlib
matplotlib.use("Agg")

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
    # Crop model (Phase 1)
    model   = joblib.load("models/crop_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    results = joblib.load("models/model_results.pkl")
    # Fertilizer model (Phase 2)
    fert_model    = joblib.load("models/fertilizer_model.pkl")
    fert_scaler   = joblib.load("models/fertilizer_scaler.pkl")
    fert_encoder  = joblib.load("models/fertilizer_label_encoder.pkl")
    fert_soil_enc = joblib.load("models/fertilizer_soil_encoder.pkl")
    fert_crop_enc = joblib.load("models/fertilizer_crop_encoder.pkl")
    fert_results  = joblib.load("models/fertilizer_model_results.pkl")
    # Yield model (Phase 3)
    yield_model      = joblib.load("models/yield_model.pkl")
    yield_scaler     = joblib.load("models/yield_scaler.pkl")
    yield_crop_enc   = joblib.load("models/yield_crop_encoder.pkl")
    yield_season_enc = joblib.load("models/yield_season_encoder.pkl")
    yield_state_enc  = joblib.load("models/yield_state_encoder.pkl")
    yield_results    = joblib.load("models/yield_model_results.pkl")
    yield_crops      = joblib.load("models/yield_crops.pkl")
    yield_seasons    = joblib.load("models/yield_seasons.pkl")
    yield_states     = joblib.load("models/yield_states.pkl")
    return (model, scaler, encoder, results,
            fert_model, fert_scaler, fert_encoder,
            fert_soil_enc, fert_crop_enc, fert_results,
            yield_model, yield_scaler, yield_crop_enc,
            yield_season_enc, yield_state_enc, yield_results,
            yield_crops, yield_seasons, yield_states)

(model, scaler, le, model_results,
 fert_model, fert_scaler, fert_le,
 fert_soil_le, fert_crop_le, fert_results,
 yield_model, yield_scaler, yield_crop_le,
 yield_season_le, yield_state_le, yield_results,
 yield_crops, yield_seasons, yield_states) = load_models()


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


# ── 5 Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌱 Predict & Explain",
    "📊 Insights Dashboard",
    "🏆 Model Comparison",
    "🧪 Fertiliser Recommendation",
    "📈 Yield Prediction"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT & EXPLAIN
# ════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Enter Soil & Weather Values")
    st.markdown("Move the sliders to match your field conditions.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        N = st.slider("Nitrogen (N)", 0, 140, 50,
                      help="Nitrogen content in soil (kg/ha)")
    with col2:
        P = st.slider("Phosphorus (P)", 5, 145, 50,
                      help="Phosphorus content in soil (kg/ha)")
    with col3:
        K = st.slider("Potassium (K)", 5, 205, 50,
                      help="Potassium content in soil (kg/ha)")
    with col4:
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, 0.5)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0, 0.5)
    with col6:
        ph = st.slider("Soil pH", 3.5, 10.0, 6.5, 0.1)
    with col7:
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 1.0)
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
        st.subheader("🔎 Why was this crop recommended?")
        st.markdown("Red = pushed **towards** this crop. Blue = pushed **against** it.")

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
            ax.set_xlabel("SHAP Value")
            ax.set_title(f"Why {predicted_crop.capitalize()}? — Feature Contributions")
            for bar, val in zip(bars, shap_df["SHAP"]):
                ax.text(val + (0.001 if val >= 0 else -0.001),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center",
                        ha="left" if val >= 0 else "right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("📝 Plain English Summary")
        top_pos = shap_df[shap_df["SHAP"] > 0].head(3)
        top_neg = shap_df[shap_df["SHAP"] < 0].head(2)

        if len(top_pos) > 0:
            st.markdown(
                f"✅ **{top_pos.iloc[0]['Feature']}** (value: {top_pos.iloc[0]['Value']}) "
                f"is the strongest reason {predicted_crop.capitalize()} is recommended."
            )
        for _, row in top_pos.iloc[1:].iterrows():
            st.markdown(f"✅ **{row['Feature']}** also supports {predicted_crop.capitalize()}.")
        for _, row in top_neg.iterrows():
            st.markdown(
                f"⚠️ **{row['Feature']}** slightly works against "
                f"{predicted_crop.capitalize()} but other factors outweigh it."
            )
        st.divider()
        st.info(
            f"**About {predicted_crop.capitalize()}:** Based on N:{N}, P:{P}, K:{K}, "
            f"Temp:{temperature}°C, Humidity:{humidity}%, pH:{ph}, Rainfall:{rainfall}mm — "
            f"{predicted_crop.capitalize()} is recommended with {confidence:.1f}% confidence."
        )


# ════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS DASHBOARD
# ════════════════════════════════════════════════════════════
with tab2:

    st.subheader("📊 Data Insights")
    st.markdown("Explore patterns in the crop dataset used to train the model.")

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
    st.markdown("### 🎯 Feature Importance")
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
    st.success(f"🏆 **Most important: {top_feat}** (score: {top_score:.3f})")

    st.divider()
    st.markdown("### 🌿 Samples Per Crop")
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
    results_df = pd.DataFrame({
        "Model":    list(model_results.keys()),
        "Accuracy": [f"{v}%" for v in model_results.values()]
    }).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1
    st.dataframe(results_df, use_container_width=True)

    st.divider()
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
                 f"{score}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

    st.divider()
    st.markdown("### 🤔 Why is Random Forest the saved model?")
    best_model = max(model_results, key=model_results.get)
    rf_score   = model_results.get("Random Forest", 0)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Best Accuracy", f"{model_results[best_model]}%", delta=best_model)
    with mc2:
        st.metric("Random Forest", f"{rf_score}%")
    with mc3:
        diff = round(model_results[best_model] - rf_score, 2)
        st.metric("Difference", f"{diff}%",
                  delta="within margin" if diff < 1 else "notable gap")

    st.info(
        "Random Forest provides **feature_importances_** natively — powering the "
        "'Why this crop?' SHAP explanation. SVM and XGBoost don't provide this. "
        "Accuracy difference is under 1% but explainability benefit is significant."
    )

    st.divider()
    algo_data = {
        "Ridge Regression": "Linear baseline. 8.64% confirms non-linear patterns.",
        "Random Forest":    "100 trees voting. Feature importance. **Primary model.**",
        "CatBoost":         "Gradient boosting for categorical variables.",
        "XGBoost":          "Gradient boosting with L1/L2 regularisation.",
        "SVM (RBF)":        "Finds best class boundary. RBF = non-linear kernel.",
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
        selected_soil = st.selectbox("Soil Type",
                                     options=list(fert_soil_le.classes_),
                                     help="Select the type of soil in your field")
    with f_col2:
        selected_crop = st.selectbox("Crop Type",
                                     options=list(fert_crop_le.classes_),
                                     help="Select the crop you are growing")

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
    fert_btn = st.button("🧪 Recommend Fertiliser", type="primary",
                         use_container_width=True, key="fert_btn")

    if fert_btn:
        soil_encoded      = fert_soil_le.transform([selected_soil])[0]
        crop_encoded      = fert_crop_le.transform([selected_crop])[0]
        fert_input        = np.array([[f_temp, f_humidity, f_rainfall, f_ph,
                                       f_N, f_P, f_K, soil_encoded, crop_encoded]])
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
        feat_names  = ["Temperature","Humidity","Rainfall","pH",
                       "N","P","K","Soil Type","Crop Type"]
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
        if f_N < 5: deficient.append("Nitrogen (N)")
        if f_P < 5: deficient.append("Phosphorus (P)")
        if f_K < 5: deficient.append("Potassium (K)")

        if deficient:
            st.markdown(f"⚠️ Low levels of: {', '.join(deficient)}. "
                        f"**{fert_name}** addresses this deficiency.")
        else:
            st.markdown(f"✅ Nutrient levels adequate. "
                        f"**{fert_name}** will maintain and optimise growth.")

        st.info(
            f"**Summary:** For **{selected_crop}** in **{selected_soil}** soil — "
            f"**{fert_name}** recommended with {fert_confidence:.1f}% confidence."
        )

        st.divider()
        st.subheader("📊 Fertiliser Model Accuracy")
        fres_df = pd.DataFrame({
            "Model":    list(fert_results.keys()),
            "Accuracy": [f"{v}%" for v in fert_results.values()]
        }).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        fres_df.index = fres_df.index + 1
        st.dataframe(fres_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — YIELD PREDICTION
# KEY DIFFERENCE: This is REGRESSION not classification.
# We predict a number (tonnes/hectare) not a category.
# ════════════════════════════════════════════════════════════
with tab5:

    st.subheader("📈 Crop Yield Prediction")
    st.markdown(
        "Select your crop, state and season, then enter field conditions "
        "to predict **expected yield in tonnes per hectare**."
    )

    # ── Info box explaining regression ───────────────────────
    st.info(
        "**What is yield prediction?** Unlike crop recommendation (which predicts "
        "a category), yield prediction is a **regression** problem — it predicts "
        "a continuous number. The model was trained on 50,000 real Indian government "
        "records covering 75 crops across 7 states."
    )

    # ── Dropdowns ─────────────────────────────────────────────
    y_col1, y_col2, y_col3 = st.columns(3)

    with y_col1:
        # Strip whitespace from season names for display
        season_display = [s.strip() for s in yield_seasons]
        selected_season = st.selectbox(
            "Season",
            options=yield_seasons,
            format_func=lambda x: x.strip(),
            help="Kharif = June-Nov, Rabi = Nov-Apr, Zaid = Apr-Jun"
        )

    with y_col2:
        selected_state = st.selectbox(
            "State",
            options=yield_states,
            help="Select your state"
        )

    with y_col3:
        selected_crop_yield = st.selectbox(
            "Crop",
            options=yield_crops,
            help="Select the crop you want to grow"
        )

    st.markdown("---")
    st.markdown("**Enter field conditions:**")

    yc1, yc2, yc3, yc4 = st.columns(4)

    with yc1:
        y_temp = st.slider("Temperature (°C)", 10.0, 45.0, 28.0, 0.5, key="y_temp")
    with yc2:
        y_humidity = st.slider("Humidity (%)", 10.0, 100.0, 70.0, 0.5, key="y_hum")
    with yc3:
        y_moisture = st.slider("Soil Moisture (%)", 10.0, 90.0, 40.0, 0.5, key="y_moist")
    with yc4:
        y_area = st.number_input(
            "Area (hectares)",
            min_value=0.1,
            max_value=10000.0,
            value=1.0,
            step=0.5,
            help="Total field area in hectares"
        )

    yc5, yc6 = st.columns(2)
    with yc5:
        y_year = st.slider("Crop Year", 2000, 2030, 2024, 1, key="y_year")
    with yc6:
        st.write("")

    st.divider()

    yield_btn = st.button(
        "📈 Predict Yield",
        type="primary",
        use_container_width=True,
        key="yield_btn"
    )

    if yield_btn:

        # 1. Encode categorical inputs
        crop_enc   = yield_crop_le.transform([selected_crop_yield])[0]
        season_enc = yield_season_le.transform([selected_season])[0]
        state_enc  = yield_state_le.transform([selected_state])[0]

        # 2. Build input array — MUST match training order:
        #    Crop, Season, State_Name, Temperature, Humidity,
        #    Soil_Moisture, Area, Crop_Year
        yield_input = np.array([[
            crop_enc, season_enc, state_enc,
            y_temp, y_humidity, y_moisture,
            y_area, y_year
        ]])

        # 3. Scale
        yield_input_scaled = yield_scaler.transform(yield_input)

        # 4. Predict yield per hectare
        yield_pred = yield_model.predict(yield_input_scaled)[0]
        yield_pred = max(0, yield_pred)   # can't have negative yield

        # 5. Calculate total production
        total_production = yield_pred * y_area

        # ── Show results ──────────────────────────────────────
        st.success(
            f"### 📈 Predicted Yield: **{yield_pred:.2f} tonnes/hectare**"
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Yield per Hectare",
                f"{yield_pred:.2f} t/ha",
                help="Predicted tonnes of crop per hectare of land"
            )
        with m2:
            st.metric(
                "Total Production",
                f"{total_production:.2f} tonnes",
                help=f"For your {y_area} hectare field"
            )
        with m3:
            # Convert to kg for smaller yields
            yield_kg = yield_pred * 1000
            st.metric(
                "In kg/hectare",
                f"{yield_kg:.0f} kg/ha",
                help="Same value expressed in kilograms"
            )

        st.divider()

        # ── Feature importance chart ──────────────────────────
        st.subheader("🔎 What factors influenced this prediction?")

        feat_names_yield = ["Crop", "Season", "State",
                            "Temperature", "Humidity",
                            "Soil Moisture", "Area", "Year"]
        imp_yield = yield_model.feature_importances_

        imp_yield_df = pd.DataFrame({
            "Feature":    feat_names_yield,
            "Importance": imp_yield
        }).sort_values("Importance", ascending=True)

        fig_y, ax_y = plt.subplots(figsize=(8, 4))
        ax_y.barh(imp_yield_df["Feature"], imp_yield_df["Importance"],
                  color=sns.color_palette("viridis", len(imp_yield_df)))
        ax_y.set_xlabel("Feature Importance Score")
        ax_y.set_title("What drives yield prediction most?")
        plt.tight_layout()
        st.pyplot(fig_y)
        plt.close()

        # ── Plain English summary ─────────────────────────────
        st.subheader("📝 Plain English Summary")

        top_yield_feat = imp_yield_df.iloc[-1]["Feature"]
        st.markdown(
            f"✅ **{top_yield_feat}** has the most influence on predicted yield "
            f"for {selected_crop_yield} in {selected_state}."
        )

        # Season insight
        season_clean = selected_season.strip()
        season_tips = {
            "Kharif":     "Kharif crops depend heavily on monsoon rainfall (June–November).",
            "Rabi":       "Rabi crops grow in winter (November–April) with irrigation.",
            "Summer":     "Summer crops need irrigation and heat-tolerant varieties.",
            "Whole Year": "Year-round crops need consistent soil moisture management.",
            "Autumn":     "Autumn crops benefit from post-monsoon soil moisture.",
            "Winter":     "Winter crops need cold tolerance and frost protection."
        }
        tip = season_tips.get(season_clean, "")
        if tip:
            st.markdown(f"🌦️ **Season note:** {tip}")

        st.info(
            f"**Summary:** **{selected_crop_yield}** grown in **{selected_state}** "
            f"during **{season_clean}** season on **{y_area} hectares** — "
            f"predicted yield is **{yield_pred:.2f} t/ha** "
            f"({total_production:.2f} tonnes total)."
        )

        st.divider()

        # ── Regression model comparison ───────────────────────
        st.subheader("📊 Yield Model Comparison — Regression Metrics")
        st.markdown(
            "Unlike classification (accuracy %), regression uses **R²**, **RMSE** and **MAE**."
        )

        comp_rows = []
        for mname, metrics in yield_results.items():
            comp_rows.append({
                "Model": mname,
                "R² Score": metrics["R2"],
                "RMSE (t/ha)": metrics["RMSE"],
                "MAE (t/ha)": metrics["MAE"]
            })
        comp_df = pd.DataFrame(comp_rows).sort_values(
            "R² Score", ascending=False).reset_index(drop=True)
        comp_df.index = comp_df.index + 1
        st.dataframe(comp_df, use_container_width=True)

        st.markdown(
            "**How to read these metrics:**\n\n"
            "- **R²**: How much yield variation the model explains. "
            "0.9089 = explains 90.89% of variation — very strong.\n\n"
            "- **RMSE**: Average prediction error in tonnes/hectare. Lower = better.\n\n"
            "- **MAE**: Average absolute error. More interpretable than RMSE."
        )