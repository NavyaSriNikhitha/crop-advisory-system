# 🌾 AI-Powered Crop Advisory System

An end-to-end AI system that helps Indian farmers make data-driven decisions about crop selection, fertilizer use, yield estimation, and farming practices.

## Features

- 🌱 **Crop Recommendation** — Predicts best crop based on soil and weather conditions (99.55% accuracy)
- 🧪 **Fertilizer Recommendation** — Suggests optimal fertilizer based on soil type and crop
- 📈 **Yield Prediction** — Estimates crop yield in tonnes per hectare (regression)
- 📊 **Insights Dashboard** — Visual EDA charts and feature importance
- 🏆 **Model Comparison** — Compares 5 ML algorithms
- 🤖 **AI Farming Chatbot** — Gemini-powered chatbot with RAG knowledge base

## Tech Stack

- **ML Models**: Scikit-learn, XGBoost, CatBoost, Random Forest
- **Explainability**: SHAP
- **AI Chatbot**: Google Gemini (gemini-2.5-flash)
- **RAG**: ChromaDB + Sentence Transformers
- **Frontend**: Streamlit
- **Language**: Python 3.11+

## Setup

### 1. Install dependencies
```bash
uv add streamlit pandas numpy scikit-learn xgboost catboost shap joblib matplotlib seaborn google-genai chromadb sentence-transformers
```

### 2. Train models (if models/ folder not present)
```bash
python crop_model.py
python generate_fertilizer_data.py
python fertilizer_model.py
python yield_model.py
```

### 3. Build RAG knowledge base (run once)
```bash
python knowledge_base.py
```

### 4. Run the app
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Main Streamlit app (6 tabs)
├── crop_model.py           # Phase 1 — crop prediction training
├── fertilizer_model.py     # Phase 2 — fertilizer prediction training
├── generate_fertilizer_data.py  # generates fertilizer dataset
├── yield_model.py          # Phase 3 — yield prediction training
├── knowledge_base.py       # Phase 5 — builds ChromaDB knowledge base
├── rag_pipeline.py         # Phase 5 — RAG search + fallback logic
├── data/                   # datasets
├── models/                 # trained ML models (.pkl files)
└── vector_db/              # ChromaDB vector store
```

## RAG System

The chatbot uses a 3-tier response system:
1. 🤖 **ML Model** — runs actual prediction model for crop/fertilizer/yield queries
2. 📚 **RAG** — searches 47 farming documents in ChromaDB for grounded answers
3. 🌐 **General** — falls back to Gemini general farming knowledge

## Environment Variables

Set your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```
Or enter it directly in the app UI.
