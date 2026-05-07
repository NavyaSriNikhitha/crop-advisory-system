# check_models.py
# Run this ONCE to see which Gemini models your API key supports
# Usage: python check_models.py
# Then copy the correct model name into app.py

import os

api_key = input("Paste your Gemini API key: ").strip()

try:
    import google.generativeai as genai
    print(f"\nSDK version: {genai.__version__}")
    
    genai.configure(api_key=api_key)
    
    print("\n✅ Available models that support generateContent:\n")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  ✅ {m.name}")
    
    print("\n📋 ALL models (including other methods):\n")
    for m in genai.list_models():
        print(f"  {m.name}  →  {m.supported_generation_methods}")

except Exception as e:
    print(f"\n❌ Error: {e}")
