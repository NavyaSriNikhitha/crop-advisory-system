# setup.py
# Downloads ML model files from Google Drive to models/ folder
# Runs automatically on Streamlit Cloud before app starts
# Run locally with: python setup.py

import os
import gdown

# Google Drive folder ID extracted from shared link
# https://drive.google.com/drive/folders/1TBzhOOahTPxs9pCIE91_k5MB5G0FQmmp
FOLDER_ID = "1TBzhOOahTPxs9pCIE91_k5MB5G0FQmmp"

def download_models():
    """Downloads all model files from Google Drive if not present locally."""

    os.makedirs("models", exist_ok=True)

    # Check if models already exist — skip download if they do
    if os.path.exists("models/crop_model.pkl"):
        print("✅ Models already present — skipping download")
        return

    print("📥 Downloading models from Google Drive...")
    print("This may take 2-3 minutes on first run...")

    try:
        gdown.download_folder(
            id=FOLDER_ID,
            output="models/",
            quiet=False,
            use_cookies=False
        )
        print("✅ All models downloaded successfully!")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Please check the Google Drive folder is publicly accessible.")

if __name__ == "__main__":
    download_models()
