# setup.py
import os
import gdown

FOLDER_ID = "1TBzhOOahTPxs9pCIE91_k5MB5G0FQmmp"

def download_models():
    os.makedirs("models", exist_ok=True)

    if os.path.exists("models/crop_model.pkl"):
        print("Models already present - skipping download")
        return

    print("Downloading models from Google Drive folder...")

    try:
        # output="." downloads into current directory/models/ correctly
        gdown.download_folder(
            id=FOLDER_ID,
            output=".",
            quiet=False,
            use_cookies=False
        )
        print("Download complete!")

    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_models()
