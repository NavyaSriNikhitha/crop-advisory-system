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
        gdown.download_folder(
            id=FOLDER_ID,
            output=".",
            quiet=False,
            use_cookies=False
        )

        # Fix double folder if it happened
        if os.path.exists("models/models"):
            import shutil
            for f in os.listdir("models/models"):
                shutil.move(f"models/models/{f}", f"models/{f}")
            os.rmdir("models/models")

        print("Download complete!")

    except Exception as e:
        print(f"Download failed: {e}")

def build_knowledge_base():
    """Build ChromaDB knowledge base if not already built."""
    if os.path.exists("vector_db"):
        print("Knowledge base already exists - skipping build")
        return

    print("Building knowledge base...")
    try:
        import knowledge_base
        print("Knowledge base built successfully!")
    except Exception as e:
        print(f"Knowledge base build failed: {e}")

if __name__ == "__main__":
    download_models()
    build_knowledge_base()
