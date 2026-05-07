# setup.py
# Downloads all ML model files from Google Drive to models/ folder
# Runs automatically when Streamlit Cloud starts the app
# Run locally with: python setup.py

import os
import gdown

# Each filename mapped to its Google Drive file ID
MODEL_FILES = {
    "crop_model.pkl":                "1ch3xWAEK2F7OMGJvzslvZLeftgfctNKX",
    "fertilizer_crop_encoder.pkl":   "1ujo2o9djgWiHZBogsd7q2bCrDsDnsMOt",
    "fertilizer_label_encoder.pkl":  "1R7ytshN1Qp5sfos3L8N52V6oqL0PIw1v",
    "fertilizer_model.pkl":          "1McxiWjbr-tephOYWa39XB0TP8H58n_aW",
    "fertilizer_model_results.pkl":  "1CFflRGhYy13LxruVEi2oVfEPZh9rWxTc",
    "fertilizer_scaler.pkl":         "1uBbFlvIOI9bpXFWtQkq_KV-qc03y7ccI",
    "fertilizer_soil_encoder.pkl":   "1t40J_wqptlj9VPUwk1TVzwK-W0OXemCC",
    "label_encoder.pkl":             "19Q9ayBsRSDV9C42aVVxoUApuinsdgZE8",
    "model_results.pkl":             "1Z-W-89-O4EJ3AO0NBX9dUFWe-cAujqaw",
    "scaler.pkl":                    "1irsqxYUp9UZluYHANQxcUCX8YTSt6oGS",
    "yield_crops.pkl":               "1UyG-iGy9GCbEE0-LRx1rT-wFWhMwg2mZ",
    "yield_crop_encoder.pkl":        "1yVTfRyKMPrfmfwmIG05DPtVnPrgL8qhI",
    "yield_model.pkl":               "1Dd0XFccrj2e9ka2twQ36WkZGON1kXQpE",
    "yield_model_results.pkl":       "1vj-2REYDc4uImhJhpV0qsatr2emf0Gwn",
    "yield_scaler.pkl":              "1W4ErIoDhxT4tNmdNo_HAHMBe8kkAeQ3Q",
    "yield_seasons.pkl":             "1uwutKFZpcJldHGhwCUZGZ9nbd0kXz4LJ",
    "yield_season_encoder.pkl":      "13x9r9XnaHjoj086VtkOOf7aZjJwZhx5j",
    "yield_states.pkl":              "18FD6nTpTP7LR9lKAdMP_UQFRXpbVlTCu",
    "yield_state_encoder.pkl":       "1mlWLnmjxikAdFxm5RSHyDeerPmEabjcE",
}

def download_models():
    """Downloads all model files from Google Drive if not present locally."""

    os.makedirs("models", exist_ok=True)

    # Check if models already downloaded
    if os.path.exists("models/crop_model.pkl"):
        print("✅ Models already present — skipping download")
        return

    print("📥 Downloading models from Google Drive...")
    print(f"Total files: {len(MODEL_FILES)}")

    failed = []

    for filename, file_id in MODEL_FILES.items():
        output_path = f"models/{filename}"
        url = f"https://drive.google.com/uc?id={file_id}"

        try:
            print(f"  Downloading {filename}...")
            gdown.download(url, output_path, quiet=True)

            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  ✅ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {filename} — file not created")
                failed.append(filename)

        except Exception as e:
            print(f"  ❌ {filename} failed: {e}")
            failed.append(filename)

    if failed:
        print(f"\n❌ Failed to download: {failed}")
    else:
        print(f"\n✅ All {len(MODEL_FILES)} models downloaded successfully!")

if __name__ == "__main__":
    download_models()
