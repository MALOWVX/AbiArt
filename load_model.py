"""
Load a Civitai model into Modal
Run: python load_model.py
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =====================
# CONFIGURE THESE:
# =====================
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "")
MODEL_ID = "1116447"  # The model version ID from Civitai URL
FILENAME = "noobai_xl.safetensors"  # What to name the file

# Your Modal endpoint (check your terminal when running modal serve)
MODAL_URL = os.environ.get(
    "MODAL_LOAD_URL",
    "https://YOUR_USERNAME--stable-diffusion-api-stablediffusion-load-civitai-model-dev.modal.run"
)

# =====================


def main():
    if not CIVITAI_TOKEN:
        print("WARNING: No CIVITAI_TOKEN set. Add it to your .env file.")
        print("  Get your token at: https://civitai.com/user/account")
        return

    download_url = f"https://civitai.com/api/download/models/{MODEL_ID}?token={CIVITAI_TOKEN}"
    
    print(f"Sending request to Modal to download: {FILENAME}")
    print("This may take several minutes for large models...")
    
    try:
        response = requests.post(
            MODAL_URL,
            json={
                "url": download_url,
                "filename": FILENAME
            },
            timeout=600  # 10 minute timeout for large files
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
    except requests.exceptions.Timeout:
        print("Request timed out - the model might still be downloading in the background")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
