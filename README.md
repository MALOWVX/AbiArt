# 🎨 AbiArt — AI Image Studio

A web-based AI image generation studio that supports multiple backends: **Hugging Face**, **Kaggle**, and **Modal** (with Automatic1111 WebUI + Civitai models).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)
![License](https://img.shields.io/badge/License-Private-red)

## ✨ Features

- **Multi-Backend Support**
  - 🤗 **Hugging Face** — Simple API-based generation (SDXL, SD 1.5, DreamShaper)
  - 📊 **Kaggle** — Connect to your own Kaggle notebook running SD WebUI
  - ⚡ **Modal** — Cloud GPU backend with Automatic1111 WebUI (full feature set)

- **Advanced Generation Options**
  - Custom models from Civitai (checkpoints, LoRAs, VAEs)
  - Multiple samplers (DPM++ 2M Karras, Euler, etc.)
  - ADetailer for automatic face/body refinement
  - Hi-Res Fix for upscaled output
  - Batch generation (up to 4 images)
  - Seed control for reproducibility

- **User Experience**
  - Preset system to save/load configurations
  - Image history with thumbnails
  - Trigger word management for LoRAs
  - One-click settings recycling
  - Responsive dark-mode interface

## 🏗 Architecture

```
YOUR COMPUTER                    MODAL CLOUD
+-----------+                    +---------------------+
|  Browser  |                    | StableDiffusion     |
|     |     |        HTTP POST   | (modal_sd.py)       |
|  Flask App| ────────────────→  | GPU: L4             |
|  app.py   |                    |   |           |     |
|  :5000    |                    |   v           v     |
+-----------+                    | Volume     A1111    |
     |                           | sd-models  WebUI    |
     |  Search                   +---------------------+
     v                                  |
+-----------+                           | Download
| Civitai   | ←————————————————————————+
| API       |
+-----------+
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/AbiArt.git
cd AbiArt
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# Your Modal endpoint URLs (from `modal serve modal_sd.py`)
MODAL_GENERATE_URL=https://your-username--stable-diffusion-api-...
MODAL_MODELS_URL=https://your-username--stable-diffusion-api-...
MODAL_LOAD_URL=https://your-username--stable-diffusion-api-...
MODAL_DELETE_URL=https://your-username--stable-diffusion-api-...

# Optional: Civitai token for model downloads
CIVITAI_TOKEN=your_token_here
```

### 3. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## ⚡ Modal Backend Setup

The Modal backend runs Automatic1111 WebUI on a cloud GPU. You need a [Modal](https://modal.com) account.

### Install Modal

```bash
pip install modal
modal setup
```

### Deploy (Development)

```bash
modal serve modal_sd.py
```

This gives you temporary endpoint URLs. Copy them to your `.env` file.

### Deploy (Production)

```bash
modal deploy modal_sd.py
```

This gives you permanent endpoint URLs.

### Download Models

Use the built-in Civitai browser in the app, or the utility script:

```bash
# Edit MODEL_ID and FILENAME in load_model.py, then:
python load_model.py
```

## 🐳 Docker Deployment

```bash
# Build
docker build -t abiart .

# Run
docker run -p 5000:5000 --env-file .env abiart
```

## ☁️ Cloud Deployment

### Render

1. Push to GitHub
2. Connect your repo on [render.com](https://render.com)
3. Set environment variables in the dashboard
4. Deploy — Render auto-detects the `Procfile`

### Railway

1. Push to GitHub
2. Connect your repo on [railway.app](https://railway.app)
3. Set environment variables
4. Deploy — Railway auto-detects the `Procfile`

## 📁 Project Structure

```
AbiArt/
├── app.py              # Flask backend (routes & API proxy)
├── modal_sd.py         # Modal GPU backend (Automatic1111)
├── load_model.py       # Utility: download models to Modal
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container deployment
├── Procfile            # PaaS deployment (Render/Railway)
├── .env.example        # Environment variables template
├── .gitignore          # Git exclusions
├── static/
│   └── style.css       # Application styles
└── templates/
    └── index.html      # Frontend UI
```

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `MODAL_GENERATE_URL` | Modal endpoint for image generation | Yes (for Modal) |
| `MODAL_MODELS_URL` | Modal endpoint for listing models | Yes (for Modal) |
| `MODAL_LOAD_URL` | Modal endpoint for downloading models | Yes (for Modal) |
| `MODAL_DELETE_URL` | Modal endpoint for deleting models | Yes (for Modal) |
| `CIVITAI_TOKEN` | Civitai API token for model downloads | Optional |
| `FLASK_PORT` | Server port (default: 5000) | No |
| `FLASK_DEBUG` | Enable debug mode (default: false) | No |

## 📝 Usage Tips

- **Hugging Face mode**: Get a free API token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **LoRA trigger words**: The app automatically fetches trigger words from Civitai when you add a LoRA
- **Presets**: Save your favorite prompt + settings combinations for quick reuse
- **Seed recycling**: Click on a seed in the info bar to reuse it, or use the ♻ Recycle button

---

Made with ❤️ by AbiArt
