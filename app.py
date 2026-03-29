"""
AbiArt — AI Image Studio
Stable Diffusion Web Interface
Supports: Hugging Face API + Kaggle Backend + Modal with Civitai Models
"""

import os
import json
import base64
import random
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# Hugging Face Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "dreamshaper": "Lykon/dreamshaper-8",
}

# Modal Backend URLs (loaded from environment)
MODAL_GENERATE_URL = os.environ.get(
    "MODAL_GENERATE_URL",
    "https://YOUR_USERNAME--stable-diffusion-api-stablediffusion-api-generate-dev.modal.run"
)
MODAL_MODELS_URL = os.environ.get(
    "MODAL_MODELS_URL",
    "https://YOUR_USERNAME--stable-diffusion-api-stablediffusion-list-models-dev.modal.run"
)
MODAL_LOAD_URL = os.environ.get(
    "MODAL_LOAD_URL",
    "https://YOUR_USERNAME--stable-diffusion-api-stablediffusion-load-civitai-model-dev.modal.run"
)
MODAL_DELETE_URL = os.environ.get(
    "MODAL_DELETE_URL",
    "https://YOUR_USERNAME--stable-diffusion-api-stablediffusion-delete-model-dev.modal.run"
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_hf():
    """Generate image using Hugging Face API"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        model_key = data.get('model', 'sdxl')
        guidance_scale = data.get('guidance_scale', 7.5)
        api_key = data.get('api_key', '')

        if not prompt:
            return jsonify({'error': 'Please provide a prompt'}), 400
        if not api_key:
            return jsonify({'error': 'API key required'}), 400

        model_id = HF_MODELS.get(model_key, HF_MODELS['sdxl'])
        api_url = f"{HF_API_URL}{model_id}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": guidance_scale,
            }
        }
        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=120)

        if response.status_code == 503:
            return jsonify({'error': 'Model is loading, try again in a few seconds'}), 503

        if response.status_code != 200:
            try:
                err = response.json().get('error', 'Generation failed')
            except Exception:
                err = 'Generation failed'
            return jsonify({'error': err}), response.status_code

        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return jsonify({'success': True, 'image': image_base64})

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kaggle/test', methods=['POST'])
def test_kaggle():
    """Test connection to Kaggle backend"""
    try:
        data = request.json
        url = data.get('url', '').rstrip('/')

        response = requests.get(f"{url}/sdapi/v1/options", timeout=10)
        
        if response.status_code == 200:
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Could not connect to SD WebUI'}), 400

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Connection timed out'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kaggle/models', methods=['POST'])
def get_kaggle_models():
    """Get available models from Kaggle backend"""
    try:
        data = request.json
        url = data.get('url', '').rstrip('/')

        response = requests.get(f"{url}/sdapi/v1/sd-models", timeout=30)
        
        if response.status_code == 200:
            models = response.json()
            return jsonify({'models': models})
        else:
            return jsonify({'error': 'Could not fetch models'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kaggle/generate', methods=['POST'])
def generate_kaggle():
    """Generate image using Kaggle backend"""
    try:
        data = request.json
        url = data.get('url', '').rstrip('/')
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        model = data.get('model', '')
        cfg_scale = data.get('cfg_scale', 7)
        steps = data.get('steps', 25)
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        loras = data.get('loras', [])

        if not prompt:
            return jsonify({'error': 'Please provide a prompt'}), 400

        # Add LoRA syntax to prompt if any active
        full_prompt = prompt
        for lora in loras:
            full_prompt = f"<lora:{lora['name']}:{lora['weight']}> " + full_prompt

        # Change model if specified
        if model:
            print(f"[Kaggle] Switching to model: {model}")
            try:
                import time
                model_resp = requests.post(f"{url}/sdapi/v1/options", json={
                    "sd_model_checkpoint": model
                }, timeout=120)
                print(f"[Kaggle] Model switch response: {model_resp.status_code}")
                if model_resp.status_code == 200:
                    time.sleep(2)
            except Exception as e:
                print(f"[Kaggle] Model switch error: {e}")

        payload = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "width": width,
            "height": height,
            "sampler_name": "Euler",
            "scheduler": "Normal",
            "seed": -1,
        }

        print(f"[Kaggle] Generating with payload: {payload}")
        response = requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=300)
        print(f"[Kaggle] Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'images' in result and len(result['images']) > 0:
                return jsonify({
                    'success': True,
                    'image': result['images'][0]
                })
            else:
                print(f"[Kaggle] No images in response: {result}")
                return jsonify({'error': 'No image generated - check model loading'}), 500
        else:
            error_text = response.text[:500] if response.text else 'Unknown error'
            print(f"[Kaggle] Generation failed: {error_text}")
            return jsonify({'error': f'Generation failed: {error_text}'}), 500

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Generation timed out (model may still be loading - try again)'}), 504
    except Exception as e:
        print(f"[Kaggle] Exception: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/civitai/search', methods=['GET'])
def search_civitai():
    """Search Civitai for models"""
    try:
        query = request.args.get('query', '')
        model_type = request.args.get('type', '')

        params = {
            'limit': 20,
            'sort': 'Highest Rated',
        }
        if query:
            params['query'] = query

        response = requests.get(
            'https://civitai.com/api/v1/models',
            params=params,
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            models = []

            for item in data.get('items', []):
                item_type = item.get('type', '')
                
                if model_type:
                    if model_type == 'Checkpoint' and item_type != 'Checkpoint':
                        continue
                    if model_type == 'LORA' and item_type != 'LORA':
                        continue
                
                version = item.get('modelVersions', [{}])[0]
                files = version.get('files', [{}])
                primary_file = files[0] if files else {}
                images = version.get('images', [{}])
                image_url = images[0].get('url', '') if images else ''

                models.append({
                    'id': item.get('id'),
                    'name': item.get('name'),
                    'type': item_type,
                    'version': version.get('name'),
                    'image': image_url,
                    'download_url': primary_file.get('downloadUrl', ''),
                    'filename': primary_file.get('name', 'model.safetensors'),
                    'trained_words': version.get('trainedWords', [])
                })

            return jsonify({'models': models})
        else:
            return jsonify({'error': 'Civitai search failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kaggle/download', methods=['POST'])
def download_to_kaggle():
    """Trigger model download on Kaggle backend"""
    try:
        data = request.json
        url = data.get('url', '').rstrip('/')
        model_url = data.get('model_url', '')
        filename = data.get('filename', '')
        model_type = data.get('type', 'checkpoint')

        return jsonify({
            'success': True,
            'message': f'To download, run in Kaggle notebook: download_civitai_model("{model_url}", "{filename}", "{model_type}")'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===============================
# MODAL BACKEND
# ===============================

@app.route('/api/modal/generate', methods=['POST'])
def generate_modal():
    """Generate image using Modal backend"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        cfg_scale = data.get('cfg_scale', 7)
        steps = data.get('steps', 25)
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        model = data.get('model', 'SDXL Base 1.0')
        loras = data.get('loras', [])
        scheduler = data.get('scheduler', 'Euler')
        try:
            seed = int(data.get('seed', -1))
        except (ValueError, TypeError):
            seed = -1
            
        if seed == -1:
            seed = random.randint(1, 2147483647)
        clip_skip = data.get('clip_skip', 1)
        vae = data.get('vae', '')
        
        # ADetailer params
        adetailer_enabled = data.get('adetailer_enabled', False)
        adetailer_same_checkpoint = data.get('adetailer_same_checkpoint', True)
        adetailer_model = data.get('adetailer_model', 'face_yolov8n.pt')
        adetailer_confidence = data.get('adetailer_confidence', 0.3)
        adetailer_prompt = data.get('adetailer_prompt', '')
        adetailer_negative = data.get('adetailer_negative', '')
        adetailer_strength = data.get('adetailer_strength', 0.4)
        adetailer_steps = data.get('adetailer_steps', 25)
        batch_count = data.get('batch_count', 1)

        # Hi-Res Fix params
        enable_hr = data.get('enable_hr', False)
        hr_upscaler = data.get('hr_upscaler', 'R-ESRGAN 4x+')
        hr_scale = data.get('hr_scale', 1.5)
        hr_denoising_strength = data.get('hr_denoising_strength', 0.35)
        hr_second_pass_steps = data.get('hr_second_pass_steps', 15)

        if not prompt:
            return jsonify({'error': 'Please provide a prompt'}), 400

        print(f"[Modal] Generating with model '{model}': {prompt[:50]}...")
        print(f"[Modal] Scheduler: {scheduler}, Seed: {seed}, Clip skip: {clip_skip}")
        if loras:
            print(f"[Modal] LoRAs: {loras}")
        if adetailer_enabled:
            print(f"[Modal] ADetailer enabled")
        if enable_hr:
            print(f"[Modal] Hi-Res Fix: {hr_upscaler}, scale={hr_scale}, denoise={hr_denoising_strength}")
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "width": width,
            "height": height,
            "model": model,
            "loras": loras,
            "scheduler": scheduler,
            "seed": seed,
            "clip_skip": clip_skip,
            "vae": vae,
            "batch_count": batch_count,
            "adetailer_enabled": adetailer_enabled,
            "adetailer_same_checkpoint": adetailer_same_checkpoint,
            "adetailer_model": adetailer_model,
            "adetailer_confidence": adetailer_confidence,
            "adetailer_prompt": adetailer_prompt,
            "adetailer_negative": adetailer_negative,
            "adetailer_strength": adetailer_strength,
            "adetailer_steps": adetailer_steps,
            "enable_hr": enable_hr,
            "hr_upscaler": hr_upscaler,
            "hr_scale": hr_scale,
            "hr_denoising_strength": hr_denoising_strength,
            "hr_second_pass_steps": hr_second_pass_steps,
        }

        timeout = 300
        if batch_count > 1: timeout = 600
        if enable_hr: timeout = 600
        response = requests.post(MODAL_GENERATE_URL, json=payload, timeout=timeout)
        print(f"[Modal] Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'image' in result:
                resp = {
                    'success': True,
                    'image': result['image'],
                    'info': result.get('info', {})
                }
                if 'images' in result:
                    resp['images'] = result['images']
                return jsonify(resp)
            else:
                return jsonify({'error': 'No image in response'}), 500
        else:
            error_text = response.text[:500] if response.text else 'Unknown error'
            return jsonify({'error': f'Modal error: {error_text}'}), 500

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Generation timed out - Modal may be cold starting, try again'}), 504
    except Exception as e:
        print(f"[Modal] Exception: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modal/models', methods=['GET'])
def get_modal_models():
    """Get list of models and LoRAs available on Modal"""
    try:
        print(f"[Modal] Fetching models from: {MODAL_MODELS_URL}")
        response = requests.get(MODAL_MODELS_URL, timeout=120)
        print(f"[Modal] Response status: {response.status_code}")
        print(f"[Modal] Response body: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                'models': data.get('models', ['SDXL Base 1.0']),
                'loras': data.get('loras', []),
                'vaes': data.get('vaes', ['Automatic'])
            })
        else:
            print(f"[Modal] Non-200 response: {response.status_code} - {response.text[:200]}")
            return jsonify({'models': ['SDXL Base 1.0'], 'loras': [], 'vaes': ['Automatic']})
    except Exception as e:
        print(f"[Modal] Error getting models: {e}")
        return jsonify({'models': ['SDXL Base 1.0'], 'loras': [], 'vaes': ['Automatic']})


@app.route('/api/modal/download', methods=['POST'])
def download_to_modal():
    """Download a model or LoRA to Modal backend"""
    try:
        data = request.json
        model_url = data.get('model_url', '')
        filename = data.get('filename', 'model.safetensors')
        is_lora = data.get('is_lora', False)
        is_vae = data.get('is_vae', False)

        if not model_url:
            return jsonify({'error': 'No model URL provided'}), 400

        civitai_token = os.environ.get('CIVITAI_TOKEN', '')
        if civitai_token and 'token=' not in model_url:
            model_url = f"{model_url}?token={civitai_token}"

        file_type = 'VAE' if is_vae else ('LoRA' if is_lora else 'checkpoint')
        print(f"[Modal] Downloading {file_type}: {filename}...")
        
        response = requests.post(
            MODAL_LOAD_URL,
            json={
                "model_url": model_url,
                "filename": filename,
                "is_lora": is_lora,
                "is_vae": is_vae
            },
            timeout=600
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success') or result.get('model'):
                return jsonify({'success': True, 'model': filename})
            else:
                return jsonify({'error': result.get('error', 'Download failed')}), 500
        else:
            return jsonify({'error': f'Modal download failed: {response.status_code}'}), 500

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Download timed out - try again'}), 504
    except Exception as e:
        print(f"[Modal] Download exception: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modal/delete', methods=['POST'])
def delete_modal_file():
    """Delete a model or LoRA from Modal"""
    try:
        data = request.json
        filename = data.get('filename', '')
        file_type = data.get('type', 'checkpoint')

        if not filename:
            return jsonify({'error': 'No filename provided'}), 400

        print(f"[Modal] Deleting {file_type}: {filename}")
        
        response = requests.post(
            MODAL_DELETE_URL,
            json={
                "filename": filename,
                "type": file_type
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return jsonify({'success': True})
            else:
                return jsonify({'error': result.get('error', 'Delete failed')}), 500
        else:
            return jsonify({'error': f'Modal delete failed: {response.status_code}'}), 500

    except Exception as e:
        print(f"[Modal] Delete exception: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    print("\n" + "="*60)
    print("AbiArt — AI Image Studio")
    print("="*60)
    print(f"\n* Open http://localhost:{port} in your browser")
    print("* Hugging Face mode: Use HF API (simple, limited models)")
    print("* Kaggle mode: Use your Kaggle backend with Civitai models")
    print("* Modal mode: Use Modal cloud GPU backend")
    print("\n" + "="*60 + "\n")
    app.run(debug=debug, port=port)
