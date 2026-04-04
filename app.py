import os
import json
import base64
import random
import sqlite3
import logging
import traceback
import requests
from collections import deque
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'abiart-secret-key-2024-change-me')

# Session permanence — keeps user logged in for 30 days if they don't explicitly log out
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# =============================
# IN-MEMORY LOG BUFFER
# =============================
_log_buffer = deque(maxlen=200)  # Keep last 200 log entries

def log(level, message, extra=None):
    """Add entry to in-memory log buffer and print to stdout"""
    entry = {
        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'level': level,
        'message': message,
    }
    if extra:
        entry['extra'] = str(extra)[:500]
    _log_buffer.append(entry)
    prefix = {'INFO': '[INFO]', 'WARN': '[WARN]', 'ERROR': '[ERROR]'}.get(level, '[LOG]')
    print(f"{prefix} {entry['time']} {message}" + (f" | {extra}" if extra else ''))

# ===============================
# DATABASE & AUTH
# ===============================
DATABASE_URL = os.environ.get('DATABASE_URL', '')
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
    print(f"[DB] Using PostgreSQL")
else:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abiart.db')
    print(f"[DB] Using SQLite at {DB_PATH}")


def get_db():
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def db_execute(conn, query, params=None):
    """Execute a query with the correct placeholder syntax"""
    if USE_POSTGRES:
        # Convert ? placeholders to %s for PostgreSQL
        query = query.replace('?', '%s')
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        cur = conn.cursor()
    cur.execute(query, params or ())
    return cur


def db_fetchone(conn, query, params=None):
    cur = db_execute(conn, query, params)
    row = cur.fetchone()
    cur.close()
    return row


def db_fetchall(conn, query, params=None):
    cur = db_execute(conn, query, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def init_db():
    conn = get_db()
    if USE_POSTGRES:
        db_execute(conn, '''CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            images_remaining INTEGER DEFAULT 3,
            is_admin BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    else:
        db_execute(conn, '''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            images_remaining INTEGER DEFAULT 3,
            is_admin BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    conn.commit()

    # Seed admin account
    admin_user = os.environ.get('ADMIN_USERNAME', 'malowvx')
    admin_pass = os.environ.get('ADMIN_PASSWORD', 'zaza1140')
    existing = db_fetchone(conn, 'SELECT id FROM users WHERE username = ?', (admin_user,))
    if not existing:
        db_execute(conn,
            'INSERT INTO users (username, password_hash, images_remaining, is_admin) VALUES (?, ?, ?, ?)',
            (admin_user, generate_password_hash(admin_pass), -1, True)
        )
        conn.commit()
        print(f"[Auth] Admin account '{admin_user}' created")
    else:
        # Ensure admin always has admin flag and unlimited credits
        db_execute(conn, 'UPDATE users SET is_admin = TRUE, images_remaining = -1 WHERE username = ?', (admin_user,))
        conn.commit()
    conn.close()


def get_current_user():
    """Get current logged-in user from session"""
    if 'user_id' not in session:
        return None
    conn = get_db()
    user = db_fetchone(conn, 'SELECT * FROM users WHERE id = ?', (session['user_id'],))
    conn.close()
    return user


def check_credits(image_count=1):
    """Check if user has enough credits. Returns (user, error_response) or (user, None)"""
    user = get_current_user()
    if not user:
        return None, (jsonify({'error': 'Please log in to generate images'}), 401)
    if user['is_admin']:
        return user, None
    if user['images_remaining'] < image_count:
        return None, (jsonify({
            'error': f'Not enough credits. You have {user["images_remaining"]} image(s) remaining, but this request needs {image_count}.'
        }), 403)
    return user, None


def deduct_credits(user_id, image_count):
    """Deduct credits after successful generation"""
    conn = get_db()
    user = db_fetchone(conn, 'SELECT * FROM users WHERE id = ?', (user_id,))
    if user and not user['is_admin']:
        new_remaining = max(0, user['images_remaining'] - image_count)
        db_execute(conn, 'UPDATE users SET images_remaining = ? WHERE id = ?', (new_remaining, user_id))
        conn.commit()
        print(f"[Auth] User '{user['username']}' used {image_count} credits, {new_remaining} remaining")
    conn.close()


# Initialize database on startup
init_db()

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


# ===============================
# AUTH ROUTES
# ===============================

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if len(password) < 4:
        return jsonify({'error': 'Password must be at least 4 characters'}), 400

    conn = get_db()
    existing = db_fetchone(conn, 'SELECT id FROM users WHERE username = ?', (username,))
    if existing:
        conn.close()
        return jsonify({'error': 'Username already taken'}), 400

    db_execute(conn, 'INSERT INTO users (username, password_hash) VALUES (?, ?)',
                 (username, generate_password_hash(password)))
    conn.commit()
    user = db_fetchone(conn, 'SELECT * FROM users WHERE username = ?', (username,))
    conn.close()

    session['user_id'] = user['id']
    session['username'] = user['username']
    session.permanent = True  # Persist session for 30 days
    log('INFO', f"New registration: '{username}'")

    return jsonify({
        'success': True,
        'user': {
            'username': user['username'],
            'images_remaining': user['images_remaining'],
            'is_admin': bool(user['is_admin'])
        }
    })


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')

    conn = get_db()
    user = db_fetchone(conn, 'SELECT * FROM users WHERE username = ?', (username,))
    conn.close()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid username or password'}), 401

    session['user_id'] = user['id']
    session['username'] = user['username']
    session.permanent = True  # Persist session for 30 days
    log('INFO', f"Login: '{username}' from {request.remote_addr}")

    return jsonify({
        'success': True,
        'user': {
            'username': user['username'],
            'images_remaining': user['images_remaining'],
            'is_admin': bool(user['is_admin'])
        }
    })


@app.route('/api/auth/me', methods=['GET'])
def get_me():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 401
    return jsonify({
        'user': {
            'username': user['username'],
            'images_remaining': user['images_remaining'],
            'is_admin': bool(user['is_admin'])
        }
    })


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/generate', methods=['POST'])
def generate_hf():
    """Generate image using Hugging Face API"""
    try:
        # Credit check (HF generates 1 image)
        user, err = check_credits(1)
        if err:
            return err

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
        deduct_credits(user['id'], 1)
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
        # Credit check (Kaggle generates 1 image)
        user, err = check_credits(1)
        if err:
            return err

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
                deduct_credits(user['id'], 1)
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
        batch_count = data.get('batch_count', 1)

        # Credit check (Modal counts per image in batch)
        user, err = check_credits(batch_count)
        if err:
            return err

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
        log('INFO', f"Modal generate: status={response.status_code}, user={session.get('username','?')}")

        if response.status_code == 200:
            result = response.json()
            if 'image' in result:
                # Count actual images generated
                actual_images = len(result.get('images', [result['image']]))
                deduct_credits(user['id'], actual_images)
                resp = {
                    'success': True,
                    'image': result['image'],
                    'info': result.get('info', {})
                }
                if 'images' in result:
                    resp['images'] = result['images']
                # Include updated credits in response
                updated_user = get_current_user()
                if updated_user:
                    resp['credits'] = {
                        'images_remaining': updated_user['images_remaining'],
                        'is_admin': bool(updated_user['is_admin'])
                    }
                return jsonify(resp)
            else:
                log('WARN', 'Modal response missing image field', response.text[:200])
                return jsonify({'error': 'No image in response'}), 500
        else:
            error_text = response.text[:500] if response.text else 'Unknown error'
            log('ERROR', f"Modal HTTP {response.status_code}", error_text)
            return jsonify({'error': f'Modal error: {error_text}'}), 500

    except requests.exceptions.Timeout:
        log('ERROR', 'Modal generation timed out', f"user={session.get('username','?')}")
        return jsonify({'error': 'Generation timed out - Modal may be cold starting, try again in ~30s'}), 504
    except Exception as e:
        tb = traceback.format_exc()
        log('ERROR', f'Modal generate exception: {str(e)}', tb[:300])
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


# ===============================
# TOOLS ROUTES
# ===============================
_wd_tagger_client = None


@app.route('/api/tools/tag-image', methods=['POST'])
def tool_tag_image():
    """Tag an image with Danbooru-style tags using WD Tagger"""
    import tempfile

    data = request.json
    image_b64 = data.get('image', '')
    threshold = float(data.get('threshold', 0.35))

    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400

    tmp_path = None
    try:
        from gradio_client import Client, handle_file

        global _wd_tagger_client
        if _wd_tagger_client is None:
            print("[Tools] Initializing WD Tagger client...")
            _wd_tagger_client = Client("SmilingWolf/wd-tagger")

        # Save to temp file
        image_bytes = base64.b64decode(image_b64)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(image_bytes)
        tmp.close()
        tmp_path = tmp.name

        result = _wd_tagger_client.predict(
            image=handle_file(tmp_path),
            model_repo="SmilingWolf/wd-swinv2-tagger-v3",
            general_thresh=threshold,
            general_mcut_enabled=False,
            character_thresh=0.85,
            character_mcut_enabled=False,
            api_name="/predict"
        )

        # Parse result — typically (tags_string, rating_html, char_html, general_html)
        tags_string = ''
        if isinstance(result, (list, tuple)) and len(result) > 0:
            tags_string = str(result[0]) if result[0] else ''
        elif isinstance(result, str):
            tags_string = result

        tags = [t.strip() for t in tags_string.split(',') if t.strip()]
        print(f"[Tools] Tagged image: {len(tags)} tags")

        return jsonify({'success': True, 'tags': tags, 'tags_string': ', '.join(tags)})

    except ImportError:
        return jsonify({'error': 'Tagger not available (gradio_client not installed)'}), 500
    except Exception as e:
        print(f"[Tools] Tagger error: {str(e)}")
        return jsonify({'error': f'Tagging failed: {str(e)}'}), 500
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.route('/api/tools/char-to-prompt', methods=['POST'])
def tool_char_to_prompt():
    """Convert character description to SD prompt using GLM 4.7"""
    data = request.json
    description = data.get('description', '').strip()
    style = data.get('style', 'anime')  # anime, realistic, fantasy

    if not description:
        return jsonify({'error': 'No description provided'}), 400

    api_endpoint = os.environ.get('GLM_API_URL', 'https://openai-nim-proxy-production-9eb6.up.railway.app/V1/chat/completions')
    glm_key = os.environ.get('GLM_API_KEY', '')

    style_hints = {
        'anime': 'Use Danbooru/booru-style tags. Include anime-specific quality tags like masterpiece, best quality, highres.',
        'realistic': 'Use tags suited for photorealistic models. Include tags like photo, realistic, detailed skin, professional lighting.',
        'fantasy': 'Use tags for fantasy illustration style. Include tags like fantasy, dramatic lighting, epic composition, painted style.'
    }

    system_prompt = f"""You are an expert Stable Diffusion prompt engineer. Convert the user's character description into an optimized comma-separated tag prompt for image generation.

Rules:
1. Output ONLY comma-separated tags, nothing else. No explanations, no markdown.
2. {style_hints.get(style, style_hints['anime'])}
3. Start with subject count (1girl, 1boy, 2girls, etc.)
4. Include: hair (color, style, length), eyes (color), expression, clothing, pose, background, lighting
5. Add quality boosters at the end: masterpiece, best quality, highres, absurdres
6. Use underscores for multi-word tags (e.g. blue_hair, school_uniform)
7. Include a negative prompt on a second line starting with "NEGATIVE:" with common bad tags
8. Keep the prompt between 20-50 tags
9. Be specific and descriptive based on what the user describes"""

    try:
        headers = {'Content-Type': 'application/json'}
        if glm_key:
            headers['Authorization'] = f'Bearer {glm_key}'

        payload = {
            'model': data.get('model', 'glm-4'),
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': description}
            ],
            'temperature': 0.7,
            'max_tokens': 500
        }

        resp = requests.post(api_endpoint, json=payload, headers=headers, timeout=30)

        if resp.status_code != 200:
            print(f"[Tools] GLM error: {resp.status_code} {resp.text[:200]}")
            return jsonify({'error': f'GLM API error: {resp.status_code}'}), 500

        result = resp.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Parse positive and negative prompts
        lines = content.strip().split('\n')
        positive = ''
        negative = ''
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('NEGATIVE:'):
                negative = stripped[9:].strip()
            elif stripped and not positive:
                positive = stripped
            elif stripped and positive and not negative:
                # Could be continuation of positive
                positive += ', ' + stripped

        print(f"[Tools] Generated prompt from description ({len(positive)} chars)")
        return jsonify({
            'success': True,
            'prompt': positive,
            'negative': negative
        })

    except requests.exceptions.Timeout:
        return jsonify({'error': 'GLM API timed out'}), 504
    except Exception as e:
        print(f"[Tools] Char-to-prompt error: {str(e)}")
        return jsonify({'error': str(e)}), 500



# ===============================
# ADMIN LOGS
# ===============================

@app.route('/admin/logs')
def admin_logs_view():
    """Admin-only HTML page showing recent log entries"""
    user = get_current_user()
    if not user or not user['is_admin']:
        return '<h1>403 Forbidden</h1>', 403

    logs = list(reversed(_log_buffer))
    level_colors = {'INFO': '#4ade80', 'WARN': '#facc15', 'ERROR': '#f87171'}

    rows = ''
    for e in logs:
        color = level_colors.get(e['level'], '#aaa')
        extra_html = f'<div style="font-size:.7rem;opacity:.6;margin-top:.2rem">{e.get("extra","")}</div>' if e.get('extra') else ''
        rows += f'''<tr>
            <td style="color:#888;white-space:nowrap;padding:.3rem .6rem">{e["time"]}</td>
            <td style="color:{color};font-weight:700;padding:.3rem .6rem">{e["level"]}</td>
            <td style="padding:.3rem .6rem">{e["message"]}{extra_html}</td>
        </tr>'''

    html = f'''<!DOCTYPE html>
<html>
<head>
<title>AbiArt Logs</title>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<style>
  body {{ background:#0a0a0f; color:#e0e0e0; font-family:monospace; margin:0; padding:1rem; }}
  h1 {{ color:#8b5cf6; font-size:1.2rem; margin-bottom:.5rem; }}
  p {{ color:#666; font-size:.75rem; margin-bottom:1rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.78rem; }}
  tr:nth-child(even) {{ background:rgba(255,255,255,.03); }}
  tr:hover {{ background:rgba(139,92,246,.08); }}
  a {{ color:#8b5cf6; text-decoration:none; }}
  .badge {{ background:#1a1a2e; border:1px solid #333; padding:.15rem .4rem; border-radius:3px; font-size:.65rem; }}
</style>
</head>
<body>
<h1>🔍 AbiArt — Server Logs</h1>
<p>Last {len(logs)} entries (max 200) · Auto-refreshes every 15s · <a href="/admin/logs/json">JSON</a></p>
<table>
<thead><tr>
  <th style="text-align:left;color:#555;padding:.3rem .6rem">Time (UTC)</th>
  <th style="text-align:left;color:#555;padding:.3rem .6rem">Level</th>
  <th style="text-align:left;color:#555;padding:.3rem .6rem">Message</th>
</tr></thead>
<tbody>{rows if rows else '<tr><td colspan="3" style="color:#555;padding:1rem">No logs yet</td></tr>'}</tbody>
</table>
</body>
</html>'''
    return html


@app.route('/admin/logs/json')
def admin_logs_json():
    """Admin-only JSON endpoint returning raw log buffer"""
    user = get_current_user()
    if not user or not user['is_admin']:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify({'logs': list(reversed(_log_buffer)), 'count': len(_log_buffer)})


# ===============================
# GLOBAL ERROR HANDLERS
# ===============================

@app.errorhandler(401)
def handle_401(e):
    log('WARN', f'401 Unauthorized: {request.path}', request.remote_addr)
    return jsonify({'error': 'Authentication required'}), 401


@app.errorhandler(404)
def handle_404(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def handle_500(e):
    log('ERROR', f'500 Internal server error: {request.path}', str(e))
    return jsonify({'error': 'Internal server error'}), 500


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
