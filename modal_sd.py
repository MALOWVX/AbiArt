"""
Stable Diffusion on Modal using Automatic1111 WebUI API
Run with: modal serve modal_sd.py
"""

import modal

# Create the Modal app
app = modal.App("stable-diffusion-api")

A1111_DIR = "/webui"
MODEL_DIR = "/models"
CHECKPOINT_DIR = f"{MODEL_DIR}/checkpoints"
LORA_DIR = f"{MODEL_DIR}/loras"
VAE_DIR = f"{MODEL_DIR}/vae"

# Fix: Stability-AI/stablediffusion.git is now private.
# Use the public fork recommended by the A1111 community.
PUBLIC_SD_REPO = "https://github.com/w-e-w/stablediffusion.git"

# Build the container image with A1111 installed
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxrender1", "libxext6", "google-perftools",
        "libgoogle-perftools-dev",
    )
    # Install PyTorch first (matching A1111's expected version)
    .pip_install("torch==2.1.2", "torchvision==0.16.2", "torchaudio")
    # Clone A1111 WebUI
    .run_commands(
        f"git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git {A1111_DIR}",
    )
    # Run launch.py --exit with STABLE_DIFFUSION_REPO env var pointing to public fork.
    # This lets launch.py handle ALL dependency installation correctly:
    #   - Clones all 5 repos (using public fork for stablediffusion)
    #   - Installs requirements_versions.txt
    #   - Installs clip, open_clip
    .env({"STABLE_DIFFUSION_REPO": PUBLIC_SD_REPO})
    .run_commands(
        f"cd {A1111_DIR} && python launch.py --exit --skip-torch-cuda-test --no-download-sd-model",
    )
    # Install xformers matching PyTorch 2.1.2+cu121
    .pip_install("xformers==0.0.23.post1")
    # Install ADetailer and ControlNet extensions
    .run_commands(
        f"git clone --depth 1 https://github.com/Bing-su/adetailer.git {A1111_DIR}/extensions/adetailer",
        "pip install ultralytics mediapipe",
        f"git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet.git {A1111_DIR}/extensions/sd-webui-controlnet",
    )
    # Download ControlNet SDXL models
    .run_commands(
        f"mkdir -p {A1111_DIR}/extensions/sd-webui-controlnet/models",
        f"wget -q -O {A1111_DIR}/extensions/sd-webui-controlnet/models/ip-adapter_sdxl_vit-h.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sdxl_vit-h.safetensors",
        f"wget -q -O {A1111_DIR}/extensions/sd-webui-controlnet/models/sai_xl_canny_2x.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_2x.safetensors",
        f"wget -q -O {A1111_DIR}/extensions/sd-webui-controlnet/models/sai_xl_depth_2x.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_depth_2x.safetensors",
        f"wget -q -O {A1111_DIR}/extensions/sd-webui-controlnet/models/thibaud_xl_openpose_2e1b.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose_2e1b.safetensors",
    )
    # Download Clip Vision encoder for IP-Adapter
    .run_commands(
        f"mkdir -p {A1111_DIR}/models/clip_vision",
        f"wget -q -O {A1111_DIR}/models/clip_vision/clip_g.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
    )
    # Remove default model directories (will be symlinked to volume at runtime)
    .run_commands(
        f"rm -rf {A1111_DIR}/models/Stable-diffusion",
        f"rm -rf {A1111_DIR}/models/Lora",
        f"rm -rf {A1111_DIR}/models/VAE",
    )
    # CRITICAL: Force numpy 1.x as the VERY LAST pip command.
    # ultralytics/mediapipe pull numpy 2.x which breaks PyTorch 2.1.2 and scikit-image 0.21.0.
    # This must be the final install so nothing can override it.
    .run_commands(
        "pip install numpy==1.26.2",
    )
    # Download SDXL VAE for sharper colors and fewer artifacts
    .run_commands(
        f"mkdir -p {A1111_DIR}/models/VAE",
        f"wget -q -O {A1111_DIR}/models/VAE/sdxl_vae.safetensors "
        "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
    )
    .run_commands("echo a1111-v17-controlnet")  # Cache buster
)

# Volume to persist models between runs
volume = modal.Volume.from_name("sd-models", create_if_missing=True)


@app.cls(
    gpu="L4",
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=600,
    container_idle_timeout=300,
)
class StableDiffusion:
    def __init__(self):
        self.a1111_process = None
        self.a1111_url = "http://127.0.0.1:7860"
        self.current_model = None
        self.ready = False

    @modal.enter()
    def setup(self):
        """Start A1111 WebUI API server"""
        import os
        import subprocess
        import time
        import requests
        import threading

        # Reload volume to see previously downloaded models
        print("Reloading volume...")
        volume.reload()

        # Create model directories on volume if they don't exist
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(LORA_DIR, exist_ok=True)
        os.makedirs(VAE_DIR, exist_ok=True)

        # Symlink A1111 model directories to our volume
        a1111_models = f"{A1111_DIR}/models"
        for link_name, target in [
            ("Stable-diffusion", CHECKPOINT_DIR),
            ("Lora", LORA_DIR),
            ("VAE", VAE_DIR),
        ]:
            link_path = f"{a1111_models}/{link_name}"
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.remove(link_path) if os.path.islink(link_path) else None
                if os.path.isdir(link_path):
                    import shutil
                    shutil.rmtree(link_path)
            os.symlink(target, link_path)
            print(f"  Linked {link_path} -> {target}")

        # Log volume contents
        print("=== Volume contents ===")
        for subdir_name, subdir_path in [("Checkpoints", CHECKPOINT_DIR), ("LoRAs", LORA_DIR), ("VAEs", VAE_DIR)]:
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                print(f"  {subdir_name}: {files if files else '(empty)'}")
        print("=== End ===")

        # Start A1111 WebUI in API-only mode
        print("Starting A1111 WebUI API...")
        env = os.environ.copy()
        # Keep using public fork at runtime too (in case A1111 tries to fetch)
        env["STABLE_DIFFUSION_REPO"] = PUBLIC_SD_REPO

        self.a1111_process = subprocess.Popen(
            [
                "python", "launch.py",
                "--api", "--nowebui",
                "--xformers",
                "--listen", "--port", "7860",
                "--skip-install",
                "--skip-torch-cuda-test",
                "--no-download-sd-model",
                "--skip-version-check",
                "--no-hashing",
                "--opt-sdp-attention",
            ],
            cwd=A1111_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Stream A1111 output to Modal logs
        def stream_output():
            for line in iter(self.a1111_process.stdout.readline, b''):
                print(f"[A1111] {line.decode().rstrip()}")

        thread = threading.Thread(target=stream_output, daemon=True)
        thread.start()

        # Wait for A1111 to be ready (up to 180 seconds)
        print("Waiting for A1111 to start...")
        for i in range(90):
            try:
                r = requests.get(f"{self.a1111_url}/sdapi/v1/samplers", timeout=3)
                if r.status_code == 200:
                    print(f"A1111 is ready! (took ~{i*2}s)")
                    self.ready = True

                    # Get current model
                    try:
                        opts = requests.get(f"{self.a1111_url}/sdapi/v1/options", timeout=5).json()
                        self.current_model = opts.get("sd_model_checkpoint", "")
                        print(f"Current model: {self.current_model}")
                    except:
                        pass
                    return
            except:
                pass
            time.sleep(2)

        print("WARNING: A1111 did not start within 180 seconds!")

    def _extract_checkpoint_name(self, model_title: str) -> str:
        """Convert A1111 model title into a checkpoint filename when possible."""
        if not model_title:
            return ""
        if "[" in model_title:
            model_title = model_title.split("[", 1)[0].strip()
        return model_title

    def _is_probably_vpred_checkpoint(self, model_name: str) -> bool:
        """Best-effort classifier for v-pred checkpoints based on naming."""
        normalized = (model_name or "").lower()
        hints = ("vpred", "v_pred", "v-pred", "v prediction", "noobai")
        return any(hint in normalized for hint in hints)

    def _find_checkpoint_file(self, model_name: str):
        """Find checkpoint path in CHECKPOINT_DIR from model title/name."""
        import os
        from pathlib import Path

        if not model_name:
            return None

        base_name = self._extract_checkpoint_name(model_name)
        candidate = Path(CHECKPOINT_DIR) / base_name
        if candidate.exists():
            return candidate

        if not os.path.exists(CHECKPOINT_DIR):
            return None

        target_root = Path(base_name).stem.lower()
        for file_name in os.listdir(CHECKPOINT_DIR):
            if not file_name.endswith((".safetensors", ".ckpt")):
                continue
            stem = Path(file_name).stem.lower()
            if stem == target_root or file_name.lower() == base_name.lower():
                return Path(CHECKPOINT_DIR) / file_name
        return None

    def _ensure_vpred_yaml(self, model_name: str) -> bool:
        """Create <checkpoint>.yaml from sd_xl_v.yaml when a likely v-pred model is used."""
        import shutil
        from pathlib import Path

        if not self._is_probably_vpred_checkpoint(model_name):
            return False

        checkpoint_path = self._find_checkpoint_file(model_name)
        if checkpoint_path is None:
            print(f"[v-pred] Could not resolve checkpoint path for model: {model_name}")
            return False

        yaml_target = checkpoint_path.with_suffix(".yaml")
        if yaml_target.exists():
            return False

        template = Path(A1111_DIR) / "configs" / "sd_xl_v.yaml"
        if not template.exists():
            print(f"[v-pred] Missing template config: {template}")
            return False

        shutil.copyfile(template, yaml_target)
        print(f"[v-pred] Created config: {yaml_target.name}")
        return True

    def _post_txt2img(self, payload: dict, timeout: int):
        import requests
        return requests.post(f"{self.a1111_url}/sdapi/v1/txt2img", json=payload, timeout=timeout)

    def _is_nan_failure(self, response_text: str) -> bool:
        text = (response_text or "").lower()
        nan_markers = ("nansexception", "nan", "inf", "black image")
        return any(marker in text for marker in nan_markers)

    @modal.web_endpoint(method="GET")
    def health(self):
        """Health check endpoint"""
        import requests
        try:
            r = requests.get(f"{self.a1111_url}/sdapi/v1/sd-models", timeout=5)
            return {
                "status": "ready" if r.status_code == 200 else "error",
                "model": self.current_model or "none",
            }
        except:
            return {"status": "starting", "model": "loading"}

    @modal.web_endpoint(method="GET")
    def list_models(self):
        """List all available models, LoRAs, and VAEs"""
        import requests
        import os

        try:
            # Get checkpoints from A1111
            models = []
            try:
                r = requests.get(f"{self.a1111_url}/sdapi/v1/sd-models", timeout=10)
                if r.status_code == 200:
                    models = [m["title"] for m in r.json()]
            except:
                pass

            # Fallback: list files from volume if A1111 returned nothing
            if not models and os.path.exists(CHECKPOINT_DIR):
                models = [f for f in os.listdir(CHECKPOINT_DIR)
                          if f.endswith(('.safetensors', '.ckpt'))]

            # Get LoRAs from A1111
            loras = []
            try:
                # Refresh LoRA list first
                requests.post(f"{self.a1111_url}/sdapi/v1/refresh-loras", timeout=10)
                r = requests.get(f"{self.a1111_url}/sdapi/v1/loras", timeout=10)
                if r.status_code == 200:
                    loras = [l["name"] for l in r.json()]
            except:
                pass

            # Fallback: list files from volume if A1111 returned nothing
            if not loras and os.path.exists(LORA_DIR):
                loras = [f for f in os.listdir(LORA_DIR)
                         if f.endswith(('.safetensors', '.ckpt', '.pt'))]

            # Get VAEs (list directory directly)
            vaes = ["Automatic"]
            if os.path.exists(VAE_DIR):
                vae_files = [f for f in os.listdir(VAE_DIR)
                             if f.endswith(('.safetensors', '.pt', '.ckpt'))]
                vaes += vae_files

            return {"models": models, "loras": loras, "vaes": vaes}

        except Exception as e:
            print(f"Error listing models: {e}")
            return {"models": [], "loras": [], "vaes": ["Automatic"]}

    @modal.web_endpoint(method="POST")
    def api_generate(self, data: dict):
        """Generate an image via A1111's txt2img API"""
        import requests
        import json

        try:
            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "")
            steps = min(int(data.get("steps", 25)), 50)
            cfg = float(data.get("cfg_scale", 7.0))
            width = int(data.get("width", 1024))
            height = int(data.get("height", 1024))
            model = data.get("model", "")
            loras = data.get("loras", [])
            sampler_name = data.get("scheduler", "Euler")
            seed = int(data.get("seed", -1))
            clip_skip = int(data.get("clip_skip", 1))
            vae_name = data.get("vae", "")
            batch_count = min(int(data.get("batch_count", 1)), 4)  # Max 4 images

            # Hi-Res Fix params
            hires_enabled = data.get("enable_hr", False)
            hires_upscaler = data.get("hr_upscaler", "R-ESRGAN 4x+")
            hires_scale = float(data.get("hr_scale", 1.5))
            hires_denoising = float(data.get("hr_denoising_strength", 0.35))
            hires_steps = int(data.get("hr_second_pass_steps", 15))

            # ADetailer params (1st pass)
            adetailer_enabled = data.get("adetailer_enabled", False)
            adetailer_model = data.get("adetailer_model", "face_yolov8n.pt")
            adetailer_confidence = float(data.get("adetailer_confidence", 0.3))
            adetailer_prompt = data.get("adetailer_prompt", "")
            adetailer_negative = data.get("adetailer_negative", "")
            adetailer_strength = float(data.get("adetailer_strength", 0.4))
            adetailer_steps = int(data.get("adetailer_steps", 25))

            # ADetailer params (2nd pass) - e.g. hands or full person
            adetailer2_enabled = data.get("adetailer2_enabled", False)
            adetailer2_model = data.get("adetailer2_model", "hand_yolov8n.pt")
            adetailer2_confidence = float(data.get("adetailer2_confidence", 0.3))
            adetailer2_prompt = data.get("adetailer2_prompt", "")
            adetailer2_negative = data.get("adetailer2_negative", "")
            adetailer2_strength = float(data.get("adetailer2_strength", 0.4))
            adetailer2_steps = int(data.get("adetailer2_steps", 25))

            # ControlNet / IP-Adapter params
            controlnet_enabled = data.get("controlnet_enabled", False)
            controlnet_image = data.get("controlnet_image", "")
            controlnet_module = data.get("controlnet_module", "none")
            controlnet_model = data.get("controlnet_model", "None")
            controlnet_weight = float(data.get("controlnet_weight", 0.8))
            controlnet_guidance_start = float(data.get("controlnet_guidance_start", 0.0))
            controlnet_guidance_end = float(data.get("controlnet_guidance_end", 1.0))

            if not prompt:
                return {"error": "No prompt provided"}

            if not self.ready:
                return {"error": "A1111 is still starting up, please wait"}

            # Append LoRA tags to prompt using A1111's native syntax
            # This supports ALL LoRA formats (standard, LoCon, LoKR, LoHa)
            lora_tags = []
            for lora in loras:
                name = lora.get("name", "")
                weight = float(lora.get("weight", 0.8))
                if name:
                    # Remove file extension for A1111's <lora:name:weight> syntax
                    lora_base = name.rsplit(".", 1)[0] if "." in name else name
                    lora_tags.append(f"<lora:{lora_base}:{weight}>")

            full_prompt = prompt
            if lora_tags:
                full_prompt = prompt + " " + " ".join(lora_tags)

            # Build override settings for model/VAE/clip skip
            override_settings = {}
            if model and model != "default" and model != "SDXL Base 1.0":
                override_settings["sd_model_checkpoint"] = model
            if vae_name and vae_name != "Automatic" and vae_name != "":
                override_settings["sd_vae"] = vae_name
            if clip_skip > 1:
                override_settings["CLIP_stop_at_last_layers"] = clip_skip

            # Ensure v-pred checkpoints have an adjacent yaml config when needed
            # (prevents common "fried/noise" outputs on some SDXL v-pred models).
            created_yaml = self._ensure_vpred_yaml(model)
            if created_yaml:
                try:
                    requests.post(f"{self.a1111_url}/sdapi/v1/refresh-checkpoints", timeout=20)
                    print("[v-pred] Refreshed checkpoints after yaml provisioning")
                except Exception as refresh_err:
                    print(f"[v-pred] Could not refresh checkpoints: {refresh_err}")

            # Build txt2img payload
            payload = {
                "prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "sampler_name": sampler_name,
                "steps": steps,
                "cfg_scale": cfg,
                "width": width,
                "height": height,
                "seed": seed,
                "n_iter": batch_count,  # Number of images to generate
                "override_settings": override_settings,
                "override_settings_restore_afterwards": True,
            }

            # Add Hi-Res Fix if enabled
            if hires_enabled:
                payload["enable_hr"] = True
                payload["hr_upscaler"] = hires_upscaler
                payload["hr_scale"] = hires_scale
                payload["denoising_strength"] = hires_denoising
                payload["hr_second_pass_steps"] = hires_steps

            # Build alwayson_scripts
            alwayson_scripts = {}

            # Add ADetailer if enabled. ADetailer accepts multiple detection
            # tabs: each extra dict in `args` is an additional pass.
            if adetailer_enabled or adetailer2_enabled:
                ad_args = [
                    True,   # ad_enable
                    False,  # skip_img2img
                ]
                if adetailer_enabled:
                    ad_args.append({
                        "ad_model": adetailer_model,
                        "ad_confidence": adetailer_confidence,
                        "ad_prompt": adetailer_prompt or "",
                        "ad_negative_prompt": adetailer_negative or "",
                        "ad_denoising_strength": adetailer_strength,
                        "ad_steps": adetailer_steps,
                        "ad_cfg_scale": cfg,
                    })
                if adetailer2_enabled:
                    ad_args.append({
                        "ad_model": adetailer2_model,
                        "ad_confidence": adetailer2_confidence,
                        "ad_prompt": adetailer2_prompt or "",
                        "ad_negative_prompt": adetailer2_negative or "",
                        "ad_denoising_strength": adetailer2_strength,
                        "ad_steps": adetailer2_steps,
                        "ad_cfg_scale": cfg,
                    })
                alwayson_scripts["ADetailer"] = {
                    "args": ad_args
                }

            # Add ControlNet if enabled and image is provided
            if controlnet_enabled and controlnet_image:
                # strip data URL prefix if present in the base64 string
                pure_b64 = controlnet_image
                if "," in pure_b64:
                    pure_b64 = pure_b64.split(",", 1)[1]
                
                cn_args = [
                    {
                        "input_image": pure_b64,
                        "module": controlnet_module,
                        "model": controlnet_model,
                        "weight": controlnet_weight,
                        "guidance_start": controlnet_guidance_start,
                        "guidance_end": controlnet_guidance_end,
                        "pixel_perfect": True,
                        "control_mode": "Balanced"
                    }
                ]
                alwayson_scripts["ControlNet"] = {
                    "args": cn_args
                }

            if alwayson_scripts:
                payload["alwayson_scripts"] = alwayson_scripts

            print(f"Generating: {prompt[:80]}...")
            print(f"  Sampler: {sampler_name}, Steps: {steps}, CFG: {cfg}, Size: {width}x{height}")
            print(f"  Seed: {seed}, Clip skip: {clip_skip}, Batch: {batch_count}")
            if lora_tags:
                print(f"  LoRAs: {lora_tags}")
            if adetailer_enabled:
                print(f"  ADetailer #1: ON (model={adetailer_model}, confidence={adetailer_confidence}, strength={adetailer_strength}, steps={adetailer_steps})")
            if adetailer2_enabled:
                print(f"  ADetailer #2: ON (model={adetailer2_model}, confidence={adetailer2_confidence}, strength={adetailer2_strength}, steps={adetailer2_steps})")
            if hires_enabled:
                print(f"  Hi-Res Fix: ON ({hires_upscaler}, scale={hires_scale}, denoise={hires_denoising}, steps={hires_steps})")
            if controlnet_enabled and controlnet_image:
                print(f"  ControlNet: ON (module={controlnet_module}, model={controlnet_model}, weight={controlnet_weight})")

            # Call A1111's txt2img API (longer timeout for hi-res)
            gen_timeout = 600 if hires_enabled else 300
            r = self._post_txt2img(payload, timeout=gen_timeout)

            # Automatic recovery for NaN/fried generations:
            # retry with a safer profile (Euler, lower CFG, no LoRA, no hi-res).
            if r.status_code != 200 and self._is_nan_failure(r.text):
                print("[safe-mode] NaN/Inf-like failure detected. Retrying with safe settings...")
                safe_override = dict(override_settings)
                if "xl" in (model or "").lower() and not safe_override.get("sd_vae"):
                    safe_override["sd_vae"] = "sdxl_vae.safetensors"

                safe_payload = {
                    "prompt": prompt,  # disable LoRA tags for retry
                    "negative_prompt": negative_prompt,
                    "sampler_name": "Euler",
                    "steps": min(steps, 30),
                    "cfg_scale": min(cfg, 4.5),
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "n_iter": batch_count,
                    "override_settings": safe_override,
                    "override_settings_restore_afterwards": True,
                }
                r = self._post_txt2img(safe_payload, timeout=300)
                if r.status_code == 200:
                    print("[safe-mode] Recovery succeeded.")
                else:
                    print(f"[safe-mode] Recovery failed: {r.text[:300] if r.text else 'Unknown error'}")

            if r.status_code == 200:
                result = r.json()
                images = result.get("images", [])
                info_str = result.get("info", "{}")

                # Parse info JSON to get actual seed
                try:
                    info_dict = json.loads(info_str) if isinstance(info_str, str) else info_str
                    actual_seed = info_dict.get("seed", seed)
                except:
                    actual_seed = seed
                    info_dict = {}

                if images:
                    print(f"Generation complete! {len(images)} image(s), Seed: {actual_seed}")
                    return {
                        "image": images[0],  # First image (backwards compatible)
                        "images": images,    # All images for batch
                        "info": {
                            "seed": actual_seed,
                            "model": model or self.current_model,
                            "scheduler": sampler_name,
                            "clip_skip": clip_skip,
                            "batch_count": batch_count,
                        }
                    }
                else:
                    return {"error": "No image generated"}
            else:
                error_text = r.text[:300] if r.text else "Unknown error"
                print(f"A1111 error ({r.status_code}): {error_text}")
                return {"error": f"Generation failed: {error_text}"}

        except Exception as e:
            print(f"Error: {str(e)}")
            return {"error": str(e)}

    @modal.web_endpoint(method="POST")
    def load_civitai_model(self, data: dict):
        """Download a model/LoRA/VAE from Civitai"""
        import os
        import requests

        try:
            model_url = data.get("model_url", "")
            filename = data.get("filename", "model.safetensors")
            is_lora = data.get("is_lora", False)
            is_vae = data.get("is_vae", False)

            # Ensure filename has a valid extension (A1111 won't recognize files without one)
            valid_extensions = ('.safetensors', '.ckpt', '.pt')
            if not filename.lower().endswith(valid_extensions):
                filename = f"{filename}.safetensors"

            if not model_url:
                return {"error": "No model URL provided"}

            # Force using civitai.red to bypass civitai.com download blocks
            if "civitai.com" in model_url:
                model_url = model_url.replace("civitai.com", "civitai.red")

            # Add Civitai token if available
            civitai_token = os.environ.get("CIVITAI_TOKEN", "")
            if civitai_token and "token=" not in model_url:
                separator = "&" if "?" in model_url else "?"
                model_url = f"{model_url}{separator}token={civitai_token}"

            # Choose target directory
            if is_vae:
                target_dir = VAE_DIR
            elif is_lora:
                target_dir = LORA_DIR
            else:
                target_dir = CHECKPOINT_DIR

            filepath = f"{target_dir}/{filename}"

            print(f"Downloading {filename} to {target_dir}...")
            r = requests.get(model_url, stream=True, timeout=30)
            r.raise_for_status()

            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded / total) * 100
                        if int(pct) % 10 == 0 and int(pct) > 0:
                            print(f"  {pct:.0f}%")

            volume.commit()

            # Refresh A1111's model lists
            try:
                requests.post(f"{self.a1111_url}/sdapi/v1/refresh-checkpoints", timeout=10)
                requests.post(f"{self.a1111_url}/sdapi/v1/refresh-loras", timeout=10)
            except:
                pass

            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"Download complete: {filename} ({size_mb:.1f} MB)")

            return {"success": True, "filename": filename, "size_mb": round(size_mb, 1)}

        except Exception as e:
            print(f"Download error: {str(e)}")
            return {"error": str(e)}

    @modal.web_endpoint(method="POST")
    def delete_model(self, data: dict):
        """Delete a model/LoRA/VAE"""
        import os
        import requests

        try:
            filename = data.get("filename", "")
            file_type = data.get("type", "checkpoint")

            if not filename:
                return {"error": "No filename provided"}

            if file_type == "lora":
                target_dir = LORA_DIR
            elif file_type == "vae":
                target_dir = VAE_DIR
            else:
                target_dir = CHECKPOINT_DIR

            filepath = f"{target_dir}/{filename}"

            # A1111 often returns names without extension — try with common extensions
            if not os.path.exists(filepath):
                for ext in ('.safetensors', '.ckpt', '.pt'):
                    candidate = f"{target_dir}/{filename}{ext}"
                    if os.path.exists(candidate):
                        filepath = candidate
                        break

            if os.path.exists(filepath):
                os.remove(filepath)
                volume.commit()

                # Refresh A1111's model lists
                try:
                    requests.post(f"{self.a1111_url}/sdapi/v1/refresh-checkpoints", timeout=10)
                    requests.post(f"{self.a1111_url}/sdapi/v1/refresh-loras", timeout=10)
                except:
                    pass

                print(f"Deleted {file_type}: {os.path.basename(filepath)}")
                return {"success": True}
            else:
                return {"error": f"File not found: {filename}"}

        except Exception as e:
            print(f"Error deleting: {str(e)}")
            return {"error": str(e)}


@app.local_entrypoint()
def main():
    """Test the model locally"""
    print("Run 'modal serve modal_sd.py' to start the API")
