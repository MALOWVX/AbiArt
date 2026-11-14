# Copie de la passerelle du notebook kaggle_backend5_comfyui.ipynb (cellule 5), utilisee par modal_comfy.py.
# Seules differences : adresse d'ecoute (ABIART_HOST) et prefixe secret (ABIART_PATH_PREFIX).
#
# AbiArt gateway : parle au site comme le faisait l'API d'Automatic1111, et calcule avec ComfyUI.
#
# Le site AbiArt ne change pas : il appelle les memes routes (/abiart/..., /sdapi/v1/...). Cette passerelle
# traduit chaque demande de generation en workflow ComfyUI et la confie a une instance ComfyUI libre
# (une instance par GPU). Bibliotheque standard + requests ; Pillow est facultatif (metadonnees de l'image).
import base64
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urljoin, urlparse

import requests

PORT = int(os.environ.get("ABIART_PORT", "7860"))
HOST = os.environ.get("ABIART_HOST", "127.0.0.1")
# Prefixe secret facultatif (ex. "/k3y") : sans lui dans l'URL, la passerelle repond 404.
PATH_PREFIX = "/" + os.environ.get("ABIART_PATH_PREFIX", "").strip("/") if os.environ.get("ABIART_PATH_PREFIX", "").strip("/") else ""
COMFY_DIR = os.environ.get("ABIART_COMFY_DIR", ".")
MODELS = os.path.join(COMFY_DIR, "models")
OUT_BASE = os.environ.get("ABIART_OUT_BASE", os.path.join(COMFY_DIR, "output"))
DEFAULT_CKPT = os.environ.get("ABIART_DEFAULT_CKPT", "")
HF_TOKEN = os.environ.get("ABIART_HF_TOKEN", "")
HF_REPO = os.environ.get("ABIART_HF_REPO", "")
MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth")

# ----------------------------------------------------------------------------------------------
# Noms affiches sur le site (ceux d'A1111) -> noms ComfyUI
# ----------------------------------------------------------------------------------------------
SAMPLERS = [
    ("Euler a", "euler_ancestral"), ("Euler", "euler"), ("DPM++ 2M", "dpmpp_2m"), ("DPM++ SDE", "dpmpp_sde"),
    ("DPM++ 2M SDE", "dpmpp_2m_sde"), ("DPM++ 3M SDE", "dpmpp_3m_sde"), ("DPM++ 2S a", "dpmpp_2s_ancestral"),
    ("DPM2", "dpm_2"), ("DPM2 a", "dpm_2_ancestral"), ("Heun", "heun"), ("LMS", "lms"), ("DDIM", "ddim"),
    ("UniPC", "uni_pc"), ("LCM", "lcm"),
]
SCHEDULERS = [
    ("Automatic", None), ("Normal", "normal"), ("Karras", "karras"), ("Exponential", "exponential"),
    ("SGM Uniform", "sgm_uniform"), ("Simple", "simple"), ("DDIM", "ddim_uniform"), ("Beta", "beta"),
    ("KL Optimal", "kl_optimal"),
]
# "Automatic" : A1111 choisit l'echeancier habituel du sampler
AUTO_SCHEDULER = {"dpmpp_2m": "karras", "dpmpp_sde": "karras", "dpmpp_2s_ancestral": "karras",
                  "dpmpp_2m_sde": "exponential", "dpmpp_3m_sde": "exponential"}
LATENT_MODES = [("Latent", "bilinear"), ("Latent (bicubic)", "bicubic"), ("Latent (nearest-exact)", "nearest-exact"),
                ("Latent (bislerp)", "bislerp")]
UPSCALER_ALIASES = {"R-ESRGAN 4x+": "RealESRGAN_x4plus", "R-ESRGAN 4x+ Anime6B": "RealESRGAN_x4plus_anime_6B"}
SUBDIRS = {"checkpoint": "checkpoints", "lora": "loras", "vae": "vae", "embedding": "embeddings"}
HF_FOLDERS = {"checkpoint": "checkpoints", "lora": "loras", "vae": "vae", "embedding": "embeddings"}


def log(msg):
    print("[AbiArt] " + msg, flush=True)


def list_models(sub, exts=MODEL_EXT):
    # Fichiers d'un dossier de modeles ComfyUI, chemins relatifs avec "/" (sous-dossiers compris).
    root = os.path.join(MODELS, sub)
    out = []
    for cur, _dirs, files in os.walk(root, followlinks=True):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.relpath(os.path.join(cur, f), root).replace("\\", "/"))
    return sorted(out, key=str.lower)


def stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def find_model(sub, wanted, exts=MODEL_EXT):
    # Retrouve un fichier a partir du nom donne par le site (avec ou sans extension, avec ou sans [hash]).
    wanted = re.sub(r"\s*\[[0-9a-fA-F]+\]\s*$", "", str(wanted or "")).strip().replace("\\", "/")
    if not wanted:
        return None
    files = list_models(sub, exts)
    for test in (lambda f: f == wanted, lambda f: os.path.basename(f) == os.path.basename(wanted),
                 lambda f: stem(f).lower() == stem(wanted).lower(), lambda f: stem(f).lower() == wanted.lower()):
        for f in files:
            if test(f):
                return f
    return None


def embedding_names():
    # Embeddings installes, sans extension, les plus longs d'abord (un nom peut en contenir un autre).
    return sorted({os.path.splitext(f)[0] for f in list_models("embeddings")}, key=len, reverse=True)


def with_embeddings(text, names):
    # Dans A1111 on ecrit juste le nom de l'embedding ; ComfyUI attend "embedding:nom".
    for name in names:
        pattern = r"(?<![\w:./-])(?:embedding:)?" + re.escape(name) + r"(?![\w.-])"
        text = re.sub(pattern, lambda _m, n=name: "embedding:" + n, text, flags=re.I)
    return text


def comfy_name(rel):
    # ComfyUI attend le separateur du systeme dans les noms de fichiers en sous-dossier.
    return rel.replace("/", os.sep)


# ----------------------------------------------------------------------------------------------
# Instances ComfyUI (une par GPU)
# ----------------------------------------------------------------------------------------------
_PORTS = [p.strip() for p in os.environ.get("ABIART_COMFY_PORTS", "8188").split(",") if p.strip()]
BACKENDS = [{"id": i, "url": "http://127.0.0.1:" + p, "out": os.path.join(OUT_BASE, "gpu%d" % i), "busy": False,
             "up": False, "checked": 0.0, "model": "", "info": None}
            for i, p in enumerate(_PORTS)]
GEN_TTL = 15 * 60
GEN_MAX_ACTIVE = 2 + 2 * len(BACKENDS)
GEN_JOBS = {}
_GEN_LOCK = threading.Lock()
_BK_COND = threading.Condition()
_LOAD_LOCK = threading.Lock()   # un seul chargement de checkpoint a la fois (memoire vive)
LOAD_LOCK_SECONDS = float(os.environ.get("ABIART_LOAD_LOCK_SECONDS", "60"))
_WAITING = []


def backend_up(b):
    if b["up"]:
        return True
    now = time.time()
    if now - b["checked"] < 10:
        return False
    b["checked"] = now
    try:
        b["up"] = requests.get(b["url"] + "/system_stats", timeout=3).status_code == 200
    except Exception:
        b["up"] = False
    return b["up"]


def object_info(b, refresh=False):
    # Description des noeuds disponibles sur une instance (sert a savoir ce qui est installe).
    if b["info"] is None or refresh:
        b["info"] = requests.get(b["url"] + "/object_info", timeout=60).json()
    return b["info"]


def any_info():
    for b in BACKENDS:
        if backend_up(b):
            try:
                return object_info(b)
            except Exception:
                pass
    return {}


def combo_options(spec):
    # Valeurs possibles d'une liste deroulante ComfyUI (ancien format [[...], {...}] ou nouveau ["COMBO", {"options": [...]}]).
    if isinstance(spec, (list, tuple)) and spec:
        if isinstance(spec[0], (list, tuple)):
            return list(spec[0])
        if len(spec) > 1 and isinstance(spec[1], dict) and isinstance(spec[1].get("options"), list):
            return list(spec[1]["options"])
    return []


def node_options(info, cls, key):
    spec = (((info.get(cls) or {}).get("input") or {}).get("required") or {}).get(key)
    return combo_options(spec)


def has_detailer(info):
    return "FaceDetailer" in info and "UltralyticsDetectorProvider" in info


def available_samplers(info):
    opts = node_options(info, "KSampler", "sampler_name")
    return [(a, c) for a, c in SAMPLERS if not opts or c in opts]


def available_schedulers(info):
    opts = node_options(info, "KSampler", "scheduler")
    return [(a, c) for a, c in SCHEDULERS if c is None or not opts or c in opts]


def ram_free_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return 99.0


# ----------------------------------------------------------------------------------------------
# Demande du site (format txt2img d'A1111) -> workflow ComfyUI
# ----------------------------------------------------------------------------------------------
LORA_TAG = re.compile(r"<lora:([^:>]+)(?::([-+]?[0-9]*\.?[0-9]+))?(?::[^>]*)?>", re.I)


class BadRequest(Exception):
    pass


def _num(value, default, cast=float, lo=None, hi=None):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        v = cast(default)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _complete(graph, info):
    # Renseigne les entrees obligatoires non fournies avec leur valeur par defaut (noeuds qui evoluent).
    for node in graph.values():
        req = (((info.get(node["class_type"]) or {}).get("input") or {}).get("required") or {})
        for key, spec in req.items():
            if key in node["inputs"] or not isinstance(spec, (list, tuple)) or not spec:
                continue
            conf = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            opts = combo_options(spec)
            if opts:
                node["inputs"][key] = conf.get("default", opts[0])
            elif spec[0] in ("INT", "FLOAT"):
                node["inputs"][key] = conf.get("default", 0)
            elif spec[0] == "BOOLEAN":
                node["inputs"][key] = conf.get("default", False)
            elif spec[0] == "STRING":
                node["inputs"][key] = conf.get("default", "")


def build_workflow(data, info, job_id):
    # Retourne (workflow, meta). meta sert au texte de parametres ecrit dans l'image et au suivi.
    prompt_raw = str(data.get("prompt") or "")
    negative = str(data.get("negative_prompt") or "")
    loras = []
    for m in LORA_TAG.finditer(prompt_raw):
        loras.append((m.group(1).strip(), _num(m.group(2), 1.0)))
    prompt = re.sub(r"\s{2,}", " ", LORA_TAG.sub("", prompt_raw)).strip(" ,") or prompt_raw

    wanted = str(data.get("abiart_model") or "").strip() or CURRENT["model"] or DEFAULT_CKPT
    ckpt = find_model("checkpoints", wanted) if wanted else None
    if not ckpt:
        files = list_models("checkpoints")
        if wanted or not files:
            raise BadRequest("Checkpoint introuvable sur le notebook : %s" % (wanted or "(aucun modele installe)"))
        ckpt = files[0]

    width = _num(data.get("width"), 1024, int, 64, 4096) // 8 * 8
    height = _num(data.get("height"), 1024, int, 64, 4096) // 8 * 8
    steps = _num(data.get("steps"), 25, int, 1, 150)
    cfg = _num(data.get("cfg_scale"), 7.0, float, 0.0, 30.0)
    seed = _num(data.get("seed"), -1, int)
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % 4294967294 + 1
    overrides = data.get("override_settings") or {}
    clip_skip = _num(overrides.get("CLIP_stop_at_last_layers"), 1, int, 1, 12)

    sampler_label = str(data.get("sampler_name") or "Euler")
    sampler = dict(SAMPLERS).get(sampler_label, sampler_label)
    sampler_opts = node_options(info, "KSampler", "sampler_name")
    if sampler_opts and sampler not in sampler_opts:
        sampler = "euler"
    sched_label = str(data.get("scheduler") or "Automatic")
    scheduler = dict(SCHEDULERS).get(sched_label, sched_label.lower().replace(" ", "_"))
    if scheduler is None:
        scheduler = AUTO_SCHEDULER.get(sampler, "normal")
    sched_opts = node_options(info, "KSampler", "scheduler")
    if sched_opts and scheduler not in sched_opts:
        scheduler = "normal"

    g = {}

    def node(cls, **inputs):
        nid = str(len(g) + 1)
        g[nid] = {"class_type": cls, "inputs": inputs}
        return nid

    ck = node("CheckpointLoaderSimple", ckpt_name=comfy_name(ckpt))
    model, clip, vae = [ck, 0], [ck, 1], [ck, 2]

    vae_name = str(overrides.get("sd_vae") or data.get("abiart_vae") or "")
    if vae_name and vae_name.lower() not in ("automatic", "none"):
        vae_file = find_model("vae", vae_name)
        if vae_file:
            vae = [node("VAELoader", vae_name=comfy_name(vae_file)), 0]

    used_loras = []
    for name, weight in loras:
        f = find_model("loras", name)
        if not f:
            raise BadRequest("LoRA introuvable sur le notebook : %s" % name)
        ln = node("LoraLoader", model=model, clip=clip, lora_name=comfy_name(f), strength_model=weight,
                  strength_clip=weight)
        model, clip = [ln, 0], [ln, 1]
        used_loras.append((stem(f), weight))

    # Clip skip : 1 = reglage par defaut du modele (avant-derniere couche en SDXL, derniere en SD 1.5).
    if clip_skip >= 2:
        clip = [node("CLIPSetLastLayer", clip=clip, stop_at_clip_layer=-clip_skip), 0]

    a1111_weights = "BNK_CLIPTextEncodeAdvanced" in info
    emb_names = embedding_names()

    def encode(text):
        text = with_embeddings(text, emb_names)
        # Poids (mot:1.2) calcules comme dans A1111 quand le noeud est installe, sinon a la facon ComfyUI.
        if a1111_weights:
            return [node("BNK_CLIPTextEncodeAdvanced", text=text, clip=clip, token_normalization="none",
                         weight_interpretation="A1111"), 0]
        return [node("CLIPTextEncode", text=text, clip=clip), 0]

    positive, negative_c = encode(prompt), encode(negative)

    latent = [node("EmptyLatentImage", width=width, height=height, batch_size=1), 0]
    samples = [node("KSampler", model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler,
                    positive=positive, negative=negative_c, latent_image=latent, denoise=1.0), 0]

    meta = {"prompt": prompt_raw, "negative": negative, "steps": steps, "sampler": sampler_label,
            "scheduler": sched_label, "cfg": cfg, "seed": seed, "width": width, "height": height, "ckpt": ckpt,
            "clip_skip": clip_skip, "loras": used_loras, "hires": None, "detailers": [],
            "a1111_weights": a1111_weights}

    # ---- Hi-Res Fix : agrandissement puis seconde passe de diffusion
    if data.get("enable_hr"):
        scale = _num(data.get("hr_scale"), 1.5, float, 1.0, 4.0)
        denoise = _num(data.get("denoising_strength"), 0.35, float, 0.05, 1.0)
        hr_steps = _num(data.get("hr_second_pass_steps") or steps, steps, int, 1, 150)
        up_label = str(data.get("hr_upscaler") or "Latent")
        tw, th = int(round(width * scale / 8)) * 8, int(round(height * scale / 8)) * 8
        latent_mode = dict(LATENT_MODES).get(up_label)
        up_file = None if latent_mode else find_model("upscale_models", UPSCALER_ALIASES.get(up_label, up_label))
        if latent_mode is None and up_file is None:
            raise BadRequest("Upscaler introuvable sur le notebook : %s" % up_label)
        if latent_mode:
            hr_latent = [node("LatentUpscaleBy", samples=samples, upscale_method=latent_mode, scale_by=scale), 0]
        else:
            img = [node("VAEDecode", samples=samples, vae=vae), 0]
            um = [node("UpscaleModelLoader", model_name=comfy_name(up_file)), 0]
            big = [node("ImageUpscaleWithModel", upscale_model=um, image=img), 0]
            fit = [node("ImageScale", image=big, upscale_method="lanczos", width=tw, height=th, crop="disabled"), 0]
            hr_latent = [node("VAEEncode", pixels=fit, vae=vae), 0]
        samples = [node("KSampler", model=model, seed=seed, steps=hr_steps, cfg=cfg, sampler_name=sampler,
                        scheduler=scheduler, positive=positive, negative=negative_c, latent_image=hr_latent,
                        denoise=denoise), 0]
        meta["hires"] = {"scale": scale, "steps": hr_steps, "upscaler": up_label, "denoise": denoise}

    image = [node("VAEDecode", samples=samples, vae=vae), 0]

    # ---- Retouche automatique (equivalent d'ADetailer) : une passe par onglet active sur le site
    tabs = []
    ad = (data.get("alwayson_scripts") or {}).get("ADetailer") or {}
    args = ad.get("args") if isinstance(ad, dict) else None
    if isinstance(args, list) and args and args[0] is not False:
        tabs = [a for a in args if isinstance(a, dict) and a.get("ad_model") and a.get("ad_model") != "None"]
    if tabs and not has_detailer(info):
        raise BadRequest("always on script ADetailer not found : la retouche automatique n'est pas installee sur ce notebook")
    for tab in tabs:
        det_name = os.path.basename(str(tab["ad_model"]))
        folder = "segm" if "-seg" in det_name else "bbox"
        if not os.path.exists(os.path.join(MODELS, "ultralytics", folder, det_name)):
            raise BadRequest("Modele de detection absent du notebook : %s" % det_name)
        d_denoise = _num(tab.get("ad_denoising_strength"), 0.4, float, 0.05, 1.0)
        d_steps_site = _num(tab.get("ad_steps"), steps, int, 1, 150) if tab.get("ad_use_steps") else steps
        # A1111 n'execute que (pas x denoise) pas reels pour une retouche : meme quantite de calcul ici.
        d_steps = max(5, int(round(d_steps_site * d_denoise)))
        d_cfg = _num(tab.get("ad_cfg_scale"), cfg, float, 0.0, 30.0) if tab.get("ad_use_cfg_scale") else cfg
        # Le texte propre a la retouche s'ajoute au prompt de l'image : s'il le remplacait, la zone serait
        # redessinee sans les mots de style ni les couleurs demandes (yeux, cheveux...).
        ad_pos = LORA_TAG.sub("", str(tab.get("ad_prompt") or "")).strip(" ,")
        ad_neg = str(tab.get("ad_negative_prompt") or "").strip(" ,")
        d_pos = encode(", ".join(t for t in (prompt, ad_pos) if t)) if ad_pos else positive
        d_neg = encode(", ".join(t for t in (negative, ad_neg) if t)) if ad_neg else negative_c
        det = node("UltralyticsDetectorProvider", model_name=folder + "/" + det_name)
        fd = node("FaceDetailer", image=image, model=model, clip=clip, vae=vae,
                  guide_size=1024, guide_size_for=True, max_size=1024,
                  seed=seed, steps=d_steps, cfg=d_cfg, sampler_name=sampler, scheduler=scheduler,
                  positive=d_pos, negative=d_neg, denoise=d_denoise,
                  feather=_num(tab.get("ad_mask_blur"), 4, int, 0, 100), noise_mask=True, force_inpaint=True,
                  bbox_threshold=_num(tab.get("ad_confidence"), 0.3, float, 0.0, 1.0),
                  bbox_dilation=_num(tab.get("ad_dilate_erode"), 4, int, -512, 512), bbox_crop_factor=3.0,
                  sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0,
                  sam_mask_hint_threshold=0.7, sam_mask_hint_use_negative="False", drop_size=10,
                  bbox_detector=[det, 0], wildcard="", cycle=1)
        image = [fd, 0]
        meta["detailers"].append({"model": det_name, "denoise": d_denoise, "steps": d_steps,
                                  "confidence": g[fd]["inputs"]["bbox_threshold"]})

    save = node("SaveImage", images=image, filename_prefix="abiart_" + job_id)
    _complete(g, info)
    meta["save_node"] = save
    return g, meta


def infotext(meta):
    # Texte de parametres au format d'A1111 (lu par le site et par les lecteurs de metadonnees habituels).
    parts = ["Steps: %d" % meta["steps"], "Sampler: %s" % meta["sampler"], "Schedule type: %s" % meta["scheduler"],
             "CFG scale: %s" % meta["cfg"], "Seed: %d" % meta["seed"], "Size: %dx%d" % (meta["width"], meta["height"]),
             "Model: %s" % stem(meta["ckpt"]), "Clip skip: %d" % meta["clip_skip"]]
    if meta["hires"]:
        h = meta["hires"]
        parts += ["Denoising strength: %s" % h["denoise"], "Hires upscale: %s" % h["scale"],
                  "Hires steps: %d" % h["steps"], "Hires upscaler: %s" % h["upscaler"]]
    for i, d in enumerate(meta["detailers"]):
        suffix = "" if i == 0 else " %s" % _ordinal(i + 1)
        parts += ["ADetailer model%s: %s" % (suffix, d["model"]), "ADetailer confidence%s: %s" % (suffix, d["confidence"]),
                  "ADetailer denoising strength%s: %s" % (suffix, d["denoise"])]
    if meta["loras"]:
        parts.append('Lora weights: "%s"' % ", ".join("%s: %s" % (n, w) for n, w in meta["loras"]))
    parts.append("Version: ComfyUI (AbiArt)")
    return "%s\nNegative prompt: %s\n%s" % (meta["prompt"], meta["negative"], ", ".join(parts))


def _ordinal(n):
    return {2: "2nd", 3: "3rd"}.get(n, "%dth" % n)


def with_parameters(png_bytes, text):
    # Remplace les metadonnees ComfyUI de l'image par le texte de parametres. Sans Pillow, l'image reste telle quelle.
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        img = Image.open(io.BytesIO(png_bytes))
        extra = PngInfo()
        extra.add_text("parameters", text)
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=extra, compress_level=4)
        return buf.getvalue()
    except Exception:
        return png_bytes


# ----------------------------------------------------------------------------------------------
# Execution d'un job sur une instance
# ----------------------------------------------------------------------------------------------
CURRENT = {"model": ""}   # checkpoint choisi par un ancien site via POST /sdapi/v1/options


def _acquire(job_id, ckpt):
    job = GEN_JOBS[job_id]
    with _BK_COND:
        _WAITING.append(job_id)
        try:
            while True:
                if job.get("cancel"):
                    return None
                if _WAITING[0] == job_id:
                    free = [b for b in BACKENDS if not b["busy"] and backend_up(b)]
                    if free:
                        pick = next((b for b in free if b["model"] == ckpt), free[0])
                        pick["busy"] = True
                        return pick
                _BK_COND.wait(timeout=1.0)
        finally:
            _WAITING.remove(job_id)
            _BK_COND.notify_all()


def _release(b):
    with _BK_COND:
        b["busy"] = False
        _BK_COND.notify_all()


def _comfy_error(resp):
    try:
        js = resp.json()
    except Exception:
        return (resp.text or "HTTP %d" % resp.status_code)[:400]
    msgs = []
    err = js.get("error")
    if isinstance(err, dict):
        msgs.append(str(err.get("message") or err.get("type") or ""))
    elif err:
        msgs.append(str(err))
    for nid, ne in (js.get("node_errors") or {}).items():
        for e in ne.get("errors") or []:
            msgs.append("%s : %s %s" % (ne.get("class_type", nid), e.get("message", ""), e.get("details", "")))
    return (" | ".join(m for m in msgs if m) or str(js))[:400]


def _history_error(entry):
    for kind, payload in (entry.get("status") or {}).get("messages") or []:
        if kind == "execution_error" and isinstance(payload, dict):
            return ("%s : %s" % (payload.get("node_type", "?"), payload.get("exception_message", "erreur"))).strip()[:400]
        if kind == "execution_interrupted":
            return "Interrupted"
    return "ComfyUI execution failed"


def _run_on(job, b, data):
    # Calcule le job sur l'instance b. Renvoie True s'il faut le relancer sur une autre instance.
    job.update(state="running", backend=b["id"], started=time.time(), updated=time.time())
    held = {"on": False}
    guard = threading.Lock()

    def unlock():
        with guard:
            if held["on"]:
                held["on"] = False
                _LOAD_LOCK.release()

    try:
        info = object_info(b)
        graph, meta = build_workflow(data, info, job["id"])
        if ram_free_gb() < 5:
            try:   # memoire vive presque pleine : ComfyUI libere les modeles qu'il garde en reserve
                requests.post(b["url"] + "/free", json={"unload_models": True, "free_memory": True}, timeout=30)
            except Exception:
                pass
        if b["model"] != meta["ckpt"]:
            # Changement de checkpoint : un seul a la fois sur la machine (pic de memoire vive pendant la
            # lecture du fichier). Le verrou est rendu au bout d'une minute, quand le modele est charge.
            job["loading_model"] = True
            _LOAD_LOCK.acquire()
            held["on"] = True
            timer = threading.Timer(LOAD_LOCK_SECONDS, unlock)
            timer.daemon = True
            timer.start()
        t_gen = time.time()
        r = requests.post(b["url"] + "/prompt", json={"prompt": graph, "client_id": "abiart"}, timeout=60)
        if r.status_code != 200:
            job.update(state="error", error="ComfyUI : " + _comfy_error(r), updated=time.time())
            return False
        pid = r.json().get("prompt_id")
        job["prompt_id"] = pid
        deadline = time.time() + 1800
        entry = None
        while time.time() < deadline:
            if job.get("cancel"):
                try:
                    requests.post(b["url"] + "/interrupt", timeout=10)
                except Exception:
                    pass
                job.update(state="cancelled", error="Cancelled", updated=time.time())
                b["model"] = meta["ckpt"]
                return False
            time.sleep(0.7)
            h = requests.get(b["url"] + "/history/" + pid, timeout=30).json()
            entry = h.get(pid)
            if entry and ((entry.get("status") or {}).get("completed") or entry.get("outputs")
                          or (entry.get("status") or {}).get("status_str") == "error"):
                break
            entry = None
        if entry is None:
            requests.post(b["url"] + "/interrupt", timeout=10)
            job.update(state="error", error="Generation trop longue (30 min)", updated=time.time())
            return False
        b["model"] = meta["ckpt"]
        if (entry.get("status") or {}).get("status_str") == "error":
            job.update(state="error", error=_history_error(entry), updated=time.time())
            return False
        files = ((entry.get("outputs") or {}).get(meta["save_node"]) or {}).get("images") or []
        if not files:
            job.update(state="error", error="ComfyUI n'a renvoye aucune image", updated=time.time())
            return False
        images = []
        text = infotext(meta)
        for f in files:
            path = os.path.join(b["out"], f.get("subfolder") or "", f["filename"])
            if os.path.exists(path):
                raw = open(path, "rb").read()
                try:
                    os.remove(path)
                except OSError:
                    pass
            else:
                raw = requests.get(b["url"] + "/view", params={"filename": f["filename"], "type": f.get("type", "output"),
                                                                "subfolder": f.get("subfolder", "")}, timeout=60).content
            images.append(base64.b64encode(with_parameters(raw, text)).decode("ascii"))
        job.update(state="done", images=images, updated=time.time(), seconds=round(time.time() - t_gen, 1),
                   info=json.dumps({"seed": meta["seed"], "infotexts": [text], "engine": "comfyui",
                                    "a1111_weights": meta["a1111_weights"]}))
        return False
    except BadRequest as e:
        job.update(state="error", error=str(e), updated=time.time())
        return False
    except requests.exceptions.ConnectionError as e:
        b["up"] = False
        b["checked"] = time.time()
        b["info"] = None
        if not job.get("cancel") and any(x is not b and backend_up(x) for x in BACKENDS):
            log("GPU %d ne repond plus : le job est relance sur un autre GPU" % b["id"])
            job.update(state="queued", updated=time.time())
            return True
        job.update(state="error", error=("ComfyUI injoignable (%s)" % type(e).__name__), updated=time.time())
        return False
    except Exception as e:
        job.update(state="error", error=("%s: %s" % (type(e).__name__, e))[:400], updated=time.time())
        return False
    finally:
        job["loading_model"] = False
        unlock()


def _gen_worker(job_id, data):
    job = GEN_JOBS[job_id]
    wanted = str(data.get("abiart_model") or "").strip() or CURRENT["model"] or DEFAULT_CKPT
    ckpt = find_model("checkpoints", wanted) or ""
    for _attempt in range(len(BACKENDS) + 1):
        b = _acquire(job_id, ckpt)
        if b is None:
            job.update(state="cancelled", error="Cancelled", updated=time.time())
            return
        try:
            retry = _run_on(job, b, data)
        finally:
            _release(b)
        if not retry:
            return
    job.update(state="error", error="No GPU backend available", updated=time.time())


def start_generate(data):
    if not isinstance(data, dict) or not data.get("prompt"):
        return {"error": "Missing prompt"}, 400
    now = time.time()
    with _GEN_LOCK:
        for jid in [j for j, v in GEN_JOBS.items() if v["state"] in ("done", "error", "cancelled") and now - v["updated"] > GEN_TTL]:
            GEN_JOBS.pop(jid, None)
        if sum(1 for v in GEN_JOBS.values() if v["state"] in ("queued", "running")) >= GEN_MAX_ACTIVE:
            return {"error": "Too many generations in progress, wait for one to finish"}, 429
        job_id = uuid.uuid4().hex[:16]
        GEN_JOBS[job_id] = {"id": job_id, "state": "queued", "created": now, "updated": now, "error": "", "cancel": False}
    threading.Thread(target=_gen_worker, args=(job_id, dict(data)), daemon=True).start()
    return {"success": True, "job_id": job_id}, 200


def gen_status(job_id):
    job = GEN_JOBS.get(job_id)
    if not job:
        return {"error": "Unknown job (the notebook may have been restarted)", "state": "error"}, 404
    out = {"state": job["state"], "error": job.get("error", "")}
    if job["state"] == "queued":
        out["queue_position"] = sum(1 for v in GEN_JOBS.values() if v["state"] == "queued" and v["created"] < job["created"])
    elif job["state"] == "running":
        out["gpu"] = job.get("backend", 0)
        out["progress"] = None
        out["loading_model"] = bool(job.get("loading_model"))
    elif job["state"] == "done":
        out.update(progress=1.0, gpu=job.get("backend", 0), seconds=job.get("seconds"), load_seconds=None,
                   images=job["images"], info=job["info"])
    return out, 200


def gen_cancel(job_id):
    job = GEN_JOBS.get(job_id)
    if not job:
        return {"error": "Unknown job"}, 404
    if job["state"] in ("queued", "running"):
        job["cancel"] = True
        if job["state"] == "queued":
            job.update(state="cancelled", error="Cancelled", updated=time.time())
    return {"success": True}, 200


def txt2img_sync(data):
    # Route historique /sdapi/v1/txt2img : meme calcul, reponse quand l'image est prete.
    payload, status = start_generate(data)
    if status != 200:
        return payload, status
    job = GEN_JOBS[payload["job_id"]]
    while job["state"] in ("queued", "running"):
        time.sleep(0.5)
    if job["state"] == "done":
        return {"images": job["images"], "info": job["info"], "parameters": {}}, 200
    return {"error": job.get("error") or "Generation failed"}, 500


# ----------------------------------------------------------------------------------------------
# Telechargement de modeles (Civitai / Hugging Face) + sauvegarde Hugging Face
# ----------------------------------------------------------------------------------------------
ALLOWED_HOSTS = ("civitai.com", "civitai.red", "huggingface.co")
MAX_BYTES = 14 * 1024 ** 3
MAX_ACTIVE_JOBS = 2
JOBS = {}
_LOCK = threading.Lock()


def _host_allowed(host):
    host = (host or "").lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_HOSTS)


def _is_public(host):
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except Exception:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def _clean_name(name):
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    name = re.sub(r"[^A-Za-z0-9._\-\(\) \[\]]", "_", name).lstrip(".")[:150]
    return name if name.lower().endswith(MODEL_EXT) else ""


def _open(url, token=""):
    current = url
    for hop in range(6):
        p = urlparse(current)
        host = p.hostname or ""
        if p.scheme != "https":
            raise ValueError("Only https URLs are allowed")
        if hop == 0 and not _host_allowed(host):
            raise ValueError("Host not allowed: " + host)
        if not _is_public(host):
            raise ValueError("Cannot reach %s from this machine, or its address is not public" % host)
        headers = {"User-Agent": "AbiArt-Kaggle/1.0"}
        if hop == 0 and token and "civitai" in host:
            headers["Authorization"] = "Bearer " + token   # jamais transmis apres une redirection
        r = requests.get(current, headers=headers, stream=True, allow_redirects=False, timeout=(15, 60))
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            current = urljoin(current, r.headers["Location"])
            r.close()
            continue
        return r
    raise ValueError("Too many redirects")


def _header_filename(resp):
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", cd, re.I)
    return m.group(1) if m else ""


def _backup_to_hf(job, final, kind, name):
    if not (HF_TOKEN and HF_REPO):
        job["backup"] = "disabled"
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
        job["backup"] = "running"
        api.upload_file(path_or_fileobj=final, path_in_repo="%s/%s" % (HF_FOLDERS[kind], name),
                        repo_id=HF_REPO, repo_type="model", commit_message="Add " + name)
        job["backup"] = "done"
        log("Sauvegarde Hugging Face terminee : %s" % name)
    except Exception as e:
        job["backup"] = "error"
        log("Sauvegarde Hugging Face impossible pour %s : %s" % (name, str(e)[:300]))


def _download_worker(job_id, url, token, kind, wanted_name):
    job = JOBS[job_id]
    tmp = None
    try:
        r = _open(url, token)
        if r.status_code in (401, 403) and token and "token=" not in url:
            r.close()
            r = _open(url + ("&" if "?" in url else "?") + "token=" + token, token)
        if r.status_code in (401, 403):
            raise RuntimeError("Download refused (HTTP %d) : %s" % (
                r.status_code, "Civitai refuse le token recu (se terminant par ...%s)" % token[-4:] if token
                else "aucun token Civitai recu par le notebook : colle-le dans le champ \"Civitai API Token\" du site"))
        if r.status_code != 200:
            raise RuntimeError("Download failed (HTTP %d)" % r.status_code)
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError("The link returned a web page, not a model file (login required?)")
        header_name = _header_filename(r)
        name = _clean_name(wanted_name) or _clean_name(header_name)
        if not name:
            if header_name:
                raise RuntimeError('Ce lien ne pointe pas vers un modele : fichier recu "%s"' % header_name)
            name = "model_%s.safetensors" % job_id[:6]
        total = int(r.headers.get("Content-Length") or 0)
        if total > MAX_BYTES:
            raise RuntimeError("File too large")
        dest_dir = os.path.join(MODELS, SUBDIRS[kind])
        os.makedirs(dest_dir, exist_ok=True)
        if total and shutil.disk_usage(dest_dir).free < total + 1024 ** 3:
            raise RuntimeError("Not enough free disk space on the Kaggle machine")
        final = os.path.join(dest_dir, name)
        job.update(filename=name, total=total)
        if os.path.exists(final) and os.path.getsize(final) > 0:
            size = os.path.getsize(final)
            job.update(state="done", done=size, total=size, note="already present")
            return
        tmp = final + ".part"
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if done > MAX_BYTES:
                    raise RuntimeError("File too large")
                job["done"] = done
        if total and done != total:
            raise RuntimeError("Incomplete download")
        os.replace(tmp, final)
        tmp = None
        job.update(state="done", done=done, total=done)
        _backup_to_hf(job, final, kind, name)
    except Exception as e:
        job.update(state="error", error=str(e))
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def start_download(data):
    url = (data.get("url") or "").strip()
    kind = (data.get("type") or "checkpoint").lower()
    if kind not in SUBDIRS:
        return {"error": "Unknown type: " + kind}, 400
    if not url:
        return {"error": "Missing url"}, 400
    wanted = _clean_name((data.get("filename") or "").strip())
    host = urlparse(url).hostname or ""
    if urlparse(url).scheme != "https" or not _host_allowed(host):
        return {"error": "Only https links from civitai.com, civitai.red or huggingface.co are allowed"}, 400
    with _LOCK:
        if sum(1 for j in JOBS.values() if j["state"] == "running") >= MAX_ACTIVE_JOBS:
            return {"error": "Too many downloads in progress, wait for one to finish"}, 429
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"state": "running", "kind": kind, "filename": wanted, "done": 0, "total": 0,
                        "error": "", "started": time.time()}
    token = (data.get("token") or "").strip().strip("\"'")
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    threading.Thread(target=_download_worker, args=(job_id, url, token, kind, wanted), daemon=True).start()
    return {"success": True, "job_id": job_id, "filename": wanted, "type": kind}, 200


def download_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "Unknown job (the notebook may have been restarted)", "state": "error"}, 404
    return {k: job.get(k) for k in ("state", "kind", "filename", "done", "total", "error", "note", "backup")}, 200


# ----------------------------------------------------------------------------------------------
# Verificateurs (recherche qualite) : detections YOLO sur une image, calculees sur le processeur
# pour ne pas prendre de memoire aux GPU. Sert a noter les images, jamais a les modifier.
# ----------------------------------------------------------------------------------------------
_DET_LOCK = threading.Lock()
_DET_MODELS = {}
DET_DEFAULT = ("hand_yolov8s.pt", "face_yolov8m.pt", "person_yolov8m-seg.pt")


def detect_available():
    try:
        import importlib.util
        return importlib.util.find_spec("ultralytics") is not None
    except Exception:
        return False


def detect(data):
    try:
        raw = base64.b64decode(str(data.get("image") or "").split(",")[-1])
    except Exception:
        raw = b""
    if not raw:
        return {"error": "Missing image"}, 400
    names = data.get("models") if isinstance(data.get("models"), list) and data.get("models") else list(DET_DEFAULT)
    min_conf = _num(data.get("min_conf"), 0.15, float, 0.01, 0.99)
    try:
        from PIL import Image
        from ultralytics import YOLO
    except Exception as e:
        return {"error": "Verificateurs indisponibles (%s: %s)" % (type(e).__name__, e)}, 501
    with _DET_LOCK:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return {"error": "Image illisible"}, 400
        out = {}
        for name in names[:8]:
            name = os.path.basename(str(name))
            path = os.path.join(MODELS, "ultralytics", "segm" if "-seg" in name else "bbox", name)
            if not os.path.exists(path):
                out[name] = {"error": "absent"}
                continue
            try:
                if name not in _DET_MODELS:
                    _DET_MODELS[name] = YOLO(path)
                res = _DET_MODELS[name].predict(img, conf=min_conf, device="cpu", verbose=False)[0]
                out[name] = {"boxes": [{"conf": round(float(c), 4), "box": [round(float(v), 1) for v in b]}
                                       for c, b in zip(res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]}
            except Exception as e:
                out[name] = {"error": ("%s: %s" % (type(e).__name__, e))[:200]}
        return {"width": img.width, "height": img.height, "detections": out}, 200


# ----------------------------------------------------------------------------------------------
# Routes HTTP
# ----------------------------------------------------------------------------------------------
def route_get(path):
    if path == "/abiart/ping":
        info = any_info()
        return {"ok": True, "version": 4, "engine": "comfyui", "jobs": True, "routing": True, "embeddings": True, "detect": detect_available(),
                "gpus": sum(1 for b in BACKENDS if backend_up(b)), "detailer": has_detailer(info),
                "a1111_weights": "BNK_CLIPTextEncodeAdvanced" in info}, 200
    m = re.fullmatch(r"/abiart/generate/([0-9a-f]+)", path)
    if m:
        return gen_status(m.group(1))
    m = re.fullmatch(r"/abiart/download/([0-9a-f]+)", path)
    if m:
        return download_status(m.group(1))
    if path == "/sdapi/v1/options":
        current = CURRENT["model"] or BACKENDS[0]["model"] or DEFAULT_CKPT
        return {"sd_model_checkpoint": find_model("checkpoints", current) or current}, 200
    if path == "/sdapi/v1/sd-models":
        return [{"title": f, "model_name": stem(f), "filename": os.path.join(MODELS, "checkpoints", f)}
                for f in list_models("checkpoints")], 200
    if path == "/sdapi/v1/loras":
        return [{"name": stem(f), "alias": stem(f), "path": os.path.join(MODELS, "loras", f)}
                for f in list_models("loras")], 200
    if path == "/sdapi/v1/embeddings":
        return {"loaded": {n: {} for n in sorted(embedding_names(), key=str.lower)}, "skipped": {}}, 200
    if path == "/sdapi/v1/samplers":
        return [{"name": a, "aliases": [c], "options": {}} for a, c in available_samplers(any_info())], 200
    if path == "/sdapi/v1/schedulers":
        return [{"name": (c or "automatic"), "label": a} for a, c in available_schedulers(any_info())], 200
    if path == "/sdapi/v1/upscalers":
        names = [stem(f) for f in list_models("upscale_models")]
        back = {v: k for k, v in UPSCALER_ALIASES.items()}
        return [{"name": back.get(n, n)} for n in names], 200
    if path == "/sdapi/v1/latent-upscale-modes":
        return [{"name": a} for a, _c in LATENT_MODES], 200
    if path == "/sdapi/v1/scripts":
        return {"txt2img": ["adetailer"] if has_detailer(any_info()) else [], "img2img": []}, 200
    if path == "/sdapi/v1/progress":
        return {"progress": 0, "eta_relative": None, "state": {}}, 200
    return {"error": "Not found"}, 404


def route_post(path, data):
    if path == "/abiart/generate":
        return start_generate(data)
    m = re.fullmatch(r"/abiart/generate/([0-9a-f]+)/cancel", path)
    if m:
        return gen_cancel(m.group(1))
    if path == "/abiart/download":
        return start_download(data)
    if path == "/abiart/detect":
        return detect(data)
    if path == "/sdapi/v1/options":
        wanted = str(data.get("sd_model_checkpoint") or "").strip()
        if wanted:
            if not find_model("checkpoints", wanted):
                return {"error": "model %s not found" % wanted}, 500
            CURRENT["model"] = wanted
        return {}, 200
    if path in ("/sdapi/v1/refresh-checkpoints", "/sdapi/v1/refresh-loras", "/sdapi/v1/refresh-vae",
                "/sdapi/v1/refresh-embeddings"):
        return {}, 200   # les dossiers sont relus a chaque demande
    if path == "/sdapi/v1/txt2img":
        return txt2img_sync(data)
    if path == "/sdapi/v1/interrupt":
        for job in list(GEN_JOBS.values()):
            if job["state"] == "running":
                job["cancel"] = True
        return {}, 200
    return {"error": "Not found"}, 404


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, payload, status):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle(self, fn, *args):
        try:
            payload, status = fn(*args)
        except Exception as e:
            payload, status = {"error": "%s: %s" % (type(e).__name__, e)}, 500
        try:
            self._send(payload, status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _path(self):
        # Chemin demande, sans le prefixe secret. None si le prefixe attendu est absent.
        path = urlparse(self.path).path.rstrip("/") or "/"
        if not PATH_PREFIX:
            return path
        if path == PATH_PREFIX or path.startswith(PATH_PREFIX + "/"):
            return path[len(PATH_PREFIX):] or "/"
        return None

    def do_GET(self):
        path = self._path()
        if path is None:
            return self._handle(lambda: ({"error": "Not found"}, 404))
        self._handle(route_get, path)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except Exception:
            data = {}
        path = self._path()
        if path is None:
            return self._handle(lambda: ({"error": "Not found"}, 404))
        self._handle(route_post, path, data if isinstance(data, dict) else {})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    for b in BACKENDS:
        os.makedirs(b["out"], exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    log("Passerelle prete sur le port %d (%d instance(s) ComfyUI prevue(s))" % (PORT, len(BACKENDS)))
    server.serve_forever()


if __name__ == "__main__":
    main()
