"""
AbiArt — banc d'essai qualité (prototype « générer, vérifier, réparer »).

Trois commandes, à lancer depuis le dossier du projet. Python standard uniquement, rien à installer.

  python research/bench.py run --url https://xxxx.trycloudflare.com
      Génère les images du jeu de prompts sur le notebook Kaggle (ComfyUI) et relève, pour chacune,
      le temps de calcul et les détections des vérificateurs (mains, visages, personnages).
      Reprend là où il s'est arrêté : une image déjà présente n'est pas recalculée.

  python research/bench.py label
      Ouvre une page locale pour juger les images à l'aveugle (ordre mélangé, aucune indication de
      réglage ni de score). C'est la mesure de référence : les vérificateurs automatiques sont
      eux-mêmes jugés contre ces réponses.

  python research/bench.py analyze
      Calcule les taux de défauts, la fiabilité des vérificateurs, le gain simulé d'un choix
      automatique entre deux images et l'effet de la retouche. Écrit aussi report.md.

Les résultats vont dans research/runs/<nom> (option --run, « e1 » par défaut).
"""
import argparse
import base64
import json
import math
import os
import queue
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"User-Agent": "Mozilla/5.0 (AbiArt-Bench)", "Accept": "application/json"}
CONF_MIN = 0.3            # seuil de confiance retenu pour compter une détection
LABEL_SEED = 20261113     # ordre de présentation des images à l'étiquetage


# ------------------------------------------------------------------------------------------------
# Fichiers
# ------------------------------------------------------------------------------------------------
def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run_dir(name):
    d = os.path.join(HERE, "runs", name)
    for sub in ("images", "meta"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def load_setup(args):
    cfg = load_json(os.path.join(HERE, args.config))
    prompts = load_json(os.path.join(HERE, args.prompts))
    if not cfg or not prompts:
        sys.exit("Fichiers config.json / prompts introuvables dans " + HERE)
    return cfg, prompts["prompts"]


def all_meta(d):
    out = {}
    for name in sorted(os.listdir(os.path.join(d, "meta"))):
        if name.endswith(".json"):
            m = load_json(os.path.join(d, "meta", name))
            if m and os.path.exists(os.path.join(d, "images", m["id"] + ".png")):
                out[m["id"]] = m
    return out


# ------------------------------------------------------------------------------------------------
# run : génération + détections
# ------------------------------------------------------------------------------------------------
def http_json(method, url, body=None, timeout=60):
    """Renvoie (code HTTP, contenu JSON ou None). Code 0 = pas de réponse."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = dict(HEADERS, **({"Content-Type": "application/json"} if data else {}))
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, code = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    except Exception:
        return 0, None
    try:
        return code, json.loads(raw)
    except Exception:
        return code, None


def build_tasks(cfg, prompts, variants):
    tasks = []
    for vname in variants:
        v = cfg["variants"][vname]
        seeds = cfg["seeds"] if v.get("seeds") == "all" else cfg["seeds"][:1]
        for p in prompts:
            if p["cat"] not in v["cats"]:
                continue
            for seed in seeds:
                tasks.append({"id": "%s_s%d_%s" % (p["id"], seed, vname), "pid": p["id"], "cat": p["cat"],
                              "chars": p["chars"], "seed": seed, "variant": vname, "prompt": p["prompt"]})
    return tasks


def build_payload(cfg, task):
    quality = (cfg.get("quality") or "").strip()
    payload = {
        "prompt": task["prompt"] + (", " + quality if quality else ""),
        "negative_prompt": cfg.get("negative", ""),
        "steps": cfg["steps"], "cfg_scale": cfg["cfg"], "width": cfg["width"], "height": cfg["height"],
        "sampler_name": cfg["sampler"], "scheduler": cfg.get("scheduler", "Automatic"), "seed": task["seed"],
        "override_settings": {"CLIP_stop_at_last_layers": cfg.get("clip_skip", 1)},
    }
    if cfg.get("model"):
        payload["abiart_model"] = cfg["model"]
    tabs = cfg["variants"][task["variant"]].get("detailers") or []
    if tabs:
        full = [dict({"ad_prompt": "", "ad_negative_prompt": "", "ad_use_steps": True, "ad_steps": cfg["steps"]}, **t)
                for t in tabs]
        payload["alwayson_scripts"] = {"ADetailer": {"args": [True, False] + full}}
    return payload


def generate(base, payload, timeout=900):
    """Renvoie (image PNG en base64, infos) ou lève RuntimeError."""
    for _ in range(60):
        code, js = http_json("POST", base + "/abiart/generate", payload, timeout=60)
        if code == 200 and js and js.get("job_id"):
            break
        if code == 429 or code == 0 or code >= 502:
            time.sleep(5)        # file pleine ou tunnel momentanément indisponible
            continue
        raise RuntimeError((js or {}).get("error") or "HTTP %d au démarrage du job" % code)
    else:
        raise RuntimeError("le backend ne répond pas")
    job, deadline, failures = js["job_id"], time.time() + timeout, 0
    while time.time() < deadline:
        time.sleep(2)
        code, st = http_json("GET", base + "/abiart/generate/" + job, timeout=30)
        if code != 200 or not st:
            failures += 1
            if code == 404 or failures >= 10:
                raise RuntimeError("job perdu (HTTP %d) : le notebook a-t-il redémarré ?" % code)
            continue
        failures = 0
        if st.get("state") == "done":
            return st["images"][0], {"seconds": st.get("seconds"), "gpu": st.get("gpu")}
        if st.get("state") in ("error", "cancelled"):
            raise RuntimeError(st.get("error") or "génération échouée")
    http_json("POST", base + "/abiart/generate/%s/cancel" % job, {}, timeout=10)
    raise RuntimeError("génération trop longue (%d s)" % timeout)


def detect(base, image_b64, models):
    for _ in range(3):
        code, js = http_json("POST", base + "/abiart/detect", {"image": image_b64, "models": models}, timeout=300)
        if code == 200 and js and "detections" in js:
            return js["detections"], None
        if code in (0, 502, 503, 504, 524):
            time.sleep(4)
            continue
        return None, (js or {}).get("error") or "HTTP %d" % code
    return None, "pas de réponse"


def cmd_run(args):
    cfg, prompts = load_setup(args)
    d = run_dir(args.run)
    base = args.url.rstrip("/")
    code, ping = http_json("GET", base + "/abiart/ping", timeout=30)
    if code != 200 or not ping or not ping.get("jobs"):
        sys.exit("Backend injoignable ou trop ancien (%s). Vérifiez l'URL et que le notebook ComfyUI tourne." % code)
    can_detect = bool(ping.get("detect"))
    workers = max(1, min(int(ping.get("gpus") or 1), 4))
    print("Backend : %s, %d GPU, vérificateurs %s" % (ping.get("engine", "?"), workers,
                                                      "disponibles" if can_detect else "ABSENTS (ancien notebook)"))
    if not can_detect:
        print("  -> les images seront générées sans détections ; relancez « run » avec le notebook à jour pour les ajouter.")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()] or list(cfg["variants"])
    unknown = [v for v in variants if v not in cfg["variants"]]
    if unknown:
        sys.exit("Variante(s) inconnue(s) dans config.json : " + ", ".join(unknown))
    tasks = build_tasks(cfg, prompts, variants)
    if args.limit:
        tasks = tasks[:args.limit]
    save_json(os.path.join(d, "setup.json"), {"config": cfg, "url_host": base.split("//")[-1].split(".")[0][:6] + "…",
                                              "started": time.strftime("%Y-%m-%d %H:%M")})
    todo = queue.Queue()
    n_gen = 0
    for t in tasks:
        meta = load_json(os.path.join(d, "meta", t["id"] + ".json"))
        has_img = os.path.exists(os.path.join(d, "images", t["id"] + ".png"))
        if meta and has_img and (meta.get("detections") or not can_detect):
            continue
        n_gen += 0 if (meta and has_img) else 1
        todo.put(t)
    total = todo.qsize()
    print("%d image(s) prévue(s), %d à générer, %d à compléter (détections seules)." % (len(tasks), n_gen, total - n_gen))
    if not total:
        return
    lock, done, errors, t0 = threading.Lock(), [0], [], time.time()

    def worker():
        while True:
            try:
                t = todo.get_nowait()
            except queue.Empty:
                return
            img_path = os.path.join(d, "images", t["id"] + ".png")
            meta_path = os.path.join(d, "meta", t["id"] + ".json")
            try:
                meta = load_json(meta_path)
                if meta and os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    note = "détections"
                else:
                    b64, info = generate(base, build_payload(cfg, t))
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(b64))
                    meta = dict(t, **info)
                    note = "%s s, GPU %s" % (info.get("seconds"), info.get("gpu"))
                if can_detect:
                    dets, err = detect(base, b64, cfg["detectors"])
                    if dets is not None:
                        meta["detections"] = dets
                    else:
                        meta["detect_error"] = err
                        note += " | détection : " + str(err)[:80]
                save_json(meta_path, meta)
            except Exception as e:
                note = "ÉCHEC : %s" % e
                with lock:
                    errors.append(t["id"])
            with lock:
                done[0] += 1
                left = (time.time() - t0) / done[0] * (total - done[0])
                print("[%3d/%d] %-24s %s   (reste ~%d min)" % (done[0], total, t["id"], note, round(left / 60)), flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("\nTerminé en %d min. %d échec(s)%s." % ((time.time() - t0) / 60, len(errors),
                                                  " : " + ", ".join(errors[:8]) if errors else ""))
    if errors:
        print("Relancez la même commande : seules les images manquantes seront refaites.")


# ------------------------------------------------------------------------------------------------
# label : étiquetage humain à l'aveugle
# ------------------------------------------------------------------------------------------------
LABEL_PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>AbiArt — évaluation à l'aveugle</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0b0c0f;--panel:#14161b;--raised:#1d2027;--text:#edeef2;--muted:rgba(237,238,242,.62);--border:rgba(255,255,255,.1);
--accent:#8f7cf7;--ok:#3ecf8e;--bad:#f2777a;--none:#8a8f9c}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:15px/1.5 Inter,system-ui,"Segoe UI",sans-serif;height:100vh;display:flex;overflow:hidden}
#stage{flex:1;display:flex;align-items:center;justify-content:center;padding:20px;min-width:0;min-height:0}
#stage img{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.6)}
aside{width:340px;flex:none;background:var(--panel);border-left:1px solid var(--border);padding:24px;display:flex;flex-direction:column;gap:18px;overflow-y:auto}
h1{font-size:17px;font-weight:650;letter-spacing:-.01em}
.bar{height:6px;border-radius:99px;background:var(--raised);overflow:hidden}
.bar>div{height:100%;background:var(--accent);width:0;transition:width .2s}
.muted{color:var(--muted);font-size:13px}
.q{border:1px solid var(--border);border-radius:14px;padding:16px;display:flex;flex-direction:column;gap:10px;opacity:.45}
.q.active{opacity:1;border-color:var(--accent)}
.q h2{font-size:14px;font-weight:600}
.choices{display:grid;gap:8px}
.choices.count{grid-template-columns:repeat(4,1fr)}
button{font:inherit;color:var(--text);background:var(--raised);border:1px solid var(--border);border-radius:10px;padding:10px 12px;cursor:pointer;text-align:left;display:flex;align-items:center;gap:10px}
button:hover{border-color:rgba(255,255,255,.3)}
button.sel{border-color:var(--accent);background:rgba(143,124,247,.18)}
kbd{font:600 12px ui-monospace,Consolas,monospace;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:2px 7px;min-width:24px;text-align:center}
.dot{width:9px;height:9px;border-radius:50%;flex:none}
.nav{display:flex;gap:8px;margin-top:auto}
.nav button{flex:1;justify-content:center}
.count button{justify-content:center}
#done{display:none;text-align:center;padding:40px 10px}
#done h2{font-size:20px;margin-bottom:8px}
@media (max-width:720px){body{flex-direction:column;overflow:auto;height:auto}aside{width:100%;border-left:none;border-top:1px solid var(--border)}#stage{flex:none;height:60vh}}
</style></head><body>
<div id="stage"><img id="img" alt="Image à évaluer"></div>
<aside>
  <div><h1>Évaluation à l'aveugle</h1><p class="muted" id="progress">Chargement…</p></div>
  <div class="bar"><div id="fill"></div></div>
  <div id="work">
    <div class="q" id="qHands">
      <h2>Les mains visibles sont-elles correctes ?</h2>
      <div class="choices">
        <button data-hands="ok"><kbd>1</kbd><span class="dot" style="background:var(--ok)"></span>Correctes</button>
        <button data-hands="bad"><kbd>2</kbd><span class="dot" style="background:var(--bad)"></span>Au moins une main ratée</button>
        <button data-hands="none"><kbd>3</kbd><span class="dot" style="background:var(--none)"></span>Aucune main visible</button>
      </div>
      <p class="muted">Ratée = mauvais nombre de doigts, doigts fusionnés ou main difforme. Jugez comme un visiteur du site, sans zoomer.</p>
    </div>
    <div class="q" id="qCount" style="margin-top:14px">
      <h2>Combien de personnages voyez-vous ?</h2>
      <div class="choices count" id="countBtns"></div>
    </div>
  </div>
  <div id="done"><h2>Terminé, merci !</h2><p class="muted">Toutes les images sont évaluées. Vous pouvez fermer cette page.</p></div>
  <div class="nav"><button id="prev"><kbd>←</kbd> Précédente</button><button id="next">Suivante <kbd>→</kbd></button></div>
</aside>
<script>
let items = [], labels = {}, cur = 0, step = 'hands';
const $ = id => document.getElementById(id);
const countBtns = $('countBtns');
for (let n = 0; n <= 7; n++) { const b = document.createElement('button'); b.dataset.count = n; b.innerHTML = '<kbd>' + n + '</kbd>'; countBtns.appendChild(b); }

function complete(i) { const l = labels[i]; return !!l && !!l.hands && (!items[i].ask_count || l.count !== undefined && l.count !== null); }
function firstTodo() { const i = items.findIndex((_, k) => !complete(k)); return i < 0 ? items.length : i; }

function render() {
  const n = items.length, doneN = items.filter((_, k) => complete(k)).length;
  $('progress').textContent = doneN + ' / ' + n + ' images évaluées';
  $('fill').style.width = (n ? 100 * doneN / n : 0) + '%';
  const finished = cur >= n;
  $('work').style.display = finished ? 'none' : '';
  $('done').style.display = finished ? 'block' : 'none';
  $('img').style.visibility = finished ? 'hidden' : 'visible';
  if (finished) return;
  const it = items[cur], l = labels[cur] || {};
  if ($('img').dataset.i !== String(cur)) { $('img').dataset.i = cur; $('img').src = '/img/' + cur; }
  $('qCount').style.display = it.ask_count ? '' : 'none';
  $('qHands').classList.toggle('active', step === 'hands');
  $('qCount').classList.toggle('active', step === 'count');
  document.querySelectorAll('[data-hands]').forEach(b => b.classList.toggle('sel', l.hands === b.dataset.hands));
  document.querySelectorAll('[data-count]').forEach(b => b.classList.toggle('sel', l.count === Number(b.dataset.count)));
}

async function save(i) {
  await fetch('/api/label', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(Object.assign({ i }, labels[i])) });
}
function go(i) { cur = Math.max(0, Math.min(items.length, i)); step = 'hands'; render(); }
function setHands(v) {
  if (cur >= items.length) return;
  labels[cur] = Object.assign({}, labels[cur], { hands: v });
  if (items[cur].ask_count) { step = 'count'; save(cur); render(); } else { save(cur); go(cur + 1); }
}
function setCount(n) {
  if (cur >= items.length || !items[cur].ask_count) return;
  labels[cur] = Object.assign({}, labels[cur], { count: n });
  save(cur);
  if (labels[cur].hands) go(cur + 1); else { step = 'hands'; render(); }
}
document.querySelectorAll('[data-hands]').forEach(b => b.onclick = () => setHands(b.dataset.hands));
document.querySelectorAll('[data-count]').forEach(b => b.onclick = () => setCount(Number(b.dataset.count)));
$('prev').onclick = () => go(cur - 1);
$('next').onclick = () => go(cur + 1);
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') return go(cur - 1);
  if (e.key === 'ArrowRight') return go(cur + 1);
  const m = /^(?:Digit|Numpad)(\d)$/.exec(e.code);   // indépendant du clavier (AZERTY sans Maj)
  if (!m) return;
  const d = Number(m[1]);
  if (step === 'hands') { if (d >= 1 && d <= 3) setHands(['ok', 'bad', 'none'][d - 1]); }
  else if (d <= 7) setCount(d);
});
fetch('/api/state').then(r => r.json()).then(s => { items = s.items; labels = s.labels; go(firstTodo()); });
</script></body></html>
"""


def label_order(d, metas, labels=None, recheck=0):
    """Ordre de présentation stable : les images déjà placées gardent leur rang, les nouvelles sont mélangées à la suite.

    recheck > 0 : glisse parmi les nouvelles ce nombre d'images « mains » déjà jugées, à rejuger sans le savoir
    (identifiant « <image>#2 »). Sert à mesurer si l'évaluateur juge pareil d'une séance à l'autre."""
    path = os.path.join(d, "order.json")
    order = [i for i in (load_json(path, []) or []) if i.split("#")[0] in metas]
    known = set(order)
    new = sorted(i for i in metas if i not in known)
    missing = recheck - sum(1 for i in order if i.endswith("#2"))
    if missing > 0 and labels:
        pool = sorted(i for i in metas if i in known and i + "#2" not in known and metas[i]["cat"] == "hands"
                      and (labels.get(i) or {}).get("hands") in ("ok", "bad"))
        random.Random(LABEL_SEED).shuffle(pool)
        new += [i + "#2" for i in pool[:missing]]
    random.Random(LABEL_SEED + len(order)).shuffle(new)
    order += new
    save_json(path, order)
    return order


def cmd_label(args):
    d = run_dir(args.run)
    metas = all_meta(d)
    if not metas:
        sys.exit("Aucune image dans %s : lancez d'abord « run »." % d)
    labels_path = os.path.join(d, "labels.json")
    order = label_order(d, metas, load_json(labels_path, {}) or {}, args.recheck)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, LABEL_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                saved = load_json(labels_path, {}) or {}
                state = {"items": [{"ask_count": metas[i.split("#")[0]]["cat"] == "count"} for i in order],
                         "labels": {str(k): saved[i] for k, i in enumerate(order) if i in saved}}
                return self._send(200, json.dumps(state).encode("utf-8"))
            if path.startswith("/img/") and path[5:].isdigit() and int(path[5:]) < len(order):
                with open(os.path.join(d, "images", order[int(path[5:])].split("#")[0] + ".png"), "rb") as f:
                    return self._send(200, f.read(), "image/png")
            self._send(404, b"{}")

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, b"{}")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                i = int(body["i"])
                entry = {}
                if body.get("hands") in ("ok", "bad", "none"):
                    entry["hands"] = body["hands"]
                if isinstance(body.get("count"), int) and 0 <= body["count"] <= 7:
                    entry["count"] = body["count"]
                with lock:
                    saved = load_json(labels_path, {}) or {}
                    saved[order[i]] = entry
                    save_json(labels_path, saved)
                self._send(200, b'{"ok":true}')
            except Exception:
                self._send(400, b'{"ok":false}')

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://localhost:%d" % args.port
    print("Page d'évaluation : %s   (%d images ; Ctrl+C pour arrêter, les réponses sont enregistrées au fur et à mesure)" % (url, len(order)))
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------------------------------------
# analyze : statistiques
# ------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def pct(k, n):
    if not n:
        return "—  (aucune donnée)"
    lo, hi = wilson(k, n)
    return "%3.0f %%  (%d/%d ; intervalle à 95 %% : %.0f–%.0f %%)" % (100 * k / n, k, n, 100 * lo, 100 * hi)


def auc(scores, positives):
    """Probabilité qu'un exemple positif ait un score plus élevé qu'un négatif (0,5 = hasard)."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def sign_test(a, b):
    """Test du signe bilatéral : probabilité d'un écart au moins aussi grand entre a et b si le hasard seul jouait."""
    n, k = a + b, min(a, b)
    if not n:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boxes(meta, model):
    det = (meta.get("detections") or {}).get(model) or {}
    return [b["conf"] for b in det.get("boxes", []) if b["conf"] >= CONF_MIN] if "boxes" in det else None


def hand_features(meta, detectors):
    out = {}
    for m in detectors:
        if not m.startswith("hand_"):
            continue
        confs = boxes(meta, m)
        if confs is None:
            continue
        short = m.replace(".pt", "")
        out[short + " : confiance minimale"] = min(confs) if confs else 0.0
        out[short + " : confiance moyenne"] = sum(confs) / len(confs) if confs else 0.0
    return out


def cmd_analyze(args):
    cfg, _prompts = load_setup(args)
    d = run_dir(args.run)
    metas = all_meta(d)
    labels = load_json(os.path.join(d, "labels.json"), {}) or {}
    detectors = cfg["detectors"]
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    def lab(i):
        return labels.get(i) or {}

    by_variant = {}
    for m in metas.values():
        by_variant.setdefault(m["variant"], []).append(m)
    n_lab = sum(1 for i in metas if lab(i).get("hands"))
    out("# Banc d'essai AbiArt — série « %s »" % args.run)
    out("")
    out("%d images, dont %d évaluées par un humain. Réglages : %s, %d pas, CFG %s, %d×%d, modèle « %s »." % (
        len(metas), n_lab, cfg["sampler"], cfg["steps"], cfg["cfg"], cfg["width"], cfg["height"],
        cfg.get("model") or "par défaut du notebook"))
    if n_lab < len(metas):
        out("ATTENTION : %d image(s) pas encore évaluée(s) ; les chiffres ci-dessous sont partiels." % (len(metas) - n_lab))

    # ---- 1. coût
    out("\n## 1. Coût de calcul")
    for v, ms in sorted(by_variant.items()):
        secs = [m["seconds"] for m in ms if isinstance(m.get("seconds"), (int, float))]
        if secs:
            out("- %-9s : %5.1f s par image en moyenne (%d images)" % (v, sum(secs) / len(secs), len(secs)))

    base = {m["id"]: m for m in by_variant.get("baseline", [])}

    # ---- 2. taux de défauts de référence
    out("\n## 2. Référence (une passe simple, jugement humain)")
    hv = [m for m in base.values() if m["cat"] == "hands" and lab(m["id"]).get("hands") in ("ok", "bad")]
    out("- Mains correctes (prompts « mains », mains visibles) : " + pct(sum(lab(m["id"])["hands"] == "ok" for m in hv), len(hv)))
    hidden = sum(1 for m in base.values() if m["cat"] == "hands" and lab(m["id"]).get("hands") == "none")
    if hidden:
        out("  (%d image(s) sans main visible, exclues)" % hidden)
    cv = [m for m in base.values() if m["cat"] == "count" and lab(m["id"]).get("count") is not None]
    out("- Bon nombre de personnages (prompts « comptage »)   : " + pct(sum(lab(m["id"])["count"] == m["chars"] for m in cv), len(cv)))
    for n in sorted({m["chars"] for m in cv}):
        sub = [m for m in cv if m["chars"] == n]
        out("    %d personnages demandés : %d/%d corrects" % (n, sum(lab(m["id"])["count"] == n for m in sub), len(sub)))

    # ---- 3. les vérificateurs disent-ils la même chose que l'humain ?
    out("\n## 3. Fiabilité des vérificateurs automatiques (hypothèse à tester, pas un acquis)")
    feats = {}
    for m in hv:
        for name, val in hand_features(m, detectors).items():
            feats.setdefault(name, []).append((val, lab(m["id"])["hands"] == "ok"))
    best_feat, best_auc = None, 0.0
    if not feats:
        out("- Mains : aucune détection enregistrée (notebook sans vérificateurs ?).")
    for name, pairs in sorted(feats.items()):
        a = auc([p[0] for p in pairs], [p[1] for p in pairs])
        if a is None:
            out("- Mains, %-36s : non calculable (il faut des mains correctes ET ratées)" % name)
            continue
        out("- Mains, %-36s : AUC %.2f sur %d images" % (name, a, len(pairs)))
        if a > best_auc:
            best_feat, best_auc = name, a
    if feats:
        out("  Lecture : 0,50 = hasard, 0,70 = signal utile, 0,85 et plus = fiable.")
    for model, label in (("face_yolov8m.pt", "visages"), ("person_yolov8m-seg.pt", "personnes")):
        sub = [(len(boxes(m, model)), lab(m["id"])["count"]) for m in cv if boxes(m, model) is not None]
        if sub:
            out("- Comptage par détection de %-9s : d'accord avec l'humain sur " % label + pct(sum(a == b for a, b in sub), len(sub)))

    # ---- 4. choix automatique entre deux images (simulation, sans nouvelle génération)
    out("\n## 4. Choisir la meilleure de deux images (simulation sur les paires de seeds)")
    seeds = cfg["seeds"]
    if len(seeds) >= 2:
        def pair(pid):
            a, b = base.get("%s_s%d_baseline" % (pid, seeds[0])), base.get("%s_s%d_baseline" % (pid, seeds[1]))
            return (a, b) if a and b else None

        pairs = [p for p in (pair(pid) for pid in sorted({m["pid"] for m in hv})) if p
                 and all(lab(x["id"]).get("hands") in ("ok", "bad") for x in p)]
        if pairs:
            ok = lambda x: lab(x["id"])["hands"] == "ok"
            n = len(pairs)
            out("Mains (%d paires) :" % n)
            out("- une seule image (situation actuelle)   : %3.0f %% correctes" % (100 * sum(ok(a) + ok(b) for a, b in pairs) / (2 * n)))
            if best_feat:
                pick = [a if hand_features(a, detectors).get(best_feat, 0) >= hand_features(b, detectors).get(best_feat, 0) else b
                        for a, b in pairs]
                out("- choix automatique (« %s ») : " % best_feat + pct(sum(ok(x) for x in pick), n))
            out("- choix parfait (plafond théorique)      : " + pct(sum(ok(a) or ok(b) for a, b in pairs), n))
            if best_feat:
                out("  Réserve : le critère est choisi ET testé sur les mêmes images, donc ce gain est optimiste."
                    " À confirmer sur des seeds neuves.")
        cpairs = [p for p in (pair(pid) for pid in sorted({m["pid"] for m in cv})) if p
                  and all(lab(x["id"]).get("count") is not None for x in p)]
        if cpairs:
            good = lambda x: lab(x["id"])["count"] == x["chars"]
            n = len(cpairs)
            out("Comptage (%d paires) :" % n)
            out("- une seule image                        : %3.0f %% correctes" % (100 * sum(good(a) + good(b) for a, b in cpairs) / (2 * n)))
            for model, label in (("face_yolov8m.pt", "visages"), ("person_yolov8m-seg.pt", "personnes")):
                if all(boxes(x, model) is not None for p in cpairs for x in p):
                    pick = [b if (len(boxes(b, model)) == b["chars"] and len(boxes(a, model)) != a["chars"]) else a for a, b in cpairs]
                    out("- choix automatique (détection de %-9s) : " % label + pct(sum(good(x) for x in pick), n))
            out("- choix parfait (plafond théorique)      : " + pct(sum(good(a) or good(b) for a, b in cpairs), n))
        if not pairs and not cpairs:
            out("Pas encore de paire complète évaluée.")

    # ---- 5. retouche : corrige-t-elle sans abîmer ? (toujours sur les mêmes seeds)
    out("\n## 5. Retouche visage + mains (même seed, avec et sans)")

    def hands_ok(i):
        v = lab(i).get("hands")
        return None if v not in ("ok", "bad") else v == "ok"

    keyed = {v: {(m["pid"], m["seed"]): m["id"] for m in ms} for v, ms in by_variant.items()}

    def hand_strength(v):
        for t in cfg["variants"].get(v, {}).get("detailers") or []:
            if str(t.get("ad_model", "")).startswith("hand_"):
                return t.get("ad_denoising_strength")
        return None

    def compare(va, vb):
        """Paires (même prompt, même seed) jugées dans les deux variantes : gagnées, perdues, correctes des deux côtés, ratées des deux côtés."""
        won = lost = both = neither = 0
        for key, ia in keyed.get(va, {}).items():
            ib = keyed.get(vb, {}).get(key)
            a, b = hands_ok(ia), hands_ok(ib) if ib else None
            if a is None or b is None:
                continue
            won += (not a) and b
            lost += a and (not b)
            both += a and b
            neither += (not a) and (not b)
        return won, lost, both, neither

    repairs = [v for v in cfg["variants"] if v in by_variant and cfg["variants"][v].get("detailers")]
    if not repairs:
        out("Aucune variante avec retouche dans cette série.")
    for v in repairs:
        won, lost, both, neither = compare("baseline", v)
        n = won + lost + both + neither
        if not n:
            out("- %s : pas encore de paire évaluée." % v)
            continue
        out("- %s (mains à %s) : " % (v, hand_strength(v)) + pct(both + won, n))
        out("    par rapport à sans retouche (%.0f %%) : %d réparées, %d abîmées, p = %.2f" % (
            100 * (both + lost) / n, won, lost, sign_test(won, lost) or 1.0))
    ref = "repair"
    for v in repairs:
        if v == ref or ref not in repairs:
            continue
        won, lost, both, neither = compare(ref, v)
        n = won + lost + both + neither
        if n:
            p = sign_test(won, lost)
            out("- %s contre %s : %d mains gagnées, %d perdues sur %d paires, p = %s" % (
                v, ref, won, lost, n, "%.2f" % p if p is not None else "— (aucune différence)"))
    common = [k for k in keyed.get("baseline", {}) if all(
        k in keyed.get(v, {}) and hands_ok(keyed[v][k]) is not None for v in ["baseline"] + repairs)]
    if len(repairs) > 1 and common:
        out("Sur les %d images jugées dans toutes les variantes :" % len(common))
        for v in ["baseline"] + repairs:
            out("    %-9s : %3.0f %% de mains correctes" % (v, 100 * sum(hands_ok(keyed[v][k]) for k in common) / len(common)))
    out("  Lecture : p est la probabilité d'un tel écart par hasard ; en dessous de 0,05 l'écart est net.")

    # ---- 6. l'évaluateur juge-t-il pareil deux fois ?
    again = [(i[:-2], v.get("hands")) for i, v in labels.items() if i.endswith("#2") and v.get("hands")]
    again = [(lab(i).get("hands"), second) for i, second in again if lab(i).get("hands")]
    if again:
        same = sum(a == b for a, b in again)
        out("\n## 6. Constance de l'évaluateur (images rejugées sans le savoir)")
        out("- Même réponse les deux fois : " + pct(same, len(again)))
        out("- Détail : %d passées de correcte à ratée, %d de ratée à correcte" % (
            sum(a == "ok" and b == "bad" for a, b in again), sum(a == "bad" and b == "ok" for a, b in again)))
        out("  Lecture : en dessous de 85 %, les écarts entre séances d'évaluation sont à prendre avec prudence.")

    out("\n## Limites de cette mesure")
    out("- Un seul évaluateur humain, un seul modèle, %d seeds : les intervalles ci-dessus sont larges." % len(seeds))
    out("- Les détecteurs servant à noter (%s) ne sont pas ceux de la retouche (yolov9c), mais viennent de la même famille." % ", ".join(
        x.replace(".pt", "") for x in detectors))
    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nRapport écrit dans " + os.path.join(d, "report.md"))


# ------------------------------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Banc d'essai qualité AbiArt")
    ap.add_argument("--run", default="e1", help="nom de la série (dossier research/runs/<nom>)")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--prompts", default="prompts_v1.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="générer les images et relever les détections")
    r.add_argument("--url", required=True, help="URL du notebook Kaggle (https://xxxx.trycloudflare.com)")
    r.add_argument("--variants", default="", help="liste séparée par des virgules ; par défaut, toutes celles de config.json")
    r.add_argument("--limit", type=int, default=0, help="ne traiter que les N premières images (essai rapide)")
    lb = sub.add_parser("label", help="évaluer les images à l'aveugle dans le navigateur")
    lb.add_argument("--port", type=int, default=5058)
    lb.add_argument("--no-browser", action="store_true")
    lb.add_argument("--recheck", type=int, default=0,
                    help="nombre d'images déjà jugées à rejuger à l'aveugle (constance de l'évaluateur)")
    sub.add_parser("analyze", help="calculer les résultats")
    args = ap.parse_args()
    {"run": cmd_run, "label": cmd_label, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
