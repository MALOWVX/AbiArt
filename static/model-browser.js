/* ============================================================
   AbiArt — fenêtre « Parcourir les modèles » (Civitai)

   Chargé après le script de index.html, dont il réutilise :
   currentBackend, showToast, updateStatus, kaggleDownloadAndWait, afterKaggleDownload,
   downloadToModal, civitaiKind, uiSetSelect, uiSetSlider, autoSaveSettings.
   ============================================================ */
(function () {
    'use strict';

    // Familles de modèles : `bases` = noms exacts utilisés par Civitai, `arch` = architecture réelle.
    // Un LoRA ou un embedding ne se charge correctement que sur un checkpoint de la même architecture.
    const FAMILIES = [
        { key: 'sd15', label: 'SD 1.5', arch: 'sd15', bases: ['SD 1.5', 'SD 1.5 LCM', 'SD 1.5 Hyper'] },
        { key: 'sdxl', label: 'SDXL', arch: 'sdxl', bases: ['SDXL 1.0', 'SDXL Lightning', 'SDXL Hyper', 'SDXL Turbo'] },
        { key: 'pony', label: 'Pony', arch: 'sdxl', bases: ['Pony'] },
        { key: 'illustrious', label: 'Illustrious', arch: 'sdxl', bases: ['Illustrious'] },
        { key: 'noobai', label: 'NoobAI', arch: 'sdxl', bases: ['NoobAI'] },
        { key: 'flux', label: 'Flux', arch: 'flux', bases: ['Flux.1 D', 'Flux.1 S'] },
    ];
    // Pour un LoRA, les familles proches de celle du checkpoint (Illustrious et NoobAI partagent leurs LoRA).
    const LORA_NEIGHBOURS = { illustrious: ['illustrious', 'noobai'], noobai: ['noobai', 'illustrious'] };

    const TYPES = [
        { key: 'Checkpoint', label: 'Checkpoints' },
        { key: 'LORA', label: 'LoRA' },
        { key: 'TextualInversion', label: 'Embeddings' },
    ];
    const TYPE_LABEL = { Checkpoint: 'Checkpoint', LORA: 'LoRA', TextualInversion: 'Embedding' };
    const LORA_TAGS = [
        { key: 'character', label: 'Personnage' }, { key: 'style', label: 'Style' }, { key: 'concept', label: 'Concept' },
        { key: 'clothing', label: 'Vêtements' }, { key: 'poses', label: 'Pose' }, { key: 'background', label: 'Décor' },
    ];
    const SORTS = [['Most Downloaded', 'Les plus téléchargés'], ['Most Liked', 'Les plus aimés'], ['Newest', 'Les plus récents']];
    const PERIODS = [['Week', 'Cette semaine'], ['Month', 'Ce mois'], ['Year', 'Cette année'], ['AllTime', 'Depuis toujours']];

    const BASES_KEY = 'abiart_model_bases';     // nom de fichier (sans extension) -> modèle de base Civitai
    const ADULT_KEY = 'abiart_show_adult';

    const S = {
        type: 'Checkpoint', families: [], tag: '', query: '', sort: 'Most Downloaded', period: 'Month',
        compat: true, adult: false, periodTouched: false,
        models: [], starter: [], cursor: '', loading: false, searchId: 0,
        detail: null, detailId: 0, versionIdx: 0, images: [], recommended: null, imageIdx: 0,
        downloading: false,
    };
    let root = null;

    // ---------- petits outils ----------
    const esc = v => String(v == null ? '' : v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    const $ = sel => root.querySelector(sel);
    const stem = name => String(name || '').replace(/\.[^.\/\\]+$/, '').toLowerCase();

    function fmtCount(n) {
        n = Number(n) || 0;
        if (n >= 1e6) return (n / 1e6).toFixed(1).replace('.', ',').replace(',0', '') + ' M';
        if (n >= 1e3) return Math.round(n / 1e3) + ' k';
        return String(n);
    }

    function fmtSize(kb) {
        kb = Number(kb) || 0;
        if (!kb) return '';
        if (kb >= 1048576) return (kb / 1048576).toFixed(1).replace('.', ',') + ' Go';
        if (kb >= 1024) return Math.round(kb / 1024) + ' Mo';
        return Math.max(1, Math.round(kb)) + ' Ko';
    }

    function backendLabel() { return currentBackend === 'modal' ? 'Modal' : 'Kaggle'; }

    // ---------- famille du checkpoint utilisé en ce moment ----------
    function familyOfBase(base) {
        const b = String(base || '');
        const exact = FAMILIES.find(f => f.bases.includes(b));
        if (exact) return exact;
        if (/^SDXL/i.test(b)) return FAMILIES[1];
        if (/^SD 1/i.test(b)) return FAMILIES[0];
        if (/^Flux/i.test(b)) return FAMILIES[5];
        return null;
    }

    function storedBases() {
        try { return JSON.parse(localStorage.getItem(BASES_KEY) || '{}') || {}; } catch (e) { return {}; }
    }

    function rememberBase(filename, base) {
        if (!filename || !base) return;
        try {
            const all = storedBases();
            all[stem(filename)] = base;
            localStorage.setItem(BASES_KEY, JSON.stringify(all));
        } catch (e) { /* stockage plein : la compatibilité retombera sur le nom du fichier */ }
    }

    function currentCheckpoint() {
        const sel = document.getElementById(currentBackend === 'modal' ? 'modalModel' : 'kaggleModel');
        const value = sel && sel.value ? sel.value : '';
        return value && value !== 'default' ? value : '';
    }

    // Famille connue si le modèle a été téléchargé depuis cette fenêtre ; sinon, devinée d'après le nom du fichier.
    function currentFamily() {
        const name = currentCheckpoint();
        if (!name) return null;
        const known = storedBases()[stem(name.split(/[\\/]/).pop())];
        if (known) return familyOfBase(known);
        const n = name.toLowerCase();
        const byKey = k => FAMILIES.find(f => f.key === k);
        if (/noob|nai[-_ ]?xl/.test(n)) return byKey('noobai');
        if (/illustrious|ilxl|[-_ ]il[-_ ]?v?\d|wai/.test(n)) return byKey('illustrious');
        if (/pony|pdxl/.test(n)) return byKey('pony');
        if (/flux/.test(n)) return byKey('flux');
        if (/xl/.test(n)) return byKey('sdxl');
        if (/sd[-_ ]?1[-_.]?5|v1[-_.]5|sd15|1\.5/.test(n)) return byKey('sd15');
        return null;
    }

    function compatBases() {
        const fam = currentFamily();
        if (!fam) return [];
        const keys = S.type === 'LORA'
            ? (LORA_NEIGHBOURS[fam.key] || [fam.key])
            : FAMILIES.filter(f => f.arch === fam.arch).map(f => f.key);
        return FAMILIES.filter(f => keys.includes(f.key)).flatMap(f => f.bases);
    }

    // ---------- modèles déjà présents sur le backend ----------
    function installedNames() {
        const prefix = currentBackend === 'modal' ? 'modal' : 'kaggle';
        const names = new Set();
        document.querySelectorAll(`select[id^="${prefix}"]`).forEach(sel => {
            if (!/(Model|Lora|Embedding)/.test(sel.id) || /Ad\d?Model/.test(sel.id)) return;
            Array.from(sel.options).forEach(o => { if (o.value) names.add(stem(o.value.split(/[\\/]/).pop())); });
        });
        (window._allModalLoras || []).forEach(l => {
            const n = typeof l === 'string' ? l : (l && (l.name || l.filename));
            if (n && prefix === 'modal') names.add(stem(String(n).split(/[\\/]/).pop()));
        });
        return names;
    }

    // ---------- construction de la fenêtre ----------
    function build() {
        root = document.createElement('div');
        root.className = 'mb-overlay';
        root.id = 'modelBrowser';
        root.setAttribute('role', 'dialog');
        root.setAttribute('aria-modal', 'true');
        root.setAttribute('aria-label', 'Parcourir les modèles');
        root.innerHTML = `
            <div class="mb-head">
                <div class="mb-title"><i class="fas fa-compass" style="color:var(--accent2);margin-right:.4rem"></i>Parcourir les modèles</div>
                <span class="mb-pill" id="mbTarget"></span>
                <label class="mb-check" title="Les images et modèles réservés aux adultes sont masqués par défaut">
                    <input type="checkbox" id="mbAdult"> Afficher le contenu adulte
                </label>
                <button class="mb-iconbtn" id="mbClose" title="Fermer" aria-label="Fermer"><i class="fas fa-xmark"></i></button>
            </div>
            <div class="mb-filters">
                <div class="mb-row">
                    <input type="text" class="mb-search" id="mbQuery" placeholder="anime, réaliste, nom d'un modèle" spellcheck="false">
                    <button class="btn btn-sm" id="mbSearchBtn" title="Rechercher" aria-label="Rechercher"><i class="fas fa-search"></i></button>
                    <select id="mbSort" aria-label="Tri">${SORTS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}</select>
                    <select id="mbPeriod" aria-label="Période">${PERIODS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}</select>
                </div>
                <div class="mb-row" id="mbTypeRow"></div>
                <div class="mb-row" id="mbExtraRow"></div>
            </div>
            <div class="mb-body">
                <div class="mb-results" id="mbResults"></div>
                <aside class="mb-detail" id="mbDetail"></aside>
            </div>`;
        document.body.appendChild(root);

        $('#mbClose').addEventListener('click', close);
        $('#mbSearchBtn').addEventListener('click', () => { S.query = $('#mbQuery').value.trim(); search(); });
        $('#mbQuery').addEventListener('keydown', e => { if (e.key === 'Enter') { S.query = e.target.value.trim(); search(); } });
        $('#mbSort').addEventListener('change', e => { S.sort = e.target.value; search(); });
        $('#mbPeriod').addEventListener('change', e => { S.period = e.target.value; S.periodTouched = true; search(); });
        $('#mbAdult').addEventListener('change', onAdultToggle);
        $('#mbTypeRow').addEventListener('click', onChipClick);
        $('#mbExtraRow').addEventListener('click', onChipClick);
        $('#mbExtraRow').addEventListener('change', e => {
            if (e.target.id !== 'mbCompat') return;
            S.compat = e.target.checked;
            if (S.compat) S.families = [];
            renderFilters(); search();
        });
        $('#mbResults').addEventListener('click', e => {
            const card = e.target.closest('.mb-card');
            if (card) { openDetail(Number(card.dataset.id)); return; }
            if (e.target.closest('#mbMore')) search(true);
        });
        $('#mbDetail').addEventListener('click', onDetailClick);
        $('#mbDetail').addEventListener('change', e => {
            if (e.target.id === 'mbVersion') { S.versionIdx = Number(e.target.value) || 0; loadImages(); }
        });
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && root.classList.contains('mb-open')) close();
        });
    }

    function onAdultToggle(e) {
        if (e.target.checked && !window.confirm('Ce contenu est réservé aux adultes.\nConfirmez-vous avoir 18 ans ou plus ?')) {
            e.target.checked = false;
            return;
        }
        S.adult = e.target.checked;
        try { localStorage.setItem(ADULT_KEY, S.adult ? '1' : '0'); } catch (err) { /* préférence non mémorisée */ }
        S.starter = [];
        search();
        if (S.detail) openDetail(S.detail.id);
    }

    function setType(key) {
        S.type = key;
        S.tag = '';
        // Peu d'embeddings sortent chaque mois : sans choix explicite, on les cherche sur toute la période
        if (!S.periodTouched) S.period = key === 'TextualInversion' ? 'AllTime' : 'Month';
        $('#mbPeriod').value = S.period;
    }

    function onChipClick(e) {
        const chip = e.target.closest('.mb-chip');
        if (!chip) return;
        const { kind, key } = chip.dataset;
        if (kind === 'type') {
            setType(key);
        } else if (kind === 'family') {
            S.families = S.families.includes(key) ? S.families.filter(k => k !== key) : S.families.concat(key);
            if (S.families.length) S.compat = false;
        } else if (kind === 'tag') {
            S.tag = S.tag === key ? '' : key;
        }
        renderFilters();
        search();
    }

    // ---------- filtres ----------
    function renderFilters() {
        $('#mbTarget').textContent = 'Destination : ' + backendLabel();
        $('#mbTypeRow').innerHTML =
            TYPES.map(t => `<button class="mb-chip${S.type === t.key ? ' mb-on' : ''}" data-kind="type" data-key="${t.key}">${t.label}</button>`).join('') +
            '<span class="mb-sep"></span>' +
            FAMILIES.map(f => `<button class="mb-chip${S.families.includes(f.key) ? ' mb-on' : ''}" data-kind="family" data-key="${f.key}">${f.label}</button>`).join('');

        let extra = '';
        if (S.type === 'LORA') {
            extra += '<span class="mb-label">Catégorie :</span>' +
                LORA_TAGS.map(t => `<button class="mb-chip${S.tag === t.key ? ' mb-on' : ''}" data-kind="tag" data-key="${t.key}">${t.label}</button>`).join('');
        }
        if (S.type !== 'Checkpoint') {
            const fam = currentFamily();
            const name = currentCheckpoint().split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
            extra += fam
                ? `<label class="mb-check" style="margin-left:auto" title="${esc(name)} est un modèle ${esc(fam.label)}">
                       <input type="checkbox" id="mbCompat"${S.compat && !S.families.length ? ' checked' : ''}> Compatibles avec mon modèle (${esc(fam.label)})
                   </label>`
                : `<span class="mb-check mb-off" style="margin-left:auto" title="Choisissez une famille ci-dessus pour filtrer">
                       <i class="fas fa-circle-question"></i> Famille de votre modèle inconnue : filtrez à la main
                   </span>`;
        }
        $('#mbExtraRow').innerHTML = extra;
        $('#mbExtraRow').style.display = extra ? '' : 'none';
    }

    function activeBases() {
        if (S.families.length) return FAMILIES.filter(f => S.families.includes(f.key)).flatMap(f => f.bases);
        if (S.type !== 'Checkpoint' && S.compat) return compatBases();
        return [];
    }

    // ---------- recherche ----------
    async function search(more) {
        if (more && (!S.cursor || S.loading)) return;
        const id = ++S.searchId;
        if (!more) { S.models = []; S.cursor = ''; }
        S.loading = true;
        renderResults();
        const params = new URLSearchParams({ type: S.type, sort: S.sort, period: S.period, adult: S.adult ? '1' : '0' });
        if (S.query) params.set('query', S.query);
        if (S.tag && S.type === 'LORA') params.set('tag', S.tag);
        const bases = activeBases();
        if (bases.length) params.set('base_model', bases.join(','));
        if (more) params.set('cursor', S.cursor);
        let error = '';
        try {
            const data = await (await fetch('/api/civitai/search?' + params.toString())).json();
            if (id !== S.searchId) return;
            if (data.error) error = data.error;
            S.models = S.models.concat(data.models || []);
            S.cursor = data.next_cursor || '';
        } catch (e) {
            if (id !== S.searchId) return;
            error = 'La recherche a échoué. Vérifiez la connexion, puis réessayez.';
        }
        S.loading = false;
        renderResults(error);
        if (showStarter() && !S.starter.length) loadStarter();
    }

    // La sélection de départ n'apparaît que sur l'écran d'accueil : checkpoints, sans recherche ni filtre.
    function showStarter() {
        return S.type === 'Checkpoint' && !S.query && !S.families.length;
    }

    async function loadStarter() {
        try {
            const data = await (await fetch('/api/civitai/starter?adult=' + (S.adult ? '1' : '0'))).json();
            S.starter = data.models || [];
            if (S.starter.length && showStarter()) renderResults();
        } catch (e) { /* la sélection de départ est facultative */ }
    }

    function cardHtml(m, installed) {
        const img = m.image
            ? `<img src="${esc(m.image)}" alt="" loading="lazy" referrerpolicy="no-referrer">`
            : (m.image_hidden
                ? '<i class="fas fa-eye-slash"></i><span>Image masquée</span>'
                : '<i class="fas fa-image"></i>');
        const size = fmtSize(m.size_kb);
        return `
            <button class="mb-card${S.detail && S.detail.id === m.id ? ' mb-on' : ''}" data-id="${Number(m.id)}" title="${esc(m.name)}">
                <div class="mb-card-img">${img}</div>
                <div class="mb-card-body">
                    <div class="mb-card-name">${esc(m.name)}</div>
                    <div class="mb-badges">
                        <span class="mb-badge">${esc(TYPE_LABEL[m.type] || m.type)}</span>
                        ${m.base_model ? `<span class="mb-badge">${esc(m.base_model)}</span>` : ''}
                        ${installed.has(stem(m.filename)) ? '<span class="mb-badge mb-ok">Installé</span>' : ''}
                    </div>
                    <div class="mb-stats">
                        <span><i class="fas fa-download"></i> ${fmtCount(m.downloads)}</span>
                        <span><i class="fas fa-heart"></i> ${fmtCount(m.likes)}</span>
                        ${size ? `<span>${size}</span>` : ''}
                    </div>
                </div>
            </button>`;
    }

    function renderResults(error) {
        const installed = installedNames();
        let html = '';
        if (showStarter() && S.starter.length) {
            html += `<div class="mb-section-title">Pour commencer : des valeurs sûres</div>
                     <div class="mb-grid">${S.starter.map(m => cardHtml(m, installed)).join('')}</div>
                     <div class="mb-section-title">${esc(sortTitle())}</div>`;
        }
        if (S.models.length) {
            html += `<div class="mb-grid">${S.models.map(m => cardHtml(m, installed)).join('')}</div>`;
        }
        if (S.loading) {
            html += '<div class="mb-empty"><i class="fas fa-spinner fa-spin"></i> Recherche sur Civitai…</div>';
        } else if (error) {
            html += `<div class="mb-empty" style="color:var(--error)">${esc(error)}</div>`;
        } else if (!S.models.length) {
            html += '<div class="mb-empty">Aucun modèle ne correspond à ces filtres. Essayez une autre famille ou une période plus longue.</div>';
        } else if (S.cursor) {
            html += '<button class="btn mb-more" id="mbMore">Afficher plus de modèles</button>';
        }
        const box = $('#mbResults');
        const scroll = box.scrollTop;
        box.innerHTML = html;
        box.scrollTop = scroll;
    }

    function sortTitle() {
        const sort = (SORTS.find(s => s[0] === S.sort) || SORTS[0])[1];
        const period = (PERIODS.find(p => p[0] === S.period) || PERIODS[1])[1].toLowerCase();
        return S.sort === 'Newest' ? sort : `${sort}, ${period}`;
    }

    // ---------- fiche d'un modèle ----------
    async function openDetail(modelId) {
        const id = ++S.detailId;
        root.classList.add('mb-has-detail');
        S.detail = { id: modelId, loading: true };
        S.versionIdx = 0; S.images = []; S.recommended = null; S.imageIdx = 0;
        $('#mbDetail').innerHTML = backButton() + '<div class="mb-empty"><i class="fas fa-spinner fa-spin"></i> Chargement du modèle…</div>';
        root.querySelectorAll('.mb-card').forEach(c => c.classList.toggle('mb-on', Number(c.dataset.id) === modelId));
        try {
            const data = await (await fetch(`/api/civitai/model/${modelId}?adult=${S.adult ? '1' : '0'}`)).json();
            if (id !== S.detailId) return;
            if (data.error || !data.versions || !data.versions.length) throw new Error(data.error || 'no version');
            S.detail = data;
            // Un LoRA ou un embedding a souvent une version par famille : on ouvre celle qui convient au filtre en cours
            const wanted = data.type === 'Checkpoint' ? [] : activeBases();
            const match = data.versions.findIndex(v => wanted.includes(v.base_model));
            if (match > 0) S.versionIdx = match;
            loadImages();
        } catch (e) {
            if (id !== S.detailId) return;
            S.detail = null;
            $('#mbDetail').innerHTML = backButton() + '<div class="mb-empty" style="color:var(--error)">Impossible de charger ce modèle. Réessayez dans un instant.</div>';
        }
    }

    // Images d'exemple de la version choisie, avec les réglages qui les ont produites
    async function loadImages() {
        const id = S.detailId;
        const version = S.detail.versions[S.versionIdx];
        S.images = version.images || [];
        S.recommended = null;
        S.imageIdx = 0;
        renderDetail();
        try {
            const data = await (await fetch(`/api/civitai/images?version_id=${Number(version.id)}&adult=${S.adult ? '1' : '0'}`)).json();
            if (id !== S.detailId || S.detail.versions[S.versionIdx] !== version) return;
            if (data.images && data.images.length) {
                S.images = data.images;
                S.recommended = data.recommended || null;
                S.imageIdx = Math.max(0, S.images.findIndex(i => i.meta));
                renderDetail();
            }
        } catch (e) { /* on garde les images de couverture, sans réglages */ }
    }

    function backButton() {
        return '<button class="btn btn-sm mb-back" id="mbBack" style="margin-bottom:.6rem"><i class="fas fa-arrow-left"></i>&nbsp;Retour à la liste</button>';
    }

    function emptyDetail() {
        $('#mbDetail').innerHTML = '<div class="mb-empty"><i class="fas fa-hand-pointer" style="font-size:1.2rem;margin-bottom:.6rem;display:block"></i>Choisissez un modèle pour voir ses images, ses réglages conseillés et le télécharger.</div>';
    }

    function settingsLine(meta) {
        const parts = [];
        if (meta.sampler) parts.push(esc(meta.sampler));
        if (meta.steps) parts.push(meta.steps + ' pas');
        if (meta.cfg) parts.push('CFG ' + String(meta.cfg).replace('.', ','));
        if (meta.width && meta.height) parts.push(`${meta.width}×${meta.height}`);
        if (meta.clip_skip) parts.push('Clip skip ' + meta.clip_skip);
        return parts.join(' · ');
    }

    function rangeText(range, unit) {
        if (!range) return '';
        const f = v => String(v).replace('.', ',');
        return range[0] === range[1] ? f(range[0]) + unit : `${f(range[0])} à ${f(range[1])}${unit}`;
    }

    function compatWarning(version) {
        if (S.detail.type === 'Checkpoint') return '';
        const mine = currentFamily();
        const theirs = familyOfBase(version.base_model);
        if (!mine || !theirs || mine.arch === theirs.arch) return '';
        return `<span class="mb-badge mb-warn"><i class="fas fa-triangle-exclamation"></i> Prévu pour ${esc(theirs.label)} : sans effet correct sur votre modèle actuel (${esc(mine.label)})</span>`;
    }

    function renderDetail() {
        const d = S.detail;
        if (!d || d.loading) return;
        const v = d.versions[S.versionIdx];
        const image = S.images[S.imageIdx];
        const meta = image && image.meta;
        const rec = S.recommended;
        const kind = civitaiKind(d.type);
        const installed = installedNames().has(stem(v.filename));
        const size = fmtSize(v.size_kb);

        const versionSelect = d.versions.length > 1
            ? `<select id="mbVersion" aria-label="Version" style="flex:none;width:auto;max-width:9rem">${d.versions.map((x, i) => `<option value="${i}"${i === S.versionIdx ? ' selected' : ''}>${esc(x.name)}</option>`).join('')}</select>`
            : '';

        const gallery = S.images.length
            ? `<div class="mb-main-img"><img src="${esc(image.full || image.url)}" alt="Image d'exemple" referrerpolicy="no-referrer"></div>
               ${S.images.length > 1 ? `<div class="mb-thumbs">${S.images.slice(0, 12).map((im, i) => `<button class="mb-thumb${i === S.imageIdx ? ' mb-on' : ''}" data-img="${i}" aria-label="Image ${i + 1}"><img src="${esc(im.url)}" alt="" loading="lazy" referrerpolicy="no-referrer"></button>`).join('')}</div>` : ''}`
            : `<div class="mb-main-img"><span><i class="fas fa-eye-slash"></i> ${S.adult ? 'Aucune image d\'exemple' : 'Aucune image tout public pour cette version'}</span></div>`;

        const imageSettings = meta
            ? `<div class="mb-block">
                   <div class="mb-block-title">Réglages de cette image</div>
                   <div class="mb-line">${settingsLine(meta) || 'Réglages non indiqués'}</div>
                   <div class="mb-prompt">${esc(meta.prompt)}</div>
                   <button class="btn btn-sm mb-wide" id="mbUseImage"><i class="fas fa-copy"></i>&nbsp;Reprendre ce prompt et ces réglages</button>
               </div>`
            : '';

        const recLine = rec ? [rec.sampler ? esc(rec.sampler) : '', rangeText(rec.steps, ' pas'), rec.cfg ? 'CFG ' + rangeText(rec.cfg, '') : '', rec.clip_skip ? 'Clip skip ' + rec.clip_skip : ''].filter(Boolean).join(' · ') : '';
        const recommended = recLine
            ? `<div class="mb-block">
                   <div class="mb-block-title">Réglages conseillés</div>
                   <div class="mb-line">${recLine}</div>
                   <div class="mb-sub">Estimation d'après ${rec.count} images d'exemple. La description de l'auteur fait foi.</div>
                   <button class="btn btn-sm mb-wide" id="mbUseRec"><i class="fas fa-sliders"></i>&nbsp;Appliquer ces réglages</button>
               </div>`
            : '';

        const words = (v.trained_words || []).length
            ? `<div class="mb-block">
                   <div class="mb-block-title">${kind === 'embedding' ? 'Mot à écrire dans le prompt' : 'Mots déclencheurs'}</div>
                   <div class="mb-words">${v.trained_words.slice(0, 20).map(w => `<span class="mb-badge">${esc(w)}</span>`).join('')}</div>
               </div>`
            : '';

        let action;
        if (currentBackend === 'modal' && kind === 'embedding') {
            action = '<div class="mb-line">Les embeddings ne sont disponibles que sur le backend Kaggle.</div>';
        } else if (!v.download_url) {
            action = '<div class="mb-line">Ce fichier n\'est pas téléchargeable depuis Civitai.</div>';
        } else if (installed && kind === 'checkpoint') {
            action = `<button class="btn-accent mb-download" id="mbUseModel"><i class="fas fa-check"></i>&nbsp;Déjà installé : utiliser ce modèle</button>`;
        } else if (installed) {
            action = `<div class="mb-line" style="color:var(--success)"><i class="fas fa-check"></i> Déjà installé sur ${backendLabel()}</div>`;
        } else {
            action = `<button class="btn-accent mb-download" id="mbDownload"${S.downloading ? ' disabled' : ''}><i class="fas fa-cloud-download-alt"></i>&nbsp;Télécharger sur ${backendLabel()}${size ? ' (' + size + ')' : ''}</button>`;
        }

        const notes = [v.notes ? 'Notes de la version ' + v.name + ' :\n' + v.notes : '', d.description].filter(Boolean).join('\n\n');

        const box = $('#mbDetail');
        const progress = box.querySelector('#mbProgress');
        const progressText = progress ? progress.textContent : '';
        box.innerHTML = `
            ${backButton()}
            <div class="mb-detail-head">
                <div style="flex:1;min-width:0">
                    <div class="mb-detail-name">${esc(d.name)}</div>
                    <div class="mb-sub">${d.creator ? 'par ' + esc(d.creator) + ' · ' : ''}<i class="fas fa-download"></i> ${fmtCount(d.downloads)} · <i class="fas fa-heart"></i> ${fmtCount(d.likes)} · <a href="${esc(d.url)}" target="_blank" rel="noopener noreferrer">Voir sur Civitai</a></div>
                </div>
                ${versionSelect}
            </div>
            ${gallery}
            <div class="mb-block">
                <div class="mb-badges">
                    <span class="mb-badge">${esc(TYPE_LABEL[d.type] || d.type)}</span>
                    ${v.base_model ? `<span class="mb-badge">${esc(v.base_model)}</span>` : ''}
                    ${size ? `<span class="mb-badge">${size}</span>` : ''}
                    ${v.format ? `<span class="mb-badge">${esc(v.format)}</span>` : ''}
                    ${v.published ? `<span class="mb-badge">${esc(v.published)}</span>` : ''}
                    ${compatWarning(v)}
                </div>
            </div>
            <div class="mb-block">
                ${action}
                <div class="mb-progress" id="mbProgress">${esc(progressText)}</div>
            </div>
            ${imageSettings}
            ${recommended}
            ${words}
            ${notes ? `<details class="mb-block"><summary>Description de l'auteur</summary><div class="mb-desc">${esc(notes)}</div></details>` : ''}`;
    }

    function onDetailClick(e) {
        if (e.target.closest('#mbBack')) { root.classList.remove('mb-has-detail'); return; }
        const thumb = e.target.closest('.mb-thumb');
        if (thumb) { S.imageIdx = Number(thumb.dataset.img) || 0; renderDetail(); return; }
        if (e.target.closest('#mbUseImage')) { const im = S.images[S.imageIdx]; if (im && im.meta) applySettings(im.meta, true); return; }
        if (e.target.closest('#mbUseRec')) { applyRecommended(); return; }
        if (e.target.closest('#mbDownload')) { download(); return; }
        if (e.target.closest('#mbUseModel')) { useInstalledModel(); }
    }

    // ---------- reprise des réglages ----------
    function setRange(sliderId, inputId, value) {
        const slider = document.getElementById(sliderId);
        if (!slider || value == null) return;
        const min = Number(slider.min), max = Number(slider.max);
        let v = Number(value);
        if (!isFinite(v)) return;
        if (isFinite(min)) v = Math.max(min, v);
        if (isFinite(max)) v = Math.min(max, v);
        uiSetSlider(sliderId, inputId, v);
    }

    function applySampler(name) {
        if (!name) return;
        if (currentBackend === 'modal') { uiSetSelect('modalScheduler', name); return; }
        // Civitai écrit « DPM++ 2M Karras » : le site sépare le sampler et son scheduler
        const m = /^(.*?)\s+(Karras|Exponential|SGM Uniform|Simple|Normal|Beta|KL Optimal)$/i.exec(name);
        uiSetSelect('kaggleSampler', m ? m[1] : name);
        if (m) uiSetSelect('kaggleScheduler', m[2]);
    }

    function applyNumbers(steps, cfg, clip) {
        if (currentBackend === 'modal') {
            setRange('modalSteps', 'modalStepsVal', steps);
            setRange('modalCfgScale', 'modalCfgVal', cfg);
            setRange('modalClipSkip', 'modalClipSkipVal', clip);
        } else {
            setRange('steps', 'stepsVal', steps);
            setRange('cfgScale', 'cfgVal', cfg);
            setRange('kaggleClipSkip', 'kaggleClipSkipVal', clip);
        }
    }

    function applySettings(meta, withPrompt) {
        let stripped = false;
        if (withPrompt) {
            // Les LoRA cités dans le prompt d'exemple ne sont pas installés ici : on retire leurs balises
            const clean = text => String(text || '').replace(/<(?:lora|lyco|hypernet)[^>]*>/gi, () => { stripped = true; return ''; }).replace(/\s*,\s*(,\s*)+/g, ', ').trim();
            const prompt = document.getElementById('promptInput');
            const negative = document.getElementById('negativePrompt');
            if (prompt) { prompt.value = clean(meta.prompt); prompt.dispatchEvent(new Event('input', { bubbles: true })); }
            if (negative) { negative.value = clean(meta.negative); negative.dispatchEvent(new Event('input', { bubbles: true })); }
            const prefix = currentBackend === 'modal' ? 'modal' : 'kaggle';
            const w = document.getElementById(prefix + 'Width'), h = document.getElementById(prefix + 'Height');
            if (w && h && meta.width && meta.height && meta.width <= 2048 && meta.height <= 2048) { w.value = meta.width; h.value = meta.height; }
        }
        applySampler(meta.sampler);
        applyNumbers(meta.steps, meta.cfg, meta.clip_skip);
        if (typeof autoSaveSettings === 'function') autoSaveSettings();
        close();
        showToast(stripped ? 'Prompt et réglages repris. Les LoRA de l\'exemple ont été retirés du prompt.' : (withPrompt ? 'Prompt et réglages repris' : 'Réglages appliqués'), 'success');
    }

    function applyRecommended() {
        const rec = S.recommended;
        if (!rec) return;
        const middle = range => range ? Math.round(((range[0] + range[1]) / 2) * 2) / 2 : null;
        applySettings({ sampler: rec.sampler, steps: rec.steps ? Math.round(middle(rec.steps)) : null, cfg: middle(rec.cfg), clip_skip: rec.clip_skip }, false);
    }

    // ---------- téléchargement ----------
    function setProgress(text) {
        const el = root.querySelector('#mbProgress');
        if (el) el.textContent = text || '';
    }

    function useInstalledModel() {
        const v = S.detail.versions[S.versionIdx];
        uiSetSelect(currentBackend === 'modal' ? 'modalModel' : 'kaggleModel', v.filename.replace(/\.[^.]+$/, ''));
        if (typeof autoSaveSettings === 'function') autoSaveSettings();
        close();
        showToast('Modèle sélectionné', 'success');
    }

    async function download() {
        if (S.downloading || !S.detail) return;
        const d = S.detail;
        const v = d.versions[S.versionIdx];
        const kind = civitaiKind(d.type);
        const model = { name: d.name, type: d.type, version: v.name, download_url: v.download_url, filename: v.filename, trained_words: v.trained_words || [] };
        S.downloading = true;
        renderDetail();
        setProgress('Démarrage du téléchargement…');
        try {
            if (currentBackend === 'modal') {
                await downloadToModal(model);
                rememberBase(v.filename, v.base_model);
                setProgress('');
            } else {
                updateStatus('Downloading...', 'warning');
                // Jeton Civitai du champ « Avancé », s'il est rempli ; sinon le serveur utilise le sien
                const tokenField = document.getElementById('civitaiToken');
                const token = tokenField ? tokenField.value.trim() : '';
                const result = await kaggleDownloadAndWait({ model_url: v.download_url, filename: v.filename, type: kind, token }, setProgress);
                rememberBase(result.filename || v.filename, v.base_model);
                afterKaggleDownload(result, model.trained_words);
                showToast(`${d.name} est installé sur Kaggle`, 'success');
                setProgress('Téléchargement terminé.');
            }
        } catch (e) {
            updateStatus('Download failed', 'error');
            setProgress('Échec du téléchargement : ' + e.message);
            showToast('Download failed: ' + e.message);
        }
        S.downloading = false;
        // les listes du site se rechargent juste après : on réaffiche pour montrer « Installé »
        [2000, 6000, 12000].forEach(delay => setTimeout(() => {
            if (root.classList.contains('mb-open') && !S.loading) { renderResults(); renderDetail(); renderFilters(); }
        }, delay));
        renderDetail();
    }

    // ---------- ouverture / fermeture ----------
    function open(type) {
        if (!root) build();
        try { S.adult = localStorage.getItem(ADULT_KEY) === '1'; } catch (e) { S.adult = false; }
        $('#mbAdult').checked = S.adult;
        if (type && TYPES.some(t => t.key === type)) setType(type);
        $('#mbSort').value = S.sort;
        $('#mbPeriod').value = S.period;
        root.classList.add('mb-open');
        document.body.style.overflow = 'hidden';
        renderFilters();
        if (!S.detail || window.innerWidth <= 900) root.classList.remove('mb-has-detail');
        if (!S.detail) emptyDetail(); else renderDetail();
        search();
        setTimeout(() => { const q = $('#mbQuery'); if (q && window.innerWidth > 900) q.focus(); }, 50);
    }

    function close() {
        if (!root) return;
        root.classList.remove('mb-open');
        document.body.style.overflow = '';
    }

    window.openModelBrowser = open;
    window.closeModelBrowser = close;
})();
