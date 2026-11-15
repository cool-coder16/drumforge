// App wiring: connects the controls, the audio engine, the canvases and storage.
(function () {
  const $ = (id) => document.getElementById(id);
  const TRACKS = DF.TRACKS;

  const state = {
    mode: 'random',                       // 'audio' (from a song) or 'random'
    slots: { audio: null, random: null }, // the current project of each mode
    project: null,
    audio: null,                          // { name, buffer, an, bpm, phase, barShift, nudge, feats }
    undo: { audio: [], random: [] },
    muted: new Set(),
    cue: { step: 0, time: 0 },            // where Play will start
    rec: false,
    styleByMode: { audio: 'auto', random: 'surprise' },
  };

  // ------------------------------------------------------------------ audio engine
  let ctx = null, master, drumGain, songGain, analyser, previewSynth, player;
  const hitQueue = [];

  function ensureAudio() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      master = ctx.createGain(); master.gain.value = 0.9;
      analyser = ctx.createAnalyser(); analyser.fftSize = 256; analyser.smoothingTimeConstant = 0.78;
      drumGain = ctx.createGain(); songGain = ctx.createGain();
      drumGain.connect(master); songGain.connect(master);
      master.connect(analyser); analyser.connect(ctx.destination);
      applyVolumes();
      previewSynth = new DF.DrumSynth(ctx, drumGain, $('kit').value);
      player = new DF.Player({
        ctx, drumGain, songGain, muted: state.muted,
        getProject: () => state.project,
        getBuffer: () => state.audio && state.audio.buffer,
        getKit: () => $('kit').value,
        onHit: (id, v, when) => hitQueue.push({ id, when }),
        onStop: () => updatePlayUI(),
      });
    }
    if (ctx.state !== 'running') ctx.resume();
  }

  function applyVolumes() {
    if (!ctx) return;
    const curve = (el) => Math.pow(+el.value / 100, 1.5) * 1.2;
    drumGain.gain.value = curve($('volDrums'));
    songGain.gain.value = curve($('volSong'));
  }

  // ------------------------------------------------------------------ UI helpers
  const fmtTime = (s) => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
  const pct = (id) => +$(id).value / 100;

  function updateOutputs() {
    ['complexity', 'ghost', 'fills', 'sync', 'swing', 'humanize'].forEach((id) => ($(id + 'Out').textContent = $(id).value + '%'));
    $('rbpmOut').textContent = $('rbpm').value;
  }

  function fillStyleSelect() {
    const sel = $('style');
    sel.innerHTML = '';
    const first = state.mode === 'audio' ? ['auto', 'Auto (follow the song)'] : ['surprise', 'Surprise me'];
    const opts = [first].concat(DF.STYLE_LIST.map((id) => [id, DF.STYLE_LABELS[id]]), [['euclid', DF.STYLE_LABELS.euclid]]);
    opts.forEach(([v, l]) => sel.append(new Option(l, v)));
    sel.value = state.styleByMode[state.mode];
  }

  function updatePlayUI() {
    const playing = player && player.playing;
    $('playIcon').setAttribute('d', playing ? 'M6 6h12v12H6z' : 'M7 4l13 8-13 8z');
    $('play').disabled = !state.project;
    if (!playing) $('posSub').textContent = state.project ? 'stopped' : 'load a song first';
  }

  function updateUndoBtn() { $('undo').disabled = state.undo[state.mode].length === 0; }

  function say(msg, isError) {
    const el = $('saveMsg');
    el.textContent = msg;
    el.style.color = isError ? '#ff8fa0' : '';
  }

  // ------------------------------------------------------------------ projects & undo
  const cloneVel = (pattern) => {
    const o = {};
    for (const k in pattern) o[k] = pattern[k].slice();
    return o;
  };

  function pushUndo() {
    const p = state.project;
    if (!p) return;
    const list = state.undo[state.mode];
    list.push({ bars: p.bars, pattern: cloneVel(p.pattern) });
    if (list.length > 30) list.shift();
    updateUndoBtn();
  }

  function doUndo() {
    const snap = state.undo[state.mode].pop();
    if (!snap || !state.project) return;
    state.project.bars = snap.bars;
    state.project.pattern = snap.pattern;
    refreshViews();
    updateUndoBtn();
  }

  function refreshViews() {
    const p = state.project, a = state.audio;
    roll.setProject(p, state.mode === 'audio' && a ? a.feats : null);
    overview.setProject(p, state.mode === 'audio' && a ? a.an.peaks : null, state.mode === 'audio' && a ? a.duration : 0);
    roll.muted = state.muted;
    updatePlayUI();
  }

  function setProject(p, withUndo) {
    if (withUndo) pushUndo();
    state.slots[state.mode] = p;
    state.project = p;
    refreshViews();
  }

  // ------------------------------------------------------------------ generating
  function readParams() {
    return {
      style: $('style').value,
      complexity: pct('complexity'), ghost: pct('ghost'), fills: pct('fills'),
      sync: pct('sync'), humanize: pct('humanize'),
      seed: $('seed').value.trim() || 'GROOVE-1',
    };
  }

  // retune: after picking a style, also set tempo + swing to suit it (random mode only).
  function generate(retune) {
    const prm = readParams();
    let bars, bpm, gridOffset = 0, feats = null;
    if (state.mode === 'audio') {
      if (!state.audio) return;
      feats = state.audio.feats; bars = feats.bars; bpm = state.audio.bpm; gridOffset = feats.T0;
    } else {
      bars = +$('bars').value; bpm = +$('rbpm').value;
    }
    const res = DF.generate(Object.assign({}, prm, { bars, feats }));
    if (state.mode === 'random' && retune) {
      const r = DF.makeRng(DF.hashString(prm.seed + '|bpm'));
      bpm = Math.round(res.bpmRange[0] + r() * (res.bpmRange[1] - res.bpmRange[0]));
      $('rbpm').value = bpm;
      $('swing').value = Math.round(res.swing * 100);
      updateOutputs();
    }
    const project = {
      name: '', mode: state.mode, bpm, bars, swing: pct('swing'), humanize: prm.humanize,
      kit: $('kit').value, gridOffset, pattern: res.pattern, styleLabel: res.label, styleId: res.styleId,
    };
    $('styleInfo').textContent = (state.mode === 'audio' && prm.style === 'auto' ? 'Groove feels like: ' : 'Style: ') + res.label;
    setProject(project, true);
    writeHash();
    say('');
  }

  // Random-mode settings live in the URL so a link recreates the same beat.
  function writeHash() {
    if (state.mode !== 'random') { history.replaceState(null, '', location.pathname + location.search); return; }
    const q = new URLSearchParams({
      s: $('style').value, b: $('rbpm').value, n: $('bars').value, c: $('complexity').value, g: $('ghost').value,
      f: $('fills').value, w: $('swing').value, h: $('humanize').value, k: $('kit').value, d: $('seed').value,
    });
    history.replaceState(null, '', '#' + q.toString());
  }

  function readHash() {
    if (!location.hash.length) return false;
    const q = new URLSearchParams(location.hash.slice(1));
    if (!q.get('d')) return false;
    const set = (id, key) => { if (q.get(key) != null) $(id).value = q.get(key); };
    state.styleByMode.random = q.get('s') || 'surprise';
    fillStyleSelect();
    set('rbpm', 'b'); set('bars', 'n'); set('complexity', 'c'); set('ghost', 'g'); set('fills', 'f');
    set('swing', 'w'); set('humanize', 'h'); set('kit', 'k'); set('seed', 'd');
    updateOutputs();
    return true;
  }

  // ------------------------------------------------------------------ mode switching
  function switchMode(mode) {
    if (player) player.stop();
    state.mode = mode;
    document.body.className = 'mode-' + mode;
    $('tab-audio').classList.toggle('active', mode === 'audio');
    $('tab-random').classList.toggle('active', mode === 'random');
    $('audioBox').hidden = mode !== 'audio';
    fillStyleSelect();
    state.project = state.slots[mode];
    state.cue = { step: 0, time: 0 };
    roll.cueStep = -1;
    roll.viewStart = 0;
    refreshViews();
    updateUndoBtn();
    if (mode === 'random') {
      if (!state.project) generate(!state.fromLink); else writeHash();
      $('styleInfo').textContent = state.project ? 'Style: ' + state.project.styleLabel : '';
    } else {
      writeHash();
      $('styleInfo').textContent = state.project ? 'Groove feels like: ' + state.project.styleLabel : '';
    }
    updatePlayUI();
  }

  // ------------------------------------------------------------------ loading audio
  function showProgress(text, frac) {
    $('progress').hidden = false;
    $('progText').textContent = text;
    $('progBar').style.width = Math.round((frac || 0) * 100) + '%';
  }

  async function loadFile(file) {
    if (!file) return;
    if (state.mode !== 'audio') switchMode('audio');
    ensureAudio();
    if (player) player.stop();
    showProgress('Decoding ' + file.name + '…', 0);
    try {
      const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
      await analyseBuffer(buffer, file.name);
    } catch (err) {
      console.error(err);
      showProgress('Could not read that file as audio. Try an mp3, wav or m4a.', 0);
    }
  }

  async function analyseBuffer(buffer, name) {
    const long = buffer.duration > 600 ? ' (long file: using the first 10 minutes)' : '';
    showProgress('Analysing rhythm…' + long, 0);
    let an;
    try {
      an = await DF.analyzeAudio(buffer, (f) => showProgress('Analysing rhythm…' + long, f));
    } catch (err) {
      showProgress(err.message, 0);
      return;
    }
    state.audio = { name, buffer, duration: Math.min(buffer.duration, an.duration), an, bpm: an.bpm, phase: an.phase, barShift: an.barShift, nudge: 0 };
    rebuildGrid();
    $('seed').value = DF.randomSeed();
    state.undo.audio = [];
    state.slots.audio = null;
    state.cue = { step: 0, time: 0 };
    state.styleByMode.audio = 'auto';
    fillStyleSelect();
    syncAudioUI();
    $('progress').hidden = true;
    $('analysis').hidden = false;
    $('swing').value = 0;
    updateOutputs();
    generate(false);
    roll.viewStart = 0;
  }

  function rebuildGrid() {
    const a = state.audio;
    a.feats = DF.buildGrid(a.an, a.bpm, a.phase, a.barShift, a.nudge / 1000);
  }

  function syncAudioUI() {
    const a = state.audio;
    if (!a) return;
    $('fileChip').textContent = a.name + ' · ' + fmtTime(a.duration);
    $('abpm').value = a.bpm;
    $('nudge').value = a.nudge;
    $('nudgeOut').textContent = a.nudge + ' ms';
  }

  // Tempo / downbeat / nudge changed: recompute the grid then redo the drum part.
  function regrid(retempo) {
    const a = state.audio;
    if (!a) return;
    if (player) player.stop();
    if (retempo) {
      a.bpm = DF.clamp(a.bpm, 40, 260);
      a.phase = DF.findPhase(a.an, a.bpm);
      a.barShift = DF.pickDownbeat(a.an, a.bpm, a.phase);
      a.nudge = 0;
    }
    rebuildGrid();
    syncAudioUI();
    state.cue = { step: 0, time: 0 };
    generate(false);
  }

  // ------------------------------------------------------------------ transport
  function startPlayback() {
    if (!state.project) return;
    ensureAudio();
    if (state.mode === 'audio') player.start(0, state.cue.time);
    else player.start(state.cue.step);
    updatePlayUI();
  }

  function togglePlay() {
    if (player && player.playing) player.stop(); else startPlayback();
  }

  function seekStep(step, time) {
    const p = state.project;
    if (!p) return;
    const sd = 60 / p.bpm / 4;
    state.cue = { step, time: time != null ? time : p.gridOffset + step * sd };
    if (state.mode === 'random') state.cue.time = 0;
    roll.cueStep = step;
    if (!roll.visibleStep(step)) roll.setView(step - 2);
    if (player && player.playing) startPlayback();
  }

  // ------------------------------------------------------------------ pads
  function buildPads() {
    const wrap = $('pads');
    TRACKS.forEach((t) => {
      const b = document.createElement('button');
      b.className = 'pad'; b.dataset.id = t.id; b.style.setProperty('--c', t.color);
      const bar = document.createElement('i'); bar.style.background = t.color;
      const kbd = document.createElement('kbd'); kbd.textContent = t.key;
      b.append(bar, kbd, document.createTextNode(t.label));
      b.addEventListener('pointerdown', (e) => { e.preventDefault(); padHit(t.id); });
      wrap.append(b);
    });
  }

  function flashPad(id) {
    const el = document.querySelector('.pad[data-id="' + id + '"]');
    if (!el) return;
    el.classList.add('hit');
    setTimeout(() => el.classList.remove('hit'), 90);
  }

  function padHit(id) {
    ensureAudio();
    previewSynth.trigger(id, ctx.currentTime + 0.005, 0.9);
    flashPad(id); roll.hit(id);
    // Record: write the hit into the nearest step while the pattern plays.
    if (state.rec && player && player.playing && state.project) {
      const pos = player.stepPos();
      const total = state.project.bars * 16;
      const s = Math.round(pos) % total;
      if (pos >= 0 && s >= 0 && s < total) {
        state.project.pattern[id][s] = 0.9;
        overview.refresh();
      }
    }
  }

  // ------------------------------------------------------------------ save / export
  function download(blob, filename) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  function projectName() {
    const typed = $('pname').value.trim();
    if (typed) return typed;
    if (state.mode === 'audio' && state.audio) return state.audio.name.replace(/\.[^.]+$/, '') + ' drums';
    const p = state.project;
    return p ? p.styleLabel + ' ' + p.bpm + 'bpm' : 'drums';
  }
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'drumforge';

  async function withBusy(btn, label, fn) {
    const old = btn.textContent;
    btn.classList.add('busy'); btn.textContent = label;
    try { await fn(); } catch (err) { console.error(err); say('Export failed: ' + err.message, true); }
    btn.classList.remove('busy'); btn.textContent = old;
  }

  function renderLibrary() {
    const ul = $('library');
    ul.innerHTML = '';
    const items = DF.library.list();
    if (!items.length) {
      const li = document.createElement('li');
      li.className = 'empty'; li.textContent = 'Nothing saved yet. Make a beat you like and hit "Save to library".';
      ul.append(li);
      return;
    }
    items.forEach((it) => {
      const li = document.createElement('li');
      const nm = document.createElement('span'); nm.className = 'nm'; nm.textContent = it.name;
      const meta = document.createElement('span'); meta.className = 'meta';
      meta.textContent = it.bpm + ' BPM · ' + it.bars + ' bars · ' + (it.styleLabel || '') + ' · ' + new Date(it.saved).toLocaleDateString();
      const mk = (label, fn) => { const b = document.createElement('button'); b.className = 'btn'; b.textContent = label; b.onclick = fn; return b; };
      li.append(nm, meta,
        mk('Load', () => loadFromLibrary(it)),
        mk('MIDI', () => download(DF.makeMidi(DF.library.revive(it), null, false), slug(it.name) + '.mid')),
        mk('Delete', () => { DF.library.remove(it.id); renderLibrary(); }));
      ul.append(li);
    });
  }

  function loadFromLibrary(item) {
    if (state.mode !== 'random') switchMode('random');
    if (player) player.stop();
    const p = DF.library.revive(item);
    if (![...$('bars').options].some((o) => +o.value === p.bars)) $('bars').append(new Option(p.bars, p.bars));
    $('bars').value = p.bars; $('rbpm').value = p.bpm;
    $('swing').value = Math.round(p.swing * 100); $('humanize').value = Math.round(p.humanize * 100);
    if (DF.KITS[p.kit]) $('kit').value = p.kit;
    $('pname').value = item.name;
    updateOutputs();
    setProject(p, true);
    $('styleInfo').textContent = 'Loaded from library: ' + item.name;
    say('Loaded "' + item.name + '". Press play.');
  }

  // ------------------------------------------------------------------ hero visualiser
  const viz = $('viz'), vg = viz.getContext('2d');
  let freq = null;
  function drawViz(t) {
    const dpr = window.devicePixelRatio || 1, w = viz.clientWidth, h = viz.clientHeight;
    if (viz.width !== Math.floor(w * dpr)) { viz.width = Math.floor(w * dpr); viz.height = Math.floor(h * dpr); }
    vg.setTransform(dpr, 0, 0, dpr, 0, 0);
    vg.clearRect(0, 0, w, h);
    const n = Math.max(24, Math.floor(w / 14));
    if (analyser && !freq) freq = new Uint8Array(analyser.frequencyBinCount);
    if (analyser) analyser.getByteFrequencyData(freq);
    const grad = vg.createLinearGradient(0, h, 0, 0);
    grad.addColorStop(0, 'rgba(255,90,60,0.55)'); grad.addColorStop(1, 'rgba(255,182,39,0.05)');
    vg.fillStyle = grad;
    const bw = w / n;
    for (let i = 0; i < n; i++) {
      const live = freq ? freq[Math.floor((i / n) * freq.length * 0.7)] / 255 : 0;
      const idle = 0.1 + 0.07 * Math.sin(t / 700 + i * 0.4);
      const bh = Math.max(live, idle) * h * 0.9;
      vg.fillRect(i * bw + 1, h - bh, bw - 2, bh);
    }
  }

  // ------------------------------------------------------------------ main render loop
  let roll, overview;
  function frame(t) {
    requestAnimationFrame(frame);
    if (ctx) { // pop hits that are due now -> flash pads and rows
      const ct = ctx.currentTime;
      for (let i = hitQueue.length - 1; i >= 0; i--) {
        if (hitQueue[i].when <= ct) { flashPad(hitQueue[i].id); roll.hit(hitQueue[i].id); hitQueue.splice(i, 1); }
      }
    }
    const playing = player && player.playing;
    const pos = playing ? player.stepPos() : -1;
    roll.playStep = pos;
    const p = state.project;
    if (playing && pos >= 0 && performance.now() - roll.lastInteract > 2500) {
      // Follow the playhead: turn the page when it nears the right edge.
      if (pos < roll.viewStart || pos > roll.viewStart + roll.visSteps * 0.88) roll.viewStart = DF.clamp(pos - roll.visSteps * 0.12, 0, roll.maxView);
    }
    if (p) {
      overview.view = { start: roll.viewStart, len: roll.visSteps };
      overview.playFrac = pos >= 0 ? overview.stepToFrac(pos) : -1;
      if (playing) {
        const shown = Math.max(0, pos);
        $('pos').textContent = 'Bar ' + (Math.floor(shown / 16) + 1) + ' · Beat ' + (Math.floor((shown % 16) / 4) + 1);
        $('posSub').textContent = state.mode === 'audio'
          ? fmtTime(player.songTime()) + ' / ' + fmtTime(state.audio.duration) + ' · ' + p.bpm + ' BPM'
          : p.bpm + ' BPM · looping ' + p.bars + ' bar' + (p.bars > 1 ? 's' : '');
      } else {
        const c = Math.max(0, state.cue.step);
        $('pos').textContent = 'Bar ' + (Math.floor(c / 16) + 1) + ' · Beat ' + (Math.floor((c % 16) / 4) + 1);
      }
    }
    roll.draw(); overview.draw(); drawViz(t);
  }

  // ------------------------------------------------------------------ init & events
  function init() {
    Object.keys(DF.KITS).forEach((k) => $('kit').append(new Option(k, k)));
    buildPads();

    roll = new DF.Roll($('roll'), {
      onEditStart: () => pushUndo(),
      onEdit: () => overview.refresh(),
      onSeek: (step) => seekStep(step),
      onMute: (id) => { if (state.muted.has(id)) state.muted.delete(id); else state.muted.add(id); },
      onAudition: (id, v) => { ensureAudio(); previewSynth.trigger(id, ctx.currentTime + 0.01, v || 0.9); flashPad(id); },
      onView: () => {},
    });
    overview = new DF.Overview($('overview'), {
      onSeek: (frac) => {
        if (!state.project) return;
        const t = overview.fracToTarget(frac);
        seekStep(t.step, t.time != null ? t.time : undefined);
        roll.setView(t.step - roll.visSteps * 0.15);
      },
    });

    // tabs
    $('tab-audio').onclick = () => switchMode('audio');
    $('tab-random').onclick = () => switchMode('random');

    // upload
    $('chooseBtn').onclick = (e) => { e.stopPropagation(); $('file').click(); };
    $('drop').onclick = () => $('file').click();
    $('drop').onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } };
    $('file').onchange = (e) => { loadFile(e.target.files[0]); e.target.value = ''; };
    $('demoBtn').onclick = async (e) => {
      e.stopPropagation();
      ensureAudio();
      showProgress('Rendering demo song…', 0.1);
      const buf = await DF.renderDemo();
      await analyseBuffer(buf, 'Demo song (synth loop)');
    };
    // Drag & drop anywhere on the page.
    ['dragenter', 'dragover'].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); $('drop').classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => document.addEventListener(ev, (e) => { if (ev === 'drop' || e.target === document.documentElement) $('drop').classList.remove('over'); }));
    document.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

    // tempo / grid corrections
    $('abpm').onchange = () => { state.audio.bpm = +$('abpm').value || state.audio.bpm; regrid(true); };
    $('bpmHalf').onclick = () => { state.audio.bpm = Math.round((state.audio.bpm / 2) * 100) / 100; regrid(true); };
    $('bpmDouble').onclick = () => { state.audio.bpm = Math.round(state.audio.bpm * 2 * 100) / 100; regrid(true); };
    $('shiftBeat').onclick = () => { state.audio.barShift = (state.audio.barShift + 1) % 4; regrid(false); };
    $('shiftHalf').onclick = () => { state.audio.barShift = (state.audio.barShift + 2) % 4; regrid(false); };
    $('nudge').oninput = () => ($('nudgeOut').textContent = $('nudge').value + ' ms');
    $('nudge').onchange = () => { state.audio.nudge = +$('nudge').value; regrid(false); };

    // generator controls
    document.querySelectorAll('[data-regen]').forEach((el) => {
      el.addEventListener('input', updateOutputs);
      el.addEventListener('change', () => generate(false));
    });
    $('style').onchange = () => { state.styleByMode[state.mode] = $('style').value; generate(true); };
    $('bars').onchange = () => generate(false);
    $('rbpm').oninput = () => {
      updateOutputs();
      if (state.mode === 'random' && state.project) { state.project.bpm = +$('rbpm').value; }
    };
    $('rbpm').onchange = writeHash;
    $('swing').oninput = () => { updateOutputs(); if (state.project) state.project.swing = pct('swing'); };
    $('humanize').oninput = () => { updateOutputs(); if (state.project) state.project.humanize = pct('humanize'); };
    $('kit').onchange = () => {
      if (previewSynth) previewSynth.setKit($('kit').value);
      if (player) player.setKit($('kit').value);
      if (state.project) state.project.kit = $('kit').value;
      writeHash();
    };
    $('seed').onchange = () => generate(false);
    $('dice').onclick = () => { $('seed').value = DF.randomSeed(); generate(true); };
    $('regen').onclick = () => generate(false);
    $('mutate').onclick = () => {
      if (!state.project) return;
      pushUndo();
      DF.mutate(state.project.pattern, state.project.bars * 16, DF.makeRng((Math.random() * 4294967296) >>> 0), 0.12);
      overview.refresh();
    };
    $('undo').onclick = doUndo;

    // transport
    $('play').onclick = togglePlay;
    $('volDrums').oninput = applyVolumes;
    $('volSong').oninput = applyVolumes;
    $('zoom').oninput = () => roll.setZoom(+$('zoom').value);
    $('rec').onclick = () => {
      state.rec = !state.rec;
      $('rec').classList.toggle('on', state.rec);
      if (state.rec) pushUndo();
    };

    // saving
    $('save').onclick = () => {
      const p = state.project;
      if (!p) return;
      p.name = projectName();
      try {
        DF.library.save(p);
        renderLibrary();
        say('Saved "' + p.name + '" to your library.');
      } catch (err) {
        say('Could not save: browser storage is full or blocked. Download the MIDI or WAV instead.', true);
      }
    };
    $('dlMidi').onclick = () => {
      if (!state.project) return;
      download(DF.makeMidi(state.project, state.muted, true), slug(projectName()) + '.mid');
      say('MIDI saved. Open it in any DAW and assign a drum kit (General MIDI channel 10).');
    };
    $('dlWav').onclick = () => withBusy($('dlWav'), 'Rendering…', async () => {
      const p = state.project;
      if (!p) return;
      const loops = state.mode === 'random' ? Math.max(1, Math.ceil(8 / p.bars)) : 1;
      download(await DF.renderWav(p, { muted: state.muted, loops, buffer: state.audio && state.audio.buffer, mixSong: false }), slug(projectName()) + '-drums.wav');
      say(state.mode === 'random' ? 'Rendered ' + loops * p.bars + ' bars of drums.' : 'Rendered the drums, aligned with your song.');
    });
    $('dlMix').onclick = () => withBusy($('dlMix'), 'Rendering…', async () => {
      const p = state.project;
      if (!p || !state.audio) return;
      const blob = await DF.renderWav(p, { muted: state.muted, loops: 1, buffer: state.audio.buffer, mixSong: true, songVol: +$('volSong').value / 100, drumVol: +$('volDrums').value / 100 });
      download(blob, slug(projectName()) + '-mix.wav');
      say('Rendered your song with the drums on top.');
    });
    $('copyLink').onclick = async () => {
      writeHash();
      try { await navigator.clipboard.writeText(location.href); say('Link copied. It recreates this generated beat (manual edits are not included).'); }
      catch (e) { say('Copy this page address to share: ' + location.href); }
    };

    // keyboard
    document.addEventListener('keydown', (e) => {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' && e.target.type === 'text' || tag === 'textarea') return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); doUndo(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
      if (e.code === 'Space' && tag !== 'button' && tag !== 'select') { e.preventDefault(); togglePlay(); return; }
      const tr = TRACKS.find((t) => t.key === e.key.toLowerCase());
      if (tr && tag !== 'select') padHit(tr.id);
    });

    updateOutputs();
    renderLibrary();

    // start in random mode (or restore from a share link)
    state.fromLink = readHash(); // a share link already carries tempo + swing, so don't retune
    if (!state.fromLink) $('seed').value = DF.randomSeed();
    switchMode('random');
    state.fromLink = false;
    requestAnimationFrame(frame);

    // Handy for debugging in the console.
    DF.debug = { state, generate, get roll() { return roll; }, get player() { return player; }, ensureAudio, analyseBuffer };
  }

  window.addEventListener('DOMContentLoaded', init);
})();
