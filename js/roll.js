// Canvas drawing for the two big views:
//   Roll     - the step sequencer grid (click / drag to add or erase hits)
//   Overview - the whole song at a glance (waveform + drum density), click to seek
(function (DF) {
  const RULER_H = 24, ROW_H = 30;
  const css = { bg: '#0d0d16', row: '#12121e', rowAlt: '#151524', line: '#1f1f33', beat: '#2c2c46', bar: '#50507a', text: '#8d8dad' };

  function setupCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return g;
  }

  DF.Roll = class Roll {
    // cb = { onEditStart(), onEdit(), onSeek(step), onMute(id), onAudition(id), onView() }
    constructor(canvas, cb) {
      this.c = canvas; this.cb = cb;
      this.project = null; this.feats = null; this.muted = new Set();
      this.viewStart = 0; this.cellW = 22;
      this.playStep = -1; this.cueStep = -1; this.flash = {}; this.lastInteract = 0;
      this.paint = null; this.dragging = false; this.lw = 108;
      this.resize();
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      canvas.addEventListener('pointerdown', (e) => this._down(e));
      canvas.addEventListener('pointermove', (e) => this._move(e));
      canvas.addEventListener('pointerup', () => (this.dragging = false));
      canvas.addEventListener('pointercancel', () => (this.dragging = false));
      canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        this.scrollBy(d / this.cellW);
      }, { passive: false });
    }

    get total() { return this.project ? this.project.bars * 16 : 0; }
    get visSteps() { return (this.W - this.lw) / this.cellW; }
    get maxView() { return Math.max(0, this.total - this.visSteps); }
    visibleStep(s) { return s >= this.viewStart && s <= this.viewStart + this.visSteps - 1; }

    resize() {
      const w = Math.floor(this.c.parentElement.clientWidth) || 600;
      this.lw = w < 480 ? 84 : 108; // narrower label column on phones
      this.W = w; this.H = RULER_H + DF.TRACKS.length * ROW_H;
      this.g = setupCanvas(this.c, this.W, this.H);
      this.viewStart = DF.clamp(this.viewStart, 0, this.maxView);
    }

    setProject(p, feats) { this.project = p; this.feats = feats || null; this.viewStart = DF.clamp(this.viewStart, 0, this.maxView); }
    scrollBy(steps) { this.lastInteract = performance.now(); this.setView(this.viewStart + steps); }
    setView(v) { this.viewStart = DF.clamp(v, 0, this.maxView); if (this.cb.onView) this.cb.onView(); }
    setZoom(cellW) {
      const center = this.viewStart + this.visSteps / 2;
      this.cellW = DF.clamp(cellW, 8, 48);
      this.viewStart = DF.clamp(center - this.visSteps / 2, 0, this.maxView);
    }
    hit(id) { this.flash[id] = performance.now(); }

    _pos(e) { const r = this.c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
    _cell(x, y) {
      if (x < this.lw || y < RULER_H) return null;
      const step = Math.floor(this.viewStart + (x - this.lw) / this.cellW);
      const row = Math.floor((y - RULER_H) / ROW_H);
      if (step < 0 || step >= this.total || row < 0 || row >= DF.TRACKS.length) return null;
      return { step, row };
    }

    _down(e) {
      if (!this.project) return;
      const { x, y } = this._pos(e);
      if (y < RULER_H && x >= this.lw) { // ruler click = seek
        const step = Math.floor(this.viewStart + (x - this.lw) / this.cellW);
        this.cb.onSeek(DF.clamp(step, 0, this.total - 1));
        return;
      }
      if (x < this.lw && y >= RULER_H) { // label: M button mutes, anywhere else previews the sound
        const row = Math.floor((y - RULER_H) / ROW_H);
        if (row >= 0 && row < DF.TRACKS.length) {
          const id = DF.TRACKS[row].id;
          if (x > this.lw - 28) this.cb.onMute(id); else this.cb.onAudition(id);
        }
        return;
      }
      const cell = this._cell(x, y);
      if (!cell) return;
      const id = DF.TRACKS[cell.row].id, arr = this.project.pattern[id];
      // First cell decides whether this drag paints or erases.
      this.paint = arr[cell.step] > 0 ? 0 : e.shiftKey ? 0.4 : 0.85;
      this.cb.onEditStart();
      this.dragging = true;
      this.c.setPointerCapture(e.pointerId);
      this._apply(cell);
    }
    _move(e) {
      if (!this.dragging) return;
      const { x, y } = this._pos(e);
      const cell = this._cell(x, y);
      if (cell) this._apply(cell);
    }
    _apply(cell) {
      const id = DF.TRACKS[cell.row].id, arr = this.project.pattern[id];
      if (arr[cell.step] === this.paint) return;
      arr[cell.step] = this.paint;
      if (this.paint > 0 && this.cb.onAudition) this.cb.onAudition(id, this.paint);
      this.cb.onEdit();
    }

    draw() {
      const g = this.g, p = this.project;
      g.fillStyle = css.bg;
      g.fillRect(0, 0, this.W, this.H);
      if (!p) return;
      const cw = this.cellW, total = this.total;
      const first = Math.max(0, Math.floor(this.viewStart)), last = Math.min(total - 1, Math.ceil(this.viewStart + this.visSteps));
      const xOf = (i) => this.lw + (i - this.viewStart) * cw;
      const now = performance.now();

      g.save();
      g.beginPath(); g.rect(this.lw, 0, this.W - this.lw, this.H); g.clip();

      // row backgrounds
      for (let r = 0; r < DF.TRACKS.length; r++) {
        g.fillStyle = r % 2 ? css.rowAlt : css.row;
        g.fillRect(this.lw, RULER_H + r * ROW_H, this.W - this.lw, ROW_H);
      }
      // grid lines + ruler
      g.font = '11px ui-monospace, Menlo, monospace';
      g.textBaseline = 'middle';
      for (let i = first; i <= last + 1; i++) {
        const x = Math.round(xOf(i)) + 0.5;
        const isBar = i % 16 === 0, isBeat = i % 4 === 0;
        if (!isBeat && cw < 14) continue;
        g.strokeStyle = isBar ? css.bar : isBeat ? css.beat : css.line;
        g.beginPath(); g.moveTo(x, isBar ? 0 : RULER_H); g.lineTo(x, this.H); g.stroke();
        if (isBar && i < total) { // bar numbers: bold and bright
          g.fillStyle = '#e6e6fa';
          g.font = 'bold 12px ui-monospace, Menlo, monospace';
          g.fillText(String(i / 16 + 1), x + 5, RULER_H / 2);
          g.font = '11px ui-monospace, Menlo, monospace';
        } else if (isBeat && cw >= 18 && i < total) { // beat ticks: small and dim
          g.fillStyle = '#4a4a68';
          g.fillText('.' + (((i % 16) >> 2) + 1), x + 3, RULER_H / 2);
        }
      }
      // faint "what the analyser heard" bars under kick / snare / hat rows (audio mode)
      if (this.feats) {
        const rows = [['kick', this.feats.L], ['snare', this.feats.M], ['hat', this.feats.H]];
        for (const [id, arr] of rows) {
          const r = DF.TRACKS.findIndex((t) => t.id === id), col = DF.TRACKS[r].color;
          g.fillStyle = col + '30';
          for (let i = first; i <= last; i++) {
            const h = Math.min(1, arr[i]) * (ROW_H - 4);
            if (h > 1.5) g.fillRect(xOf(i) + 2, RULER_H + (r + 1) * ROW_H - 2 - h, cw - 4, h);
          }
        }
      }
      // notes
      for (let r = 0; r < DF.TRACKS.length; r++) {
        const tr = DF.TRACKS[r], arr = p.pattern[tr.id], dim = this.muted.has(tr.id);
        const y = RULER_H + r * ROW_H;
        for (let i = first; i <= last; i++) {
          const v = arr[i];
          if (v <= 0) continue;
          const x = xOf(i);
          const age = this.playStep >= 0 && i === Math.floor(this.playStep) ? 1 : 0;
          g.globalAlpha = dim ? 0.2 : 0.4 + 0.6 * v;
          g.fillStyle = tr.color;
          g.beginPath(); g.roundRect(x + 1.5, y + 4, cw - 3, ROW_H - 8, Math.min(4, cw / 3)); g.fill();
          if (age && !dim) { g.globalAlpha = 0.7; g.fillStyle = '#fff'; g.fill(); }
        }
        g.globalAlpha = 1;
      }
      // cue marker (where Play will start from) when stopped
      if (this.playStep < 0 && this.cueStep > 0) {
        const x = xOf(this.cueStep);
        g.strokeStyle = '#ffb627'; g.setLineDash([4, 4]);
        g.beginPath(); g.moveTo(x, 0); g.lineTo(x, this.H); g.stroke();
        g.setLineDash([]);
      }
      // playhead
      if (this.playStep >= 0) {
        const x = xOf(this.playStep);
        g.strokeStyle = '#ffffff'; g.lineWidth = 2;
        g.beginPath(); g.moveTo(x, 0); g.lineTo(x, this.H); g.stroke();
        g.lineWidth = 1;
      }
      g.restore();

      // labels (drawn last so scrolled notes slide under them)
      for (let r = 0; r < DF.TRACKS.length; r++) {
        const tr = DF.TRACKS[r], y = RULER_H + r * ROW_H;
        g.fillStyle = '#10101a';
        g.fillRect(0, y, this.lw, ROW_H);
        g.fillStyle = css.line;
        g.fillRect(0, y + ROW_H - 1, this.lw, 1);
        const fl = Math.max(0, 1 - (now - (this.flash[tr.id] || 0)) / 180);
        const muted = this.muted.has(tr.id);
        g.fillStyle = tr.color;
        g.globalAlpha = muted ? 0.25 : 0.55 + 0.45 * fl;
        g.beginPath(); g.arc(14, y + ROW_H / 2, 4 + 3 * fl, 0, 7); g.fill();
        g.globalAlpha = 1;
        g.fillStyle = muted ? '#555570' : '#d0d0e8';
        g.font = '12px system-ui, sans-serif';
        const narrow = this.lw < 100; // short names so they don't hit the mute button
        g.fillText(narrow ? tr.label.replace('Open Hat', 'O.Hat').replace('Tom ', 'T.') : tr.label, narrow ? 24 : 28, y + ROW_H / 2);
        // mute button
        g.strokeStyle = muted ? '#ff5a3c' : '#3a3a58';
        g.strokeRect(this.lw - 24.5, y + 7.5, 18, 15);
        g.fillStyle = muted ? '#ff5a3c' : '#6a6a8a';
        g.font = '10px system-ui, sans-serif';
        g.fillText('M', this.lw - 19, y + ROW_H / 2 + 0.5);
      }
      g.fillStyle = '#10101a';
      g.fillRect(0, 0, this.lw, RULER_H);
      g.fillStyle = css.text;
      g.font = '10px system-ui, sans-serif';
      g.fillText('BAR', 10, RULER_H / 2);
    }
  };

  DF.Overview = class Overview {
    // cb.onSeek({ step } | { time }) ; cb.onScroll(centerStep)
    constructor(canvas, cb) {
      this.c = canvas; this.cb = cb;
      this.project = null; this.peaks = null; this.duration = 0;
      this.view = { start: 0, len: 1 }; this.playFrac = -1; this.dens = [];
      this.resize();
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      canvas.addEventListener('pointerdown', (e) => {
        const r = canvas.getBoundingClientRect();
        this.cb.onSeek(DF.clamp((e.clientX - r.left) / r.width, 0, 0.9999));
      });
    }
    resize() {
      this.W = Math.floor(this.c.parentElement.clientWidth) || 600; this.H = 84;
      this.g = setupCanvas(this.c, this.W, this.H);
    }
    setProject(p, peaks, duration) { this.project = p; this.peaks = peaks || null; this.duration = duration || 0; this.refresh(); }
    // Recompute how busy each bar is (for the little bar chart).
    refresh() {
      const p = this.project;
      if (!p) return;
      this.dens = [];
      for (let b = 0; b < p.bars; b++) {
        let n = 0;
        for (const tr of DF.TRACKS) for (let s = b * 16; s < b * 16 + 16; s++) if (p.pattern[tr.id][s] > 0) n++;
        this.dens.push(Math.min(1, n / 40));
      }
    }
    draw() {
      const g = this.g, p = this.project, W = this.W, H = this.H;
      g.fillStyle = '#0d0d16'; g.fillRect(0, 0, W, H);
      if (!p) return;
      const sd = 60 / p.bpm / 4, total = p.bars * 16;
      // Map a step to an x position: song time in audio mode, plain step otherwise.
      const span = this.duration || total * sd;
      const offset = this.duration ? p.gridOffset : 0;
      const xOfStep = (s) => ((offset + s * sd) / span) * W;
      if (this.peaks) {
        g.fillStyle = '#3b3b5e';
        const n = this.peaks.length / 2;
        for (let x = 0; x < W; x++) {
          const b = Math.min(n - 1, Math.floor((x / W) * n));
          const mn = this.peaks[b * 2], mx = this.peaks[b * 2 + 1];
          const top = H * 0.42 - mx * H * 0.36, bot = H * 0.42 - mn * H * 0.36;
          g.fillRect(x, top, 1, Math.max(1, bot - top));
        }
      }
      // drum density per bar
      for (let b = 0; b < p.bars; b++) {
        const x0 = xOfStep(b * 16), x1 = xOfStep((b + 1) * 16);
        const h = this.dens[b] * (H * 0.4);
        g.fillStyle = 'rgba(255,90,60,0.85)';
        g.fillRect(x0 + 0.5, H - h - 2, Math.max(1, x1 - x0 - 1), h);
        if (p.bars <= 32) { g.fillStyle = '#5a5a80'; g.font = '10px system-ui'; g.fillText(String(b + 1), x0 + 4, 11); }
      }
      // visible window
      const vx0 = xOfStep(this.view.start), vx1 = xOfStep(this.view.start + this.view.len);
      g.fillStyle = 'rgba(255,255,255,0.08)';
      g.fillRect(vx0, 0, Math.max(3, vx1 - vx0), H);
      g.strokeStyle = 'rgba(255,255,255,0.35)';
      g.strokeRect(vx0 + 0.5, 0.5, Math.max(3, vx1 - vx0) - 1, H - 1);
      if (this.playFrac >= 0) {
        g.fillStyle = '#fff';
        g.fillRect(this.playFrac * W - 1, 0, 2, H);
      }
    }
    // Convert a click fraction to a seek target.
    fracToTarget(frac) {
      const p = this.project;
      const sd = 60 / p.bpm / 4, total = p.bars * 16;
      if (this.duration) {
        const time = frac * this.duration;
        return { time, step: DF.clamp(Math.floor((time - p.gridOffset) / sd), 0, total - 1) };
      }
      return { step: DF.clamp(Math.floor(frac * total), 0, total - 1) };
    }
    stepToFrac(step) {
      const p = this.project;
      const sd = 60 / p.bpm / 4, total = p.bars * 16;
      return this.duration ? (p.gridOffset + step * sd) / this.duration : step / total;
    }
  };
})((window.DF = window.DF || {}));
