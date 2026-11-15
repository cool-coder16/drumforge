// The neural drummer, running in the browser with no ML library.
//
// This file re-implements, in plain JavaScript, the small transformer trained in
// drumforge-ml/model.py (4 layers, ~1.7M parameters, trained on ~9,500 real songs from the
// Lakh MIDI dataset). It writes the drum part one 16th-note step at a time: at every step the
// network sees the drums it has already played, the audio features around that step, where it is
// in the bar / phrase, and optional conditioning (style, density, tempo), and outputs a probability
// for every drum on that step. A KV cache holds the last 96 steps, so a whole song is cheap.
(function (DF) {
  const N_TRACKS = 9, N_CLASSES = 4, OFFS = [-4, -3, -2, -1, 0, 1, 2, 3, 4];

  // float16 -> float32 lookup (weights are shipped as fp16 to halve the download)
  let F16 = null;
  function f16table() {
    if (F16) return F16;
    F16 = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
      F16[h] = e === 0 ? s * m * Math.pow(2, -24) : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
    }
    return F16;
  }

  // erf via Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7), for the exact GELU the model was trained with
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x * 0.7071067811865476));

  // y = W x + b, with W stored row-major as (out, in). Four independent accumulators let the JS engine
  // overlap the multiply-adds (this loop is where nearly all the inference time goes).
  function matvec(W, x, b, out, nOut, nIn) {
    const n4 = nIn & ~3;
    for (let o = 0, base = 0; o < nOut; o++, base += nIn) {
      let s0 = b ? b[o] : 0, s1 = 0, s2 = 0, s3 = 0, i = 0;
      for (; i < n4; i += 4) {
        s0 += W[base + i] * x[i]; s1 += W[base + i + 1] * x[i + 1];
        s2 += W[base + i + 2] * x[i + 2]; s3 += W[base + i + 3] * x[i + 3];
      }
      for (; i < nIn; i++) s0 += W[base + i] * x[i];
      out[o] = s0 + s1 + s2 + s3;
    }
  }

  function layerNorm(x, w, b, out) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
    const inv = 1 / Math.sqrt(v / n + 1e-5);
    for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  }

  class Net {
    constructor(cfg, buf) {
      this.cfg = cfg;
      const table = f16table(), u16 = new Uint16Array(buf), W = {};
      for (const t of cfg.tensors) {
        const n = t.shape.reduce((a, b) => a * b, 1), arr = new Float32Array(n);
        for (let i = 0; i < n; i++) arr[i] = table[u16[t.offset + i]];
        W[t.name] = arr;
      }
      this.W = W;
      this.d = cfg.d; this.h = cfg.n_heads; this.hd = cfg.d / cfg.n_heads;
      this.win = cfg.window; this.L = cfg.n_layers; this.ff = cfg.ff; this.hh = cfg.head_hidden;
      this.slopes = Array.from({ length: this.h }, (_, i) => Math.pow(2, (-8 * (i + 1)) / this.h));
      const d = this.d;
      this.x = new Float32Array(d); this.a = new Float32Array(d); this.qkv = new Float32Array(3 * d);
      this.att = new Float32Array(d); this.tmp = new Float32Array(d); this.f1 = new Float32Array(this.ff);
      this.scores = new Float32Array(this.win); this.hid = new Float32Array(d);
      this.z = new Float32Array(this.hh); this.lg = new Float32Array(N_CLASSES);
      this.reset();
    }

    reset() { // empty KV cache
      const n = this.win * this.d;
      this.K = Array.from({ length: this.L }, () => new Float32Array(n));
      this.V = Array.from({ length: this.L }, () => new Float32Array(n));
      this.pos = 0;
    }

    // Copy / restore the network's memory, so a bar can be tried several ways and rewound.
    snapshot() { return { K: this.K.map((a) => a.slice()), V: this.V.map((a) => a.slice()), pos: this.pos }; }
    restore(sn) { sn.K.forEach((a, i) => this.K[i].set(a)); sn.V.forEach((a, i) => this.V[i].set(a)); this.pos = sn.pos; }

    // Conditioning vector (style + density + tempo embeddings), constant over a generation.
    cond(style, dens, tempo) {
      const W = this.W, d = this.d, c = new Float32Array(d);
      for (let i = 0; i < d; i++) c[i] = W['style_emb.weight'][style * d + i] + W['dens_emb.weight'][dens * d + i] + W['tempo_emb.weight'][tempo * d + i];
      return c;
    }

    // Run one time step through the trunk; returns the hidden state used by the track heads.
    step(prev, feat, stepInBar, barInPhrase, cond, maxDist) {
      const W = this.W, d = this.d, x = this.x, a = this.a;
      for (let i = 0; i < d; i++) x[i] = cond[i] + W['pos_step.weight'][stepInBar * d + i] + W['pos_bar.weight'][barInPhrase * d + i] + W['feat.bias'][i];
      const pe = W['prev_emb.weight'];
      for (let j = 0; j < N_TRACKS; j++) {
        const o = (j * N_CLASSES + prev[j]) * d;
        for (let i = 0; i < d; i++) x[i] += pe[o + i];
      }
      const fw = W['feat.weight'], nf = this.cfg.n_feat;
      for (let i = 0; i < d; i++) {
        let s = 0;
        const base = i * nf;
        for (let k = 0; k < nf; k++) s += fw[base + k] * feat[k];
        x[i] += s;
      }
      const slot = this.pos % this.win, n = Math.min(this.pos + 1, this.win, maxDist || this.win);
      for (let l = 0; l < this.L; l++) {
        const p = 'blocks.' + l + '.';
        layerNorm(x, W[p + 'ln1.weight'], W[p + 'ln1.bias'], a);
        matvec(W[p + 'qkv.weight'], a, W[p + 'qkv.bias'], this.qkv, 3 * d, d);
        this.K[l].set(this.qkv.subarray(d, 2 * d), slot * d);
        this.V[l].set(this.qkv.subarray(2 * d, 3 * d), slot * d);
        const K = this.K[l], V = this.V[l], sc = this.scores, att = this.att, hd = this.hd, inv = 1 / Math.sqrt(hd);
        for (let h = 0; h < this.h; h++) {
          const qo = h * hd;
          let mx = -Infinity;
          for (let dist = 0; dist < n; dist++) {
            const ko = ((((this.pos - dist) % this.win) + this.win) % this.win) * d + h * hd;
            let s = 0;
            for (let i = 0; i < hd; i++) s += this.qkv[qo + i] * K[ko + i];
            s = s * inv - this.slopes[h] * dist;
            sc[dist] = s;
            if (s > mx) mx = s;
          }
          let sum = 0;
          for (let dist = 0; dist < n; dist++) { sc[dist] = Math.exp(sc[dist] - mx); sum += sc[dist]; }
          for (let i = 0; i < hd; i++) att[h * hd + i] = 0;
          for (let dist = 0; dist < n; dist++) {
            const w = sc[dist] / sum, vo = ((((this.pos - dist) % this.win) + this.win) % this.win) * d + h * hd;
            for (let i = 0; i < hd; i++) att[h * hd + i] += w * V[vo + i];
          }
        }
        matvec(W[p + 'proj.weight'], att, W[p + 'proj.bias'], this.tmp, d, d);
        for (let i = 0; i < d; i++) x[i] += this.tmp[i];
        layerNorm(x, W[p + 'ln2.weight'], W[p + 'ln2.bias'], a);
        matvec(W[p + 'fc1.weight'], a, W[p + 'fc1.bias'], this.f1, this.ff, d);
        for (let i = 0; i < this.ff; i++) this.f1[i] = gelu(this.f1[i]);
        matvec(W[p + 'fc2.weight'], this.f1, W[p + 'fc2.bias'], this.tmp, d, this.ff);
        for (let i = 0; i < d; i++) x[i] += this.tmp[i];
      }
      layerNorm(x, W['ln_f.weight'], W['ln_f.bias'], this.hid);
      this.pos++;
      return this.hid;
    }

    // Logits (4 classes) for track j given the classes already chosen for tracks < j at this step.
    head(h, j, chosen) {
      const W = this.W, d = this.d, hh = this.hh, nin = d + N_TRACKS * N_CLASSES;
      const w1 = W['head1.' + j + '.weight'], b1 = W['head1.' + j + '.bias'], z = this.z;
      for (let o = 0; o < hh; o++) {
        const base = o * nin;
        let s = b1[o];
        for (let i = 0; i < d; i++) s += w1[base + i] * h[i];
        for (let t = 0; t < j; t++) s += w1[base + d + t * N_CLASSES + chosen[t]];
        z[o] = gelu(s);
      }
      matvec(W['head2.' + j + '.weight'], z, W['head2.' + j + '.bias'], this.lg, N_CLASSES, hh);
      return this.lg;
    }
  }

  const tickCh = (() => {
    const ch = new MessageChannel(), q = [];
    ch.port1.onmessage = () => { const r = q.shift(); if (r) r(); };
    return () => new Promise((r) => { q.push(r); ch.port2.postMessage(0); });
  })();

  const AI = (DF.AI = { net: null, cfg: null, loading: null });

  // Fetch the weights (once). `base` is the folder that holds drummer.json / drummer.bin.
  AI.load = function (base) {
    if (AI.loading) return AI.loading;
    AI.loading = (async () => {
      const [cfg, bin] = await Promise.all([fetch(base + 'drummer.json').then((r) => r.json()), fetch(base + 'drummer.bin').then((r) => r.arrayBuffer())]);
      AI.cfg = cfg;
      AI.net = new Net(cfg, bin);
      AI.styles = cfg.styles;
      return AI.net;
    })();
    AI.loading.catch(() => { AI.loading = null; });
    return AI.loading;
  };
  AI.Net = Net;

  // ---- learned downbeat detector (a small MLP, see drumforge-ml/downbeat.py) ----
  // Looks at a window of beats of low/mid/high onset strengths and says where in the bar the first beat is.
  // Sliding it over the whole song and adding up the log-probabilities picks which beat is beat 1.
  AI.loadDownbeat = async function (url) {
    const j = await fetch(url).then((r) => r.json()), T = {};
    for (const k in j.tensors) T[k] = Float32Array.from(j.tensors[k].data);
    AI.db = { W: j.window, floor: j.floor, h1: j.h1 || 192, h2: j.h2 || 96, T };
    return AI.db;
  };

  // grid = DF.buildGrid(...) built with barShift 0. Returns { shift, confidence } where `shift` is how many
  // beats after the grid's first beat the first bar line falls (0..3).
  AI.downbeat = function (grid) {
    const m = AI.db;
    if (!m) return null;
    const nBeats = Math.floor(grid.steps / 4), W = m.W;
    if (nBeats < W + 4) return null;
    const X = new Float32Array(nBeats * 12);
    for (let b = 0; b < nBeats; b++) {
      for (let q = 0; q < 4; q++) {
        const s = b * 4 + q;
        [grid.L[s], grid.M[s], grid.H[s]].forEach((v, k) => { X[b * 12 + q * 3 + k] = v < m.floor ? 0 : Math.min(1.2, v); });
      }
    }
    const H1 = m.h1, H2 = m.h2, h1 = new Float32Array(H1), h2 = new Float32Array(H2), lg = new Float32Array(4), score = [0, 0, 0, 0];
    for (let st = 0; st + W <= nBeats; st += 2) {
      const x = X.subarray(st * 12, (st + W) * 12);
      matvec(m.T['f1.weight'], x, m.T['f1.bias'], h1, H1, W * 12);
      for (let i = 0; i < H1; i++) h1[i] = gelu(h1[i]);
      matvec(m.T['f2.weight'], h1, m.T['f2.bias'], h2, H2, H1);
      for (let i = 0; i < H2; i++) h2[i] = gelu(h2[i]);
      matvec(m.T['f3.weight'], h2, m.T['f3.bias'], lg, 4, H2);
      const mx = Math.max(lg[0], lg[1], lg[2], lg[3]);
      const lse = mx + Math.log(Math.exp(lg[0] - mx) + Math.exp(lg[1] - mx) + Math.exp(lg[2] - mx) + Math.exp(lg[3] - mx));
      // window starting at grid beat `st` has true position (st + d) % 4 if the bar offset is d
      for (let d = 0; d < 4; d++) score[d] += lg[(st + d) % 4] - lse;
    }
    let best = 0;
    for (let d = 1; d < 4; d++) if (score[d] > score[best]) best = d;
    const mx = Math.max(...score), ex = score.map((v) => Math.exp((v - mx) / Math.max(1, nBeats / 8))), conf = ex[best] / ex.reduce((a, b) => a + b, 0);
    return { shift: (4 - best) % 4, confidence: conf, scores: score };
  };

  const bucket = (x, cuts) => { let b = 0; while (b < cuts.length && x >= cuts[b]) b++; return b; };
  const VEL = [0, 0.3, 0.68, 0.97]; // velocity for each class: off / ghost / normal / accent

  // Build the 34-number feature vector for step s (mirrors Data.batch in train.py).
  function featureVector(s, f, gain, out) {
    const n = f.steps, bar = Math.floor(s / 16);
    let k = 0;
    for (const o of OFFS) {
      const i = Math.min(n - 1, Math.max(0, s + o));
      out[k++] = f.L[i] * gain; out[k++] = f.M[i] * gain; out[k++] = f.H[i] * gain;
    }
    const E = f.E[bar], En = f.E[Math.min(f.bars - 1, bar + 1)], Ep = f.E[Math.max(0, bar - 1)];
    out[k++] = E; out[k++] = En - E; out[k++] = E - Ep;
    out[k++] = f.barMean[bar * 3] * gain; out[k++] = f.barMean[bar * 3 + 1] * gain; out[k++] = f.barMean[bar * 3 + 2] * gain;
    out[k++] = 1; // has audio
    return out;
  }

  // Preprocess the browser's analysis grid the way the training features look: exact zeros between
  // onsets (a small floor is removed) plus per-bar means.
  AI.prepareFeatures = function (feats, floor) {
    const fl = floor == null ? 0.2 : floor, cl = (a) => Float32Array.from(a, (v) => (v < fl ? 0 : Math.min(1.2, v)));
    const L = cl(feats.L), M = cl(feats.M), H = cl(feats.H), bars = feats.bars, barMean = new Float32Array(bars * 3);
    for (let b = 0; b < bars; b++) {
      for (let p = 0; p < 16; p++) {
        barMean[b * 3] += L[b * 16 + p] / 16; barMean[b * 3 + 1] += M[b * 16 + p] / 16; barMean[b * 3 + 2] += H[b * 16 + p] / 16;
      }
    }
    return { L, M, H, E: feats.E, bars, steps: bars * 16, barMean };
  };


  // Generate a drum pattern.
  //   o = { bars, bpm, seed, feats?, styleId (-1 = auto), density (-1 = auto, else 0..7), temperature, follow (0..1.5),
  //         ghostBias, fillBias, fills (0..1 chance of a fill at phrase ends), humanize, onProgress(frac) }
  // Returns { pattern: {trackId: Float32Array}, styleId, fillBars }
  //
  // How it works:
  //  * Without audio the network first writes several 2-bar grooves from scratch; the dull / empty ones are
  //    filtered out by hit count and by the network's own likelihood, and one is picked at random.
  //  * Then it plays on, bar by bar. If a bar collapses (goes nearly silent) it is re-sampled.
  //  * At phrase ends it may write a fill: candidates are sampled with limited attention and a busier density
  //    token, and the network's memory is then rewound so later bars go back to the groove.
  AI.generate = async function (o) {
    const net = AI.net, cfg = AI.cfg;
    if (!net) throw new Error('AI model not loaded');
    const rng = DF.makeRng(DF.hashString(String(o.seed)));
    const bars = o.bars, steps = bars * 16, K = cfg.n_styles;
    const useAudio = !!o.feats && (o.follow == null || o.follow > 0);
    const feats = useAudio ? AI.prepareFeatures(o.feats, o.floor) : null;
    const gain = o.follow == null ? 1 : o.follow;
    let styleId = o.styleId == null ? -1 : o.styleId;
    if (styleId === -2) styleId = Math.floor(rng() * K); // "surprise"
    const styleTok = styleId >= 0 ? styleId : K;
    const densTok = o.density != null && o.density >= 0 ? Math.min(7, o.density) : 8;
    const tempoTok = o.bpm ? bucket(o.bpm, cfg.tempo_cuts) : 6;
    const cond = net.cond(styleTok, densTok, tempoTok);
    const condFill = net.cond(styleTok, Math.min(7, (densTok === 8 ? 4 : densTok) + 3), tempoTok);
    const temp = Math.max(0.05, o.temperature == null ? 0.9 : o.temperature);
    const ghostBias = o.ghostBias || 0, fillBias = o.fillBias || 0, hum = o.humanize == null ? 0.3 : o.humanize;
    const fillChance = o.fills == null ? 0 : o.fills, minPfrac = o.minP == null ? 0.06 : o.minP;
    const ids = DF.TRACKS.map((t) => t.id);
    const out = new Uint8Array(steps * N_TRACKS); // the chosen classes for every step and track

    const noFeat = new Float32Array(cfg.n_feat), fv = new Float32Array(cfg.n_feat), probs = new Float32Array(N_CLASSES);
    const stepFeat = (s) => (feats ? featureVector(s, feats, gain, fv) : noFeat);
    const chosen = new Uint8Array(N_TRACKS);

    // Sample steps [s0, s1) into `dst`, continuing from the drums `prevIn` played just before. opt: cond, maxDist,
    // temp, crashBias (extra push toward a crash on the first step), lp (accumulates log-probs).
    function sampleRange(dst, s0, s1, prevIn, opt) {
      const prev = Uint8Array.from(prevIn), c = (opt && opt.cond) || cond, T_ = (opt && opt.temp) || temp;
      for (let s = s0; s < s1; s++) {
        const bar = Math.floor(s / 16), sib = s % 16;
        const silent = feats && feats.E[bar] < 0.04; // keep the song's silent bars silent
        const h = net.step(prev, stepFeat(s), sib, bar % 8, c, opt && opt.maxDist);
        chosen.fill(0);
        if (!silent) {
          for (let j = 0; j < N_TRACKS; j++) {
            const lg = net.head(h, j, chosen);
            let mx = -Infinity, mx1 = -Infinity;
            for (let k = 0; k < N_CLASSES; k++) {
              let v = lg[k];
              if (k === 1) v += ghostBias;
              // fills: nudge toms / crash toward playing near the end of a 4-bar phrase
              if (k > 0 && (j === 5 || j === 6) && bar % 4 === 3 && sib >= 10) v += fillBias;
              if (k > 0 && j === 7 && sib === 0 && opt && opt.crashBias) v += opt.crashBias;
              probs[k] = v / T_;
              if (probs[k] > mx) mx = probs[k];
              if (lg[k] > mx1) mx1 = lg[k];
            }
            let sum = 0, z1 = 0;
            for (let k = 0; k < N_CLASSES; k++) { probs[k] = Math.exp(probs[k] - mx); sum += probs[k]; z1 += Math.exp(lg[k] - mx1); }
            const thr = minPfrac * Math.max(probs[0], probs[1], probs[2], probs[3]);
            sum = 0;
            for (let k = 0; k < N_CLASSES; k++) { if (probs[k] < thr) probs[k] = 0; sum += probs[k]; }
            let r = rng() * sum, pick = N_CLASSES - 1;
            for (let k = 0; k < N_CLASSES; k++) { r -= probs[k]; if (r <= 0) { pick = k; break; } }
            chosen[j] = pick;
            if (opt && opt.lp) opt.lp.v += lg[pick] - mx1 - Math.log(z1);
          }
        }
        for (let j = 0; j < N_TRACKS; j++) { prev[j] = chosen[j]; dst[s * N_TRACKS + j] = chosen[j]; }
      }
      return prev;
    }
    // Teacher-forced: push already-decided steps [s0, s1) of `src` through the network to rebuild its memory.
    function feedRange(src, s0, s1, prevIn, c) {
      const prev = Uint8Array.from(prevIn);
      for (let s = s0; s < s1; s++) {
        net.step(prev, stepFeat(s), s % 16, Math.floor(s / 16) % 8, c || cond);
        for (let j = 0; j < N_TRACKS; j++) prev[j] = src[s * N_TRACKS + j];
      }
      return prev;
    }
    const hitsIn = (arr, s0, s1) => { let n = 0; for (let i = s0 * N_TRACKS; i < s1 * N_TRACKS; i++) n += arr[i] > 0; return n; };

    // Expected hits per bar for the requested density bucket (to reject dull or over-busy candidates).
    const cuts = cfg.dens_cuts, dBucket = densTok === 8 ? 4 : densTok;
    const lo = dBucket === 0 ? cuts[0] * 0.6 : cuts[dBucket - 1], hi = dBucket >= cuts.length ? cuts[cuts.length - 1] * 1.4 : cuts[dBucket];
    const okBar = (n) => n >= lo * 0.55 && n <= hi * 1.6;

    let prev = new Uint8Array(N_TRACKS), startBar = 0;
    net.reset();

    // 1. Opening groove from scratch (audio-free mode)
    if (!useAudio && bars >= 2) {
      const cands = [];
      for (let k = 0; k < 10; k++) {
        net.reset();
        const lp = { v: 0 }, buf = new Uint8Array(32 * N_TRACKS);
        sampleRange(buf, 0, 32, new Uint8Array(N_TRACKS), { lp });
        const h0 = hitsIn(buf, 0, 16), h1 = hitsIn(buf, 16, 32);
        const kicks = (() => { let n = 0; for (let s = 0; s < 32; s++) n += buf[s * N_TRACKS] > 0; return n; })();
        if (okBar(h0) && okBar(h1) && kicks >= 2 && Math.abs(h0 - h1) <= 0.4 * Math.max(h0, h1)) cands.push({ buf, lp: lp.v / 32 });
        if (o.onProgress) o.onProgress(0.05 * (k + 1) / 10);
      }
      let pick = null;
      if (cands.length) {
        cands.sort((a, b) => b.lp - a.lp);
        pick = cands[Math.floor(rng() * Math.max(1, Math.ceil(cands.length / 2)))]; // random among the more typical half
      }
      net.reset();
      if (pick) {
        out.set(pick.buf, 0);
        prev = feedRange(out, 0, 32, prev);
        startBar = 2;
      }
    }

    // 2. Bar by bar
    const fillBars = [];
    let crashNext = 0;
    const refHits = [];
    for (let bar = startBar; bar < bars; bar++) {
      const s0 = bar * 16, s1 = s0 + 16, snapStart = net.snapshot(), prevStart = Uint8Array.from(prev);
      const phraseEnd = bars >= 2 && bar > 0 && (bars >= 4 ? bar % 4 === 3 : bar === bars - 1);
      const buildUp = feats && bar + 1 < bars && feats.E[bar + 1] - feats.E[bar] > 0.22;
      const wantFill = phraseEnd && (!feats || feats.E[bar] > 0.25) && (buildUp || rng() < fillChance);
      const silent = feats && feats.E[bar] < 0.04;

      const copt = crashNext ? { crashBias: crashNext } : null;
      crashNext = 0;
      let snapMid = null, prevMid = null;
      prev = sampleRange(out, s0, s0 + 8, prevStart, copt);
      if (wantFill && !silent) { snapMid = net.snapshot(); prevMid = Uint8Array.from(prev); }
      prev = sampleRange(out, s0 + 8, s1, prev, null);

      // collapse guard: a bar that went (nearly) silent although the song isn't gets re-sampled, else copies the last bar
      if (!feats && refHits.length >= 1 && !silent) {
        const ref = refHits[refHits.length - 1];
        let tries = 0;
        while (hitsIn(out, s0, s1) < 0.5 * ref && tries++ < 4) {
          net.restore(snapStart);
          prev = sampleRange(out, s0, s1, prevStart, copt);
          snapMid = null;
        }
        if (hitsIn(out, s0, s1) < 0.5 * ref && bar > 0) {
          net.restore(snapStart);
          out.copyWithin(s0 * N_TRACKS, (s0 - 16) * N_TRACKS, s0 * N_TRACKS);
          prev = feedRange(out, s0, s1, prevStart);
          snapMid = null;
        }
      }
      if (!silent) refHits.push(hitsIn(out, s0, s1));

      // phrase-end fill: pick the busiest of a few candidate second halves, then rewind so the groove carries on
      if (snapMid) {
        const normal = out.slice(s0 * N_TRACKS, s1 * N_TRACKS), baseline = (() => { let n = 0; for (let s = 8; s < 16; s++) for (const j of [1, 5, 6]) n += normal[s * N_TRACKS + j] > 0; return n; })();
        let best = null;
        for (let k = 0; k < 8; k++) {
          net.restore(snapMid);
          const buf = new Uint8Array(steps * N_TRACKS);
          sampleRange(buf, s0 + 8, s1, prevMid, { cond: condFill, maxDist: 8, temp: temp * 1.15 });
          let score = 0;
          for (let s = s0 + 8; s < s1; s++) for (const j of [1, 5, 6]) score += buf[s * N_TRACKS + j] > 0;
          if (score >= 3 && score >= baseline + 2 && (!best || score > best.score)) best = { buf, score };
        }
        net.restore(snapMid);
        if (best) {
          out.set(best.buf.subarray((s0 + 8) * N_TRACKS, s1 * N_TRACKS), (s0 + 8) * N_TRACKS);
          fillBars.push(bar);
          crashNext = 3; // the downbeat after a fill: nudge the network toward a crash
        }
        // rebuild the network's memory from the normal groove, so the fill isn't copied by later bars
        const fillPart = out.slice((s0 + 8) * N_TRACKS, s1 * N_TRACKS);
        out.set(normal.subarray(8 * N_TRACKS), (s0 + 8) * N_TRACKS);
        prev = feedRange(out, s0 + 8, s1, prevMid);
        out.set(fillPart, (s0 + 8) * N_TRACKS);
      }
      if (o.onProgress) { o.onProgress((bar + 1) / bars); if (bar % 2 === 1) await tickCh(); }
    }

    // classes -> velocities
    const pat = {};
    ids.forEach((id, j) => {
      pat[id] = new Float32Array(steps);
      for (let s = 0; s < steps; s++) { const c = out[s * N_TRACKS + j]; if (c > 0) pat[id][s] = DF.clamp(VEL[c] * (1 - hum * 0.25 * rng()), 0.08, 1); }
    });
    return { pattern: pat, styleId, fillBars };
  };
})(typeof window !== 'undefined' ? (window.DF = window.DF || {}) : (globalThis.DF = globalThis.DF || {}));
