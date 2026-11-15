// Head-to-head on held-out TEST songs (artists the model never saw). Features are what the browser's real
// analyser measured on rendered audio. Each system sees only the song and must reproduce the part the song's
// real drummer played.        node eval_compare.js ../models [N] [temperature]
const fs = require('fs');
globalThis.window = globalThis;
globalThis.DF = {};
const root = __dirname + '/../js/';
for (const f of ['rng.js', 'synth.js', 'generate.js', 'ai.js']) new Function('window', fs.readFileSync(root + f, 'utf8'))(globalThis);

(async () => {
  const models = process.argv[2], N = +(process.argv[3] || 100), TEMP = +(process.argv[4] || 0.7);
  const cfg = JSON.parse(fs.readFileSync(models + '/drummer.json', 'utf8'));
  DF.AI.hd = JSON.parse(fs.readFileSync(models + '/has_drums.json', 'utf8'));
  const bin = fs.readFileSync(models + '/drummer.bin');
  DF.AI.cfg = cfg;
  DF.AI.net = new DF.AI.Net(cfg, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  const all = fs.readFileSync(__dirname + '/data/audio_test.jsonl', 'utf8').trim().split('\n').map((l) => JSON.parse(l)).slice(0, N);
  const ids = DF.TRACKS.map((t) => t.id);
  const toFeats = (r) => ({ L: Float32Array.from(r.L), M: Float32Array.from(r.M), H: Float32Array.from(r.H), E: Float32Array.from(r.E), bars: r.E.length, steps: r.E.length * 16 });

  const naive = (r) => {
    const n = r.E.length * 16, p = {}; ids.forEach((id) => (p[id] = new Float32Array(n)));
    for (let s = 0; s < n; s++) { const q = s % 16; if (q === 0 || q === 8) p.kick[s] = 1; if (q === 4 || q === 12) p.snare[s] = 1; if (q % 2 === 0) p.hat[s] = 1; }
    return { pattern: p };
  };
  const systems = {
    'naive fixed rock beat (ignores the song)': async (r) => naive(r),
    'classic rules engine': async (r, i) => DF.generate({ bars: r.E.length, style: 'auto', complexity: 0.5, ghost: 0.35, fills: 0.55, humanize: 0.3, sync: 0.7, seed: 'e' + i, feats: toFeats(r) }),
    'neural drummer, audio switched OFF': async (r, i) => DF.AI.generate({ bars: r.E.length, bpm: r.bpm, seed: 'e' + i, temperature: TEMP, follow: 0 }),
    'neural drummer, always listening': async (r, i) => DF.AI.generate({ bars: r.E.length, bpm: r.bpm, seed: 'e' + i, temperature: TEMP, feats: toFeats(r), follow: 1 }),
    // what the app does: first decide whether the song has drums; if not, don't trust the accompaniment's onsets
    'neural drummer, adaptive (the app)': async (r, i) => {
      const f = toFeats(r), p = DF.AI.detectDrums(f);
      return DF.AI.generate({ bars: r.E.length, bpm: r.bpm, seed: 'e' + i, temperature: TEMP, feats: f, follow: 1, drumless: p !== null && p < 0.5 });
    },
  };

  const results = {};
  for (const variant of ['nd', 'wd']) {
    const set = all.filter((r) => r.variant === variant);
    results[variant] = { n: set.length };
    for (const [name, fn] of Object.entries(systems)) {
      const tp = {}, fp = {}, fn_ = {};
      ids.forEach((id) => (tp[id] = fp[id] = fn_[id] = 0));
      const dx = [], dy = [];
      for (let i = 0; i < set.length; i++) {
        const e = set[i], r = await fn(e, i), bars = e.E.length;
        const gx = [], gy = [];
        for (let b = 0; b < bars; b++) { let a = 0, c = 0; for (let q = 0; q < 16; q++) for (let j = 0; j < 9; j++) { a += r.pattern[ids[j]][b * 16 + q] > 0; c += e.D[b * 16 + q][j] > 0; } gx.push(a); gy.push(c); }
        const mx = gx.reduce((a, b) => a + b) / bars, my = gy.reduce((a, b) => a + b) / bars;
        for (let b = 0; b < bars; b++) { dx.push(gx[b] - mx); dy.push(gy[b] - my); }
        for (let s = 0; s < bars * 16; s++) for (let j = 0; j < 9; j++) {
          const pred = r.pattern[ids[j]][s] > 0, truth = e.D[s][j] > 0;
          if (pred && truth) tp[ids[j]]++; else if (pred) fp[ids[j]]++; else if (truth) fn_[ids[j]]++;
        }
      }
      const f1 = (ks) => { const a = ks.reduce((x, k) => x + tp[k], 0), b = ks.reduce((x, k) => x + fp[k], 0), c = ks.reduce((x, k) => x + fn_[k], 0); return (2 * a) / Math.max(1, 2 * a + b + c); };
      let sxy = 0, sxx = 0, syy = 0;
      dx.forEach((x, k) => { sxy += x * dy[k]; sxx += x * x; syy += dy[k] * dy[k]; });
      results[variant][name] = { kick: f1(['kick']), snare: f1(['snare']), hat: f1(['hat', 'ohat']), all: f1(ids), barDensityCorr: sxy / Math.sqrt(Math.max(1e-9, sxx * syy)) };
      console.log(`[${variant === 'nd' ? 'accompaniment only' : 'mix incl. drums  '}, n=${set.length}] ${name.padEnd(42)}`, Object.entries(results[variant][name]).map(([k, v]) => `${k} ${v.toFixed(3)}`).join('  '));
    }
  }
  fs.writeFileSync(__dirname + '/eval_results.json', JSON.stringify(results, null, 1));
  process.exit(0);
})();
