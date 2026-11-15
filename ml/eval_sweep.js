// Sweep sampling temperature and feature gain ("follow") for the neural drummer on held-out test excerpts.
//   node eval_sweep.js ../models [N]
const fs = require('fs');
globalThis.window = globalThis; globalThis.DF = {};
for (const f of ['rng.js', 'synth.js', 'ai.js']) new Function('window', fs.readFileSync(__dirname + '/../js/' + f, 'utf8'))(globalThis);
(async () => {
  const models = process.argv[2], N = +(process.argv[3] || 100);
  const cfg = JSON.parse(fs.readFileSync(models + '/drummer.json', 'utf8')), bin = fs.readFileSync(models + '/drummer.bin');
  DF.AI.cfg = cfg; DF.AI.net = new DF.AI.Net(cfg, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  const all = fs.readFileSync(__dirname + '/data/audio_test.jsonl', 'utf8').trim().split('\n').map((l) => JSON.parse(l)).slice(0, N);
  const ids = DF.TRACKS.map((t) => t.id);
  const toFeats = (r) => ({ L: Float32Array.from(r.L), M: Float32Array.from(r.M), H: Float32Array.from(r.H), E: Float32Array.from(r.E), bars: r.E.length, steps: r.E.length * 16 });
  const configs = [[0.9, 1.0], [0.9, 0.5], [0.7, 1.0], [0.7, 0.5], [0.5, 1.0], [0.5, 0.5], [0.7, 0]];
  for (const [T, follow] of configs) {
    const res = {};
    for (const variant of ['nd', 'wd']) {
      let tp = 0, fp = 0, fn = 0, ktp = 0, kfp = 0, kfn = 0, stp = 0, sfp = 0, sfn = 0;
      const set = all.filter((r) => r.variant === variant);
      for (let i = 0; i < set.length; i++) {
        const e = set[i];
        const r = await DF.AI.generate({ bars: e.E.length, bpm: e.bpm, seed: 'e' + i, temperature: T, feats: toFeats(e), follow });
        for (let s = 0; s < e.E.length * 16; s++) for (let j = 0; j < 9; j++) {
          const p = r.pattern[ids[j]][s] > 0, t = e.D[s][j] > 0;
          if (p && t) tp++; else if (p) fp++; else if (t) fn++;
        }
      }
      res[variant] = (2 * tp / Math.max(1, 2 * tp + fp + fn)).toFixed(3);
    }
    console.log(`T=${T} follow=${follow}   no-drums F1 ${res.nd}   with-drums F1 ${res.wd}`);
  }
  process.exit(0);
})();
