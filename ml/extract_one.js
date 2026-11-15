// Run the REAL browser analyser (js/analyze.js) on rendered audio and print the 16th-note features.
//   node extract_one.js <audio.f32> <bpm> <nudgeSeconds> <bars>
// The grid is built from the known tempo of the excerpt (bar 1 at t=0), exactly like buildGrid does
// in the app after tempo detection, so the features are what the browser would hand the neural net.
const fs = require('fs');
globalThis.window = globalThis;
globalThis.DF = {};
const root = __dirname + '/../js/';
for (const f of ['rng.js', 'analyze.js']) new Function('window', fs.readFileSync(root + f, 'utf8'))(globalThis);

(async () => {
  const [file, bpm, nudge, bars] = [process.argv[2], +process.argv[3], +process.argv[4], +process.argv[5]];
  const raw = fs.readFileSync(file);
  const data = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const buffer = { sampleRate: 44100, length: data.length, numberOfChannels: 1, getChannelData: () => data };
  const an = await DF.analyzeAudio(buffer);
  const g = DF.buildGrid(an, bpm, 0, 0, nudge);
  // buildGrid puts bar 1 at the earliest bar start >= -half a step; with phase 0 that is t=0 (+nudge)
  const n = bars * 16, r3 = (a) => Array.from(a.slice(0, n), (v) => Math.round(v * 1000) / 1000);
  // the analyser's MessageChannel keeps node alive, so exit explicitly once the output is flushed
  process.stdout.write(JSON.stringify({ L: r3(g.L), M: r3(g.M), H: r3(g.H), E: r3(g.E.slice(0, bars)), T0: g.T0, detected: an.bpm }), () => process.exit(0));
})();
