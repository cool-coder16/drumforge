// Checks that js/ai.js reproduces the PyTorch model's logits.
//   node parity_test.js runs/main ../models
const fs = require('fs');
globalThis.window = globalThis;
globalThis.DF = {};
const root = __dirname + '/../js/';
for (const f of ['rng.js', 'synth.js']) { try { new Function('window', fs.readFileSync(root + f, 'utf8'))(globalThis); } catch (e) { /* synth needs audio; not required here */ } }
// ai.js only needs DF.makeRng / DF.TRACKS / DF.clamp for generation, not for the parity forward pass
new Function(fs.readFileSync(root + 'ai.js', 'utf8'))();

const [run, models] = [process.argv[2], process.argv[3]];
const cfg = JSON.parse(fs.readFileSync(models + '/drummer.json', 'utf8'));
const bin = fs.readFileSync(models + '/drummer.bin');
const net = new DF.AI.Net(cfg, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const ref = JSON.parse(fs.readFileSync(run + '/parity.json', 'utf8'));

const cond = net.cond(ref.style, ref.dens, ref.tempo);
let maxErr = 0, n = 0, agree = 0;
const t0 = Date.now();
for (let t = 0; t < ref.T; t++) {
  const h = net.step(Uint8Array.from(ref.prev[t]), Float32Array.from(ref.feat[t]), t % 16, Math.floor(t / 16) % 8, cond);
  const tgt = Uint8Array.from(ref.target[t]);
  for (let j = 0; j < 9; j++) {
    const lg = net.head(h, j, tgt);
    let best = 0, rbest = 0;
    for (let c = 0; c < 4; c++) {
      maxErr = Math.max(maxErr, Math.abs(lg[c] - ref.logits[t][j][c]));
      if (lg[c] > lg[best]) best = c;
      if (ref.logits[t][j][c] > ref.logits[t][j][rbest]) rbest = c;
    }
    agree += best === rbest; n++;
  }
}
console.log(`max |logit error| = ${maxErr.toExponential(2)}   argmax agreement ${agree}/${n}   (${Date.now() - t0} ms for ${ref.T} steps)`);
process.exit(maxErr < 0.02 ? 0 : 1);
