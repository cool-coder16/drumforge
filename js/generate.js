// The groove engine. Given a style, some knobs and (optionally) features from
// an analysed song, it writes a drum pattern.
//
// Core idea: every style is a table of PROBABILITIES for each of the 16 sixteenth
// steps in a bar ("how likely is a kick here?"). For each step we roll a die and
// compare. The dice are rolled ONCE per bar position (not per bar), so bars repeat
// like a real groove, with a little jitter so they breathe. When a song is
// analysed, probabilities are pushed up wherever the song actually has a hit.
(function (DF) {
  // "9000 0090" -> [1, 0, 0, 0, 0, 0, 1, 0]. Digit = probability in ninths (9 = always).
  const P = (s) => s.replace(/\s+/g, '').split('').map((ch) => parseInt(ch, 10) / 9);
  const ZERO = '0000 0000 0000 0000';
  const EIGHTHS = '9090 9090 9090 9090';
  const SIXTEENTHS = '9999 9999 9999 9999';

  function style(label, bpm, swing, spec) {
    const base = { kick: ZERO, snare: ZERO, ghost: ZERO, clap: ZERO, hat: EIGHTHS, hat16: SIXTEENTHS, ohat: ZERO, perc: ZERO };
    const o = { label, bpm, swing };
    for (const k in base) o[k] = P(spec[k] || base[k]);
    return o;
  }

  DF.STYLES = {
    rock: style('Rock', [95, 140], 0, {
      kick: '9000 0020 6030 0040', snare: '0000 9000 0000 9000', ghost: '0010 0020 0010 0030', ohat: '0000 0000 0000 0010',
    }),
    pop: style('Pop', [95, 125], 0, {
      kick: '9000 0030 4030 0030', snare: '0000 9000 0000 9000', clap: '0000 5000 0000 5000', ghost: '0000 0010 0000 0020', ohat: '0000 0000 0000 0020',
    }),
    house: style('House', [118, 130], 0, {
      kick: '9000 9000 9000 9000', clap: '0000 9000 0000 9000', hat: '5959 5959 5959 5959', hat16: '7979 7979 7979 7979',
      ohat: '0090 0090 0090 0090', perc: '0002 0000 0002 0000',
    }),
    techno: style('Techno', [126, 140], 0, {
      kick: '9000 9000 9000 9000', clap: '0000 7000 0000 7000', hat: '0090 0090 0090 0090', hat16: '9999 9999 9999 9999',
      ohat: '0020 0020 0020 0020', perc: '0003 0000 0003 0000',
    }),
    hiphop: style('Boom Bap', [78, 98], 0.3, {
      kick: '9000 0030 0070 0030', snare: '0000 9000 0000 9000', ghost: '0000 0000 0020 0020', hat16: '9595 9595 9595 9595', ohat: '0000 0000 0000 0020',
    }),
    trap: style('Trap', [130, 160], 0, {
      kick: '9000 0030 0060 0040', snare: '0000 0000 9000 0000', clap: '0000 0000 9000 0000', hat: SIXTEENTHS,
      ohat: '0000 0000 0000 0030', perc: '0003 0000 0003 0000',
    }),
    dnb: style('Drum & Bass', [160, 176], 0, {
      kick: '9000 0000 0090 0000', snare: '0000 9000 0000 9000', ghost: '0000 0020 0020 0020', hat: EIGHTHS, hat16: '9595 9595 9595 9595',
    }),
    funk: style('Funk', [92, 112], 0.2, {
      kick: '9000 0040 0060 3050', snare: '0000 9000 0000 9000', ghost: '0030 0030 3040 0030', hat: SIXTEENTHS,
      ohat: '0000 0000 0000 0050',
    }),
    breakbeat: style('Breakbeat', [120, 140], 0.1, {
      kick: '9020 0000 6090 0000', snare: '0000 9000 0000 9000', ghost: '0020 0030 0020 4030', hat16: '9595 9595 9595 9595',
    }),
    reggae: style('One Drop', [68, 88], 0, {
      kick: '0000 0000 9000 0000', snare: '0000 0000 9000 0000', perc: '0000 5000 0000 5000', ghost: '0000 0000 0000 0000',
    }),
  };
  DF.STYLE_LIST = Object.keys(DF.STYLES);
  DF.STYLE_LABELS = { euclid: 'Euclidean (weird)' };
  DF.STYLE_LIST.forEach((id) => (DF.STYLE_LABELS[id] = DF.STYLES[id].label));

  // Euclidean rhythms spread k hits as evenly as possible over 16 steps.
  function makeEuclid(rng, complexity) {
    const eu = (k, rot) => {
      const a = [];
      for (let i = 0; i < 16; i++) a.push((((i - rot + 16) % 16) * k) % 16 < k ? 1 : 0);
      return a;
    };
    const z = new Array(16).fill(0);
    return {
      label: 'Euclidean', bpm: [90, 150], swing: 0,
      kick: eu(rng.int(3, 6), rng.pick([0, 0, 2])),
      snare: eu(rng.int(2, 3), rng.pick([4, 4, 2, 6])),
      ghost: z, clap: eu(rng.int(1, 3), rng.int(0, 15)),
      hat: eu(5 + Math.round(complexity * 8), rng.int(0, 3)),
      hat16: eu(9 + Math.round(complexity * 5), rng.int(0, 3)),
      ohat: eu(rng.int(1, 3), rng.int(0, 15)), perc: eu(rng.int(3, 7), rng.int(0, 15)),
    };
  }

  // Average the kick/snare/hat energy per bar position over the loud bars.
  // This is the song's own "groove fingerprint".
  function deriveGroove(f) {
    const kick = new Float32Array(16), snare = new Float32Array(16), hat = new Float32Array(16);
    let n = 0;
    for (let b = 0; b < f.bars; b++) {
      if (f.E[b] < 0.35) continue;
      for (let p = 0; p < 16; p++) {
        kick[p] += f.L[b * 16 + p]; snare[p] += f.M[b * 16 + p]; hat[p] += f.H[b * 16 + p];
      }
      n++;
    }
    for (const arr of [kick, snare, hat]) {
      const mx = Math.max(...arr) || 1;
      for (let p = 0; p < 16; p++) arr[p] = n ? arr[p] / mx : 0;
    }
    return { kick: Array.from(kick), snare: Array.from(snare), hat: Array.from(hat) };
  }

  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na * nb) || 1);
  }

  // Which built-in style does the song's groove fingerprint most resemble?
  function matchStyle(g) {
    const dense = g.hat.reduce((x, y) => x + y, 0) / 16 > 0.6;
    const vec = (kick, snare, hat) => kick.concat(snare, hat);
    const target = vec(g.kick, g.snare, g.hat);
    let best = 'rock', bestSim = -1;
    DF.STYLE_LIST.forEach((id) => {
      const t = DF.STYLES[id];
      const sn = t.snare.map((v, i) => Math.max(v, t.ghost[i] * 0.3));
      const sim = cosine(target, vec(t.kick, sn, dense ? t.hat16 : t.hat));
      if (sim > bestSim) { best = id; bestSim = sim; }
    });
    return best;
  }

  // A fill replaces the end of a bar. Each entry is [step, track, velocity].
  const FILLS = [
    () => { // snare roll that builds over the last two beats
      const out = [];
      for (let p = 8; p < 16; p++) if (p >= 12 || p % 2 === 0) out.push([p, 'snare', 0.35 + 0.65 * ((p - 8) / 7)]);
      return { from: 8, hits: out };
    },
    () => ({ from: 10, hits: [[10, 'tomH', 0.85], [11, 'tomH', 0.7], [12, 'tomL', 0.9], [13, 'tomL', 0.75], [14, 'snare', 0.8], [15, 'snare', 1]] }),
    () => ({ from: 10, hits: [[10, 'snare', 0.7], [11, 'kick', 0.9], [13, 'snare', 0.8], [14, 'kick', 0.8], [15, 'snare', 0.95]] }),
    () => ({ from: 12, hits: [[12, 'tomH', 0.9], [13, 'tomH', 0.8], [14, 'tomL', 0.9], [15, 'tomL', 1]] }),
    () => ({ from: 8, hits: [[12, 'kick', 0.9], [14, 'clap', 0.8], [15, 'snare', 0.9]] }),
  ];

  // Main entry point.
  //   o = { bars, style, complexity, ghost, fills, humanize, seed, feats?, sync? }
  // Returns { pattern: {trackId: Float32Array(bars*16)}, styleId, label }
  DF.generate = function (o) {
    const rng = DF.makeRng(DF.hashString(String(o.seed)));
    const bars = o.bars, steps = bars * 16;
    const feats = o.feats || null;
    const sync = feats ? o.sync : 0;
    const c = o.complexity, hum = o.humanize == null ? 0.3 : o.humanize;
    const pat = {};
    DF.TRACKS.forEach((t) => (pat[t.id] = new Float32Array(steps)));

    // ---- 1. choose the style template ----
    let styleId = o.style;
    if (styleId === 'surprise') styleId = rng.pick(DF.STYLE_LIST.concat(['euclid']));
    let derived = null;
    if (feats) derived = deriveGroove(feats);
    if (styleId === 'auto') styleId = derived ? matchStyle(derived) : rng.pick(DF.STYLE_LIST);
    let tpl = styleId === 'euclid' ? makeEuclid(rng, c) : DF.STYLES[styleId];
    if (o.style === 'auto' && derived) {
      // Blend the matched style's kick with the song's own kick pattern.
      tpl = Object.assign({}, tpl, {
        kick: tpl.kick.map((v, i) => (v >= 0.95 ? v : 0.6 * v + 0.4 * derived.kick[i] * 0.9)),
      });
    }

    // ---- 2. dice rolled once per bar position, so bars repeat like a groove ----
    const roll = {};
    DF.TRACKS.forEach((t) => (roll[t.id] = Array.from({ length: 16 }, () => rng())));
    const offScale = 0.25 + 1.5 * c; // complexity scales the "optional" notes
    const prob = (pr) => (pr >= 0.95 ? 1 : Math.min(0.97, pr * offScale));
    const ghostScale = 0.15 + 2.2 * o.ghost;
    const jitterAmt = 0.14 + 0.1 * c;

    // Typical hat density per bar of the song (to choose 8ths vs 16ths).
    let hMed = 0;
    const hDens = new Float32Array(bars);
    if (feats) {
      for (let b = 0; b < bars; b++) {
        let s = 0;
        for (let p = 0; p < 16; p++) s += feats.H[b * 16 + p];
        hDens[b] = s / 16;
      }
      const sorted = Array.from(hDens).filter((_, b) => feats.E[b] > 0.2).sort((a, b) => a - b);
      hMed = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
    }
    const hHi = feats ? Math.max(0.2, ...Array.from(hDens)) : 1; // reference "very hat-heavy" level

    const vel = (base) => DF.clamp(base * (1 - hum * 0.28 * rng()), 0.08, 1);
    let prevWasFill = false;

    // ---- 3. bar by bar ----
    for (let bar = 0; bar < bars; bar++) {
      const base = bar * 16;
      const E = feats ? feats.E[bar] : 1;
      const wasFill = prevWasFill;
      prevWasFill = false;
      if (E < 0.04) continue;                                    // silence: no drums
      const gate = E < 0.12 ? (E - 0.04) / 0.08 : 1;            // fade drums in gently
      const lvl = 0.5 + 0.5 * E;                                 // quiet bars play fewer optional notes
      const jit = () => (rng() - 0.5) * jitterAmt;
      const hat16 = feats
        ? (sync > 0.3 ? hDens[bar] > hMed * 1.15 && hDens[bar] > 0.25 : rng() < 0.5)
        : false;
      const hatDensity = feats ? DF.clamp(0.3 + 1.4 * (hDens[bar] / hHi), 0.25, 1) : 1;
      const hatArr = hat16 || (!feats && c > 0.75 && bar % 2 === 1) ? tpl.hat16 : tpl.hat;

      for (let p = 0; p < 16; p++) {
        const s = base + p;
        const backbeat = p === 4 || p === 12;

        // KICK
        let pk = prob(tpl.kick[p]);
        let ob = 0;
        if (feats) {
          ob = DF.sstep(feats.L[s], 0.4, 0.8);
          // A real kick/bass note is mostly low-band energy. A snare also leaks some low end,
          // but it sits under strong mid/high energy, so weight by how much the low band dominates.
          const dom = feats.L[s] / (feats.L[s] + 0.6 * feats.M[s] + 0.4 * feats.H[s] + 0.05);
          ob *= DF.sstep(dom, 0.3, 0.55);
          if (tpl.snare[p] >= 0.55) ob *= 0.5;
          pk = DF.clamp(pk * (1 - 0.5 * sync) + ob * sync, 0, 1);
        }
        pk *= gate * (tpl.kick[p] >= 0.95 ? 1 : lvl);
        // (pk > 0 guard: the jitter can push roll+jit below 0, which must never fire a 0% note)
        if (pk >= 0.95 || (pk > 0 && pk > roll.kick[p] + jit())) pat.kick[s] = vel((p % 4 === 0 ? 0.92 : 0.78) * (feats ? 0.85 + 0.15 * ob : 1));

        // SNARE (main hit + quiet ghost notes)
        if (tpl.snare[p] >= 0.55) {
          if (rng() < gate) pat.snare[s] = vel(0.95);
        } else {
          let pg = tpl.ghost[p] * ghostScale + tpl.snare[p] * offScale;
          if (feats) pg = pg * (1 - 0.4 * sync) + sync * 0.5 * DF.sstep(feats.M[s], 0.5, 0.9) * (backbeat ? 0 : 1);
          pg = Math.min(0.9, pg) * gate * lvl;
          if (pg > 0 && pg > roll.snare[p] + jit()) pat.snare[s] = DF.clamp(0.26 + 0.14 * rng(), 0.1, 0.5);
        }

        // CLAP
        const pc = prob(tpl.clap[p]) * gate;
        if (pc > 0 && (pc >= 0.95 || pc > roll.clap[p] + jit())) pat.clap[s] = vel(0.85);

        // PERC
        const pp = prob(tpl.perc[p]) * gate * lvl;
        if (pp > 0 && (pp >= 0.95 || pp > roll.perc[p] + jit())) pat.perc[s] = vel(0.6);

        // OPEN HAT
        const po = prob(tpl.ohat[p]) * gate * lvl;
        if (po > 0 && (po >= 0.95 || po > roll.ohat[p] + jit())) pat.ohat[s] = vel(0.75);

        // CLOSED HAT: grid from the template, thinned out where the song has few hats.
        let ph = hatArr[p];
        if (ph > 0) {
          ph = ph >= 0.95 ? ph : Math.min(0.97, ph * (0.6 + 0.8 * c));
          ph *= feats ? 1 - sync + sync * hatDensity : 1;
          ph *= gate;
          if ((ph >= 0.95 || ph > roll.hat[p] + jit()) && !pat.ohat[s]) {
            const accent = p % 4 === 0 ? 0.9 : p % 2 === 0 ? 0.7 : 0.45;
            const heard = feats ? 0.75 + 0.25 * DF.sstep(feats.H[s], 0.2, 0.8) : 1;
            pat.hat[s] = vel(accent * heard);
          }
        }
      }

      // CRASH: start of a loop, after a fill, or when the song suddenly gets louder
      const jump = feats && bar > 0 ? E - feats.E[bar - 1] : 0;
      if (o.fills > 0.05 && E > 0.3) {
        if ((bar === 0 && !feats && rng() < 0.5) || (wasFill && rng() < 0.75) || (jump > 0.22 && rng() < 0.85)) {
          pat.crash[base] = vel(0.9);
          if (!pat.kick[base]) pat.kick[base] = vel(0.9);
        }
      }

      // FILL: at the end of a phrase (every 4 bars; stronger every 8)
      const phraseEnd = bars >= 4 ? bar % 4 === 3 : bars > 1 && bar === bars - 1;
      const buildUp = feats && bar + 1 < bars && feats.E[bar + 1] - E > 0.22;
      if (E > 0.25 && o.fills > 0 && (phraseEnd || buildUp)) {
        const strong = bar % 8 === 7 || bar === bars - 1; // end of an 8-bar phrase (or of the loop)
        const pf = buildUp ? Math.max(0.8, o.fills) : DF.clamp(o.fills * (strong ? 1.4 : 0.8), 0, 1);
        if (rng() < pf) {
          const fill = rng.pick(FILLS)();
          for (let p = fill.from; p < 16; p++) {
            ['kick', 'snare', 'clap', 'hat', 'ohat', 'perc', 'tomH', 'tomL'].forEach((id) => (pat[id][base + p] = 0));
          }
          fill.hits.forEach(([p, id, v]) => (pat[id][base + p] = vel(v)));
          prevWasFill = true;
        }
      }
    }
    return { pattern: pat, styleId, label: tpl.label, swing: tpl.swing, bpmRange: tpl.bpm };
  };

  // Randomly flip a few non-essential notes, keeping the groove's skeleton.
  DF.mutate = function (pattern, steps, rng, amount) {
    const choices = [['hat', 0.3, 0.5], ['kick', 0.22, 0.7], ['snare', 0.2, 0.3], ['perc', 0.1, 0.5], ['ohat', 0.1, 0.7], ['clap', 0.05, 0.7], ['tomH', 0.03, 0.6]];
    const n = Math.max(2, Math.ceil(steps * amount * 0.25));
    for (let i = 0; i < n; i++) {
      const s = rng.int(0, steps - 1), p = s % 16;
      let r = rng(), pick = choices[0];
      for (const ch of choices) { if (r < ch[1]) { pick = ch; break; } r -= ch[1]; }
      const [id, , v] = pick;
      if (id === 'kick' && p === 0) continue;                       // keep beat 1
      if (id === 'snare' && (p === 4 || p === 12) && pattern.snare[s] > 0.6) continue; // keep the backbeat
      pattern[id][s] = pattern[id][s] > 0 ? 0 : v * (0.8 + 0.2 * rng());
    }
  };
})((window.DF = window.DF || {}));
