// Audio analysis: turns a decoded song into a tempo, a beat grid and, for
// every 16th-note step, "how much kick / snare / hat energy starts here".
//
// Pipeline (all plain JS, runs in the browser):
//   1. STFT (windowed FFT) over the whole song
//   2. "Spectral flux" per frequency band = how much energy was ADDED since
//      the last frame. Peaks in flux are note onsets.
//        low  30-160 Hz   -> where kicks / bass hits land
//        mid  200-4k Hz   -> snare body / claps / vocals
//        high 5k-16k Hz   -> hi-hats / cymbals
//   3. Autocorrelation -> rough tempo, then a fine "comb filter" search over
//      tempo + phase to lock the beat grid to the song.
//   4. Sample the flux on the 16th-note grid -> features the generator uses.
(function (DF) {
  const N = 2048;          // FFT window (samples)
  const HOP = 512;         // samples between windows
  const MAX_SECONDS = 600; // analyse at most 10 minutes
  // A flux peak shows up a bit after the real onset because the window is
  // centred ahead of it. This constant (in samples) shifts frame times back so
  // detected beats line up with the audio. Calibrated with the self-test.
  const LAT = 845;

  function makeFFT(n) {
    const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2), rev = new Uint32Array(n);
    const bits = Math.log2(n) | 0;
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos((2 * Math.PI * i) / n);
      sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      rev[i] = r;
    }
    // In-place iterative radix-2 FFT (Cooley-Tukey).
    return function (re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) {
          let t = re[i]; re[i] = re[j]; re[j] = t;
          t = im[i]; im[i] = im[j]; im[j] = t;
        }
      }
      for (let size = 2; size <= n; size *= 2) {
        const half = size / 2, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const l = j + half;
            const tre = re[l] * cos[k] + im[l] * sin[k];
            const tim = -re[l] * sin[k] + im[l] * cos[k];
            re[l] = re[j] - tre; im[l] = im[j] - tim;
            re[j] += tre; im[j] += tim;
          }
        }
      }
    };
  }

  // Yield to the browser so the progress bar can repaint. A MessageChannel is used
  // instead of setTimeout because background tabs throttle setTimeout to ~1/second.
  const tick = (() => {
    const ch = new MessageChannel(), waiting = [];
    ch.port1.onmessage = () => { const r = waiting.shift(); if (r) r(); };
    return () => new Promise((r) => { waiting.push(r); ch.port2.postMessage(0); });
  })();

  function percentile(arr, p) {
    const c = Float32Array.from(arr).sort();
    return c[Math.min(c.length - 1, Math.floor(p * (c.length - 1)))] || 0;
  }

  // Subtract a local average so only sudden rises (onsets) remain.
  function detrend(x, fps) {
    const n = x.length, w = Math.max(2, Math.round(fps * 0.35));
    const cs = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) cs[i + 1] = cs[i] + x[i];
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - w), b = Math.min(n, i + w + 1);
      out[i] = Math.max(0, x[i] - (cs[b] - cs[a]) / (b - a));
    }
    return out;
  }

  function smooth3(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      out[i] = 0.25 * (x[i - 1] || 0) + 0.5 * x[i] + 0.25 * (x[i + 1] || 0);
    }
    return out;
  }

  // Average onset strength at beat positions phase, phase+P, phase+2P ... (frames).
  function combScore(env, period, phase) {
    let s = 0, c = 0;
    const n = env.length - 1;
    for (let x = phase; x < n; x += period) {
      const i = x | 0, f = x - i;
      s += env[i] * (1 - f) + env[i + 1] * f;
      c++;
    }
    return c ? s / c : 0;
  }

  // Rough tempo via autocorrelation (how well does the onset curve match a
  // copy of itself shifted by one beat?), favouring tempos near 120 BPM.
  function estimatePeriod(env, fps) {
    const lagLo = (fps * 60) / 180, lagHi = (fps * 60) / 70;
    const maxLag = Math.ceil(lagHi * 4) + 2, n = env.length;
    const ac = new Float32Array(maxLag + 2);
    for (let l = 1; l <= maxLag; l++) {
      let s = 0;
      for (let i = 0; i + l < n; i++) s += env[i] * env[i + l];
      ac[l] = s / (n - l);
    }
    const acAt = (L) => {
      const i = Math.floor(L);
      if (i + 1 > maxLag) return 0;
      const f = L - i;
      return ac[i] * (1 - f) + ac[i + 1] * f;
    };
    let bestL = lagLo, best = -1;
    for (let L = lagLo; L <= lagHi; L += 0.1) {
      const bpm = (fps * 60) / L;
      const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120) / 0.8, 2));
      // Also count the bar-length / 2-bar lags: a real beat repeats at 2x, 3x, 4x.
      const sc = (acAt(L) + 0.6 * acAt(2 * L) + 0.35 * acAt(3 * L) + 0.3 * acAt(4 * L)) * (0.35 + 0.65 * prior);
      if (sc > best) { best = sc; bestL = L; }
    }
    return bestL;
  }

  // Beat score = onset strength ON the beats minus a share of the strength halfway
  // between them. This stops the grid locking onto the off-beats of syncopated music.
  function beatScore(env, period, phase) {
    return combScore(env, period, phase) - 0.35 * combScore(env, period, phase + period / 2);
  }

  function bestPhase(envS, period) {
    let best = -1e9, bestPh = 0;
    for (let ph = 0; ph < period; ph += 0.25) {
      const s = beatScore(envS, period, ph);
      if (s > best) { best = s; bestPh = ph; }
    }
    return bestPh;
  }

  const frameToSec = (f, sr) => (f * HOP + LAT) / sr;
  const secToFrame = (t, sr) => (t * sr - LAT) / HOP;

  // Time (seconds) of some beat, expressed within [0, beatLength).
  DF.findPhase = function (an, bpm) {
    const periodF = (an.fps * 60) / bpm;
    const ph = bestPhase(an.envS, periodF);
    const beat = 60 / bpm;
    return ((frameToSec(ph, an.sr) % beat) + beat) % beat;
  };

  // Which of the 4 beats in a bar is "beat 1"? Kick on 1 & 3, snare on 2 & 4.
  DF.pickDownbeat = function (an, bpm, phaseSec) {
    const beat = 60 / bpm, nBeats = Math.floor((an.duration - phaseSec) / beat);
    const at = (env, t) => {
      const f0 = Math.max(0, Math.floor(secToFrame(t - beat * 0.12, an.sr)));
      const f1 = Math.min(env.length - 1, Math.ceil(secToFrame(t + beat * 0.12, an.sr)));
      let m = 0;
      for (let f = f0; f <= f1; f++) if (env[f] > m) m = env[f];
      return m;
    };
    const low = new Float32Array(4), mid = new Float32Array(4), tot = new Float32Array(4), cnt = new Float32Array(4);
    for (let b = 0; b < nBeats; b++) {
      const t = phaseSec + b * beat;
      low[b % 4] += at(an.env.low, t); mid[b % 4] += at(an.env.mid, t); tot[b % 4] += at(an.env.all, t); cnt[b % 4]++;
    }
    let best = -1e9, bestS = 0;
    for (let s = 0; s < 4; s++) {
      const avg = (arr, k) => arr[(s + k) % 4] / Math.max(1, cnt[(s + k) % 4]);
      const l = (k) => avg(low, k), m = (k) => avg(mid, k);
      // Beat 1 tends to have the strongest kick/bass, snares on 2 & 4, and the biggest overall onset
      // (chord changes, crashes and new bass notes usually land on the downbeat).
      const score = l(0) + 0.5 * l(2) + 0.4 * (m(1) + m(3)) + 0.5 * avg(tot, 0) - 0.2 * (l(1) + l(3));
      if (score > best) { best = score; bestS = s; }
    }
    return bestS;
  };

  DF.analyzeAudio = async function (buffer, onProgress) {
    const sr = buffer.sampleRate;
    const len = Math.min(buffer.length, Math.floor(sr * MAX_SECONDS));
    if (len < N * 8) throw new Error('That audio is too short to analyse.');
    const nch = buffer.numberOfChannels;
    const mono = new Float32Array(len); // mix every channel down to mono
    for (let ch = 0; ch < nch; ch++) {
      const d = buffer.getChannelData(ch);
      for (let i = 0; i < len; i++) mono[i] += d[i] / nch;
    }
    const fps = sr / HOP;
    const frames = Math.floor((len - N) / HOP) + 1;
    const fft = makeFFT(N);
    const win = new Float32Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)); // Hann window
    const bin = (f) => DF.clamp(Math.round((f * N) / sr), 1, N / 2 - 1);
    const bands = [[30, 160], [200, 4000], [5000, 16000]].map(([a, b]) => [bin(a), bin(b)]);
    const re = new Float32Array(N), im = new Float32Array(N), prev = new Float32Array(N / 2);
    const env = [new Float32Array(frames), new Float32Array(frames), new Float32Array(frames)];
    const rms = new Float32Array(frames);

    for (let f = 0; f < frames; f++) {
      const off = f * HOP;
      for (let i = 0; i < N; i++) { re[i] = mono[off + i] * win[i]; im[i] = 0; }
      fft(re, im);
      for (let b = 0; b < 3; b++) {
        const [lo, hi] = bands[b];
        let s = 0;
        for (let k = lo; k <= hi; k++) {
          const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]) / (N / 4);
          const lm = Math.log1p(mag * 200);          // log-compress: quiet details count too
          const d = lm - prev[k];
          if (d > 0) s += d;                         // keep only increases (half-wave rectify)
          prev[k] = lm;
        }
        env[b][f] = s / Math.sqrt(hi - lo + 1);
      }
      let e = 0;
      const end = Math.min(len, off + HOP);
      for (let i = off; i < end; i++) e += mono[i] * mono[i];
      rms[f] = Math.sqrt(e / Math.max(1, end - off));
      if (f % 250 === 0) { if (onProgress) onProgress(f / frames); await tick(); }
    }

    // Normalise each band so its loud onsets are ~1.
    const names = ['low', 'mid', 'high'];
    const E = {};
    for (let b = 0; b < 3; b++) {
      const scale = percentile(env[b], 0.98) || 1;
      for (let f = 0; f < frames; f++) env[b][f] = Math.min(1.6, env[b][f] / scale);
      E[names[b]] = env[b];
    }
    const all = new Float32Array(frames);
    for (let f = 0; f < frames; f++) all[f] = E.low[f] + 0.9 * E.mid[f] + 0.6 * E.high[f];
    E.all = detrend(all, fps);
    const envS = smooth3(E.all);

    if (onProgress) onProgress(1);
    await tick();

    // Tempo: rough guess from autocorrelation, then a fine search of tempo + phase.
    const P0 = estimatePeriod(E.all, fps);
    let best = { score: -1e9, P: P0, ph: 0 };
    for (let d = -0.03; d <= 0.03001; d += 0.0002) {
      const P = P0 * (1 + d);
      for (let ph = 0; ph < P; ph += 0.25) {
        const s = beatScore(envS, P, ph) * (1 - 2 * Math.abs(d)); // mild preference for the unrefined guess
        if (s > best.score) best = { score: s, P, ph };
      }
    }
    let bpm = (fps * 60) / best.P;
    if (Math.abs(bpm - Math.round(bpm)) < 0.3) bpm = Math.round(bpm); // most songs sit on whole BPMs
    bpm = Math.round(bpm * 100) / 100;

    const an = {
      sr, fps, frames, duration: len / sr, env: E, envS, rms, bpm,
      peaks: computePeaks(mono, 1800),
    };
    an.phase = DF.findPhase(an, bpm);
    an.barShift = DF.pickDownbeat(an, bpm, an.phase);
    return an;
  };

  // min/max per bucket, for drawing the waveform overview.
  function computePeaks(mono, buckets) {
    const peaks = new Float32Array(buckets * 2), size = mono.length / buckets;
    for (let b = 0; b < buckets; b++) {
      const a = Math.floor(b * size), e = Math.min(mono.length, Math.floor((b + 1) * size));
      let mn = 0, mx = 0;
      const stride = Math.max(1, ((e - a) / 64) | 0);
      for (let i = a; i < e; i += stride) {
        const v = mono[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      peaks[b * 2] = mn; peaks[b * 2 + 1] = mx;
    }
    return peaks;
  }

  // Sample the analysis on a 16th-note grid for a chosen tempo/phase/bar shift.
  // Cheap, so it is re-run whenever the user corrects BPM / downbeat / nudge.
  DF.buildGrid = function (an, bpm, phaseSec, barShift, nudgeSec) {
    const beat = 60 / bpm, sd = beat / 4;
    let T0 = phaseSec + barShift * beat + (nudgeSec || 0);
    // Put bar 1 at the earliest bar start that is (almost) at or after time 0. A tiny
    // negative value is allowed so a downbeat at 0:00 isn't pushed a whole bar later.
    const lowest = -sd * 0.5;
    while (T0 < lowest) T0 += 4 * beat;
    while (T0 - 4 * beat >= lowest) T0 -= 4 * beat;
    const bars = Math.min(400, Math.max(1, Math.ceil((an.duration - T0) / (4 * beat))));
    const steps = bars * 16;
    const L = new Float32Array(steps), M = new Float32Array(steps), H = new Float32Array(steps);
    const feat = [[an.env.low, L], [an.env.mid, M], [an.env.high, H]];
    for (let s = 0; s < steps; s++) {
      const t = T0 + s * sd;
      let f0 = Math.ceil(secToFrame(t - 0.38 * sd, an.sr));
      let f1 = Math.floor(secToFrame(t + 0.38 * sd, an.sr));
      if (f1 < f0) f0 = f1 = Math.round(secToFrame(t, an.sr));
      f0 = Math.max(0, f0); f1 = Math.min(an.frames - 1, f1);
      for (const [src, dst] of feat) {
        let m = 0;
        for (let f = f0; f <= f1; f++) if (src[f] > m) m = src[f];
        dst[s] = m;
      }
    }
    // Rescale so a "strong" hit on this grid is ~1, and keep only real onsets: a step that
    // is much weaker than a neighbour is usually just that neighbour's decay tail.
    for (const arr of [L, M, H]) {
      const sc = percentile(arr, 0.9) || 1;
      for (let i = 0; i < arr.length; i++) arr[i] = Math.min(1.2, arr[i] / sc);
      const orig = Float32Array.from(arr);
      for (let i = 0; i < arr.length; i++) {
        const nb = Math.max(orig[i - 1] || 0, orig[i + 1] || 0);
        if (orig[i] < 0.6 * nb) arr[i] = orig[i] * 0.3;
      }
    }
    // Loudness of each bar (0..1), used to decide where drums play and how busy.
    const E = new Float32Array(bars), barLen = 4 * beat;
    for (let b = 0; b < bars; b++) {
      const f0 = Math.max(0, Math.floor(secToFrame(T0 + b * barLen, an.sr)));
      const f1 = Math.min(an.frames - 1, Math.floor(secToFrame(T0 + (b + 1) * barLen, an.sr)));
      let s = 0, c = 0;
      for (let f = f0; f <= f1; f++) { s += an.rms[f]; c++; }
      E[b] = c ? s / c : 0;
    }
    const ref = percentile(E, 0.9) || 1;
    for (let b = 0; b < bars; b++) E[b] = DF.clamp(Math.sqrt(E[b] / ref), 0, 1);
    return { bpm, T0, bars, steps, stepDur: sd, L, M, H, E };
  };
})((window.DF = window.DF || {}));
