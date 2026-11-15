// The drum kit. No samples are used: every drum is synthesised from
// oscillators and filtered noise with the Web Audio API, so there are no
// files to download and the same code can render offline to a WAV.
(function (DF) {
  // One entry per row in the sequencer. `midi` = General MIDI drum note.
  DF.TRACKS = [
    { id: 'kick', label: 'Kick', key: 'a', color: '#ff5a3c', midi: 36 },
    { id: 'snare', label: 'Snare', key: 's', color: '#ffb627', midi: 38 },
    { id: 'clap', label: 'Clap', key: 'd', color: '#ff4fa3', midi: 39 },
    { id: 'hat', label: 'Hat', key: 'f', color: '#3ee6c1', midi: 42 },
    { id: 'ohat', label: 'Open Hat', key: 'g', color: '#4cc3ff', midi: 46 },
    { id: 'tomH', label: 'Tom Hi', key: 'h', color: '#a78bfa', midi: 50 },
    { id: 'tomL', label: 'Tom Lo', key: 'j', color: '#8b7bff', midi: 45 },
    { id: 'crash', label: 'Crash', key: 'k', color: '#f5f06a', midi: 49 },
    { id: 'perc', label: 'Perc', key: 'l', color: '#c4f261', midi: 56 },
  ];

  // Each kit is just a set of numbers that shape the same synth voices.
  DF.KITS = {
    '808 Classic': {
      drive: 1.4, lp: 15000,
      kick: { f0: 170, f1: 44, decay: 0.8, click: 0.12 },
      snare: { tone: 190, toneGain: 0.5, toneDecay: 0.1, hp: 1500, noiseGain: 0.7, decay: 0.2 },
      clap: { bp: 1200, decay: 0.28 },
      hat: { hp: 8000, decay: 0.04 }, ohat: { hp: 7000, decay: 0.32 },
      tom: { base: 105, decay: 0.4 }, crash: { hp: 4500, decay: 1.6 },
      perc: { f1: 540, f2: 810, decay: 0.1 },
    },
    '909 Club': {
      drive: 2.4, lp: 16000,
      kick: { f0: 210, f1: 50, decay: 0.45, click: 0.3 },
      snare: { tone: 200, toneGain: 0.4, toneDecay: 0.08, hp: 2500, noiseGain: 0.8, decay: 0.16 },
      clap: { bp: 1500, decay: 0.22 },
      hat: { hp: 9000, decay: 0.05 }, ohat: { hp: 8000, decay: 0.28 },
      tom: { base: 120, decay: 0.3 }, crash: { hp: 5500, decay: 1.4 },
      perc: { f1: 600, f2: 900, decay: 0.08 },
    },
    'Studio Kit': {
      drive: 0.7, lp: 14000,
      kick: { f0: 130, f1: 58, decay: 0.3, click: 0.35 },
      snare: { tone: 230, toneGain: 0.35, toneDecay: 0.09, hp: 1200, noiseGain: 0.75, decay: 0.22 },
      clap: { bp: 1000, decay: 0.3 },
      hat: { hp: 6500, decay: 0.06 }, ohat: { hp: 6000, decay: 0.4 },
      tom: { base: 135, decay: 0.28 }, crash: { hp: 3800, decay: 2.0 },
      perc: { f1: 480, f2: 720, decay: 0.12 },
    },
    'Lo-fi Tape': {
      drive: 3.2, lp: 5200,
      kick: { f0: 120, f1: 48, decay: 0.36, click: 0.1 },
      snare: { tone: 175, toneGain: 0.5, toneDecay: 0.1, hp: 900, noiseGain: 0.6, decay: 0.17 },
      clap: { bp: 900, decay: 0.2 },
      hat: { hp: 5000, decay: 0.05 }, ohat: { hp: 4500, decay: 0.22 },
      tom: { base: 100, decay: 0.3 }, crash: { hp: 3000, decay: 1.0 },
      perc: { f1: 420, f2: 640, decay: 0.1 },
    },
  };

  // tanh-style soft clipper curve for a little analog-ish warmth.
  function driveCurve(amount) {
    const n = 1024, curve = new Float32Array(n), k = Math.max(0.01, amount);
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      curve[i] = Math.tanh(k * x) / Math.tanh(k);
    }
    return curve;
  }

  DF.DrumSynth = class DrumSynth {
    // `ctx` can be a live AudioContext or an OfflineAudioContext (for WAV export).
    constructor(ctx, dest, kitName) {
      this.ctx = ctx;
      // 2 seconds of white noise, reused by every noisy drum.
      this.noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
      const d = this.noiseBuf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      // Signal chain: bus -> soft clipper -> low-pass -> compressor -> dest
      this.bus = ctx.createGain();
      this.shaper = ctx.createWaveShaper();
      this.shaper.oversample = '2x';
      this.lp = ctx.createBiquadFilter();
      this.lp.type = 'lowpass';
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -14; comp.ratio.value = 4;
      comp.attack.value = 0.003; comp.release.value = 0.12;
      this.bus.connect(this.shaper); this.shaper.connect(this.lp);
      this.lp.connect(comp); comp.connect(dest);
      this.lastOpen = null;
      this.setKit(kitName);
    }

    setKit(name) {
      this.kit = DF.KITS[name] || DF.KITS['808 Classic'];
      this.shaper.curve = driveCurve(this.kit.drive);
      this.lp.frequency.value = this.kit.lp;
    }

    trigger(id, when, vel) {
      const v = Math.pow(DF.clamp(vel == null ? 1 : vel, 0.05, 1), 1.3); // gentle velocity curve
      const fn = this['_' + id];
      if (fn) fn.call(this, when, v);
    }

    // ---- small helpers ----
    _noise(t, dur) {
      const s = this.ctx.createBufferSource();
      s.buffer = this.noiseBuf; s.loop = true;
      s.start(t, Math.random() * 1.5); s.stop(t + dur + 0.03);
      return s;
    }
    // Amplitude envelope: instant attack, exponential decay.
    _env(t, peak, decay) {
      const g = this.ctx.createGain();
      g.gain.setValueAtTime(0.0001, t);
      g.gain.linearRampToValueAtTime(Math.max(peak, 0.0002), t + 0.001);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.001 + decay);
      return g;
    }
    _filter(type, freq, q) {
      const f = this.ctx.createBiquadFilter();
      f.type = type; f.frequency.value = freq; f.Q.value = q || 0.7;
      return f;
    }

    // ---- the drums ----
    _kick(t, v) {
      const k = this.kit.kick, c = this.ctx;
      const o = c.createOscillator(), g = this._env(t, v, k.decay);
      o.type = 'sine';
      // The "thump": pitch falls fast from f0 to f1.
      o.frequency.setValueAtTime(k.f0, t);
      o.frequency.exponentialRampToValueAtTime(k.f1, t + 0.06);
      o.connect(g); g.connect(this.bus);
      o.start(t); o.stop(t + k.decay + 0.05);
      if (k.click) { // short noise burst = the beater "click"
        const n = this._noise(t, 0.015), hp = this._filter('highpass', 2500);
        const cg = this._env(t, v * k.click, 0.012);
        n.connect(hp); hp.connect(cg); cg.connect(this.bus);
      }
    }

    _snare(t, v) {
      const s = this.kit.snare, c = this.ctx;
      const o = c.createOscillator(), og = this._env(t, v * s.toneGain, s.toneDecay);
      o.type = 'triangle';
      o.frequency.setValueAtTime(s.tone * 1.3, t);
      o.frequency.exponentialRampToValueAtTime(s.tone, t + 0.04);
      o.connect(og); og.connect(this.bus); o.start(t); o.stop(t + s.toneDecay + 0.05);
      const n = this._noise(t, s.decay), hp = this._filter('highpass', s.hp);
      const ng = this._env(t, v * s.noiseGain, s.decay);
      n.connect(hp); hp.connect(ng); ng.connect(this.bus);
    }

    _clap(t, v) {
      const k = this.kit.clap, g = this.ctx.createGain();
      const n = this._noise(t, 0.1 + k.decay), bp = this._filter('bandpass', k.bp, 0.9);
      // Three quick bursts then a tail = several hands clapping slightly apart.
      g.gain.setValueAtTime(v * 0.9, t);
      g.gain.exponentialRampToValueAtTime(v * 0.15, t + 0.009);
      g.gain.setValueAtTime(v * 0.9, t + 0.011);
      g.gain.exponentialRampToValueAtTime(v * 0.15, t + 0.02);
      g.gain.setValueAtTime(v * 0.9, t + 0.022);
      g.gain.exponentialRampToValueAtTime(v * 0.15, t + 0.031);
      g.gain.setValueAtTime(v, t + 0.033);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.033 + k.decay);
      n.connect(bp); bp.connect(g); g.connect(this.bus);
    }

    _hat(t, v) {
      const h = this.kit.hat;
      // A closed hat chokes (cuts off) any open hat still ringing.
      if (this.lastOpen && t > this.lastOpen.t) this.lastOpen.choke.gain.setTargetAtTime(0, t, 0.004);
      const n = this._noise(t, h.decay), hp = this._filter('highpass', h.hp, 0.9);
      const g = this._env(t, v * 0.5, h.decay);
      n.connect(hp); hp.connect(g); g.connect(this.bus);
    }

    _ohat(t, v) {
      const h = this.kit.ohat;
      const n = this._noise(t, h.decay), hp = this._filter('highpass', h.hp, 0.9);
      const g = this._env(t, v * 0.45, h.decay);
      const choke = this.ctx.createGain(); // lets a later closed hat cut this one off
      n.connect(hp); hp.connect(g); g.connect(choke); choke.connect(this.bus);
      this.lastOpen = { t, choke };
    }

    _tom(t, v, freq) {
      const k = this.kit.tom, o = this.ctx.createOscillator(), g = this._env(t, v * 0.9, k.decay);
      o.type = 'sine';
      o.frequency.setValueAtTime(freq * 1.5, t);
      o.frequency.exponentialRampToValueAtTime(freq, t + 0.07);
      o.connect(g); g.connect(this.bus); o.start(t); o.stop(t + k.decay + 0.05);
      const n = this._noise(t, 0.02), ng = this._env(t, v * 0.12, 0.015);
      n.connect(ng); ng.connect(this.bus);
    }
    _tomH(t, v) { this._tom(t, v, this.kit.tom.base * 1.6); }
    _tomL(t, v) { this._tom(t, v, this.kit.tom.base); }

    _crash(t, v) {
      const k = this.kit.crash;
      const n = this._noise(t, k.decay), hp = this._filter('highpass', k.hp, 0.6);
      const g = this._env(t, v * 0.5, k.decay);
      n.connect(hp); hp.connect(g); g.connect(this.bus);
    }

    _perc(t, v) { // cowbell-ish: two detuned square waves through a band-pass
      const p = this.kit.perc, c = this.ctx;
      const bp = this._filter('bandpass', (p.f1 + p.f2) / 2, 2.5), g = this._env(t, v * 0.5, p.decay);
      [p.f1, p.f2].forEach((f) => {
        const o = c.createOscillator();
        o.type = 'square'; o.frequency.value = f;
        o.connect(bp); o.start(t); o.stop(t + p.decay + 0.05);
      });
      bp.connect(g); g.connect(this.bus);
    }
  };
})((window.DF = window.DF || {}));
