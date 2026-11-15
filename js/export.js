// Getting your drum part OUT: MIDI file, WAV render, and the browser library.
// Also contains the built-in demo song (rendered on the fly, no file needed).
(function (DF) {
  // ---------- MIDI ----------
  // A minimal Standard MIDI File writer (format 0, one track, GM drum channel 10).
  function vlq(n) { // variable-length quantity, MIDI's compact number format
    const out = [n & 0x7f];
    while ((n >>= 7)) out.unshift((n & 0x7f) | 0x80);
    return out;
  }

  DF.makeMidi = function (project, muted, leadIn) {
    const PPQ = 480, tps = PPQ / 4; // ticks per 16th step
    const lead = leadIn && project.mode === 'audio' ? Math.max(0, Math.round((project.gridOffset / (60 / project.bpm)) * PPQ)) : 0;
    const ev = [];
    for (const tr of DF.TRACKS) {
      if (muted && muted.has(tr.id)) continue;
      const arr = project.pattern[tr.id];
      for (let s = 0; s < arr.length; s++) {
        if (arr[s] <= 0) continue;
        const tick = lead + Math.round(s * tps + ((s & 1) ? project.swing * 0.5 * tps : 0));
        ev.push({ tick, bytes: [0x99, tr.midi, Math.max(1, Math.round(arr[s] * 127))], order: 1 });
        ev.push({ tick: tick + 60, bytes: [0x89, tr.midi, 0], order: 0 });
      }
    }
    ev.sort((a, b) => a.tick - b.tick || a.order - b.order);
    const mpqn = Math.round(60000000 / project.bpm); // microseconds per quarter note
    const data = [
      0, 0xff, 0x03, 0x09, ...'DrumForge'.split('').map((c) => c.charCodeAt(0)),
      0, 0xff, 0x51, 0x03, (mpqn >> 16) & 255, (mpqn >> 8) & 255, mpqn & 255,
      0, 0xff, 0x58, 0x04, 4, 2, 24, 8,
    ];
    let last = 0;
    for (const e of ev) { data.push(...vlq(e.tick - last), ...e.bytes); last = e.tick; }
    data.push(0, 0xff, 0x2f, 0);
    const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
    const head = [0x4d, 0x54, 0x68, 0x64, ...u32(6), 0, 0, 0, 1, (PPQ >> 8) & 255, PPQ & 255];
    const trk = [0x4d, 0x54, 0x72, 0x6b, ...u32(data.length), ...data];
    return new Blob([new Uint8Array([...head, ...trk])], { type: 'audio/midi' });
  };

  // ---------- WAV ----------
  function encodeWav(buffer) {
    const nch = buffer.numberOfChannels, len = buffer.length, sr = buffer.sampleRate;
    const ch = [];
    let peak = 0;
    for (let c = 0; c < nch; c++) {
      ch.push(buffer.getChannelData(c));
      for (let i = 0; i < len; i++) peak = Math.max(peak, Math.abs(ch[c][i]));
    }
    const gain = peak > 0.98 ? 0.98 / peak : 1; // avoid clipping
    const out = new DataView(new ArrayBuffer(44 + len * nch * 2));
    const wr = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
    wr(0, 'RIFF'); out.setUint32(4, 36 + len * nch * 2, true); wr(8, 'WAVE'); wr(12, 'fmt ');
    out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, nch, true);
    out.setUint32(24, sr, true); out.setUint32(28, sr * nch * 2, true);
    out.setUint16(32, nch * 2, true); out.setUint16(34, 16, true);
    wr(36, 'data'); out.setUint32(40, len * nch * 2, true);
    let o = 44;
    for (let i = 0; i < len; i++) {
      for (let c = 0; c < nch; c++) {
        out.setInt16(o, DF.clamp(ch[c][i] * gain, -1, 1) * 32767, true);
        o += 2;
      }
    }
    return new Blob([out], { type: 'audio/wav' });
  }

  // Render the drum part (optionally mixed over the original song) to a WAV.
  DF.renderWav = async function (project, opts) {
    const sr = 44100, sd = 60 / project.bpm / 4;
    const total = project.bars * 16;
    const isSong = project.mode === 'audio' && opts.buffer;
    const loops = isSong ? 1 : opts.loops || 1;
    let dur = (isSong ? project.gridOffset : 0) + total * loops * sd;
    if (isSong && opts.mixSong) dur = Math.max(dur, opts.buffer.duration);
    dur += 2.5; // let the last crash ring out
    const off = new OfflineAudioContext(2, Math.ceil(sr * dur), sr);
    const master = off.createGain();
    master.gain.value = 0.9;
    master.connect(off.destination);
    const drumGain = off.createGain();
    drumGain.gain.value = opts.drumVol == null ? 1 : opts.drumVol;
    drumGain.connect(master);
    const synth = new DF.DrumSynth(off, drumGain, project.kit);
    const t0 = isSong ? project.gridOffset : 0;
    for (let abs = 0; abs < total * loops; abs++) {
      const t = Math.max(0, t0 + abs * sd + DF.swingOffset(abs, project.swing, sd));
      DF.triggerStep(synth, project, abs % total, t, opts.muted, null);
    }
    if (isSong && opts.mixSong) {
      const src = off.createBufferSource();
      const g = off.createGain();
      g.gain.value = opts.songVol == null ? 0.8 : opts.songVol;
      src.buffer = opts.buffer;
      src.connect(g); g.connect(master);
      src.start(0);
    }
    return encodeWav(await off.startRendering());
  };

  // ---------- Library (localStorage) ----------
  const KEY = 'drumforge.library.v1';
  DF.library = {
    list() {
      try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; }
    },
    _write(items) {
      localStorage.setItem(KEY, JSON.stringify(items));
    },
    // Only non-empty steps are stored, as [step, velocity%] pairs, to keep entries small.
    save(project, muted) {
      const pattern = {};
      for (const tr of DF.TRACKS) {
        const arr = project.pattern[tr.id], list = [];
        for (let s = 0; s < arr.length; s++) if (arr[s] > 0) list.push(s, Math.round(arr[s] * 100));
        pattern[tr.id] = list;
      }
      const item = {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        name: project.name || 'Untitled', saved: Date.now(),
        mode: project.mode, bpm: project.bpm, bars: project.bars, swing: project.swing,
        humanize: project.humanize, kit: project.kit, styleLabel: project.styleLabel, pattern,
      };
      const items = this.list();
      items.unshift(item);
      this._write(items); // may throw if storage is full; the caller shows the error
      return item;
    },
    remove(id) { this._write(this.list().filter((i) => i.id !== id)); },
    // Turn a stored entry back into a playable project (always loops).
    revive(item) {
      const steps = item.bars * 16, pattern = {};
      for (const tr of DF.TRACKS) {
        const arr = new Float32Array(steps), list = item.pattern[tr.id] || [];
        for (let i = 0; i < list.length; i += 2) if (list[i] < steps) arr[list[i]] = list[i + 1] / 100;
        pattern[tr.id] = arr;
      }
      return {
        name: item.name, mode: 'random', bpm: item.bpm, bars: item.bars, swing: item.swing || 0,
        humanize: item.humanize == null ? 0.3 : item.humanize, kit: item.kit, gridOffset: 0,
        styleLabel: item.styleLabel, pattern,
      };
    },
  };

  // ---------- Demo song ----------
  // A little synthetic track (bass + plucks + pad, NO drums) so people can try
  // the analyser without owning an mp3. Rendered offline in about a second.
  DF.renderDemo = async function () {
    const sr = 44100, bpm = 112, beat = 60 / bpm, bars = 16;
    const dur = bars * 4 * beat + 1.5;
    const off = new OfflineAudioContext(2, Math.ceil(sr * dur), sr);
    const out = off.createGain();
    out.gain.value = 0.7;
    out.connect(off.destination);
    const roots = [55, 43.65, 65.41, 49];
    const chords = [[220, 261.63, 329.63], [174.61, 220, 261.63], [261.63, 329.63, 392], [196, 246.94, 293.66]];
    const note = (type, freq, t, len, peak, cutoff, dest) => {
      const o = off.createOscillator(), g = off.createGain(), f = off.createBiquadFilter();
      o.type = type; o.frequency.value = freq;
      f.type = 'lowpass'; f.frequency.value = cutoff;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.linearRampToValueAtTime(peak, t + 0.005);
      g.gain.exponentialRampToValueAtTime(0.0001, t + len);
      o.connect(f); f.connect(g); g.connect(dest || out);
      o.start(t); o.stop(t + len + 0.05);
    };
    for (let bar = 0; bar < bars; bar++) {
      const ch = Math.floor(bar / 2) % 4, e = bar < 4 ? 0.55 : bar < 12 ? 1 : 0.85;
      const t0 = bar * 4 * beat, sd = beat / 4;
      // Bass: a syncopated riff.
      // Bass: steady pulse on every beat plus a few syncopated extras.
      [0, 4, 6, 8, 11, 12, 14].forEach((p) => note('sawtooth', roots[ch], t0 + p * sd, sd * 2.2, (p % 4 === 0 ? 0.7 : 0.35) * e, 380));
      // Soft sub pulse on beat 1 of each bar to mark the start of the bar.
      note('sine', roots[ch] * 1.5, t0, 0.25, 0.6 * e, 200);
      // Plucked chord arpeggio on the 8th notes.
      for (let p = 0; p < 16; p += 2) note('triangle', chords[ch][(p / 2) % 3] * 2, t0 + p * sd, 0.22, 0.12 * e, 3000);
      // Pad chord, every 2 bars.
      if (bar % 2 === 0) chords[ch].forEach((f) => note('sine', f, t0, beat * 7.5, 0.06 * e, 1500));
      // Short high "tick" on the off-beats for some hat-like content in the mix.
      if (bar >= 8) for (let p = 2; p < 16; p += 4) note('square', 5200, t0 + p * sd, 0.03, 0.02, 9000);
    }
    return off.startRendering();
  };
})((window.DF = window.DF || {}));
