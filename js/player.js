// Playback. Uses the standard "look-ahead scheduler" pattern for tight timing:
// a JS timer wakes every 25 ms and schedules any notes that fall in the next
// ~150 ms on the (very precise) Web Audio clock, instead of playing notes
// directly from the (jittery) JS timer.
(function (DF) {
  // Swing delays every second 16th note. swing is 0..1 (0.67 = triplet shuffle).
  DF.swingOffset = (step, swing, stepDur) => (step & 1 ? swing * 0.5 * stepDur : 0);

  // Trigger every drum that sits on one step. Shared by live playback and WAV export.
  DF.triggerStep = function (synth, project, s, t, muted, onHit) {
    for (const tr of DF.TRACKS) {
      if (muted && muted.has(tr.id)) continue;
      const v = project.pattern[tr.id][s];
      if (v > 0) {
        // Humanize: nudge each hit a few ms late (never early, so it can't be in the past).
        const when = t + (project.humanize ? Math.random() * 0.012 * project.humanize : 0);
        synth.trigger(tr.id, when, v);
        if (onHit) onHit(tr.id, v, when);
      }
    }
  };

  DF.Player = class Player {
    // engine = { ctx, drumGain, songGain, getProject(), getBuffer(), getKit(), muted:Set, onHit(id,v,t), onStop() }
    constructor(engine) {
      this.e = engine;
      this.playing = false;
      this.timer = null;
      this.src = null;
      this.session = null;
    }

    // fromStep: first step to play (loop mode). songPos: start position in seconds (audio mode).
    start(fromStep, songPos) {
      const e = this.e, ctx = e.ctx;
      this.stop(true);
      const p = e.getProject();
      const buf = p.mode === 'audio' ? e.getBuffer() : null;
      const sd = 60 / p.bpm / 4;
      const t0 = ctx.currentTime + 0.08;

      // A fresh synth + gain per play session, so Stop can cut it off instantly.
      const gain = ctx.createGain();
      gain.connect(e.drumGain);
      this.session = { gain, synth: new DF.DrumSynth(ctx, gain, e.getKit()) };

      this.buf = buf;
      this.t0 = t0;
      this.songPos = songPos || 0;
      if (buf) {
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(e.songGain);
        src.start(t0, this.songPos);
        src.onended = () => { if (this.src === src) this.stop(); };
        this.src = src;
        this.nextAbs = Math.max(0, Math.ceil((this.songPos - p.gridOffset) / sd - 1e-6));
        this.nextTime = t0 + (p.gridOffset + this.nextAbs * sd - this.songPos);
      } else {
        this.nextAbs = fromStep || 0;
        this.nextTime = t0;
      }
      this.playing = true;
      this.timer = setInterval(() => this._tick(), 25);
      this._tick();
    }

    _tick() {
      const e = this.e, p = e.getProject(), ctx = e.ctx;
      const total = p.bars * 16, sd = 60 / p.bpm / 4;
      const horizon = ctx.currentTime + 0.15;
      while (this.nextTime < horizon) {
        if (this.buf && this.nextAbs >= total) break; // song mode doesn't loop
        const s = this.nextAbs % total;
        const t = this.nextTime + DF.swingOffset(this.nextAbs, p.swing, sd);
        DF.triggerStep(this.session.synth, p, s, t, e.muted, e.onHit);
        this.nextAbs++;
        this.nextTime += sd;
      }
    }

    stop(silent) {
      if (this.timer) clearInterval(this.timer);
      this.timer = null;
      if (this.src) {
        const s = this.src;
        this.src = null;
        try { s.stop(); } catch (err) { /* already stopped */ }
        s.disconnect();
      }
      if (this.session) {
        const { gain } = this.session, ctx = this.e.ctx;
        gain.gain.setTargetAtTime(0, ctx.currentTime, 0.01); // fade out so nothing clicks
        setTimeout(() => gain.disconnect(), 400);
        this.session = null;
      }
      const was = this.playing;
      this.playing = false;
      if (was && !silent && this.e.onStop) this.e.onStop();
    }

    setKit(name) { if (this.session) this.session.synth.setKit(name); }

    // Current position in (fractional) steps, or -1 when stopped.
    stepPos() {
      if (!this.playing) return -1;
      const p = this.e.getProject(), ctx = this.e.ctx, sd = 60 / p.bpm / 4;
      const total = p.bars * 16;
      let pos;
      if (this.buf) pos = (this.songTime() - p.gridOffset) / sd;
      else pos = this.nextAbs + (ctx.currentTime - this.nextTime) / sd;
      if (!this.buf) pos = ((pos % total) + total) % total;
      return pos;
    }

    // Position in the song in seconds (audio mode).
    songTime() {
      return this.songPos + Math.max(0, this.e.ctx.currentTime - this.t0);
    }
  };
})((window.DF = window.DF || {}));
