"""Render training MIDI excerpts to audio, run the browser's analyser on them, keep the features.

This closes the gap between "features computed from MIDI note lists" (clean: a hi-hat only touches the
high band) and "features the browser computes from real audio" (smeared: every drum leaks into every
band). Each excerpt = BARS bars of one song:
    accompaniment audio (sine-harmonic synth, timbre by GM family)  [+ synthesised drums]
 -> js/analyze.js  -> L/M/H onset strengths per 16th step + bar loudness
 paired with the drum grid D the song's real drummer played.

python render.py --split train --n 1500 --out data/audio_train.jsonl
"""
import argparse, json, os, pickle, subprocess, sys, tempfile, warnings
import numpy as np
import pretty_midi
from multiprocessing import Pool
from preprocess import PITCH2TRACK

warnings.filterwarnings("ignore")
SR, BARS = 44100, 24
_tmpl = {}


def sustained(prog):
    return not (prog < 16 or 24 <= prog < 40 or 104 <= prog < 120)    # organ/strings/brass/reeds/pads/leads sustain


def note_template(pitch, sus, bright):
    key = (pitch, sus, round(bright, 1))
    if key in _tmpl:
        return _tmpl[key]
    f = 440.0 * 2 ** ((pitch - 69) / 12)
    n = int(SR * (0.6 if sus else 1.2))
    t = np.arange(n) / SR
    y = np.zeros(n, np.float32)
    for h in range(1, 12):
        if f * h > 9000:
            break
        y += (np.sin(2 * np.pi * f * h * t) / h ** bright).astype(np.float32)
    if not sus:
        y *= np.exp(-t * (2.0 if pitch < 50 else 4.5)).astype(np.float32)
    y /= max(np.abs(y).max(), 1e-6)
    _tmpl[key] = y
    return y


def drum_shot(track, rng):
    dur = [0.5, 0.35, 0.3, 0.12, 0.6, 0.45, 0.5, 1.5, 0.15][track]
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.uniform(-1, 1, n).astype(np.float32)
    hp = lambda x, k: np.diff(x, n=k, prepend=np.zeros(k, np.float32))
    if track == 0:       # kick: pitch-dropping sine + beater click
        f0, f1, k = rng.uniform(110, 200), rng.uniform(40, 60), rng.uniform(25, 55)
        ph = 2 * np.pi * (f1 * t + (f0 - f1) * (1 - np.exp(-k * t)) / k)
        y = np.sin(ph) * np.exp(-t * rng.uniform(6, 13)) + 0.25 * hp(noise, 1) * np.exp(-t * 300)
    elif track == 1:     # snare: noise + body
        y = 0.6 * noise * np.exp(-t * rng.uniform(14, 28)) + 0.4 * np.sin(2 * np.pi * rng.uniform(170, 230) * t) * np.exp(-t * 28)
    elif track == 2:     # clap: three bursts
        env = sum(np.exp(-np.clip(t - o, 0, None) * 120) * (t >= o) for o in (0, 0.011, 0.022)) * 0.4 + np.exp(-t * rng.uniform(15, 30)) * (t > 0.03)
        y = hp(noise, 1) * env
    elif track in (3, 4):  # hats: high-passed noise
        y = hp(noise, 2) * np.exp(-t * (rng.uniform(55, 110) if track == 3 else rng.uniform(9, 16)))
        y *= 0.5
    elif track in (5, 6):  # toms
        f = rng.uniform(150, 220) if track == 5 else rng.uniform(80, 130)
        y = np.sin(2 * np.pi * (f * t + 0.5 * f * np.exp(-t * 40) / 40)) * np.exp(-t * rng.uniform(7, 12))
    elif track == 7:     # crash
        y = hp(noise, 1) * np.exp(-t * rng.uniform(1.5, 3)) * 0.5
    else:                # perc: cowbell-ish
        y = (np.sign(np.sin(2 * np.pi * 540 * t)) + np.sign(np.sin(2 * np.pi * 810 * t))) * 0.25 * np.exp(-t * 25)
    return (y / max(np.abs(y).max(), 1e-6)).astype(np.float32)


def render(pm, t_s, t_e, rng, with_drums, bright, dgain):
    n = int((t_e - t_s) * SR) + SR
    out = np.zeros(n, np.float32)
    shots = [drum_shot(k, rng) for k in range(9)] if with_drums else None
    for inst in pm.instruments:
        if inst.is_drum and not with_drums:
            continue
        sus = sustained(inst.program)
        for note in inst.notes:
            if note.end < t_s or note.start >= t_e:
                continue
            i0 = int((note.start - t_s) * SR)
            if i0 < 0:
                continue
            if inst.is_drum:
                tr = PITCH2TRACK.get(note.pitch)
                if tr is None:
                    continue
                s = shots[tr]
                i1 = min(n, i0 + len(s))
                out[i0:i1] += s[: i1 - i0] * dgain * (note.velocity / 127.0) ** 1.2
            else:
                dur = min(note.end - note.start, 3.0)
                m = max(int(dur * SR), int(0.06 * SR))
                tmpl = note_template(note.pitch, sus, bright)
                y = np.resize(tmpl, m) if len(tmpl) < m else tmpl[:m].copy()
                fa, fr = min(m, int(0.012 * SR)), min(m, int(0.05 * SR))
                y[:fa] *= np.linspace(0, 1, fa, dtype=np.float32)
                y[m - fr:] *= np.linspace(1, 0, fr, dtype=np.float32)
                amp = (note.velocity / 127.0) * min(2.5, (200.0 / (440.0 * 2 ** ((note.pitch - 69) / 12))) ** 0.25)
                i1 = min(n, i0 + m)
                out[i0:i1] += y[: i1 - i0] * amp * 0.3
    out /= max(np.abs(out).max(), 1e-6) / 0.9
    return out[: int((t_e - t_s) * SR)]


def work(task):
    path, split, k, b0, seed, D, variant = task
    rng = np.random.default_rng(seed)
    try:
        pm = pretty_midi.PrettyMIDI(path)
        beats = pm.get_beats()
        if len(beats) < (b0 + BARS) * 4 + 1:
            return None
        t_s, t_e = beats[b0 * 4], beats[(b0 + BARS) * 4]
        bar_d = np.diff(beats[b0 * 4:(b0 + BARS) * 4 + 1:4])
        if bar_d.std() / bar_d.mean() > 0.004:       # tempo must be steady in the excerpt
            return None
        bpm = 60.0 / (bar_d.mean() / 4)
        bright, dgain = rng.uniform(0.8, 1.8), rng.uniform(0.7, 1.5)
        audio = render(pm, t_s, t_e, rng, variant == "wd", bright, dgain)
        with tempfile.NamedTemporaryFile(suffix=".f32", delete=True, dir="/tmp") as f:
            f.write(audio.tobytes())
            f.flush()
            nudge = float(rng.uniform(-0.012, 0.012))
            r = subprocess.run(["node", "extract_one.js", f.name, str(bpm), str(nudge), str(BARS)], capture_output=True, timeout=300)
        feats = json.loads(r.stdout)
        return dict(path=path, split=split, k=k, b0=b0, bpm=bpm, variant=variant, D=D.tolist(), **feats)
    except Exception as e:
        print("skip", path, repr(e)[:80], file=sys.stderr)
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    songs = pickle.load(open(f"data/{a.split}.pkl", "rb"))
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(songs))
    tasks = []
    for si in order:
        s = songs[si]
        bars = len(s["D"]) // 16
        if bars < BARS + 2:
            continue
        b0 = int(rng.integers(0, bars - BARS))
        D = s["D"][b0 * 16:(b0 + BARS) * 16]
        if (D > 0).sum() < 60:
            continue
        tasks.append((s["path"], a.split, int(si), b0, int(rng.integers(1 << 30)), D, "wd" if rng.random() < 0.5 else "nd"))
        if len(tasks) >= a.n * 2:
            break
    print(len(tasks), "candidate excerpts; target", a.n, flush=True)
    done = 0
    with open(a.out, "a") as fo, Pool(a.procs) as pool:
        for res in pool.imap_unordered(work, tasks, chunksize=1):
            if res is None:
                continue
            fo.write(json.dumps(res) + "\n")
            fo.flush()
            done += 1
            if done % 25 == 0:
                print(done, "done", flush=True)
            if done >= a.n:
                pool.terminate()
                break
    print("finished", done)
