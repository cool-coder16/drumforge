"""Turn Lakh Clean MIDI files into training examples.

For every song we keep a 16th-note grid of:
  D       (steps, 9)  drum classes 0..3 per DrumForge track (0 off, 1 ghost, 2 normal, 3 accent)
  F_nd    (steps, 3)  "audio-like" onset strength per band from the NON-drum instruments
  F_d     (steps, 3)  the same, but derived from the drums (kick->low, snare->mid, hats->high)
  E_nd    (bars,)     loudness-ish per bar without drums, 0..1
  E_d     (bars,)     loudness-ish per bar with drums, 0..1
  bpm

The features mimic what js/analyze.js measures from real audio (spectral flux in a low,
mid and high band), so a model trained on them can be fed the browser's analysis.
"""
import glob, os, signal, sys, warnings, hashlib
import numpy as np
import pretty_midi
from multiprocessing import Pool

warnings.filterwarnings("ignore")

# DrumForge track order: kick, snare, clap, hat, ohat, tomH, tomL, crash, perc
PITCH2TRACK = {}
for p in (35, 36): PITCH2TRACK[p] = 0
for p in (38, 40): PITCH2TRACK[p] = 1
PITCH2TRACK[39] = 2
for p in (42, 44, 51, 53, 59): PITCH2TRACK[p] = 3
PITCH2TRACK[46] = 4
for p in (47, 48, 50): PITCH2TRACK[p] = 5
for p in (41, 43, 45): PITCH2TRACK[p] = 6
for p in (49, 52, 55, 57): PITCH2TRACK[p] = 7
for p in [37, 54, 56] + list(range(60, 82)): PITCH2TRACK[p] = 8

# how a drum hit shows up in the (low, mid, high) bands of audio
DRUM_BANDS = {0: (1.0, 0.3, 0.1), 1: (0.35, 1.0, 0.5), 2: (0.1, 0.8, 0.6), 3: (0.0, 0.1, 0.9),
              4: (0.0, 0.1, 1.0), 5: (0.5, 0.8, 0.2), 6: (0.7, 0.6, 0.1), 7: (0.0, 0.3, 1.0), 8: (0.0, 0.5, 0.6)}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def nearest_step(times, steps):
    """index of the nearest step time for each time, plus the distance."""
    i = np.searchsorted(steps, times)
    i = np.clip(i, 1, len(steps) - 1)
    left, right = steps[i - 1], steps[i]
    pick = np.where(np.abs(times - left) <= np.abs(times - right), i - 1, i)
    return pick, np.abs(times - steps[pick])


def process(path):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(40)
    try:
        return _process(path)
    except Exception:
        return None
    finally:
        signal.alarm(0)


def _process(path):
    pm = pretty_midi.PrettyMIDI(path)
    for ts in pm.time_signature_changes:
        if (ts.numerator, ts.denominator) != (4, 4):
            return None
    drums = [i for i in pm.instruments if i.is_drum]
    others = [i for i in pm.instruments if not i.is_drum]
    if not drums or not others:
        return None
    beats = pm.get_beats()
    if len(beats) < 33:
        return None
    # four steps per beat
    steps = np.concatenate([beats[:-1, None] + (beats[1:] - beats[:-1])[:, None] * np.arange(4)[None, :] / 4.0]).reshape(-1)
    n_steps = (len(steps) // 16) * 16
    if n_steps < 16 * 8:
        return None
    steps = steps[:n_steps]
    step_dur = np.median(np.diff(steps))

    dn = [(n.start, n.pitch, n.velocity) for d in drums for n in d.notes if n.pitch in PITCH2TRACK]
    if len(dn) < 64:
        return None
    t = np.array([x[0] for x in dn]); pit = np.array([x[1] for x in dn]); vel = np.array([x[2] for x in dn], dtype=float)
    idx, dist = nearest_step(t, steps)
    on_grid = dist < 0.3 * step_dur
    if on_grid.mean() < 0.85:           # shuffled / triplet / free-time: skip
        return None
    # only keep notes inside the grid
    keep = on_grid & (t <= steps[-1] + step_dur)
    idx, pit, vel = idx[keep], pit[keep], vel[keep]
    p90 = max(np.percentile(vel, 90), 1.0)
    rel = vel / p90
    cls = np.where(rel < 0.55, 1, np.where(rel < 1.05, 2, 3)).astype(np.uint8)
    D = np.zeros((n_steps, 9), np.uint8)
    F_d = np.zeros((n_steps, 3), np.float32)
    for s, p, c, v in zip(idx, pit, cls, rel):
        tr = PITCH2TRACK[p]
        D[s, tr] = max(D[s, tr], c)
        b = DRUM_BANDS[tr]
        for k in range(3):
            F_d[s, k] = max(F_d[s, k], b[k] * min(1.2, v))
    bars = n_steps // 16
    hits_per_bar = (D > 0).sum() / bars
    if hits_per_bar < 6 or hits_per_bar > 90:
        return None

    # non-drum features: onset strength per pitch band at each step
    on = [(n.start, n.pitch, n.velocity) for i in others for n in i.notes]
    if len(on) < 200:
        return None
    ot = np.array([x[0] for x in on]); op = np.array([x[1] for x in on]); ov = np.array([x[2] for x in on], dtype=float) / 127.0
    oi, _ = nearest_step(ot, steps)
    inside = ot <= steps[-1] + step_dur
    oi, op, ov = oi[inside], op[inside], ov[inside]
    F_nd = np.zeros((n_steps, 3), np.float32)
    count = np.zeros((n_steps, 3), np.float32)
    band = np.where(op < 50, 0, np.where(op < 80, 1, 2))
    for s, b, v in zip(oi, band, ov):
        F_nd[s, b] = max(F_nd[s, b], v)
        count[s, b] += 1
    F_nd = np.minimum(1.2, F_nd + 0.12 * np.maximum(count - 1, 0))

    # per-bar loudness proxies (sum of note velocities, then sqrt-normalised like js/analyze.js)
    bar_of = lambda s: s // 16
    nd_sum = np.zeros(bars); d_sum = np.zeros(bars)
    for s, v in zip(oi, ov):
        if bar_of(s) < bars: nd_sum[bar_of(s)] += v
    for s, v in zip(idx, rel):
        if bar_of(s) < bars: d_sum[bar_of(s)] += 0.5 * min(v, 1.2)

    def norm(x):
        ref = max(np.percentile(x, 90), 1e-6)
        return np.clip(np.sqrt(x / ref), 0, 1).astype(np.float32)

    E_nd = norm(nd_sum); E_d = norm(nd_sum + d_sum)
    bpm = 60.0 / np.median(np.diff(beats))
    return dict(D=D, F_nd=F_nd.astype(np.float16), F_d=F_d.astype(np.float16), E_nd=E_nd, E_d=E_d, bpm=np.float32(bpm))


def artist_split(path):
    # split by artist directory so versions of the same song never straddle train/val/test
    artist = path.split("clean_midi/")[1].split("/")[0]
    h = int(hashlib.md5(artist.encode()).hexdigest(), 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


if __name__ == "__main__":
    files = sorted(glob.glob("data/clean_midi/**/*.mid", recursive=True) + glob.glob("data/clean_midi/**/*.midi", recursive=True))
    print(len(files), "files", flush=True)
    out = {"train": [], "val": [], "test": []}
    ok = 0
    with Pool(8) as pool:
        for n, (path, res) in enumerate(zip(files, pool.imap(process, files, chunksize=8))):
            if res is not None:
                res["path"] = path
                out[artist_split(path)].append(res)
                ok += 1
            if n % 500 == 0:
                print(n, "/", len(files), "kept", ok, flush=True)
    import pickle
    for k, v in out.items():
        with open(f"data/{k}.pkl", "wb") as f:
            pickle.dump(v, f)
        print(k, len(v), "songs", sum(len(x["D"]) // 16 for x in v), "bars")
