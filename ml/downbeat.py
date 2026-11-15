"""Learned downbeat detector: "which of the 4 beats in a bar is beat 1?"

Input : 16 consecutive beats of analyser features (each beat = 4 sixteenth steps x low/mid/high onset
        strength = 12 numbers), starting at an arbitrary beat of the song.
Output: P(position within the bar of the FIRST beat of the window) over {0,1,2,3}.
The app slides this window over the whole song and adds up the log-probabilities, which is far more
reliable than any single window.

python downbeat.py data/audio_train.jsonl,data/audio_train2.jsonl data/audio_val.jsonl out.json
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W = int(os.environ.get('DBW', 16))          # window length in beats
H1, H2 = int(os.environ.get('DBH1', 192)), int(os.environ.get('DBH2', 96))
FLOOR = 0.2


def load(paths):
    recs = [json.loads(l) for p in paths.split(",") for l in open(p)]
    fl = lambda a: np.where(a < FLOOR, 0, np.minimum(a, 1.2)).astype(np.float32)
    X = np.stack([np.stack([fl(np.array(r[k], np.float32)) for k in "LMH"], -1).reshape(-1, 12) for r in recs])   # (n, beats, 12)
    E = np.stack([np.array(r["E"], np.float32) for r in recs])
    return X, E


def windows(X, rng, B):
    ri = rng.integers(0, len(X), B)
    st = rng.integers(0, X.shape[1] - W + 1, B)
    idx = st[:, None] + np.arange(W)[None]
    w = X[ri[:, None], idx]                    # (B, W, 12)
    return w, st % 4                           # true bar start is beat 0 of every excerpt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # features of the window + the beat-to-beat change (a bar's first beat often "restarts" the pattern)
        self.f1, self.f2, self.f3 = nn.Linear(W * 12, H1), nn.Linear(H1, H2), nn.Linear(H2, 4)

    def forward(self, x):                      # x (B, W, 12)
        return self.f3(F.gelu(self.f2(F.gelu(self.f1(x.flatten(1))))))


def augment(w, rng):
    s = 0.8 + 0.4 * rng.random((len(w), 1, 1)).astype(np.float32)
    w = w * s * (1 + 0.08 * rng.standard_normal(w.shape).astype(np.float32))
    return np.clip(w, 0, 1.3)


if __name__ == "__main__":
    Xtr, _ = load(sys.argv[1])
    Xva, _ = load(sys.argv[2])
    print("train excerpts", len(Xtr), "val", len(Xva))
    rng = np.random.default_rng(0)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.02)
    steps = 6000
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * i / steps))
        w, y = windows(Xtr, rng, 256)
        loss = F.cross_entropy(net(torch.from_numpy(augment(w, rng))), torch.from_numpy(y))
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 1000 == 0 or i == steps - 1:
            with torch.no_grad():
                wv, yv = windows(Xva, np.random.default_rng(9), 4000)
                pv = net(torch.from_numpy(wv)).argmax(1).numpy()
            print(f"step {i} loss {loss.item():.3f}  val window acc {(pv == yv).mean():.3f}  "
                  f"(parity-correct {((pv % 2) == (yv % 2)).mean():.3f})", flush=True)

    # song level: slide windows over a whole excerpt, sum log-probs, pick the bar offset
    ok = ok2 = n = 0
    net.eval()
    with torch.no_grad():
        for x in Xva:
            for delta in range(4):             # pretend the detected grid is off by `delta` beats
                xs = x[delta:]                 # grid beat 0 = true beat `delta`
                starts = np.arange(0, len(xs) - W + 1, 2)
                lp = F.log_softmax(net(torch.from_numpy(np.stack([xs[s:s + W] for s in starts]))), 1).numpy()
                # window starting at grid beat j has true position (j + delta) % 4
                score = [sum(lp[k, (starts[k] + d) % 4] for k in range(len(starts))) for d in range(4)]
                best = int(np.argmax(score))   # best offset d: true position = (j + d) % 4  ->  d == delta
                ok += best == delta; ok2 += (best % 2) == (delta % 2); n += 1
    print(f"SONG-LEVEL downbeat accuracy on held-out excerpts: {ok / n:.3f}  (right parity {ok2 / n:.3f}, chance 0.25)")

    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    out = {k: dict(shape=list(v.shape), data=np.round(v.flatten(), 4).tolist()) for k, v in sd.items()}
    json.dump(dict(window=W, floor=FLOOR, h1=H1, h2=H2, tensors=out, song_acc=ok / n), open(sys.argv[3], "w"), separators=(",", ":"))
    print("wrote", sys.argv[3])
