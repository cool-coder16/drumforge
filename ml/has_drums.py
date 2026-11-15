"""Does this song already contain drums? A logistic regression on 56 numbers from the analyser features.

For drumless songs the network's own groove sense beats following the (weakly informative) accompaniment,
so the app switches strategy based on this. Trained on rendered excerpts whose label is known (variant nd/wd).

python has_drums.py data/audio_train.jsonl,... data/audio_val.jsonl data/audio_test.jsonl ../models/has_drums.json
(js/ai.js `detectDrums` recomputes exactly these features.)
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

FL = 0.2


def feats(r):
    a = np.stack([np.array(r[k], np.float32) for k in "LMH"], -1)
    a = np.where(a < FL, 0, np.minimum(a, 1.2))
    bars = len(a) // 16
    b = a[: bars * 16].reshape(bars, 16, 3)
    pat = b.mean(0).reshape(-1)                      # the average bar: 16 steps x (low, mid, high)  -> 48 numbers
    on = b.mean((0, 1))                              # overall onset density per band               -> 3
    rep = np.mean([np.corrcoef(b[i].reshape(-1), b[i + 1].reshape(-1))[0, 1]
                   if b[i].std() > 0 and b[i + 1].std() > 0 else 0 for i in range(bars - 1)])   # bar-to-bar repetition
    hb = b[:, :, 2]
    return np.concatenate([pat, on, [rep, b[:, ::4, 0].mean() - b[:, 2::4, 0].mean(),   # kick on beats vs off-beats
                                     b[:, 4::8, 1].mean() - b[:, 0::8, 1].mean(),        # backbeat in the mid band
                                     hb.mean(), hb[:, ::2].mean() - hb[:, 1::2].mean()]])  # hat-band density / even-odd


def load(paths):
    R = [json.loads(l) for p in paths.split(",") for l in open(p)]
    return np.stack([feats(r) for r in R]), np.array([r["variant"] == "wd" for r in R])


if __name__ == "__main__":
    Xtr, ytr = load(sys.argv[1]); Xva, yva = load(sys.argv[2]); Xte, yte = load(sys.argv[3])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=5000, C=0.5).fit((Xtr - mu) / sd, ytr)
    acc = lambda X, y: float((clf.predict((X - mu) / sd) == y).mean())
    print(f"train {len(ytr)} excerpts | validation accuracy {acc(Xva, yva):.3f} | TEST accuracy {acc(Xte, yte):.3f}")
    json.dump(dict(mu=np.round(mu, 5).tolist(), sd=np.round(sd, 5).tolist(), w=np.round(clf.coef_[0], 5).tolist(), b=float(clf.intercept_[0]),
                   val_acc=acc(Xva, yva), test_acc=acc(Xte, yte)), open(sys.argv[4], "w"), separators=(",", ":"))
