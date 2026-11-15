"""Train the DrumForge neural drummer on Lakh Clean MIDI (see preprocess.py).

python train.py --steps 20000 --out runs/main
"""
import argparse, json, math, os, pickle, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from model import Drummer, FEAT_OFFSETS, N_TRACKS, N_CLASSES, N_FEAT

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T = 192                      # training window = 12 bars
OFFS = np.array(FEAT_OFFSETS)
TEMPO_CUTS = [80, 100, 120, 140, 170]


def load(split):
    with open(f"data/{split}.pkl", "rb") as f:
        return pickle.load(f)


def song_vec(s):
    bars = len(s["D"]) // 16
    return (s["D"][: bars * 16].reshape(bars, 16, N_TRACKS) > 0).mean(0).reshape(-1)


class Data:
    def __init__(self, songs, kmeans, dens_cuts=None):
        songs = [s for s in songs if len(s["D"]) >= T]
        self.n = len(songs)
        self.len = np.array([len(s["D"]) for s in songs])
        self.bars = self.len // 16
        self.off = np.concatenate([[0], np.cumsum(self.len)[:-1]])
        self.boff = np.concatenate([[0], np.cumsum(self.bars)[:-1]])
        self.D = np.concatenate([s["D"] for s in songs])
        self.Fnd = np.concatenate([s["F_nd"] for s in songs])
        self.Fd = np.concatenate([np.maximum(s["F_d"], s["F_nd"]) for s in songs])   # drums-included variant
        self.End = np.concatenate([s["E_nd"][: b] for s, b in zip(songs, self.bars)])
        self.Ed = np.concatenate([s["E_d"][: b] for s, b in zip(songs, self.bars)])
        mean_bar = lambda F, b: F[: b * 16].reshape(b, 16, 3).mean(1)
        self.Bnd = np.concatenate([mean_bar(s["F_nd"].astype(np.float32), b) for s, b in zip(songs, self.bars)])
        self.Bd = np.concatenate([mean_bar(np.maximum(s["F_d"], s["F_nd"]).astype(np.float32), b) for s, b in zip(songs, self.bars)])
        self.bpm = np.array([float(s["bpm"]) for s in songs])
        self.tempo = np.digitize(self.bpm, TEMPO_CUTS)
        self.style = kmeans.predict(np.stack([song_vec(s) for s in songs]))
        self.dens_cuts = dens_cuts if dens_cuts is not None else self._dens_cuts()

    def _dens_cuts(self):
        rng = np.random.default_rng(0)
        _, _, D, *_ = self._raw(4000, rng)
        hpb = (D > 0).sum((1, 2)) / (T // 16)
        return np.quantile(hpb, np.linspace(0, 1, 9)[1:-1])   # 7 cut points -> 8 buckets

    def _raw(self, B, rng):
        w = np.maximum(self.bars - T // 16 + 1, 0).astype(float)
        si = rng.choice(self.n, B, p=w / w.sum())
        sb = rng.integers(0, self.bars[si] - T // 16 + 1)
        g = (self.off[si] + sb * 16)[:, None] + np.arange(T)[None]
        return si, sb, self.D[g], g

    def batch(self, B, rng, train=True, force=None):
        """force: None | dict(no_audio=bool, inc_drums=bool) to pin the audio condition (for evaluation)."""
        si, sb, Dw, g = self._raw(B, rng)
        lo, hi = self.off[si], self.off[si] + self.len[si] - 1
        # previous-step drums (true previous step when the window doesn't start at the song start)
        p0 = np.where((sb > 0)[:, None], self.D[np.maximum(g[:, 0] - 1, 0)], 0)
        prev = np.concatenate([p0[:, None], Dw[:, :-1]], 1)
        gi = np.clip(g[:, :, None] + OFFS[None, None, :], lo[:, None, None], hi[:, None, None])
        gb = self.boff[si][:, None] + sb[:, None] + (np.arange(T) // 16)[None]
        blo, bhi = self.boff[si][:, None], (self.boff[si] + self.bars[si] - 1)[:, None]
        gbn, gbp = np.minimum(gb + 1, bhi), np.maximum(gb - 1, blo)

        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        Fnd, Fd = t(self.Fnd[gi]).float(), t(self.Fd[gi]).float()       # (B,T,9,3)
        End, Ed = t(self.End[gb]).float(), t(self.Ed[gb]).float()
        Endn, Edn, Endp, Edp = t(self.End[gbn]).float(), t(self.Ed[gbn]).float(), t(self.End[gbp]).float(), t(self.Ed[gbp]).float()
        Bnd, Bd = t(self.Bnd[gb]).float(), t(self.Bd[gb]).float()

        r = torch.rand
        inc = r(B, device=dev) < 0.5
        no_audio = r(B, device=dev) < 0.15
        if force is not None:
            inc = torch.full((B,), force["inc_drums"], device=dev)
            no_audio = torch.full((B,), force["no_audio"], device=dev)
        sel = lambda a, b, extra=0: torch.where(inc.view(B, *([1] * (a.dim() - 1))), b, a)
        Fw, E, En, Ep, Fbar = sel(Fnd, Fd), sel(End, Ed), sel(Endn, Edn), sel(Endp, Edp), sel(Bnd, Bd)
        if train:   # make the features look like noisy real-audio analysis
            s = 0.6 + 0.7 * r(B, 1, 1, 1, device=dev)
            Fw = Fw * s * (1 + 0.12 * torch.randn_like(Fw))
            Fw = Fw * (r(*Fw.shape, device=dev) > 0.05) + (r(*Fw.shape, device=dev) < 0.03) * (0.2 + 0.5 * r(*Fw.shape, device=dev))
            Fw = Fw.clamp(0, 1.3)
            Fbar = Fbar * s.view(B, 1, 1)
            jit = lambda e: (e * (0.9 + 0.2 * r(B, 1, device=dev)) + 0.03 * torch.randn_like(e)).clamp(0, 1)
            E, En, Ep = jit(E), jit(En), jit(Ep)
        keep = (~no_audio).float()
        feat = torch.cat([Fw.flatten(2) * keep.view(B, 1, 1), ((E)).unsqueeze(-1) * keep.view(B, 1, 1),
                          (En - E).unsqueeze(-1) * keep.view(B, 1, 1), (E - Ep).unsqueeze(-1) * keep.view(B, 1, 1),
                          Fbar * keep.view(B, 1, 1), keep.view(B, 1, 1).expand(B, T, 1)], -1)
        assert feat.shape[-1] == N_FEAT

        # conditioning tokens, randomly dropped so the model also works "unconditioned"
        hpb = (Dw > 0).sum((1, 2)) / (T // 16)
        dens = np.digitize(hpb, self.dens_cuts)
        style = torch.from_numpy(self.style[si]).to(dev)
        n_styles = self.n_styles
        style = torch.where(r(B, device=dev) < 0.5, torch.full_like(style, n_styles), style)
        dens = torch.where(r(B, device=dev) < 0.5, torch.full((B,), 8, device=dev), t(dens))
        tempo = torch.where(r(B, device=dev) < 0.3, torch.full((B,), 6, device=dev), t(self.tempo[si]))
        sib = t(np.tile(np.arange(T) % 16, (B, 1)))
        bip = t((sb[:, None] + np.arange(T)[None] // 16) % 8)
        prev = t(prev).long()
        if train:   # blank random steps of the drum history so the model must use audio/position cues too
            q = 0.45 * r(B, 1, device=dev) * (r(B, 1, device=dev) < 0.6)
            prev = prev * (~(r(B, T, device=dev) < q)).long().unsqueeze(-1)
        return dict(prev=prev, feat=feat, sib=sib, bip=bip, style=style, dens=dens.long(), tempo=tempo.long(), target=t(Dw).long())


def run(model, b):
    return model(b["prev"], b["feat"], b["sib"], b["bip"], b["style"], b["dens"], b["tempo"], b["target"])


@torch.no_grad()
def evaluate(model, data, label, n_batches=12, B=32):
    model.eval()
    res = {}
    for name, force in [("full(nodrums)", dict(no_audio=False, inc_drums=False)), ("full(+drums)", dict(no_audio=False, inc_drums=True)),
                        ("no-audio", dict(no_audio=True, inc_drums=False))]:
        rng = np.random.default_rng(123)
        nll, early, tp, fp, fn = 0.0, 0.0, np.zeros(N_TRACKS), np.zeros(N_TRACKS), np.zeros(N_TRACKS)
        for _ in range(n_batches):
            b = data.batch(B, rng, train=False, force=force)
            lg = run(model, b)
            nll += F.cross_entropy(lg.reshape(-1, N_CLASSES), b["target"].reshape(-1)).item()
            early += F.cross_entropy(lg[:, :32].reshape(-1, N_CLASSES), b["target"][:, :32].reshape(-1)).item()
            pred, tgt = lg.argmax(-1) > 0, b["target"] > 0
            tp += (pred & tgt).sum((0, 1)).cpu().numpy(); fp += (pred & ~tgt).sum((0, 1)).cpu().numpy(); fn += (~pred & tgt).sum((0, 1)).cpu().numpy()
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        res[name] = (nll / n_batches, f1, early / n_batches)
    print(f"[{label}] " + " | ".join(f"{k}: nll {v[0]:.4f} (first-2-bars {v[2]:.4f}) F1 kick {v[1][0]:.2f} snare {v[1][1]:.2f} hat {v[1][3]:.2f}" for k, v in res.items()), flush=True)
    model.train()
    return res


FLOOR = 0.2   # js/ai.js zeroes features below this before feeding the network; training mirrors it


class AudioData:
    """Excerpts whose features came from the real browser analyser run on rendered audio (see render.py)."""

    def __init__(self, path, kmeans, songs_split, dens_cuts, n_styles):
        recs = [json.loads(l) for p_ in path.split(",") for l in open(p_)]
        self.n, self.n_styles, self.dens_cuts = len(recs), n_styles, dens_cuts
        self.nb = len(recs[0]["E"])
        self.ns = self.nb * 16
        self.D = np.stack([np.array(r["D"], np.uint8) for r in recs])
        fl = lambda a: np.where(a < FLOOR, 0, np.minimum(a, 1.2)).astype(np.float32)
        self.F = np.stack([np.stack([fl(np.array(r[k], np.float32)) for k in "LMH"], -1) for r in recs])   # (n, steps, 3)
        self.E = np.stack([np.array(r["E"], np.float32) for r in recs])
        self.Fbar = self.F.reshape(self.n, self.nb, 16, 3).mean(2)
        style_all = kmeans.predict(np.stack([song_vec(s) for s in songs_split]))
        self.style = np.array([style_all[r["k"]] for r in recs])
        self.tempo = np.digitize([r["bpm"] for r in recs], TEMPO_CUTS)
        self.b0 = np.array([r["b0"] for r in recs])
        self.variant = np.array([r["variant"] for r in recs])

    def batch(self, B, rng, train=True, force_no_audio=None):
        ri = rng.integers(0, self.n, B)
        sb = rng.integers(0, self.nb - T // 16 + 1, B)
        g = (sb * 16)[:, None] + np.arange(T)[None]
        Dw = self.D[ri[:, None], g]
        p0 = np.where((sb > 0)[:, None], self.D[ri, np.maximum(sb * 16 - 1, 0)], 0)
        prev = np.concatenate([p0[:, None], Dw[:, :-1]], 1)
        gi = np.clip(g[:, :, None] + OFFS[None, None, :], 0, self.ns - 1)
        gb = sb[:, None] + (np.arange(T) // 16)[None]
        gbn, gbp = np.minimum(gb + 1, self.nb - 1), np.maximum(gb - 1, 0)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        Fw = t(self.F[ri[:, None, None], gi]).float()
        E, En, Ep = (t(self.E[ri[:, None], x]).float() for x in (gb, gbn, gbp))
        Fbar = t(self.Fbar[ri[:, None], gb]).float()
        r = torch.rand
        if train:
            s = 0.85 + 0.3 * r(B, 1, 1, 1, device=dev)
            Fw = (Fw * s * (1 + 0.05 * torch.randn_like(Fw))).clamp(0, 1.3)
            Fbar = Fbar * s.view(B, 1, 1)
            jit = lambda e: (e * (0.95 + 0.1 * r(B, 1, device=dev))).clamp(0, 1)
            E, En, Ep = jit(E), jit(En), jit(Ep)
        no_audio = (r(B, device=dev) < 0.1) if force_no_audio is None else torch.full((B,), force_no_audio, device=dev)
        keep = (~no_audio).float().view(B, 1, 1)
        feat = torch.cat([Fw.flatten(2) * keep, E.unsqueeze(-1) * keep, (En - E).unsqueeze(-1) * keep, (E - Ep).unsqueeze(-1) * keep,
                          Fbar * keep, keep.expand(B, T, 1)], -1)
        hpb = (Dw > 0).sum((1, 2)) / (T // 16)
        style = torch.from_numpy(self.style[ri]).to(dev)
        style = torch.where(r(B, device=dev) < 0.5, torch.full_like(style, self.n_styles), style)
        dens = torch.where(r(B, device=dev) < 0.5, torch.full((B,), 8, device=dev), t(np.digitize(hpb, self.dens_cuts)))
        tempo = torch.where(r(B, device=dev) < 0.3, torch.full((B,), 6, device=dev), t(self.tempo[ri]))
        prev = t(prev).long()
        if train:
            q = 0.45 * r(B, 1, device=dev) * (r(B, 1, device=dev) < 0.6)
            prev = prev * (~(r(B, T, device=dev) < q)).long().unsqueeze(-1)
        return dict(prev=prev, feat=feat, sib=t(np.tile(np.arange(T) % 16, (B, 1))), bip=t((self.b0[ri][:, None] + sb[:, None] + np.arange(T)[None] // 16) % 8),
                    style=style, dens=dens.long(), tempo=tempo.long(), target=t(Dw).long())


@torch.no_grad()
def evaluate_audio(model, adata, label, n_batches=10, B=32):
    model.eval()
    out = []
    for name, na in (("audio features", False), ("no audio", True)):
        rng = np.random.default_rng(321)
        nll, tp, fp, fn = 0.0, np.zeros(N_TRACKS), np.zeros(N_TRACKS), np.zeros(N_TRACKS)
        for _ in range(n_batches):
            b = adata.batch(B, rng, train=False, force_no_audio=na)
            lg = run(model, b)
            nll += F.cross_entropy(lg.reshape(-1, N_CLASSES), b["target"].reshape(-1)).item()
            pred, tgt = lg.argmax(-1) > 0, b["target"] > 0
            tp += (pred & tgt).sum((0, 1)).cpu().numpy(); fp += (pred & ~tgt).sum((0, 1)).cpu().numpy(); fn += (~pred & tgt).sum((0, 1)).cpu().numpy()
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        out.append((nll / n_batches, f1))
        print(f"[{label}] REAL-ANALYSER {name}: nll {nll / n_batches:.4f} F1 kick {f1[0]:.2f} snare {f1[1]:.2f} hat {f1[3]:.2f}", flush=True)
    model.train()
    return out


def cat_batches(a, b):
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


def position_baseline(train_data, val_data):
    """Best model that only knows (step-in-bar, track): per-position class frequencies."""
    counts = np.ones((16, N_TRACKS, N_CLASSES))
    D = train_data.D[: (len(train_data.D) // 16) * 16].reshape(-1, 16, N_TRACKS)
    for c in range(N_CLASSES):
        counts[:, :, c] += (D == c).sum(0)
    p = counts / counts.sum(-1, keepdims=True)
    rng = np.random.default_rng(123)
    tot = 0.0
    for _ in range(12):
        _, _, Dw, _ = val_data._raw(32, rng)
        pos = np.arange(T) % 16
        pr = p[pos[None, :, None], np.arange(N_TRACKS)[None, None, :], Dw]
        tot += -np.log(pr).mean()
    return tot / 12


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--styles", type=int, default=12)
    ap.add_argument("--init", default="", help="start from this checkpoint (fine-tuning)")
    ap.add_argument("--audio-train", default="")
    ap.add_argument("--audio-val", default="")
    ap.add_argument("--audio-mix", type=float, default=0.5, help="share of each batch drawn from analyser-feature excerpts")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--minutes", type=float, default=0, help="stop early after this many minutes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    train_songs, val_songs = load("train"), load("val")
    print(len(train_songs), "train songs,", len(val_songs), "val songs; device", dev, flush=True)
    km = KMeans(a.styles, n_init=4, random_state=0).fit(np.stack([song_vec(s) for s in train_songs if len(s["D"]) >= T]))
    train, val = Data(train_songs, km), None
    train.n_styles = a.styles
    val = Data(val_songs, km, train.dens_cuts)
    val.n_styles = a.styles
    print("position-only baseline NLL:", round(position_baseline(train, val), 4), flush=True)

    # describe the discovered styles (used to name them in the app)
    styles = []
    for k in range(a.styles):
        m = train.style == k
        vecs = np.stack([song_vec(s) for s, ok in zip([s for s in train_songs if len(s["D"]) >= T], m) if ok])
        mv = vecs.mean(0).reshape(16, N_TRACKS)
        styles.append(dict(id=k, songs=int(m.sum()), bpm=float(np.median(train.bpm[m])), kick=mv[:, 0].round(2).tolist(),
                           snare=mv[:, 1].round(2).tolist(), hat=mv[:, 3].round(2).tolist(), ohat=mv[:, 4].round(2).tolist(),
                           clap=mv[:, 2].round(2).tolist(), tom=(mv[:, 5] + mv[:, 6]).round(2).tolist(), crash=mv[:, 7].round(2).tolist(),
                           perc=mv[:, 8].round(2).tolist(), density=float(mv.sum())))
    meta = dict(styles=styles, dens_cuts=train.dens_cuts.tolist(), tempo_cuts=TEMPO_CUTS, window=96)
    json.dump(meta, open(f"{a.out}/meta.json", "w"))

    model = Drummer(d=a.d, n_layers=a.layers, n_heads=6, ff=a.d * 3, window=96, n_styles=a.styles, dropout=a.dropout).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["state"])
        print("initialised from", a.init, flush=True)
    print("params:", sum(p.numel() for p in model.parameters()), flush=True)
    atrain = AudioData(a.audio_train, km, train_songs, train.dens_cuts, a.styles) if a.audio_train else None
    aval = AudioData(a.audio_val, km, val_songs, train.dens_cuts, a.styles) if a.audio_val else None
    if aval is not None:
        evaluate_audio(model, aval, "start")
    if atrain is not None:
        print("audio excerpts:", atrain.n, "train /", aval.n if aval else 0, "val", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    sched = lambda i: min(1, (i + 1) / 300) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, i / a.steps))))
    rng = np.random.default_rng(1)
    t0, ema, best_nll = time.time(), None, 1e9
    for i in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(i)
        if atrain is not None:
            na = int(round(a.batch * a.audio_mix))
            b = cat_batches(train.batch(a.batch - na, rng), atrain.batch(na, rng))
        else:
            b = train.batch(a.batch, rng)
        loss = F.cross_entropy(run(model, b).reshape(-1, N_CLASSES), b["target"].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if i % 100 == 0:
            print(f"step {i} loss {ema:.4f} lr {a.lr * sched(i):.2e} {(time.time() - t0) / 60:.1f}min", flush=True)
        if (i + 1) % a.eval_every == 0 or i == a.steps - 1:
            evaluate(model, val, f"step {i + 1}")
            if aval is not None:
                res = evaluate_audio(model, aval, f"step {i + 1}")
                if res[0][0] < best_nll:       # keep the checkpoint that is best on real-analyser validation data
                    best_nll = res[0][0]
                    torch.save(dict(cfg=model.cfg, state=model.state_dict(), step=i + 1), f"{a.out}/best.pt")
                    print(f"  new best (audio-val nll {best_nll:.4f}) at step {i + 1}", flush=True)
            torch.save(dict(cfg=model.cfg, state=model.state_dict(), step=i + 1), f"{a.out}/ckpt.pt")
        if a.minutes and (time.time() - t0) / 60 > a.minutes:
            print("time budget reached", flush=True)
            evaluate(model, val, f"final step {i + 1}")
            torch.save(dict(cfg=model.cfg, state=model.state_dict(), step=i + 1), f"{a.out}/ckpt.pt")
            break
    print("done", flush=True)
