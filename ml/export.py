"""Export a trained checkpoint for the browser.

python export.py runs/main ../models

Writes   drummer.bin   all weights as float16, concatenated
         drummer.json  config + tensor index + style names + bucket cuts
         parity.json   (in the run dir) inputs and PyTorch logits for js/ai.js to be checked against
"""
import json, sys
import numpy as np
import torch
from model import Drummer, N_FEAT, N_TRACKS, N_CLASSES

run, out = sys.argv[1], sys.argv[2]
ck = torch.load(f"{run}/ckpt.pt", map_location="cpu")
meta = json.load(open(f"{run}/meta.json"))
model = Drummer(**ck["cfg"])
model.load_state_dict(ck["state"])
model.eval()

# Names for the discovered style clusters (k-means on drum fingerprints of 9.5k songs; named by hand
# from each cluster's average kick/snare/hat pattern). Cluster ids are stable for a given dataset + seed.
NAMES = {0: "Pop rock + shaker", 1: "Half-time hats", 2: "Funk-pop", 3: "Laid-back rock", 4: "Soft pop", 5: "Sparse & jazzy",
         6: "Club (offbeat open hats)", 7: "16th-hat groove", 8: "Disco four-on-the-floor", 9: "Busy 16ths funk",
         10: "Classic rock", 11: "Offbeat open hat"}

# fp16 round trip so the parity vector is computed with exactly the weights the browser will use
sd = {k: v.half().float() for k, v in ck["state"].items()}
model.load_state_dict(sd)

tensors, blob, off = [], [], 0
for name, t in ck["state"].items():
    a = t.detach().cpu().numpy().astype(np.float16)
    tensors.append(dict(name=name, shape=list(a.shape), offset=off))
    blob.append(a.tobytes())
    off += a.size
open(f"{out}/drummer.bin", "wb").write(b"".join(blob))
cfg = dict(ck["cfg"])
cfg.update(n_feat=N_FEAT, tensors=tensors, window=ck["cfg"]["window"], dens_cuts=meta["dens_cuts"], tempo_cuts=meta["tempo_cuts"],
           styles=[dict(id=s["id"], name=NAMES.get(s["id"], f"Style {s['id']}"), songs=s["songs"], bpm=s["bpm"],
                     proto={k: s[k] for k in ("kick", "snare", "clap", "hat", "ohat", "tom", "crash", "perc")}) for s in meta["styles"]],
           train_step=ck.get("step", 0))
json.dump(cfg, open(f"{out}/drummer.json", "w"))
print("wrote", out, f"{off * 2 / 1e6:.2f} MB weights,", len(tensors), "tensors, trained steps", ck.get("step"))

# ---- parity vector: a fixed teacher-forced run ----
g = torch.Generator().manual_seed(7)
T = 40
target = torch.randint(0, N_CLASSES, (1, T, N_TRACKS), generator=g) * (torch.rand(1, T, N_TRACKS, generator=g) < 0.3)
prev = torch.cat([torch.zeros(1, 1, N_TRACKS, dtype=torch.long), target[:, :-1]], 1)
feat = torch.rand(1, T, N_FEAT, generator=g) * (torch.rand(1, T, N_FEAT, generator=g) < 0.5)
sib = torch.arange(T).remainder(16).unsqueeze(0)
bip = (torch.arange(T) // 16).remainder(8).unsqueeze(0)
style, dens, tempo = torch.tensor([3]), torch.tensor([5]), torch.tensor([2])
with torch.no_grad():
    logits = model(prev, feat, sib, bip, style, dens, tempo, target)
json.dump(dict(T=T, prev=prev[0].tolist(), target=target[0].tolist(), feat=feat[0].tolist(), style=3, dens=5, tempo=2,
               logits=logits[0].tolist()), open(f"{run}/parity.json", "w"))
print("parity vector written")
