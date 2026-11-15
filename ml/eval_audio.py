"""Score a checkpoint on held-out excerpts whose features come from the real analyser.
python eval_audio.py runs/main/ckpt.pt data/audio_val.jsonl"""
import sys, numpy as np, torch
from sklearn.cluster import KMeans
from model import Drummer
import train as T

ck = torch.load(sys.argv[1], map_location="cpu")
train_songs, val_songs = T.load("train"), T.load("val")
km = KMeans(ck["cfg"]["n_styles"], n_init=4, random_state=0).fit(np.stack([T.song_vec(s) for s in train_songs if len(s["D"]) >= T.T]))
tr = T.Data(train_songs, km)
model = Drummer(**ck["cfg"]).to(T.dev)
model.load_state_dict(ck["state"])
ad = T.AudioData(sys.argv[2], km, val_songs, tr.dens_cuts, ck["cfg"]["n_styles"])
print("checkpoint step", ck.get("step"), "| excerpts", ad.n, "| variants nd/wd:", (ad.variant == "nd").sum(), (ad.variant == "wd").sum())
T.evaluate_audio(model, ad, "snapshot", n_batches=20)
