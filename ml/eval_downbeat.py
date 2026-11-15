"""Song-level downbeat accuracy of a shipped downbeat.json on held-out excerpts.
python eval_downbeat.py ../models/downbeat.json data/audio_test.jsonl"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
import downbeat as D

m = json.load(open(sys.argv[1]))
D.W, D.H1, D.H2 = m["window"], m["h1"], m["h2"]
net = D.Net()
net.load_state_dict({k: torch.tensor(v["data"], dtype=torch.float32).view(v["shape"]) for k, v in m["tensors"].items()})
net.eval()
X, _ = D.load(sys.argv[2])
ok = ok2 = n = 0
with torch.no_grad():
    for x in X:
        for delta in range(4):
            xs = x[delta:]
            starts = np.arange(0, len(xs) - D.W + 1, 2)
            lp = F.log_softmax(net(torch.from_numpy(np.stack([xs[s:s + D.W] for s in starts]))), 1).numpy()
            score = [sum(lp[k, (starts[k] + d) % 4] for k in range(len(starts))) for d in range(4)]
            best = int(np.argmax(score))
            ok += best == delta; ok2 += (best % 2) == (delta % 2); n += 1
print(f"{len(X)} held-out excerpts: downbeat accuracy {ok / n:.3f} (right parity {ok2 / n:.3f}; chance 0.25 / 0.50)")
