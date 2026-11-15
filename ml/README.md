# Training the DrumForge neural drummer

Everything needed to reproduce the models in `../models/`. Written for an Apple-silicon Mac (PyTorch on MPS); it also runs on CPU, just slower.

```
Lakh Clean MIDI ──preprocess.py──► 16th-note drum grids + features from the MIDI notes   (data/{train,val,test}.pkl)
        │
        └──render.py──► synthesised audio ──extract_one.js (the site's analyze.js)──► real analyser features   (data/audio_*.jsonl)
                                                   │
train.py     MIDI-feature pre-training, then fine-tuning with the analyser features  ──► runs/*/best.pt
downbeat.py  small MLP: which beat is beat 1                                         ──► downbeat.json
export.py    fp16 weights + config for the browser                                   ──► ../models/drummer.{bin,json}
parity_test.js   JS network == PyTorch network (logit error ~5e-6)
eval_compare.js  neural vs classic rules vs baselines on held-out test songs
```

## Setup

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install torch numpy scikit-learn mido pretty_midi
mkdir -p data && cd data
curl -O http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz && tar xzf clean_midi.tar.gz && cd ..
```

## Steps

```bash
# 1. MIDI -> grids. Keeps 4/4 songs whose drums sit on the 16th grid (~60% of files). Split by artist.
python preprocess.py

# 2. Pre-train on MIDI-derived features (~8k steps, the run was stopped there on purpose)
python train.py --steps 28000 --out runs/main            # use the step-8000 checkpoint

# 3. Render excerpts to audio and measure them with the real analyser (needs node on PATH)
python render.py --split train --n 2500 --seed 3 --out data/audio_train.jsonl   # repeat with other seeds
python render.py --split val   --n 200  --seed 1 --out data/audio_val.jsonl
python render.py --split test  --n 200  --seed 2 --out data/audio_test.jsonl

# 4. Fine-tune on analyser features (keeps the best checkpoint by audio-validation loss)
python train.py --init runs/main/ckpt_step8000_midi.pt --audio-train data/audio_train.jsonl,... \
    --audio-val data/audio_val.jsonl --dropout 0.1 --lr 4e-4 --steps 6000 --eval-every 500 --out runs/final

# 5. Downbeat model, export, checks
DBW=24 DBH1=256 DBH2=128 python downbeat.py data/audio_train.jsonl,... data/audio_val.jsonl ../models/downbeat.json
python export.py runs/final ../models        # export.py reads ckpt.pt; copy best.pt over it first
node parity_test.js runs/final ../models
node eval_compare.js ../models 200 0.9
```

## Model

`model.py`. A causal transformer (4 layers, d=192, 6 heads, ALiBi positional bias, sliding attention window of 96 steps = 6 bars). Input at step *t*: the drums of step *t-1* (9 tracks × 4 velocity classes), 34 audio features (low/mid/high onset strengths for steps *t-4..t+4*, bar loudness and its change, the bar's mean onset strengths, a has-audio flag), step-in-bar, bar-in-phrase, and style / density / tempo tokens that are randomly dropped during training so the model also works unconditioned. Output: for each of the 9 tracks, 4 classes. Tracks are decoded in order and each head sees the tracks already chosen at that step. 1.7M parameters.

Training tricks that mattered:

* **History corruption**: random steps of the drum history are blanked during training, so the model can't only copy and has to use the audio and position.
* **Feature augmentation**: scale, multiplicative noise, dropped and false onsets, on the MIDI-derived features.
* **Feature dropout**: 15% of MIDI-path samples have no audio at all (the audio-free mode).
* **Rendered-audio fine-tune**: see below.

## The domain gap, and the fix

Features computed straight from MIDI notes are clean (a hi-hat only adds "high" energy); features measured from real audio are smeared (every drum leaks into every band, the analyser is log-compressed). A model pre-trained on the clean features barely used real analyser features (validation loss 0.0630 with audio vs 0.0645 without). `render.py` fixes this at the source: it renders 24-bar excerpts of the training MIDI to audio (sine-harmonic accompaniment with timbre by GM family, synthesised drums), runs the site's own `analyze.js` on them via `extract_one.js` (with a random ±12 ms grid error), and keeps the resulting low/mid/high features next to the real drummer's part. Fine-tuning on these brought it to 0.0517 vs 0.0639.

Overfitting: fine-tuning on a few thousand excerpts overfits within a couple of epochs (validation loss on analyser features rises while training loss falls), so the script evaluates every 500 steps, keeps the best checkpoint, and uses dropout 0.1 and more rendered excerpts.

## Honest caveats

* The rendered audio is synthetic. The only real-recording test was Apple's GarageBand loops (not redistributable); the evaluation numbers below are on rendered audio of held-out songs.
* F1 against one drummer's actual part is a coarse measure. A fixed rock beat scores high on it.
* The 12 style names were assigned by hand after clustering (`export.py`, `NAMES`).
* Lakh MIDI files are user-transcribed; their quality varies.

## Results

`eval_results.json` (produced by `eval_compare.js`) and the table on the site. Downbeat: `python eval_downbeat.py ../models/downbeat.json data/audio_test.jsonl`.

Data: Lakh MIDI Dataset, Colin Raffel, CC BY 4.0.
