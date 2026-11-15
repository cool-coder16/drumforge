# DrumForge

Upload a song and a neural network writes a drum part that follows its groove. Or skip the audio and roll the dice for a random beat. Play it, edit it, save it, export it.

**Live:** https://cool-coder16.github.io/drumforge/

Everything runs in the browser (plain HTML/CSS/JS plus the Web Audio API). No build step, no server, no ML library, and your audio is never uploaded.

## The AI

The default **AI drummer** is a small transformer (4 layers, 1.7M parameters) that I trained for this project and that runs in your browser in plain JavaScript.

| | |
|---|---|
| What it does | Writes the drum part one 16th note at a time: kick, snare, clap, hats, toms, crash and percussion together, with 4 velocity levels (off / ghost / normal / accent). |
| What it sees | The song's low / mid / high onset strengths around the step, the bar's loudness, the drums it has already played, its position in the bar and in an 8-bar phrase, and optional style / density / tempo tokens. |
| Training data | The [Lakh MIDI dataset](https://colinraffel.com/projects/lmd/) (Clean subset): 17,243 files, of which 10,614 were usable (4/4, drums on the 16th-note grid). Split by artist: 9,523 train / 558 validation / 533 test songs (about 1.04M training bars). |
| Training | Stage 1: 8,000 steps on features computed directly from the MIDI. Stage 2: fine-tuning (dropout 0.1, best checkpoint kept by validation loss) on 12,500 24-bar excerpts that were **rendered to audio and analysed by this page's own analyser**, so the network sees the same kind of features it gets from your mp3 (see "the domain gap" below). |
| Styles | The 12 selectable styles were not written by hand: they are k-means clusters of the drum patterns of the training songs, named afterwards by looking at their average pattern. |
| Downbeat | A second, tiny neural network (an MLP) decides which beat is beat 1 (80% right on held-out test excerpts). |
| Drums or not | A logistic regression on the analyser features decides whether the song already contains drums (90% on held-out test excerpts). Songs with drums are followed closely; for drumless songs the network writes from its own groove sense and only uses the song's loudness for silences and build-ups. A selector lets you override it. |

Run the **Classic rules** engine instead (the switch at the top of the controls) for the original hand-written probability tables.

### Results

See `ml/eval_results.json` and the table at the bottom of the live page (it is rendered from `models/eval.json`). Numbers are F1 on drum hits against what the real drummer played, on held-out **test** songs by artists the model never saw, with only the song available as input.

| System | No drums in the song: overall / kick / snare / hats | Drums in the mix: overall / kick / snare / hats |
|---|---|---|
| naive fixed rock beat (ignores the song) | 0.53 / 0.57 / 0.64 / 0.66 | 0.56 / 0.56 / 0.62 / 0.72 |
| classic rules engine | 0.42 / 0.50 / 0.41 / 0.52 | 0.52 / 0.60 / 0.48 / 0.67 |
| neural drummer, audio switched OFF | 0.54 / 0.58 / 0.63 / 0.66 | 0.55 / 0.56 / 0.60 / 0.71 |
| neural drummer, always listening | 0.46 / 0.57 / 0.52 / 0.56 | 0.63 / 0.70 / 0.73 / 0.73 |
| neural drummer, adaptive (the app) | 0.53 / 0.58 / 0.63 / 0.66 | 0.63 / 0.69 / 0.71 / 0.73 |

(107 drumless and 93 with-drums test excerpts of 24 bars; sampling temperature 0.7, the app's default.)

Notes on reading this honestly:

* On drumless songs the network does **not** beat a fixed rock beat on this metric (it ties it, and clearly beats the rule engine): accompaniment alone says very little about what a drummer would play, and the network's own held-out loss shows it (0.0520 with the accompaniment features vs 0.0530 without). It is clearly better when the song already has drums, which is the common case for an uploaded mp3.
* A fixed rock beat scores high because real drummers play a lot of rock. Beating it is a real bar to clear; reproducing a specific drummer's part exactly is not the goal of the app, so treat F1 as a sanity check rather than a quality score.
* Downbeat detection (which beat is beat 1) is correct on 80% of held-out 24-bar excerpts (92% get the parity right, chance is 25% / 50%). If bar 1 is in the wrong place, use the **Shift** buttons.
* The audio-free mode samples several short candidate grooves from the network, filters dull ones by hit count and by the network's own likelihood, and keeps one at random; fills are sampled from the network at phrase ends the same way. It is the network's own output, but the selection step is mine.

### The domain gap (the main thing that went wrong, and how it was fixed)

A model trained only on features computed from MIDI note lists (where a hi-hat touches only the "high" channel and a kick only the "low" one) barely used the audio at all when given real analyser features (validation loss 0.0630 with audio vs 0.0645 without), because real audio smears every drum across every band. The fix was to render training excerpts to audio (simple synthesised accompaniment and drums), run the browser's own `analyze.js` over them, and fine-tune on those features: 0.0494 with audio vs 0.0626 without (best checkpoint, held-out validation excerpts). `ml/render.py` and `ml/extract_one.js` do this.

## Features

- **From a song:** drop an mp3 / wav / m4a. DrumForge finds the tempo and bar positions, then the AI writes a drum part for it. Silent bars stay silent, and fills and crashes land at phrase ends and build-ups.
- **Random beat:** pick a learned style (or "surprise me"), tempo and length, then **Roll the dice**. Seeds are shareable: **Copy share link** recreates the same beat.
- **Controls:** complexity (hits per bar), ghost notes, fills, how closely to follow the song, creativity (sampling temperature), swing, humanize, kit, seed.
- **Play:** synthesised drum kits (808 Classic, 909 Club, Studio Kit, Lo-fi Tape). In song mode the drums play on top of your track.
- **Edit:** piano-roll sequencer, click or drag to add and erase hits (Shift = ghost note), mute rows, keyboard pads (`A S D F G H J K L`), live pad recording, undo, mutate.
- **Save and export:** save to a browser library (localStorage), or download **MIDI** (General MIDI drums) and **WAV** (drums only, or song + drums).

If the detected tempo or bar 1 is wrong, use ÷2 / ×2 for tempo, Shift 1 beat / Shift ½ bar for the downbeat, and Nudge for tiny timing tweaks.

## How it works

1. **Analysis** (`js/analyze.js`): STFT, then spectral flux in three bands (low 30-160 Hz, mid 200-4k Hz, high 5-16k Hz) to find onsets; autocorrelation + a comb-filter search lock the tempo and beat grid; the flux is sampled on the 16th-note grid.
2. **Downbeat** (`js/ai.js`, `models/downbeat.json`): an MLP slides over 24-beat windows and votes for the bar offset.
3. **Drum generation** (`js/ai.js`, `models/drummer.bin`): the transformer, with a sliding-window KV cache so a whole song is cheap. About 1.5-4 ms per 16th note on a laptop CPU (a few seconds for a whole song; the app shows progress).
4. **Sound** (`js/synth.js`): every drum is synthesised from oscillators and filtered noise; no sample files.
5. **Playback** (`js/player.js`): a look-ahead scheduler on the Web Audio clock.

The JavaScript network is checked against the PyTorch original: `ml/parity_test.js` compares the logits (max error about 5e-6).

## Reproducing the training

All the code is in `ml/` (see `ml/README.md`). In short: download the Lakh Clean MIDI archive, `preprocess.py`, `render.py`, `train.py`, `downbeat.py`, `export.py`. Training (both stages) took well under an hour of GPU time on an Apple M3, plus about 45 minutes of CPU time to render and analyse the excerpts.

## Run locally

```bash
python3 -m http.server 8000
```

then open <http://localhost:8000>.

## Limits

- Best on music with a steady tempo. Tempo-changing or free-time music will drift off the grid.
- The network was trained on 4/4 pop / rock / dance MIDI. Swung, triplet and odd-meter music is outside its data.
- Its sense of the song is the three-band onset features above: it can't hear lyrics, timbre or instrumentation. The rendered training audio is synthetic, so real recordings are always a (small) step from what it practised on.
- Only the first 10 minutes of a file are analysed. Tempo can come out at half or double speed; use ÷2 / ×2.
- Audio-free beats mostly repeat the groove they open with, as real drum parts do. Fills and variation come from the sampling step described above.

## Credits

Trained on the Lakh MIDI Dataset (Colin Raffel, CC BY 4.0). Evaluation on real recordings used Apple's built-in GarageBand loops, which are not included in this repository.
