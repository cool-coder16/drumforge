# DrumForge

Upload a song and DrumForge writes a drum part that follows its groove. Or skip the audio and roll the dice for a random beat. Play it, edit it, save it, export it.

Everything runs in the browser (plain HTML/CSS/JS plus the Web Audio API). No build step, no dependencies, no server, and your audio is never uploaded.

## Features

- **From a song:** drop an mp3 / wav / m4a. DrumForge finds the tempo and bar positions, listens for kick-, snare- and hat-like hits, and writes a drum part that locks to them. Fills land at phrase ends, crashes where the song gets louder, and drums fade in and out with the song's energy.
- **Random beat:** pick a style (rock, pop, house, techno, boom bap, trap, drum & bass, funk, breakbeat, one-drop, euclidean), tempo and length, then hit **Roll the dice**. Seeds are shareable: **Copy share link** recreates the same beat.
- **Play:** synthesised drum kits (808 Classic, 909 Club, Studio Kit, Lo-fi Tape). In song mode the drums play on top of your track with separate volume faders.
- **Edit:** piano-roll style sequencer. Click or drag to add and erase hits, Shift for ghost notes, mute rows, play the pads with your keyboard (`A S D F G H J K L`), record pad hits live, undo, mutate.
- **Save and export:** save to a library in your browser (localStorage), or download **MIDI** (General MIDI drums) and **WAV** (drums only, or song + drums).

Generator controls: style, complexity, ghost notes, fills & crashes, how closely to follow the song, swing, humanize, kit, seed.

If the detected tempo or bar 1 is wrong, use the fix-up buttons: ÷2 / ×2 for tempo, Shift 1 beat / Shift ½ bar for the downbeat, and Nudge for tiny timing tweaks.

## How it works

1. **Analysis** (`js/analyze.js`): STFT over the song, then spectral flux (energy that was *added* since the last frame) in three bands (low 30-160 Hz, mid 200-4k Hz, high 5-16k Hz) to find onsets. Autocorrelation guesses the tempo; a comb-filter search over tempo and phase locks the beat grid; a small heuristic picks which beat is beat 1. The flux is then sampled on the 16th-note grid.
2. **Generation** (`js/generate.js`): each style is a table of probabilities for every 16th step. Dice are rolled once per bar *position*, so bars repeat like a groove (with a little jitter). Where the song has a real hit, the probability is boosted. This is a rule-based engine, not a trained ML model.
3. **Sound** (`js/synth.js`): every drum is synthesised from oscillators and filtered noise, so there are no sample files.
4. **Playback** (`js/player.js`): a look-ahead scheduler on the Web Audio clock for tight timing.

## Run locally

Any static file server works, for example:

```bash
python3 -m http.server 8000
```

then open <http://localhost:8000>. (Opening `index.html` straight from the file system also works in most browsers.)

## Limits

- Best on music with a steady tempo. Tempo-changing or free-time music will drift off the grid.
- Only the first 10 minutes of a file are analysed.
- Tempo can come out at half or double speed; use ÷2 / ×2.
- Songs with no clear drum or bass pulse (ambient, solo piano) give a vaguer result.

## Files

```
index.html        page
css/style.css     styling
js/rng.js         seeded random numbers
js/synth.js       drum kits + synth voices
js/analyze.js     FFT, onset detection, tempo + beat grid
js/generate.js    style templates + pattern generator
js/player.js      look-ahead playback scheduler
js/export.js      MIDI, WAV, localStorage library, demo song
js/roll.js        sequencer + overview canvases
js/app.js         UI wiring
```
