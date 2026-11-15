// Tiny shared helpers: a seeded random number generator and string hashing.
// Seeded = the same seed always gives the same "random" numbers, which is
// what lets a seed (and a share link) reproduce the exact same beat.
(function (DF) {
  // Mulberry32: a small, fast, good-enough PRNG.
  DF.makeRng = function (seed) {
    let a = seed >>> 0;
    const rng = function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    rng.chance = (p) => rng() < p;
    rng.range = (lo, hi) => lo + (hi - lo) * rng();
    rng.int = (lo, hi) => Math.floor(lo + (hi - lo + 1) * rng());
    rng.pick = (arr) => arr[Math.floor(rng() * arr.length)];
    return rng;
  };

  // FNV-1a hash: turns any text ("GROOVE-4821") into a 32-bit number.
  DF.hashString = function (str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  };

  const WORDS = ['KICK', 'SNARE', 'GROOVE', 'FUNK', 'BOOM', 'CLAP', 'SWING', 'RIDE',
    'FLOW', 'PULSE', 'THUMP', 'CRASH', 'SHUFFLE', 'BREAK', 'DRIVE', 'ROLL'];
  DF.randomSeed = function () {
    return WORDS[Math.floor(Math.random() * WORDS.length)] + '-' +
      Math.floor(1000 + Math.random() * 9000);
  };

  DF.clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  // smoothstep: 0 below `a`, 1 above `b`, smooth in between.
  DF.sstep = (x, a, b) => {
    const t = DF.clamp((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
  };
})((window.DF = window.DF || {}));
