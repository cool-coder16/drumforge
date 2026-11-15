"""DrumForge neural drummer.

A small causal transformer that writes drums one 16th-note step at a time.

Input at step t (predicting the drums of step t):
  * the drums of step t-1        (9 tracks x 4 velocity classes, embedded and summed)
  * audio-like features around t (low/mid/high onset strengths at t-4..t+4, bar energy, ...)
  * position: step in bar (16), bar in 8-bar phrase (8)
  * conditioning: style cluster, density bucket, tempo bucket, "has audio" flag
Output at step t: 9 tracks x 4 classes (off / ghost / normal / accent). The 9 tracks are
decoded in order, each seeing the tracks already chosen at the same step, so a step is a
coherent chord of drum hits rather than 9 independent coin flips.

Attention is causal with a sliding window and ALiBi distance bias, so the same weights run on
a whole song of any length with a fixed-size KV cache (this is how js/ai.js runs it).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TRACKS, N_CLASSES = 9, 4
FEAT_OFFSETS = list(range(-4, 5))          # feature look-around, in steps
N_FEAT = len(FEAT_OFFSETS) * 3 + 3 + 3 + 1  # window(27) + bar-energy trio(3) + bar mean L/M/H(3) + has_audio(1)


def alibi_slopes(n_heads):
    return [2 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)]


class Block(nn.Module):
    def __init__(self, d, n_heads, ff, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)       # training-time only; identical to nothing at inference
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.h = n_heads

    def forward(self, x, bias):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class Drummer(nn.Module):
    def __init__(self, d=192, n_layers=4, n_heads=6, ff=576, window=96, n_styles=12, head_hidden=96, dropout=0.0):
        super().__init__()
        self.cfg = dict(d=d, n_layers=n_layers, n_heads=n_heads, ff=ff, window=window, n_styles=n_styles, head_hidden=head_hidden, dropout=dropout)
        self.window = window
        self.prev_emb = nn.Embedding(N_TRACKS * N_CLASSES, d)
        self.feat = nn.Linear(N_FEAT, d)
        self.pos_step = nn.Embedding(16, d)
        self.pos_bar = nn.Embedding(8, d)
        self.style_emb = nn.Embedding(n_styles + 1, d)   # last index = unknown
        self.dens_emb = nn.Embedding(9, d)               # 8 buckets + unknown
        self.tempo_emb = nn.Embedding(7, d)              # 6 buckets + unknown
        self.blocks = nn.ModuleList([Block(d, n_heads, ff, dropout) for _ in range(n_layers)])
        self.in_drop = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(d)
        # per-track heads: hidden state + one-hot of the tracks already chosen at this step
        self.head1 = nn.ModuleList([nn.Linear(d + N_TRACKS * N_CLASSES, head_hidden) for _ in range(N_TRACKS)])
        self.head2 = nn.ModuleList([nn.Linear(head_hidden, N_CLASSES) for _ in range(N_TRACKS)])
        self.register_buffer("slopes", torch.tensor(alibi_slopes(n_heads)).view(1, n_heads, 1, 1), persistent=False)

    def attn_bias(self, T, device):
        i = torch.arange(T, device=device)
        dist = (i[:, None] - i[None, :]).float()                   # >= 0 means key is in the past
        bias = -self.slopes * dist.clamp(min=0)                    # ALiBi
        mask = (dist < 0) | (dist >= self.window)                  # causal + sliding window
        return bias.masked_fill(mask, float("-inf"))

    def trunk(self, prev, feat, step_in_bar, bar_in_phrase, style, dens, tempo):
        """prev (B,T,9) long classes of the previous step; feat (B,T,N_FEAT); others (B,T) or (B,)."""
        B, T, _ = prev.shape
        idx = prev + torch.arange(N_TRACKS, device=prev.device) * N_CLASSES
        x = self.prev_emb(idx).sum(2) + self.feat(feat) + self.pos_step(step_in_bar) + self.pos_bar(bar_in_phrase)
        x = x + (self.style_emb(style) + self.dens_emb(dens) + self.tempo_emb(tempo)).unsqueeze(1)
        x = self.in_drop(x)
        bias = self.attn_bias(T, prev.device)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln_f(x)

    def forward(self, prev, feat, step_in_bar, bar_in_phrase, style, dens, tempo, target):
        """Teacher-forced logits for every track: (B,T,9,4)."""
        h = self.trunk(prev, feat, step_in_bar, bar_in_phrase, style, dens, tempo)
        onehot = F.one_hot(target, N_CLASSES).float().flatten(2)   # (B,T,36)
        outs = []
        for j in range(N_TRACKS):
            ctx = onehot.clone()
            ctx[..., j * N_CLASSES:] = 0                            # track j only sees tracks < j
            z = F.gelu(self.head1[j](torch.cat([h, ctx], -1)))
            outs.append(self.head2[j](z))
        return torch.stack(outs, 2)
