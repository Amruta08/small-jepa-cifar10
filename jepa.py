#!/usr/bin/env python3
"""
jepa_small.py
=============
Controlled small-scale study of JEPA-style training on CIFAR.

Research question
-----------------
How much of a small ViT's downstream accuracy comes from JEPA training rather than from
the (untrained) architecture, and does the masking recipe matter at 32x32 resolution?

Conditions (--cond)
-------------------
raw_pixels        probe on flattened pixels (weakest baseline)
untrained         randomly initialised encoder, no training (the key control)
jepa_multiblock   I-JEPA recipe: 4 target blocks, 1 large context block
jepa_block        single 4x4 target block, context = complement (the original draft's geometry)
jepa_random       25% random target patches, context = complement (same ratio as jepa_block)
pixel_multiblock  I-JEPA masks and predictor, but the target is pixels instead of latents

Faithfulness notes (what follows I-JEPA, what does not)
-------------------------------------------------------
* Target encoder sees the FULL image; targets are its (layer-normed) outputs at the target
  patches. EMA momentum rises linearly 0.996 -> 1.0. Context encoder sees only context
  tokens (masked tokens are dropped, not zeroed). Predictor = narrow ViT fed with context
  tokens + positional mask tokens. Fixed 2D sin-cos positional embeddings.
  Default loss is mean squared error (the I-JEPA paper text says L2).
* Deviations (scaled down for CIFAR / short schedules): tiny ViT (dim 192, depth 6, patch 4),
  AdamW with 5-epoch warmup + cosine, batch 256, crop(pad 4)+flip augmentation instead of
  random-resized-crop, gradient clipping at 1.0.

Usage
-----
  python jepa_small.py selftest
  python jepa_small.py sweep --mode smoke                 # ~minutes, checks everything works
  python jepa_small.py sweep --mode fast --jobs 2         # 3 seeds x 6 conditions, 100 epochs
  python jepa_small.py sweep --mode full --jobs 2         # 5 seeds, 200 epochs
  python jepa_small.py run --cond jepa_multiblock --seed 0 --epochs 100
  python jepa_small.py aggregate --out_dir runs
All runs are resumable: finished runs are skipped and unfinished runs resume from the last checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONDS = ["raw_pixels", "untrained", "jepa_multiblock", "jepa_block", "jepa_random", "pixel_multiblock"]
# condition -> (masking scheme, target type)
TRAINED = {
    "jepa_multiblock": ("multiblock", "latent"),
    "jepa_block": ("block", "latent"),
    "jepa_random": ("random", "latent"),
    "pixel_multiblock": ("multiblock", "pixel"),
}
LABEL_FRACS = (0.01, 0.1, 1.0)

MODES = {
    "smoke": dict(epochs=2, eval_epochs="1,2", seeds=[0], train_subset=2000, n_val=500, batch_size=64,
                  dim=64, depth=2, heads=2, pred_dim=32, pred_depth=1, pred_heads=2, warmup_epochs=1),
    "fast": dict(epochs=100, eval_epochs="25,50,100", seeds=[0, 1, 2]),
    "full": dict(epochs=200, eval_epochs="50,100,200", seeds=[0, 1, 2, 3, 4]),
}

HPARAM_KEYS = ["dataset", "data_root", "out_dir", "epochs", "eval_epochs", "batch_size", "lr", "min_lr",
               "warmup_epochs", "wd_start", "wd_end", "ema_start", "ema_end", "patch", "dim", "depth", "heads",
               "pred_dim", "pred_depth", "pred_heads", "loss", "grad_clip", "train_subset", "n_val",
               "split_seed", "knn_k", "device"]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_settings(dev: torch.device, disabled: bool):
    """Return (enabled, dtype, use_scaler)."""
    if dev.type != "cuda" or disabled:
        return False, torch.float32, False
    if torch.cuda.get_device_capability(dev)[0] >= 8:
        return True, torch.bfloat16, False
    return True, torch.float16, True  # e.g. T4 on free Colab


def make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def make_synthetic(n_train=6000, n_test=1500, n_classes=10):
    """Class-dependent coloured low-frequency patterns + noise. Only for testing the code."""
    g = torch.Generator().manual_seed(1234)
    protos = torch.rand(n_classes, 3, 8, 8, generator=g)

    def gen(n):
        y = torch.randint(0, n_classes, (n,), generator=g)
        base = F.interpolate(protos[y], size=32, mode="bilinear", align_corners=False)
        x = (base + 0.15 * torch.randn(n, 3, 32, 32, generator=g)).clamp(0, 1)
        return (x * 255).to(torch.uint8), y

    xtr, ytr = gen(n_train)
    xte, yte = gen(n_test)
    return xtr, ytr, xte, yte, n_classes


def load_data(args, dev):
    if args.dataset == "synthetic":
        xtr, ytr, xte, yte, ncls = make_synthetic()
    else:
        import torchvision
        cls = torchvision.datasets.CIFAR10 if args.dataset == "cifar10" else torchvision.datasets.CIFAR100
        tr = cls(args.data_root, train=True, download=True)
        te = cls(args.data_root, train=False, download=True)
        xtr = torch.from_numpy(np.asarray(tr.data)).permute(0, 3, 1, 2).contiguous()
        ytr = torch.tensor(tr.targets, dtype=torch.long)
        xte = torch.from_numpy(np.asarray(te.data)).permute(0, 3, 1, 2).contiguous()
        yte = torch.tensor(te.targets, dtype=torch.long)
        ncls = 10 if args.dataset == "cifar10" else 100
    # fixed train/val split (independent of the run seed, so every run sees the same split)
    perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(args.split_seed))
    va_idx, tr_idx = perm[: args.n_val], perm[args.n_val:]
    if args.train_subset > 0:
        tr_idx = tr_idx[: args.train_subset]
    d = dict(xtr=xtr[tr_idx], ytr=ytr[tr_idx], xva=xtr[va_idx], yva=ytr[va_idx], xte=xte, yte=yte, ncls=ncls)
    d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    f = d["xtr"].float() / 255
    d["mean"] = f.mean((0, 2, 3)).view(1, 3, 1, 1)
    d["std"] = f.std((0, 2, 3)).view(1, 3, 1, 1)
    return d


def normalize(xb_uint8, data):
    return (xb_uint8.float() / 255 - data["mean"]) / data["std"]


def augment(x):
    """Random crop (reflect-pad 4) + horizontal flip, per sample, on the GPU."""
    B, dev = x.size(0), x.device
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (B,), device=dev)
    j = torch.randint(0, 9, (B,), device=dev)
    ar = torch.arange(32, device=dev)
    rows, cols = i[:, None] + ar[None, :], j[:, None] + ar[None, :]
    xc = xp[torch.arange(B, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None],
            rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(B, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], xc.flip(3), xc)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def sincos_1d(dim, pos):
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2.0))
    out = pos.reshape(-1)[:, None].double() * omega[None, :]
    return torch.cat([out.sin(), out.cos()], 1)


def sincos_2d(dim, grid):
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, yy), sincos_1d(dim // 2, xx)], 1).float()  # (grid*grid, dim)


def take(t, idx):
    """Gather tokens: t (B,N,D), idx (B,K) -> (B,K,D)."""
    return t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.size(-1)))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
        return x + self.mlp(self.n2(x))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    def __init__(self, img=32, patch=4, dim=192, depth=6, heads=3):
        super().__init__()
        self.grid = img // patch
        self.n = self.grid ** 2
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.register_buffer("pos", sincos_2d(dim, self.grid)[None])
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(init_weights)

    def forward(self, x, keep=None):
        t = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        if keep is not None:
            t = take(t, keep)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


class Predictor(nn.Module):
    """Narrow ViT: context tokens + positional mask tokens -> predictions at target positions."""

    def __init__(self, enc_dim, out_dim, grid, dim=96, depth=4, heads=3):
        super().__init__()
        self.inp = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_2d(dim, grid)[None])
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
        self.apply(init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx_list):
        B = ctx_tokens.size(0)
        pos = self.pos.expand(B, -1, -1)
        c = self.inp(ctx_tokens) + take(pos, ctx_idx)
        outs = []
        for t_idx in tgt_idx_list:
            m = self.mask_token.expand(B, t_idx.size(1), -1) + take(pos, t_idx)
            x = torch.cat([c, m], 1)
            for b in self.blocks:
                x = b(x)
            outs.append(self.out(self.norm(x[:, c.size(1):])))
        return outs


# --------------------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------------------
class MaskGen:
    """Vectorised mask sampling. All index tensors are returned as int64 numpy arrays."""

    def __init__(self, grid, scheme, rng, block_scale=0.25, random_ratio=0.25, min_ctx=10):
        # min_ctx=10 mirrors min_keep_enc=10 of the official I-JEPA mask collator
        self.g, self.n, self.scheme, self.rng = grid, grid * grid, scheme, rng
        self.block_scale, self.random_ratio, self.min_ctx = block_scale, random_ratio, min_ctx

    def _block_size(self, scale, ar):
        # one block shape per batch (as in the official I-JEPA mask collator); positions vary per sample
        s, a = self.rng.uniform(*scale), self.rng.uniform(*ar)
        n_keep = int(self.n * s)
        h = int(round(math.sqrt(n_keep * a)))
        w = int(round(math.sqrt(n_keep / a)))
        return min(max(h, 1), self.g - 1), min(max(w, 1), self.g - 1)

    def _place(self, h, w, B):
        g = self.g
        top = self.rng.randint(0, g - h + 1, size=B)
        left = self.rng.randint(0, g - w + 1, size=B)
        r, c = np.arange(g)[None, :, None], np.arange(g)[None, None, :]
        return ((r >= top[:, None, None]) & (r < (top + h)[:, None, None]) &
                (c >= left[:, None, None]) & (c < (left + w)[:, None, None]))

    def sample(self, B):
        if self.scheme == "multiblock":
            th, tw = self._block_size((0.15, 0.2), (0.75, 1.5))
            tgt_masks = [self._place(th, tw, B) for _ in range(4)]
            union = np.logical_or.reduce(tgt_masks)
            ch, cw = self._block_size((0.85, 1.0), (1.0, 1.0))
            ctx = (self._place(ch, cw, B) & ~union).reshape(B, self.n)
            bad = ctx.sum(1) < self.min_ctx
            if bad.any():  # rare: fall back to the complement of the targets
                ctx[bad] = (~union).reshape(B, self.n)[bad]
            minc = int(ctx.sum(1).min())
            scores = np.where(ctx, self.rng.rand(B, self.n), 2.0)  # random subset of size minc per sample
            ctx_idx = np.sort(np.argsort(scores, 1)[:, :minc], 1)
            tgt_idx = [np.nonzero(m.reshape(B, -1))[1].reshape(B, -1) for m in tgt_masks]
        elif self.scheme == "block":
            side = max(1, int(round(math.sqrt(self.n * self.block_scale))))
            m = self._place(side, side, B)
            tgt_idx = [np.nonzero(m.reshape(B, -1))[1].reshape(B, -1)]
            ctx_idx = np.nonzero(~m.reshape(B, -1))[1].reshape(B, -1)
        elif self.scheme == "random":
            k = int(round(self.n * self.random_ratio))
            order = np.argsort(self.rng.rand(B, self.n), 1)
            tgt_idx = [np.sort(order[:, :k], 1)]
            ctx_idx = np.sort(order[:, k:], 1)
        else:
            raise ValueError(self.scheme)
        return ctx_idx.astype(np.int64), [t.astype(np.int64) for t in tgt_idx]


def patchify_norm(x, p):
    """Image -> per-patch normalised pixel vectors (B, n, p*p*3), used as pixel targets."""
    B, C, H, W = x.shape
    g = H // p
    x = x.reshape(B, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)
    return (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + 1e-6).sqrt()


# --------------------------------------------------------------------------------------
# Evaluation: linear probe, kNN, collapse diagnostics
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract(enc, X, data, dev, amp_cfg, bs=1000):
    enc.eval()
    en, dtype, _ = amp_cfg
    out = []
    for i in range(0, len(X), bs):
        x = normalize(X[i:i + bs], data)
        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=en):
            t = enc(x)
        out.append(t.float().mean(1))  # average-pooled patch representation
    return torch.cat(out)


def raw_features(X, data):
    return normalize(X, data).flatten(1)


def linear_probe(ftr, ytr, fva, yva, fte, yte, ncls, seed, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1), steps=300):
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr, fva, fte = (ftr - mu) / sd, (fva - mu) / sd, (fte - mu) / sd
    best = None
    for lam in lams:
        torch.manual_seed(seed)
        lin = nn.Linear(ftr.size(1), ncls).to(ftr.device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            loss = F.cross_entropy(lin(ftr), ytr) + lam * lin.weight.pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            va = (lin(fva).argmax(1) == yva).float().mean().item()
            te = (lin(fte).argmax(1) == yte).float().mean().item()
        if best is None or va > best[0]:
            best = (va, te, lam)
    return dict(val_acc=best[0], test_acc=best[1], lam=best[2])


@torch.no_grad()
def knn_acc(ftr, ytr, fq, yq, ncls, k=20, T=0.07):
    ftr, fq = F.normalize(ftr, dim=1), F.normalize(fq, dim=1)
    correct = 0
    for i in range(0, len(fq), 1000):
        w, idx = (fq[i:i + 1000] @ ftr.T).topk(k, dim=1)
        votes = torch.zeros(len(w), ncls, device=fq.device).scatter_add_(1, ytr[idx], (w / T).exp())
        correct += (votes.argmax(1) == yq[i:i + 1000]).sum().item()
    return correct / len(fq)


@torch.no_grad()
def collapse_stats(f):
    """Collapse diagnostics on pooled features f (N, D).
    erank_centered: effective rank of the mean-centred features, exp(entropy of normalised singular values)
                    (RankMe-style; PRIMARY metric - collapse drives it towards 1).
    erank:          same on the uncentred features (can be dominated by a shared mean offset).
    spread:         1 - ||mean of L2-normalised features||^2, in [0, 1] (all features identical -> 0,
                    isotropic features -> 1). Pooled transformer features share a large mean offset, so
                    absolute values are small; use it to compare conditions, not as a threshold."""
    fn = F.normalize(f, dim=1)
    out = dict(spread=float(1.0 - fn.mean(0).pow(2).sum().item()))
    for name, m in (("erank", f), ("erank_centered", f - f.mean(0, keepdim=True))):
        s = torch.linalg.svdvals(m.float())
        p = s / s.sum() + 1e-12
        out[name] = float(torch.exp(-(p * p.log()).sum()).item())
    return out


def evaluate(feats_fn, data, args, seed, epoch, extra=None):
    ftr, fva, fte = feats_fn(data["xtr"]), feats_fn(data["xva"]), feats_fn(data["xte"])
    ncls, g = data["ncls"], torch.Generator().manual_seed(seed)
    res = dict(epoch=epoch)
    for frac in LABEL_FRACS:
        n = max(ncls, int(round(frac * len(ftr)))) if frac < 1.0 else len(ftr)
        sub = torch.randperm(len(ftr), generator=g)[:n].to(ftr.device)
        lp = linear_probe(ftr[sub], data["ytr"][sub], fva, data["yva"], fte, data["yte"], ncls, seed)
        res[f"linear_{frac}"] = lp["test_acc"]
        res[f"val_linear_{frac}"] = lp["val_acc"]
    res["knn"] = knn_acc(ftr, data["ytr"], fte, data["yte"], ncls, k=args.knn_k)
    res.update(collapse_stats(fva))
    if extra:
        res.update(extra)
    return res


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def param_groups(*modules):
    decay, no_decay = [], []
    for m in modules:
        for p in m.parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [dict(params=decay, weight_decay=0.0), dict(params=no_decay, weight_decay=0.0)], decay, no_decay


def train_and_eval(args, data, dev, run_dir, scheme, target_type, results):
    amp_cfg = amp_settings(dev, args.no_amp)
    use_amp, amp_dtype, use_scaler = amp_cfg
    enc = Encoder(patch=args.patch, dim=args.dim, depth=args.depth, heads=args.heads).to(dev)
    latent = target_type == "latent"
    tgt_enc = None
    if latent:
        tgt_enc = copy.deepcopy(enc)
        for p in tgt_enc.parameters():
            p.requires_grad_(False)
    out_dim = args.dim if latent else args.patch * args.patch * 3
    pred = Predictor(args.dim, out_dim, enc.grid, args.pred_dim, args.pred_depth, args.pred_heads).to(dev)

    _, decay, no_decay = param_groups(enc, pred)  # no weight decay on biases / norm parameters
    groups = [dict(params=decay, weight_decay=args.wd_start), dict(params=no_decay, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
    scaler = make_scaler(use_scaler)
    mask_rng = np.random.RandomState(args.seed + 12345)
    maskgen = MaskGen(enc.grid, scheme, mask_rng)

    n_train = len(data["xtr"])
    spe = n_train // args.batch_size
    total = args.epochs * spe
    warm = args.warmup_epochs * spe
    eval_epochs = sorted({int(e) for e in str(args.eval_epochs).split(",") if e})
    loss_fn = F.mse_loss if args.loss == "l2" else F.smooth_l1_loss

    start_epoch, history = 0, []
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        enc.load_state_dict(ck["enc"]); pred.load_state_dict(ck["pred"]); opt.load_state_dict(ck["opt"])
        if latent:
            tgt_enc.load_state_dict(ck["tgt"])
        start_epoch, history = ck["epoch"], ck["history"]
        results["evals"] = ck.get("evals", [])
        mask_rng.set_state(ck["mask_rng"])
        print(f"  resumed from epoch {start_epoch}", flush=True)

    feats_fn = lambda X: extract(enc, X, data, dev, amp_cfg)
    if start_epoch == 0:  # epoch-0 evaluation = untrained encoder with this run's seed (paired control)
        results["evals"].append(evaluate(feats_fn, data, args, args.seed, 0, dict(train_loss=None)))

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        enc.train(); pred.train()
        perm = torch.randperm(n_train, device=dev)
        run_loss = 0.0
        for it in range(spe):
            step = epoch * spe + it
            if step < warm:
                lr = args.lr * (0.1 + 0.9 * step / max(1, warm))
            else:
                prog = (step - warm) / max(1, total - warm)
                lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * prog))
            wd = args.wd_start + (args.wd_end - args.wd_start) * step / total
            mom = args.ema_start + (args.ema_end - args.ema_start) * step / total
            opt.param_groups[0]["lr"] = opt.param_groups[1]["lr"] = lr
            opt.param_groups[0]["weight_decay"] = wd

            x = augment(normalize(data["xtr"][perm[it * args.batch_size:(it + 1) * args.batch_size]], data))
            ctx_np, tgt_np = maskgen.sample(x.size(0))
            ctx_idx = torch.from_numpy(ctx_np).to(dev)
            tgt_idx = [torch.from_numpy(t).to(dev) for t in tgt_np]
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                with torch.no_grad():
                    if latent:
                        h = tgt_enc(x)
                        h = F.layer_norm(h, (h.size(-1),))
                    else:
                        h = patchify_norm(x, args.patch)
                    targets = [take(h, t) for t in tgt_idx]
                z = enc(x, keep=ctx_idx)
                preds = pred(z, ctx_idx, tgt_idx)
                loss = sum(loss_fn(p.float(), t.float()) for p, t in zip(preds, targets)) / len(preds)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            if latent:
                with torch.no_grad():
                    for pt, pc in zip(tgt_enc.parameters(), enc.parameters()):
                        pt.mul_(mom).add_(pc.detach(), alpha=1 - mom)
            run_loss += loss.item()
        mean_loss = run_loss / spe
        history.append(dict(epoch=epoch + 1, loss=mean_loss, lr=lr))
        print(f"  epoch {epoch + 1}/{args.epochs} loss {mean_loss:.5f} lr {lr:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not math.isfinite(mean_loss):
            raise RuntimeError("loss is not finite")
        if (epoch + 1) in eval_epochs:
            ev = evaluate(feats_fn, data, args, args.seed, epoch + 1, dict(train_loss=mean_loss))
            results["evals"].append(ev)
            print("  eval:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()
                              if k in ("epoch", "linear_1.0", "linear_0.1", "knn", "erank_centered", "spread")},
                  flush=True)
        if (epoch + 1) in eval_epochs or (epoch + 1) % 5 == 0:
            ck = dict(enc=enc.state_dict(), pred=pred.state_dict(), opt=opt.state_dict(), epoch=epoch + 1,
                      history=history, evals=results["evals"], mask_rng=mask_rng.get_state())
            if latent:
                ck["tgt"] = tgt_enc.state_dict()
            torch.save(ck, ckpt_path)
    results["history"] = history
    results["train_time_sec"] = time.time() - t0


# --------------------------------------------------------------------------------------
# Single run
# --------------------------------------------------------------------------------------
def run(args):
    dev = get_device(args.device)
    seed_everything(args.seed)
    run_dir = Path(args.out_dir) / f"{args.dataset}_{args.cond}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "done.json").exists():
        print(f"[skip] {run_dir.name} already done")
        return
    print(f"[run] {run_dir.name} on {dev}", flush=True)
    data = load_data(args, dev)
    results = dict(args={k: getattr(args, k) for k in HPARAM_KEYS + ["cond", "seed"]}, evals=[], history=[],
                   env=dict(torch=torch.__version__, python=platform.python_version(),
                            device=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu")))
    if args.cond == "raw_pixels":
        results["evals"].append(evaluate(lambda X: raw_features(X, data), data, args, args.seed, 0))
    elif args.cond == "untrained":
        seed_everything(args.seed)
        enc = Encoder(patch=args.patch, dim=args.dim, depth=args.depth, heads=args.heads).to(dev)
        amp_cfg = amp_settings(dev, args.no_amp)
        results["evals"].append(evaluate(lambda X: extract(enc, X, data, dev, amp_cfg), data, args, args.seed, 0))
    else:
        scheme, ttype = TRAINED[args.cond]
        train_and_eval(args, data, dev, run_dir, scheme, ttype, results)
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    (run_dir / "done.json").write_text("{}")
    if (run_dir / "ckpt.pt").exists():  # keep the final weights only (no optimiser state) to save disk and bandwidth
        ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
        for k in ("opt", "mask_rng", "history", "evals"):
            ck.pop(k, None)
        torch.save(ck, run_dir / "final_ckpt.pt")
        (run_dir / "ckpt.pt").unlink()
    print(f"[done] {run_dir.name}", flush=True)


# --------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------
def apply_mode(args):
    for k, v in MODES[args.mode].items():
        if k == "seeds":
            if args.seeds is None:
                args.seeds = ",".join(map(str, v))
        elif getattr(args, k) == args._defaults.get(k):  # only override values the user did not set
            setattr(args, k, v)


def sweep(args):
    apply_mode(args)
    seeds = [int(s) for s in str(args.seeds).split(",")]
    conds = args.conds.split(",") if args.conds else CONDS
    jobs = []
    for s in seeds:  # seed-major order: partial completion still yields complete seeds
        for c in conds:
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--cond", c, "--seed", str(s)]
            for k in HPARAM_KEYS:
                cmd += [f"--{k}", str(getattr(args, k))]
            jobs.append((c, s, cmd))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[sweep] mode={args.mode} runs={len(jobs)} jobs={args.jobs} out_dir={args.out_dir}", flush=True)
    t0 = time.time()
    # avoid CPU thread oversubscription when several runs share the machine
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // max(1, args.jobs))))

    def work(job):
        c, s, cmd = job
        log = Path(args.out_dir) / f"{args.dataset}_{c}_s{s}"
        log.mkdir(parents=True, exist_ok=True)
        with open(log / "log.txt", "a") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"[sweep] {c} seed {s} -> exit {r.returncode} ({time.time() - t0:.0f}s elapsed)", flush=True)
        return r.returncode

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        codes = list(ex.map(work, jobs))
    failed = [(j[0], j[1]) for j, c in zip(jobs, codes) if c != 0]
    print(f"[sweep] finished in {time.time() - t0:.0f}s; failed runs: {failed or 'none'}")
    if failed:
        print("  check each run's log.txt; re-run the same command to resume.")
    else:
        aggregate(args)


# --------------------------------------------------------------------------------------
# Aggregation: tables, paired differences, figures
# --------------------------------------------------------------------------------------
def _tcrit(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(df, 2.0)


def aggregate(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.out_dir)
    rows = []
    for rj in sorted(root.glob("*/results.json")):
        r = json.loads(rj.read_text())
        for ev in r["evals"]:
            rows.append(dict(dataset=r["args"]["dataset"], cond=r["args"]["cond"], seed=r["args"]["seed"], **ev))
    if not rows:
        print("no results found in", root)
        return
    df = pd.DataFrame(rows)
    df.to_csv(root / "all_runs.csv", index=False)
    metrics = ["linear_1.0", "linear_0.1", "linear_0.01", "knn", "erank_centered", "erank", "spread", "train_loss"]
    metrics = [m for m in metrics if m in df.columns]
    out = []
    for (ds, cond, ep), g in df.groupby(["dataset", "cond", "epoch"]):
        row = dict(dataset=ds, cond=cond, epoch=ep, n_seeds=g["seed"].nunique())
        for m in metrics:
            v = g[m].dropna().values.astype(float)
            if len(v):
                row[f"{m}_mean"] = v.mean()
                row[f"{m}_std"] = v.std(ddof=1) if len(v) > 1 else float("nan")
                row[f"{m}_ci95"] = _tcrit(len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
        out.append(row)
    tab = pd.DataFrame(out)
    tab.to_csv(root / "results_table.csv", index=False)

    # final-epoch table (markdown) per dataset
    final = tab[tab.apply(lambda r: r["epoch"] == tab[(tab.dataset == r["dataset"]) & (tab.cond == r["cond"])]["epoch"].max(), axis=1)]
    lines = []
    for ds, g in final.groupby("dataset"):
        lines.append(f"### {ds} (final evaluation epoch per condition; mean +/- 95% CI over seeds)\n")
        lines.append("| condition | epoch | seeds | linear 100% | linear 10% | linear 1% | kNN | eff. rank (centred) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            def f(m):
                return f"{100 * r[m + '_mean']:.2f} +/- {100 * r[m + '_ci95']:.2f}" if m + "_mean" in r and not pd.isna(r.get(m + "_mean")) else "-"
            er = f"{r['erank_centered_mean']:.1f}" if "erank_centered_mean" in r and not pd.isna(r.get("erank_centered_mean")) else "-"
            lines.append(f"| {r['cond']} | {int(r['epoch'])} | {int(r['n_seeds'])} | {f('linear_1.0')} | "
                         f"{f('linear_0.1')} | {f('linear_0.01')} | {f('knn')} | {er} |")
        lines.append("")
    # paired differences at the final epoch (same seed => same initialisation / probe subsets)
    pairs = [("jepa_multiblock", "untrained"), ("jepa_block", "untrained"), ("jepa_random", "untrained"),
             ("pixel_multiblock", "untrained"), ("jepa_multiblock", "jepa_block"),
             ("jepa_multiblock", "jepa_random"), ("jepa_block", "jepa_random"), ("jepa_multiblock", "pixel_multiblock")]
    pd_rows = []
    for ds, g in df.groupby("dataset"):
        last = {c: g[g.cond == c].epoch.max() for c in g.cond.unique()}
        for a, b in pairs:
            if a not in last or b not in last:
                continue
            ga = g[(g.cond == a) & (g.epoch == last[a])].set_index("seed")
            gb = g[(g.cond == b) & (g.epoch == last[b])].set_index("seed")
            common = ga.index.intersection(gb.index)
            for m in ("linear_1.0", "linear_0.1", "knn"):
                d = (ga.loc[common, m] - gb.loc[common, m]).values.astype(float)
                if len(d) >= 2:
                    ci = _tcrit(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
                    pd_rows.append(dict(dataset=ds, a=a, b=b, metric=m, n_pairs=len(d), mean_diff_pp=100 * d.mean(),
                                        ci95_pp=100 * ci, excludes_zero=bool(abs(d.mean()) > ci)))
    if pd_rows:
        pdf = pd.DataFrame(pd_rows)
        pdf.to_csv(root / "paired_differences.csv", index=False)
        lines.append("### Paired differences (A - B, percentage points, final epoch; same seeds)\n")
        lines.append(f"_{len(pdf)} comparisons are listed; with 95% intervals roughly 1 in 20 will exclude 0 by chance alone, "
                     "so treat an isolated 'yes' cautiously and look for consistent patterns across metrics and seeds._\n")
        lines.append("| dataset | A | B | metric | pairs | mean diff | 95% CI | CI excludes 0 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in pdf.iterrows():
            lines.append(f"| {r.dataset} | {r.a} | {r.b} | {r.metric} | {r.n_pairs} | {r.mean_diff_pp:+.2f} | "
                         f"+/- {r.ci95_pp:.2f} | {'yes' if r.excludes_zero else 'no'} |")
    (root / "results_table.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # figures
    for ds, g in df.groupby("dataset"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for cond, gc in g.groupby("cond"):
            s = gc.groupby("epoch")["linear_1.0"].agg(["mean", "std"]).reset_index()
            if len(s) > 1:
                ax[0].errorbar(s.epoch, 100 * s["mean"], yerr=100 * s["std"].fillna(0), marker="o", capsize=3, label=cond)
            else:
                ax[0].axhline(100 * s["mean"].iloc[0], ls="--", lw=1, label=cond)
        ax[0].set_xlabel("pretraining epoch"); ax[0].set_ylabel("linear probe accuracy (%)"); ax[0].legend(fontsize=7)
        ax[0].set_title(f"{ds}: accuracy vs training (mean +/- std over seeds)")
        last = g[g.apply(lambda r: r["epoch"] == g[g.cond == r["cond"]]["epoch"].max(), axis=1)]
        # latent-target runs only: pixel-target losses are on a different scale and not comparable
        trained = last[last.train_loss.notna() & (last.epoch > 0) & last.cond.str.startswith("jepa_")]
        for cond, gc in trained.groupby("cond"):
            ax[1].scatter(gc.train_loss, 100 * gc["linear_1.0"], label=cond, s=25)
        ax[1].set_xlabel("final training loss (latent-target runs)"); ax[1].set_ylabel("linear probe accuracy (%)")
        ax[1].set_title("does training loss predict accuracy? (one dot per run)"); ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / f"fig_{ds}_curves_and_loss.png", dpi=160)
        fig.savefig(root / f"fig_{ds}_curves_and_loss.pdf")
        plt.close(fig)
    print(f"\nWrote tables and figures to {root}/")


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------
def selftest(_args=None):
    rng = np.random.RandomState(0)
    g = 8
    for scheme in ("multiblock", "block", "random"):
        mg = MaskGen(g, scheme, rng)
        for _ in range(20):
            ctx, tgts = mg.sample(32)
            assert ctx.shape[0] == 32 and ctx.min() >= 0 and ctx.max() < g * g
            for t in tgts:
                assert t.shape[0] == 32 and t.min() >= 0 and t.max() < g * g
                for b in range(32):  # no overlap between context and any target, no duplicate indices
                    assert not set(ctx[b]).intersection(t[b]), f"{scheme}: context/target overlap"
                    assert len(set(t[b])) == t.shape[1] and len(set(ctx[b])) == ctx.shape[1]
        print(f"  mask scheme {scheme}: ok (ctx tokens {ctx.shape[1]}, target blocks {[t.shape[1] for t in tgts]})")
    enc = Encoder(dim=64, depth=2, heads=2)
    pred = Predictor(64, 64, enc.grid, 32, 1, 2)
    x = torch.randn(4, 3, 32, 32)
    ctx, tgts = MaskGen(g, "multiblock", rng).sample(4)
    z = enc(x, keep=torch.from_numpy(ctx))
    outs = pred(z, torch.from_numpy(ctx), [torch.from_numpy(t) for t in tgts])
    assert z.shape == (4, ctx.shape[1], 64) and all(o.shape == (4, t.shape[1], 64) for o, t in zip(outs, tgts))
    assert enc(x).shape == (4, 64, 64)
    a = augment(x)
    assert a.shape == x.shape
    p = patchify_norm(x, 4)
    assert p.shape == (4, 64, 48) and abs(p.mean().item()) < 1e-4
    print("  shapes, patchify, augment: ok")
    torch.manual_seed(0)
    cs = collapse_stats(torch.randn(300, 32))                                    # healthy: isotropic features
    cs_rank1 = collapse_stats(torch.randn(300, 1) @ torch.randn(1, 32))          # dimensional collapse: rank 1
    cs_const = collapse_stats(torch.ones(300, 32) + 1e-4 * torch.randn(300, 32))  # complete collapse: constant
    assert cs["erank_centered"] > 25, cs
    assert cs_rank1["erank_centered"] < 2, cs_rank1
    assert cs_const["spread"] < 0.01 and cs["spread"] > 0.9, (cs, cs_const)
    print("  collapse metrics: ok (healthy erank %.1f / spread %.2f, rank-1 erank %.1f, constant spread %.4f)"
          % (cs["erank_centered"], cs["spread"], cs_rank1["erank_centered"], cs_const["spread"]))
    print("selftest passed")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def add_hparams(p):
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "synthetic"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir", default="./runs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval_epochs", type=str, default="25,50,100")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--wd_start", type=float, default=0.04)
    p.add_argument("--wd_end", type=float, default=0.4)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--pred_dim", type=int, default=96)
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--pred_heads", type=int, default=3)
    p.add_argument("--loss", default="l2", choices=["l2", "smooth_l1"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--train_subset", type=int, default=0)
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); add_hparams(p)
    p.add_argument("--cond", required=True, choices=CONDS); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("sweep"); add_hparams(p)
    p.add_argument("--mode", default="fast", choices=list(MODES)); p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seeds", default=None); p.add_argument("--conds", default=None)
    p = sub.add_parser("aggregate"); p.add_argument("--out_dir", default="./runs")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "sweep":
        args._defaults = {a.dest: a.default for a in sub.choices["sweep"]._actions}
    {"run": run, "sweep": sweep, "aggregate": aggregate, "selftest": selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
