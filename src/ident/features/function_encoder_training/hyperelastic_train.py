"""Train a function-encoder basis for HYPERELASTICITY, the same way as the granular mu(I) basis:
a neural basis Phi: x=(I1bar-3) -> R^K (3x64 SiLU) over a corpus of isotropic energy-derivative
responses W1(x) = dW/dI1bar, with the closed-form projection encoding and the Gram-identity
orthonormality loss (FUNCTION_ENCODER.md Section 2). The corpus spans LINEAR and NONLINEAR families:

    neo-Hookean : W1 = C1                      (constant -> linear stress in dev(bbar))
    Yeoh        : W1 = C1 + 2 C2 x + 3 C3 x^2  (polynomial nonlinear)
    Gent        : W1 = (mu/2)/(1 - x/Jm)       (non-polynomial locking)

The deviatoric Kirchhoff stress of any such law is tau = 2 W1(I1bar) dev(bbar), so W1 IS the
constitutive function. The trained basis is frozen to a table on an x-grid; the weak-form recovery
(sim.hyperelastic.recover_fe) sets the dictionary stresses to P_k = 2 phi_k(I1bar) dev(bbar) F^{-T}
and identifies a held-out law's coefficients through the momentum operator: encode = identify.

The simulator is never differentiated; torch trains only the basis network (invariant 1, and torch is
allowed only under this directory).

Run:  .venv/bin/python -m ident.features.function_encoder_training.hyperelastic_train
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_X = 256
X_MAX = 0.8                       # I1bar-3 range covered by the basis (drops reach ~0.5)
OUT = Path("mpm_engine/fe-weights/hyperelastic_1inv.npz")


def x_grid():
    return np.linspace(0.0, X_MAX, N_X)


def build_corpus(n_materials=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = x_grid(); W1 = []
    per = n_materials // 3
    for _ in range(per):                                           # neo-Hookean
        C1 = 10 ** rng.uniform(4.3, 5.3); W1.append(np.full_like(x, C1))
    for _ in range(per):                                           # Yeoh
        C1 = 10 ** rng.uniform(4.3, 5.3); C2 = 10 ** rng.uniform(3.5, 4.8); C3 = 10 ** rng.uniform(3.0, 4.5)
        W1.append(C1 + 2 * C2 * x + 3 * C3 * x ** 2)
    for _ in range(n_materials - 2 * per):                         # Gent (locking)
        mu = 10 ** rng.uniform(4.3, 5.3); Jm = rng.uniform(0.6, 4.0)
        W1.append(0.5 * mu / (1.0 - np.clip(x / Jm, 0, 0.9)))
    return np.array(W1), x                                         # (M, N_X)


class BasisNet(nn.Module):
    def __init__(self, K=6, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, K)]
        self.net = nn.Sequential(*layers); self.K = K

    def forward(self, x):
        return self.net(x)


def train(K=6, n_materials=2400, steps=6000, batch=128, beta=1e-2, lr=1e-3, eps_rel=1e-6, seed=0):
    torch.manual_seed(seed)
    x = x_grid()
    x_t = torch.tensor((x - x.mean()) / x.std(), dtype=torch.float64).reshape(-1, 1)
    dx = torch.tensor(np.gradient(x), dtype=torch.float64)
    # weight: emphasize the realized strain range, slight taper at the ends
    w = np.ones_like(x); W = torch.tensor(w, dtype=torch.float64); Wd = W * dx
    mus_np, _ = build_corpus(n_materials, seed=seed)
    # normalize each curve by its weighted norm so the basis learns SHAPE; theta carries scale
    mus = torch.tensor(mus_np, dtype=torch.float64)
    nrm = torch.sqrt((Wd * mus ** 2).sum(1, keepdim=True)); mus = mus / nrm
    n_train = int(0.8 * n_materials); perm = torch.randperm(n_materials)
    tr, te = perm[:n_train], perm[n_train:]
    net = BasisNet(K=K).double(); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps); eyeK = torch.eye(K, dtype=torch.float64)

    def project(Phi, mb):
        WPhi = Phi * Wd[:, None]; G = Phi.T @ WPhi
        eps = eps_rel * torch.trace(G) / K
        theta = torch.linalg.solve(G + eps * eyeK, WPhi.T @ mb.T)
        return (Phi @ theta).T, G

    def wrel(mb, rec):
        return (Wd * (mb - rec) ** 2).sum(1) / ((Wd * mb ** 2).sum(1) + 1e-30)

    hist = []
    for step in range(steps):
        bi = tr[torch.randint(0, n_train, (batch,))]; mb = mus[bi]
        Phi = net(x_t); rec, G = project(Phi, mb)
        loss = wrel(mb, rec).mean() + beta * ((G - eyeK) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == steps - 1:
            with torch.no_grad():
                rec_te, _ = project(net(x_t), mus[te]); t = wrel(mus[te], rec_te)
                hist.append(dict(step=step, test_relL2_mean=float(t.mean()), test_relL2_worst=float(t.max())))
    with torch.no_grad():
        table = net(x_t).cpu().numpy(); rec_te, G = project(net(x_t), mus[te])
        te_err = wrel(mus[te], rec_te).cpu().numpy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, x_grid=x, table=table, K=K)
    (OUT.parent / "hyperelastic_fe_train.json").write_text(json.dumps(
        dict(K=K, n_materials=n_materials, test_relL2_mean=float(te_err.mean()),
             test_relL2_worst=float(np.percentile(te_err, 99)), history=hist), indent=2))
    print(f"[hyperelastic FE] K={K} trained on {n_materials} (neo-Hookean/Yeoh/Gent)")
    print(f"  held-out W1 reconstruction relL2: mean {te_err.mean():.2e}  p99 {np.percentile(te_err,99):.2e}")
    print(f"  wrote {OUT}")
    return OUT


OUT2 = Path("mpm_engine/fe-weights/hyperelastic_2inv.npz")
X2_MAX = 0.8


def _grid2(n=14):
    g = np.linspace(0.0, X2_MAX, n)
    x1, x2 = np.meshgrid(g, g, indexing="ij")
    return x1.reshape(-1), x2.reshape(-1)                          # (n^2,), (n^2,)


def build_corpus2(n_materials=3000, seed=0):
    """Responses (W1, W2) over the (I1bar-3, I2bar-3) plane for two-invariant families.
    Adds generalized Rivlin so W2 varies over the whole plane."""
    rng = np.random.default_rng(seed)
    x1, x2 = _grid2(); W1, W2 = [], []
    per = n_materials // 5
    for _ in range(per):                                           # neo-Hookean
        C10 = 10 ** rng.uniform(4.3, 5.3); W1.append(np.full_like(x1, C10)); W2.append(np.zeros_like(x1))
    for _ in range(per):                                           # Mooney (W2 const != 0)
        C10 = 10 ** rng.uniform(4.3, 5.3); C01 = 10 ** rng.uniform(3.5, 4.8)
        W1.append(np.full_like(x1, C10)); W2.append(np.full_like(x1, C01))
    for _ in range(per):                                           # Yeoh (W1 poly in I1)
        C1 = 10 ** rng.uniform(4.3, 5.3); C2 = 10 ** rng.uniform(3.5, 4.8); C3 = 10 ** rng.uniform(3.0, 4.5)
        W1.append(C1 + 2 * C2 * x1 + 3 * C3 * x1 ** 2); W2.append(np.zeros_like(x1))
    for _ in range(per):                                           # Gent (locking in I1)
        mu = 10 ** rng.uniform(4.3, 5.3); Jm = rng.uniform(0.6, 4.0)
        W1.append(0.5 * mu / (1.0 - np.clip(x1 / Jm, 0, 0.9))); W2.append(np.zeros_like(x1))
    for _ in range(n_materials - 4 * per):                         # generalized Rivlin (W1,W2 vary in BOTH)
        C10 = 10 ** rng.uniform(4.3, 5.2); C01 = 10 ** rng.uniform(3.5, 4.6)
        C20 = 10 ** rng.uniform(3.3, 4.4); C02 = 10 ** rng.uniform(3.0, 4.2); C11 = 10 ** rng.uniform(3.0, 4.3)
        W1.append(C10 + 2 * C20 * x1 + C11 * x2); W2.append(C01 + 2 * C02 * x2 + C11 * x1)
    return np.array(W1), np.array(W2), x1, x2


class BasisNet2(nn.Module):
    """Two-invariant basis: (x1,x2) -> R^{2K}, first K are the W1 channel, next K the W2 channel."""
    def __init__(self, K=8, width=96, depth=3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2 * K)]
        self.net = nn.Sequential(*layers); self.K = K

    def forward(self, x):
        return self.net(x)


def train2(K=8, n_materials=3000, steps=8000, batch=128, beta=1e-2, lr=1e-3, eps_rel=1e-6, seed=0):
    """Train the TWO-channel (I1,I2) hyperelastic function encoder. The stacked response fingerprint
    [W1 ; W2] over the (x1,x2) grid is projected onto the stacked basis [phi^(1) ; phi^(2)]; one
    coefficient theta_k drives BOTH channels, so the encoder represents the (W1,W2) pair jointly and
    spans Mooney (W2 const), Yeoh/Gent (W2=0), and generalized Rivlin (W2 varying)."""
    torch.manual_seed(seed)
    x1, x2 = _grid2(); ng = len(x1)
    xin = np.stack([x1, x2], 1)
    xt = torch.tensor((xin - xin.mean(0)) / (xin.std(0) + 1e-9), dtype=torch.float64)
    Wd = torch.ones(2 * ng, dtype=torch.float64) * (X2_MAX / 14) ** 2   # uniform area weight, stacked
    W1, W2, _, _ = build_corpus2(n_materials, seed=seed)
    F = np.concatenate([W1, W2], 1)                                # (M, 2*ng) stacked fingerprint
    mus = torch.tensor(F, dtype=torch.float64)
    nrm = torch.sqrt((Wd * mus ** 2).sum(1, keepdim=True)); mus = mus / nrm
    ntr = int(0.8 * n_materials); perm = torch.randperm(n_materials); tr, te = perm[:ntr], perm[ntr:]
    net = BasisNet2(K=K).double(); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps); eyeK = torch.eye(K, dtype=torch.float64)

    def stacked(Phi2):                                             # Phi2: (ng, 2K) -> (2ng, K)
        return torch.cat([Phi2[:, :K], Phi2[:, K:]], 0)

    def project(Phi, mb):
        WPhi = Phi * Wd[:, None]; G = Phi.T @ WPhi
        eps = eps_rel * torch.trace(G) / K
        theta = torch.linalg.solve(G + eps * eyeK, WPhi.T @ mb.T)
        return (Phi @ theta).T, G

    def wrel(mb, rec):
        return (Wd * (mb - rec) ** 2).sum(1) / ((Wd * mb ** 2).sum(1) + 1e-30)

    hist = []
    for step in range(steps):
        bi = tr[torch.randint(0, ntr, (batch,))]; mb = mus[bi]
        Phi = stacked(net(xt)); rec, G = project(Phi, mb)
        loss = wrel(mb, rec).mean() + beta * ((G - eyeK) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == steps - 1:
            with torch.no_grad():
                rec_te, _ = project(stacked(net(xt)), mus[te]); t = wrel(mus[te], rec_te)
                hist.append(dict(step=step, test_relL2_mean=float(t.mean()), test_relL2_worst=float(t.max())))
    with torch.no_grad():
        tab = net(xt).cpu().numpy()                                # (ng, 2K)
        rec_te, _ = project(stacked(net(xt)), mus[te]); te_err = wrel(mus[te], rec_te).cpu().numpy()
    OUT2.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT2, x1=x1, x2=x2, table=tab, K=K, n=14)
    (OUT2.parent / "hyperelastic_fe2_train.json").write_text(json.dumps(
        dict(K=K, n_materials=n_materials, test_relL2_mean=float(te_err.mean()),
             test_relL2_worst=float(np.percentile(te_err, 99)), history=hist), indent=2))
    print(f"[hyperelastic FE2 (I1,I2)] K={K} over 5 families (incl. Mooney W2 + generalized Rivlin)")
    print(f"  held-out (W1,W2) recon relL2: mean {te_err.mean():.2e}  p99 {np.percentile(te_err,99):.2e}")
    print(f"  wrote {OUT2}")
    return OUT2


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "i2":
        train2()
    else:
        train()
