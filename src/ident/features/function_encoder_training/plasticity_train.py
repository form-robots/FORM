"""Train the yield-surface function-encoder basis h(p).

The yield condition is written sqrt(J2(dev tau)) = h(p) with p the Kirchhoff
pressure, and h the learned function. This one family contains the perfect
plasticity zoo the NCLaw comparison needs: von Mises is the flat curve
h(p) = c, cohesionless Drucker-Prager is the line through the origin clamped
at zero in tension, cohesion shifts the kink, cap models saturate. Rate
dependence and hardening are deliberately outside: they would need internal
variables, and the comparison's materials are perfect plasticity.

Mirrors viscous_train.py (BasisNet plus the weighted closed-form projection),
but over a LINEAR pressure grid in Pa, tension included, because a von Mises
solid yields under tension while a cohesionless cone does not, and that
difference at p < 0 is part of what the basis must span. Saves a frozen
tabulation with key p_grid that experiments.fe_ls.baseline.load_table_fe
loads; the recovered surface is h(p) = sum_k theta_k Phi_k(p), linear in
theta, so the weak-form solve stays convex.

Run from the engine root:

    .venv/bin/python -m ident.features.function_encoder_training.plasticity_train
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ident.features.function_encoder_training.train import BasisNet

# Identification observes settled-flow pressures. The rollout reads the
# surface at impact pressures far above them, so the grid extends to 3e5 Pa.
# At 3e4 Pa a cone table diverges from true Drucker-Prager by 7.7 cm; at
# 3e5 Pa the gap is 7e-6 m (sand dataset).
P_MIN, P_MAX, N_GRID = -2.0e4, 3.0e5, 161


def p_grid() -> np.ndarray:
    return np.linspace(P_MIN, P_MAX, N_GRID)


def build_corpus(n: int, seed: int = 0) -> tuple[np.ndarray, list[dict]]:
    """n unit-norm yield-surface shapes over the pressure grid, five families."""
    rng = np.random.default_rng(seed)
    p = p_grid()
    rows, descs = [], []
    per = n // 5
    # von Mises: flat, tension included
    for _ in range(per):
        rows.append(np.ones_like(p))
        descs.append({"kind": "von_mises"})
    # affine cone: max(a + b p, 0); a/b sets the kink, a <= 0 is cohesionless
    for _ in range(per):
        b = rng.uniform(0.2, 3.0)
        a = rng.uniform(-0.5, 0.5) * b * P_MAX * 0.3
        rows.append(np.clip(a + b * p, 0.0, None))
        descs.append({"kind": "affine_cone", "a": float(a), "b": float(b)})
    # smooth kink: softplus of the same lines, kink sharpness varying
    for _ in range(per):
        b = rng.uniform(0.2, 3.0)
        a = rng.uniform(-0.3, 0.3) * b * P_MAX * 0.3
        w = rng.uniform(0.02, 0.2) * P_MAX
        rows.append(b * w * np.logaddexp(0.0, (p + a / b) / w))
        descs.append({"kind": "smooth_cone", "b": float(b), "w": float(w)})
    # saturating cap: rises then levels off
    for _ in range(per):
        p0 = rng.uniform(0.0, 0.4) * P_MAX
        w = rng.uniform(0.1, 0.5) * P_MAX
        rows.append(np.clip(np.tanh((p - p0) / w), 0.0, None))
        descs.append({"kind": "cap", "p0": float(p0), "w": float(w)})
    # sub- and super-linear pressure strengthening
    while len(rows) < n:
        e = rng.uniform(0.4, 1.5)
        rows.append(np.clip(p, 0.0, None) ** e / P_MAX ** e)
        descs.append({"kind": "power", "exponent": float(e)})
    E = np.stack(rows)
    dp = np.gradient(p)
    norm = np.sqrt(np.sum(dp * E ** 2, axis=1, keepdims=True))
    return E / np.maximum(norm, 1e-30), descs


def train(K: int = 8, n_materials: int = 2500, steps: int = 6000, batch: int = 128,
          beta: float = 1e-2, lr: float = 1e-3, eps_rel: float = 1e-6,
          seed: int = 0, out_path: str = "fe-weights/yield_surface.npz",
          device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    p = p_grid()
    s_t = torch.tensor((p - p.mean()) / p.std(), dtype=torch.float64,
                       device=device).reshape(-1, 1)
    Wd = torch.tensor(np.gradient(p), dtype=torch.float64, device=device)
    E_np, descs = build_corpus(n_materials, seed=seed)
    E = torch.tensor(E_np, dtype=torch.float64, device=device)
    n_train = int(0.8 * n_materials)
    perm = torch.randperm(n_materials)
    tr, te_idx = perm[:n_train], perm[n_train:]
    net = BasisNet(K=K).to(device).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    eyeK = torch.eye(K, dtype=torch.float64, device=device)

    def project(Phi, b):
        WPhi = Phi * Wd[:, None]
        G = Phi.T @ WPhi
        eps = eps_rel * torch.trace(G) / K
        theta = torch.linalg.solve(G + eps * eyeK, WPhi.T @ b.T)
        return (Phi @ theta).T, G

    def wrel(b, r):
        return torch.sum(Wd * (b - r) ** 2, 1) / (torch.sum(Wd * b ** 2, 1) + 1e-30)

    for step in range(steps):
        bi = tr[torch.randint(0, n_train, (batch,))]
        Phi = net(s_t)
        recon, G = project(Phi, E[bi])
        loss = wrel(E[bi], recon).mean() + beta * ((G - eyeK) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        table = net(s_t).cpu().numpy()
        Phi = net(s_t)
        recon_te, _ = project(Phi, E[te_idx])
        te = wrel(E[te_idx], recon_te).cpu().numpy()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, p_grid=p, table=table, K=K)
    per_family: dict = {}
    te_full = np.full(n_materials, np.nan)
    te_full[te_idx.numpy()] = te
    for kind in sorted(set(d["kind"] for d in descs)):
        idx = [i for i, d in enumerate(descs) if d["kind"] == kind]
        vals = te_full[idx]
        vals = vals[np.isfinite(vals)]
        per_family[kind] = {"n_test": int(vals.size),
                            "wrel_mean": float(vals.mean()),
                            "wrel_worst": float(vals.max())}
    report = {"K": K, "n_materials": n_materials, "steps": steps,
              "test_wrel_mean": float(te.mean()), "test_wrel_worst": float(te.max()),
              "per_family": per_family, "table_path": str(out),
              "p_grid_pa": [P_MIN, P_MAX, N_GRID]}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
