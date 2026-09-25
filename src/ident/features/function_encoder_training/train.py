"""Train the function-encoder basis by closed-form projection (FUNCTION_ENCODER.md 2).

Network Phi: s -> R^K (3 hidden layers of 64, SiLU). For each corpus material m
with samples mu_m on the s-grid and diagonal weight W from w(s):

    theta_m = (Phi^T W Phi + eps I)^{-1} Phi^T W mu_m            (closed form)
    L = mean_m || mu_m - Phi theta_m ||^2_W / || mu_m ||^2_W  +  beta || G - I ||_F^2

with G = Phi^T W Phi the weighted Gram matrix. Backprop flows into the network
only; theta_m is an explicit (differentiable) function of the network through
the solve. The simulator is never differentiated (invariant 1).

The deployed artifact is the frozen 256-point tabulation table = Phi(s_grid),
saved to an npz that ident.features.function_encoder.FunctionEncoderDict loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ident.features.function_encoder_training.corpus import (
    N_S,
    build_corpus,
    s_grid,
    weight_vector,
)


class BasisNet(nn.Module):
    def __init__(self, K: int = 8, width: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, K)]
        self.net = nn.Sequential(*layers)
        self.K = K

    def forward(self, s_norm: torch.Tensor) -> torch.Tensor:
        return self.net(s_norm)  # (N, K)


def train(
    K: int = 8,
    n_materials: int = 2000,
    steps: int = 6000,
    batch: int = 128,
    beta: float = 1e-2,
    lr: float = 1e-3,
    eps_rel: float = 1e-6,
    seed: int = 0,
    realized_hist=None,
    out_path: str | Path = "mpm_engine/fe-weights/granular_mu_i.npz",
    device: str = "cpu",
):
    torch.manual_seed(seed)
    s = s_grid()
    s_t = torch.tensor((s - s.mean()) / s.std(), dtype=torch.float64, device=device).reshape(-1, 1)
    w = weight_vector(realized_hist)
    W = torch.tensor(w, dtype=torch.float64, device=device)        # (N_S,)
    ds = torch.tensor(np.gradient(s), dtype=torch.float64, device=device)
    Wd = W * ds                                                     # trapezoid-weighted

    mus_np, descs = build_corpus(n_materials, seed=seed)
    mus = torch.tensor(mus_np, dtype=torch.float64, device=device)  # (M, N_S)
    n_train = int(0.8 * n_materials)
    perm = torch.randperm(n_materials)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    net = BasisNet(K=K).to(device).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    eyeK = torch.eye(K, dtype=torch.float64, device=device)

    def project(Phi, mu_batch):
        # Phi: (N_S, K); mu_batch: (B, N_S). Weighted normal equations.
        WPhi = Phi * Wd[:, None]                       # (N_S, K)
        G = Phi.T @ WPhi                               # (K, K)
        eps = eps_rel * torch.trace(G) / K
        rhs = WPhi.T @ mu_batch.T                      # (K, B)
        theta = torch.linalg.solve(G + eps * eyeK, rhs)  # (K, B)
        recon = (Phi @ theta).T                        # (B, N_S)
        return recon, G

    def wrel_l2(mu_b, recon):
        num = torch.sum(Wd * (mu_b - recon) ** 2, dim=1)
        den = torch.sum(Wd * mu_b ** 2, dim=1)
        return num / (den + 1e-30)

    history = []
    for step in range(steps):
        bi = train_idx[torch.randint(0, n_train, (batch,))]
        mu_b = mus[bi]
        Phi = net(s_t)                                 # (N_S, K)
        recon, G = project(Phi, mu_b)
        fit = wrel_l2(mu_b, recon).mean()
        gram_pen = ((G - eyeK) ** 2).sum()
        loss = fit + beta * gram_pen
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                Phi = net(s_t)
                recon_te, _ = project(Phi, mus[test_idx])
                te = wrel_l2(mus[test_idx], recon_te)
                history.append({"step": step, "train_fit": float(fit),
                                "gram_pen": float(gram_pen),
                                "test_relL2_mean": float(te.mean()),
                                "test_relL2_worst": float(te.max())})

    # freeze tabulation
    with torch.no_grad():
        table = net(s_t).cpu().numpy()                 # (N_S, K)
        Phi = net(s_t)
        recon_te, G = project(Phi, mus[test_idx])
        te = wrel_l2(mus[test_idx], recon_te).cpu().numpy()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, s_grid=s, table=table, weight=w, K=K)

    # per-family span coverage
    kinds = {}
    for j, d in enumerate(descs):
        if j in set(test_idx.tolist()):
            kinds.setdefault(d["kind"], []).append(None)
    te_full = np.full(n_materials, np.nan)
    te_full[test_idx.numpy()] = te
    per_family = {}
    for kind in set(d["kind"] for d in descs):
        idx = [j for j, d in enumerate(descs) if d["kind"] == kind and not np.isnan(te_full[j])]
        if idx:
            per_family[kind] = {"n": len(idx),
                                "relL2_mean": float(np.mean(te_full[idx])),
                                "relL2_worst": float(np.max(te_full[idx]))}

    report = {
        "K": K, "n_materials": n_materials, "steps": steps, "beta": beta,
        "test_relL2_mean": float(np.mean(te)), "test_relL2_worst": float(np.max(te)),
        "gram_offdiag_max": float(np.max(np.abs(G.cpu().numpy() - np.eye(K)))),
        "per_family": per_family,
        "history": history,
        "acceptance_2pct": bool(np.max(te) < 0.02),
        "table_path": str(out_path),
    }
    with open(out_path.parent / "fe_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


if __name__ == "__main__":
    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rep = train(K=K)
    print(json.dumps({k: v for k, v in rep.items() if k != "history"}, indent=2, default=float))
    print("FE acceptance (<2% worst held-out):", rep["acceptance_2pct"])
