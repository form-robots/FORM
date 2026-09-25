"""Train the viscous (eta_app(gd)) function-encoder basis.

Mirrors train.py (reuses BasisNet + the weighted closed-form projection), but on
the viscous corpus over s = log10 gd. Saves a frozen tabulation that
ident.features.function_encoder.FunctionEncoderDict loads (with the shear rate gd
passed as its "I" argument and p = |gd| in the assembly, so the recovered
function is eta_app(gd) = sum_k theta_k Phi_k(gd)). The basis is the LEARNED,
linear-in-theta alternative to the rigid 2-parameter Bingham dictionary.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from ident.features.function_encoder_training.train import BasisNet
from ident.features.function_encoder_training.viscous_corpus import build_corpus, s_grid


def train(K=8, n_materials=2500, steps=6000, batch=128, beta=1e-2, lr=1e-3,
          eps_rel=1e-6, seed=0, out_path="mpm_engine/fe-weights/viscous.npz",
          device="cpu"):
    torch.manual_seed(seed)
    s = s_grid()
    s_t = torch.tensor((s - s.mean()) / s.std(), dtype=torch.float64, device=device).reshape(-1, 1)
    ds = torch.tensor(np.gradient(s), dtype=torch.float64, device=device)
    Wd = ds  # uniform weight (corpus laws are unit-norm shapes)
    E_np, descs = build_corpus(n_materials, seed=seed)
    E = torch.tensor(E_np, dtype=torch.float64, device=device)
    n_train = int(0.8 * n_materials)
    perm = torch.randperm(n_materials); tr, te_idx = perm[:n_train], perm[n_train:]
    net = BasisNet(K=K).to(device).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    eyeK = torch.eye(K, dtype=torch.float64, device=device)

    def project(Phi, b):
        WPhi = Phi * Wd[:, None]; G = Phi.T @ WPhi
        eps = eps_rel * torch.trace(G) / K
        theta = torch.linalg.solve(G + eps * eyeK, WPhi.T @ b.T)
        return (Phi @ theta).T, G

    def wrel(b, r):
        return torch.sum(Wd * (b - r) ** 2, 1) / (torch.sum(Wd * b ** 2, 1) + 1e-30)

    for step in range(steps):
        bi = tr[torch.randint(0, n_train, (batch,))]
        Phi = net(s_t); recon, G = project(Phi, E[bi])
        loss = wrel(E[bi], recon).mean() + beta * ((G - eyeK) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    with torch.no_grad():
        table = net(s_t).cpu().numpy()
        Phi = net(s_t); recon_te, G = project(Phi, E[te_idx])
        te = wrel(E[te_idx], recon_te).cpu().numpy()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, s_grid=s, table=table, K=K)
    per_family = {}
    te_full = np.full(n_materials, np.nan); te_full[te_idx.numpy()] = te
    for kind in set(d["kind"] for d in descs):
        idx = [j for j, d in enumerate(descs) if d["kind"] == kind and not np.isnan(te_full[j])]
        if idx:
            per_family[kind] = {"n": len(idx), "relL2_mean": float(np.mean(te_full[idx])),
                                "relL2_worst": float(np.max(te_full[idx]))}
    report = {"K": K, "n_materials": n_materials, "steps": steps,
              "test_relL2_mean": float(np.mean(te)), "test_relL2_worst": float(np.max(te)),
              "gram_offdiag_max": float(np.max(np.abs(G.cpu().numpy() - np.eye(K)))),
              "per_family": per_family, "table_path": str(out_path)}
    json.dump(report, open(out_path.parent / "fe_viscous_report.json", "w"), indent=2, default=float)
    print(json.dumps({k: report[k] for k in ["test_relL2_mean", "test_relL2_worst",
                                              "gram_offdiag_max", "per_family"]}, indent=2, default=float))
    return report


if __name__ == "__main__":
    train()
