"""Compile the warp kernel modules once, ahead of timed runs.

Warp compiles a module on first launch and caches it; any edit under
src/warpmpm/kernels changes the hash and the next run pays the compile
(about 4 s on CPU, about 2 minutes for CUDA). This script pays it now, for
the standard configurations, so a sweep never eats it mid-run.

    warpmpm-prewarm [--device cuda:0]

Run it once after `pip install` and after every pull that touches
src/warpmpm/kernels; the README's install line includes it.
"""
import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = ap.parse_args().device

    import numpy as np
    import torch
    import warp as wp
    wp.init()
    from warpmpm.kernels import MPM_Simulator_WARP

    pts = (np.random.rand(64, 3) * 0.2 + 0.4).astype(np.float32)
    configs = [
        ("plain", {}, {}),
        ("nclaw_semantics", {}, {"freeslip_bound": 3, "mass_eps": 1e-7,
                                 "empty_node_gravity": True, "mls_transfer": True,
                                 "particle_clip_cells": 0.5}),
    ]
    for name, extra, semantics in configs:
        t0 = time.time()
        s = MPM_Simulator_WARP(len(pts), device=dev)
        s.load_initial_data_from_torch(torch.from_numpy(pts),
                                       torch.full((len(pts),), 1e-6),
                                       n_grid=20, grid_lim=1.0, device=dev)
        s.set_parameters_dict({"material": "jelly", "density": 1000.0,
                               "E": 1e5, "nu": 0.2, "g": [0, 0, -9.8], **extra},
                              device=dev)
        s.finalize_mu_lam(device=dev)
        if semantics:
            s.set_grid_semantics(**semantics)
        s.p2g2p(0, 1e-4, device=dev)
        print(f"warmed {name} on {dev} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
