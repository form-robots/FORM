"""Guards for the function-encoder least-squares row (experiments/fe_ls).

Three of these lock traps that were live during the build and cost real time, so
they are the ones worth a test: a metric that rewards divergence, a metric that
reports a bitwise-identical replay as centimetres off, and a regularization
weight chosen by a criterion the data cannot support. The fourth checks the bake
that hands a recovered curve to the engine.

The granular fixture reads the trained basis from fe-weights/granular_mu_i.npz.
"""
from __future__ import annotations

import numpy as np
import pytest
from experiments.fe_ls.baseline import (
    CV_TIE_TOL,
    REG_GRID,
    _inbox_score,
    bake_mu_table,
    load_granular_fe,
    regularized_solve,
)


@pytest.fixture(scope="module")
def granular():
    return load_granular_fe()


def _dump(path, x):
    np.savez(path, x=np.asarray(x, dtype=np.float32))
    return path


def test_bake_puts_the_curve_on_the_engine_grid(granular):
    fe, prior = granular
    theta_bar = prior[0]
    baked = bake_mu_table(fe, theta_bar)
    assert baked["n_points"] == 256
    assert (baked["smin"], baked["smax"]) == (-4.0, 0.0)
    # the table is the curve itself, sampled on the pinned log10 I grid
    s = np.linspace(-4.0, 0.0, 256)
    expect = fe.phi(10.0 ** s) @ theta_bar
    assert np.allclose(baked["table"], np.clip(expect, 0.0, None), atol=1e-12)
    assert baked["n_negative_clipped"] == int((expect < 0.0).sum())


def test_bake_reports_negative_samples_rather_than_hiding_them(granular):
    fe, prior = granular
    baked = bake_mu_table(fe, -np.asarray(prior[0], float))
    assert baked["n_negative_clipped"] > 0
    assert baked["mu_min"] < 0.0
    assert min(baked["table"]) == 0.0        # clipped, and the clip is reported


def test_inbox_score_matches_the_nclaw_metric_when_nothing_escapes(tmp_path):
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.2, 0.8, size=(11, 40, 3))
    pred = truth + rng.normal(0.0, 1e-3, size=truth.shape)
    got = _inbox_score(_dump(tmp_path / "t.npz", truth),
                       _dump(tmp_path / "p.npz", pred))
    diff = truth.astype(np.float32) - pred.astype(np.float32)
    per_frame = (diff ** 2).mean(axis=(1, 2))
    assert got["n_particles_escaped_in_truth"] == 0
    assert got["mse_inbox"] == pytest.approx(float(per_frame[::5].mean()), rel=1e-6)


def test_inbox_score_drops_the_particles_the_truth_puts_outside_the_box(tmp_path):
    """One escaped particle carries the whole mean, which is the reason this exists."""
    truth = np.full((6, 10, 3), 0.5)
    truth[3, 0, 0] = 5.0                     # the blub artifact, in miniature
    pred = truth.copy()
    pred[3, 0, 0] = 0.5                      # a replay that does not follow it out
    got = _inbox_score(_dump(tmp_path / "t.npz", truth),
                       _dump(tmp_path / "p.npz", pred))
    assert got["n_particles_escaped_in_truth"] == 1
    assert got["n_particles_inbox"] == 9
    assert got["mse_inbox"] == 0.0
    assert got["rmse_inbox_mm"] == 0.0


def test_cv_selection_prefers_a_prior_to_an_exactly_fitting_noise_solve(granular):
    """A square-ish system fits its own noise; the held-out score must reject it.

    This is the discarded rule's failure in miniature: the in-sample residual is
    zero at weight zero and the cross-validated score is not, so a rule that
    reads the residual keeps the overfit and a rule that reads the held-out
    score does not.
    """
    fe, prior = granular
    rng = np.random.default_rng(0)
    K = fe.K
    n_windows, per_window = 6, 3
    groups = np.repeat(np.arange(n_windows), per_window)
    A = rng.normal(size=(n_windows * per_window, K))
    b = A @ prior[0] + rng.normal(0.0, 0.5, size=A.shape[0])
    res = regularized_solve(A, b, prior, lambda th: fe.phi(np.array([0.01])) @ th,
                            groups=groups, log=lambda *a: None)
    assert res["n_cv_folds"] == n_windows
    assert res["reg_weight"] > 0.0
    scores = [s["cv"] for s in res["reg_sweep"]]
    assert scores[0] is not None
    best = min(scores)
    picked = next(s for s in res["reg_sweep"]
                  if s["weight"] == res["reg_weight"])["cv"]
    assert picked <= CV_TIE_TOL * best
    assert len(res["reg_sweep"]) == len(REG_GRID)
