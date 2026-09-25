import os

import warp as wp

# The simulator is never differentiated (project invariant), so adjoint
# codegen is off by default; it cuts module compile time 2.5x. Set
# WARPMPM_ENABLE_BACKWARD=1 to compile adjoints for a project that
# consciously retires the invariant.
wp.set_module_options({"enable_backward":
                       os.environ.get("WARPMPM_ENABLE_BACKWARD", "0") == "1"})

from warpmpm.kernels.warp_utils import *  # noqa: F401,F403
import numpy as np
import math


# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)

@wp.func
def kirchoff_stress_stvk(F: wp.mat33, J: float, mu: float, lam: float):
    """St Venant-Kirchhoff with the same volumetric term the corotated form uses:
    tau = 2 mu F E + lam J (J - 1) I with E = (F^T F - I) / 2.

    This is a StVK implementation.
    """
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    E_green = 0.5 * (wp.transpose(F) * F - id)
    return 2.0 * mu * F * E_green + id * lam * J * (J - 1.0)


@wp.func
def kirchoff_stress_volume_linear(J: float, lam: float):
    """Purely volumetric elasticity, tau = lam J (J - 1) I, so Cauchy pressure is
    -lam (J - 1): linear in the volume change with no deviatoric term at all."""
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return id * (lam * J * (J - 1.0))


@wp.func
def kirchoff_stress_volume_ziran(J: float, mu: float, lam: float, gamma: float):
    """Volumetric equation of state tau = kappa (J - J^(1-gamma)) I with
    kappa = 2 mu / 3 + lam."""
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    kappa = 2.0 / 3.0 * mu + lam
    Jc = wp.max(J, 1.0e-6)
    return id * (kappa * (J - 1.0 / wp.pow(Jc, gamma - 1.0)))


@wp.func
def kirchoff_stress_water(
    J: float, bulk: float
):
    gamma = 1.1 # gamma is set to be a liitle greater than 1 for weakly compressible fluids
    pressure = -bulk * (wp.pow(J, -gamma) - 1.)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cauchy_stress = id * pressure
    return J * cauchy_stress

@wp.func
def kirchoff_stress_newtonian(
    J: float, bulk: float, L: wp.mat33, eta: float, tau_y: float,
    pk: float, pn: float
):
    # weakly-compressible generalized-Newtonian / Bingham / Herschel-Bulkley fluid:
    # Cauchy = -p I + 2 eta_app dev(D),
    #   eta_app = eta + tau_y / |gd|_eps + pk * |gd|_eps^(pn - 1)
    # with p from the water EOS, D = sym(L), and |gd|_eps = sqrt(2 dev(D):dev(D)
    # + eps^2) the regularized deviatoric shear rate. tau_y=0, pk=0 -> Newtonian;
    # pk>0, pn<1 -> shear-thinning power law. Kirchhoff = J * Cauchy.
    # eps matches MATH_REFERENCE.md and the identifier's EPS_GAMMA_DEFAULT: a law
    # identified at one regularization and replayed at another is a different law
    # at low shear.
    eps = 0.02
    gamma = 1.1
    pressure = -bulk * (wp.pow(J, -gamma) - 1.0)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    D = 0.5 * (L + wp.transpose(L))
    trd = (D[0, 0] + D[1, 1] + D[2, 2]) / 3.0
    D_dev = D - id * trd
    dd = (D_dev[0, 0] * D_dev[0, 0] + D_dev[0, 1] * D_dev[0, 1] + D_dev[0, 2] * D_dev[0, 2]
          + D_dev[1, 0] * D_dev[1, 0] + D_dev[1, 1] * D_dev[1, 1] + D_dev[1, 2] * D_dev[1, 2]
          + D_dev[2, 0] * D_dev[2, 0] + D_dev[2, 1] * D_dev[2, 1] + D_dev[2, 2] * D_dev[2, 2])
    gd = wp.sqrt(2.0 * dd + eps * eps)
    eta_app = eta + tau_y / gd + pk * wp.pow(gd, pn - 1.0)
    cauchy = id * pressure + 2.0 * eta_app * D_dev
    return J * cauchy

@wp.func
def kirchoff_stress_tabulated(
    J: float, bulk: float, L: wp.mat33,
    table: wp.array(dtype=float), smin: float, smax: float, n: int
):
    # Same weakly-compressible generalized-Newtonian form as kirchoff_stress_newtonian,
    # but eta_app(gd) is read from a TABLE on s = log10(gd) in [smin, smax] (n uniform
    # samples) with clamped linear interpolation, instead of the parametric HB formula.
    # This lets an FE-recovered eta_app(gd) curve be re-simulated directly. eps and the
    # EOS match the newtonian kernel exactly (0.02, MATH_REFERENCE.md).
    eps = 0.02
    gamma = 1.1
    pressure = -bulk * (wp.pow(J, -gamma) - 1.0)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    D = 0.5 * (L + wp.transpose(L))
    trd = (D[0, 0] + D[1, 1] + D[2, 2]) / 3.0
    D_dev = D - id * trd
    dd = (D_dev[0, 0] * D_dev[0, 0] + D_dev[0, 1] * D_dev[0, 1] + D_dev[0, 2] * D_dev[0, 2]
          + D_dev[1, 0] * D_dev[1, 0] + D_dev[1, 1] * D_dev[1, 1] + D_dev[1, 2] * D_dev[1, 2]
          + D_dev[2, 0] * D_dev[2, 0] + D_dev[2, 1] * D_dev[2, 1] + D_dev[2, 2] * D_dev[2, 2])
    gd = wp.sqrt(2.0 * dd + eps * eps)
    s = wp.log(gd) * 0.43429448190325176          # log10(gd)
    # map s -> fractional index in [0, n-1], clamped (flat extrapolation at both ends)
    idx = (s - smin) / (smax - smin) * float(n - 1)
    idx = wp.clamp(idx, 0.0, float(n - 1) - 1.0e-6)
    i0 = int(wp.floor(idx))
    frac = idx - float(i0)
    eta_app = table[i0] * (1.0 - frac) + table[i0 + 1] * frac
    cauchy = id * pressure + 2.0 * eta_app * D_dev
    return J * cauchy

@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def yield_table_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int):
    """Learned perfect-plasticity return map: sqrt(J2(dev tau)) <= h(p).

    h is tabulated on a LINEAR Kirchhoff-pressure grid in
    [eta_table_smin, eta_table_smax] Pa (the table storage is reused; a
    tabulated-yield material never coexists with a tabulated mu(I) or viscous
    material in one sim). A flat table reproduces von_mises_return_mapping
    exactly (cap on ||dev eps|| of h sqrt(2) / (2 mu) = sigma_y / (2 mu) when
    h = sigma_y / sqrt(2)); a line through the origin reproduces the
    cohesionless Drucker-Prager cone including its tension cutoff, which here
    is the branch h <= 0 with a positive strain trace.
    """
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)
    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    tr = epsilon[0] + epsilon[1] + epsilon[2]
    mean = tr / 3.0
    p_kirch = -(2.0 * model.mu[p] / 3.0 + model.lam[p]) * tr

    pmin = model.eta_table_smin
    pmax = model.eta_table_smax
    nt = model.eta_table_n
    x = wp.clamp(p_kirch, pmin, pmax)
    t = (x - pmin) / (pmax - pmin) * wp.float(nt - 1)
    i0 = wp.int(wp.floor(t))
    if i0 > nt - 2:
        i0 = nt - 2
    frac = t - wp.float(i0)
    h = model.eta_table[i0] * (1.0 - frac) + model.eta_table[i0 + 1] * frac

    if h <= 0.0 and tr > 0.0:
        # cohesionless tension: stress free, exactly the DP cutoff
        return U * wp.transpose(V)
    cap = wp.max(h, 0.0) * 0.70710678 / model.mu[p]   # h sqrt(2) / (2 mu)
    eps_hat = epsilon - wp.vec3(mean, mean, mean)
    n = wp.length(eps_hat) + 1e-12
    if n > cap:
        epsilon = epsilon - ((n - cap) / n) * eps_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]), 0.0, 0.0,
            0.0, wp.exp(epsilon[1]), 0.0,
            0.0, 0.0, wp.exp(epsilon[2]),
        )
        return U * sig_elastic * wp.transpose(V)
    return F_trial


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening[mat] > 0.5:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi[mat] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial

@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening[mat] * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening[mat] > 0.5:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi[mat] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def kirchoff_stress_hencky(
    U: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    # Kirchhoff stress for Hencky elasticity, coaxial with the left stretch:
    # tau = U diag(2 mu eps_i + lam tr(eps)) U^T with eps = log(sig)
    sig_c = wp.vec3(
        wp.max(sig[0], 1e-6), wp.max(sig[1], 1e-6), wp.max(sig[2], 1e-6)
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    tr_eps = eps[0] + eps[1] + eps[2]
    tau0 = 2.0 * mu * eps[0] + lam * tr_eps
    tau1 = 2.0 * mu * eps[1] + lam * tr_eps
    tau2 = 2.0 * mu * eps[2] + lam * tr_eps
    tau_diag = wp.mat33(tau0, 0.0, 0.0, 0.0, tau1, 0.0, 0.0, 0.0, tau2)
    return U * tau_diag * wp.transpose(U)


@wp.func
def mu_i_return_mapping(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int, dt: float
):
    # Local mu(I) rheology: Jop-Forterre-Pouliquen scalar mu(I) law combined
    # with a Dunatunga-Kamrin style Hencky multiplicative return.
    # Predictor is Hencky elastic, with a scalar plastic correction on the yield
    # surface tau_bar = mu(I) p where I = gamma_dot_p d sqrt(rho_s / p).
    # The return is deviatoric (non-dilatant): J and pressure are preserved.
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    sig_c = wp.vec3(
        wp.max(wp.abs(sig[0]), 1e-6),
        wp.max(wp.abs(sig[1]), 1e-6),
        wp.max(wp.abs(sig[2]), 1e-6),
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    tr_eps = eps[0] + eps[1] + eps[2]

    F_elastic = F_trial

    if tr_eps >= 0.0:
        # volumetric expansion: cohesionless free separation, stress free,
        # no tension memory
        F_elastic = U * wp.transpose(V)
    else:
        G = model.mu[p]
        lam = model.lam[p]
        J = sig_c[0] * sig_c[1] * sig_c[2]

        mean_eps = tr_eps / 3.0
        dev0 = eps[0] - mean_eps
        dev1 = eps[1] - mean_eps
        dev2 = eps[2] - mean_eps
        dev_norm = wp.sqrt(dev0 * dev0 + dev1 * dev1 + dev2 * dev2)

        p_K = -(2.0 * G / 3.0 + lam) * tr_eps  # > 0 in compression
        tau_bar_K = 2.0 * G * dev_norm / wp.sqrt(2.0)
        p_C = p_K / J

        mu_s = model.muI_mu_s[mat]
        dmu = model.muI_delta_mu[mat]
        I0 = model.muI_I0[mat]

        if tau_bar_K > mu_s * p_K:
            # plastic: bisect g(gdp) = tau_bar_K - G dt gdp - mu(I) p_K = 0,
            # strictly decreasing, sign change on [0, tau_bar_K / (G dt)]
            I_coef = model.muI_d[mat] * wp.sqrt(model.muI_rho_s[mat] / p_C)
            lo = float(0.0)
            hi = tau_bar_K / (G * dt)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                I_mid = mid * I_coef
                mu_mid = mu_s + dmu * I_mid / (I_mid + I0)
                g_mid = tau_bar_K - G * dt * mid - mu_mid * p_K
                if g_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
            gdp = 0.5 * (lo + hi)
            tau_bar_new = tau_bar_K - G * dt * gdp
            scale = tau_bar_new / wp.max(tau_bar_K, 1e-20)
            e0 = dev0 * scale + mean_eps
            e1 = dev1 * scale + mean_eps
            e2 = dev2 * scale + mean_eps
            sig_new = wp.mat33(
                wp.exp(e0), 0.0, 0.0, 0.0, wp.exp(e1), 0.0, 0.0, 0.0, wp.exp(e2)
            )
            F_elastic = U * sig_new * wp.transpose(V)

    return F_elastic


@wp.func
def mu_i_tabulated_return_mapping(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int, dt: float
):
    # IDENTICAL to mu_i_return_mapping (Dunatunga-Kamrin deviatoric return) except the
    # friction law mu(I) is read from a TABLE on s = log10(I) in [smin, smax] (n uniform
    # samples, clamped linear interpolation, flat extrapolation) instead of the parametric
    # Pouliquen formula mu_s + dmu I/(I+I0). This lets an NN-EUCLID- (or FE-) recovered
    # mu(I) curve be re-simulated directly. The mu-table is stored in model.eta_table
    # (re-used; a granular tabulated material and a viscous tabulated material are never
    # active in the same run). Everything else (pressure, bisection, stress) matches
    # material 9 exactly.
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    sig_c = wp.vec3(
        wp.max(wp.abs(sig[0]), 1e-6),
        wp.max(wp.abs(sig[1]), 1e-6),
        wp.max(wp.abs(sig[2]), 1e-6),
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    tr_eps = eps[0] + eps[1] + eps[2]

    F_elastic = F_trial

    if tr_eps >= 0.0:
        F_elastic = U * wp.transpose(V)
    else:
        G = model.mu[p]
        lam = model.lam[p]
        J = sig_c[0] * sig_c[1] * sig_c[2]

        mean_eps = tr_eps / 3.0
        dev0 = eps[0] - mean_eps
        dev1 = eps[1] - mean_eps
        dev2 = eps[2] - mean_eps
        dev_norm = wp.sqrt(dev0 * dev0 + dev1 * dev1 + dev2 * dev2)

        p_K = -(2.0 * G / 3.0 + lam) * tr_eps
        tau_bar_K = 2.0 * G * dev_norm / wp.sqrt(2.0)
        p_C = p_K / J

        smin = model.eta_table_smin
        smax = model.eta_table_smax
        nt = model.eta_table_n

        # mu at zero shear (table value at smin) sets the static yield check
        mu_static = model.eta_table[0]
        if tau_bar_K > mu_static * p_K:
            I_coef = model.muI_d[mat] * wp.sqrt(model.muI_rho_s[mat] / p_C)
            lo = float(0.0)
            hi = tau_bar_K / (G * dt)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                I_mid = mid * I_coef
                # mu(I_mid) by clamped linear interpolation on s = log10(I)
                s = wp.log(wp.max(I_mid, 1e-12)) * 0.43429448190325176
                idx = (s - smin) / (smax - smin) * float(nt - 1)
                idx = wp.clamp(idx, 0.0, float(nt - 1) - 1.0e-6)
                i0 = int(wp.floor(idx))
                frac = idx - float(i0)
                mu_mid = model.eta_table[i0] * (1.0 - frac) + model.eta_table[i0 + 1] * frac
                g_mid = tau_bar_K - G * dt * mid - mu_mid * p_K
                if g_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
            gdp = 0.5 * (lo + hi)
            tau_bar_new = tau_bar_K - G * dt * gdp
            scale = tau_bar_new / wp.max(tau_bar_K, 1e-20)
            e0 = dev0 * scale + mean_eps
            e1 = dev1 * scale + mean_eps
            e2 = dev2 * scale + mean_eps
            sig_new = wp.mat33(
                wp.exp(e0), 0.0, 0.0, 0.0, wp.exp(e1), 0.0, 0.0, 0.0, wp.exp(e2)
            )
            F_elastic = U * sig_new * wp.transpose(V)

    return F_elastic


@wp.func
def j_cs_of_I(model: MPMModelStruct, mat: int, I: float):
    # critical-state STRESS-FREE volume ratio the material relaxes toward at inertial
    # number I. Phi_c(I) = Phi_init*(1 - chi I/(I+I0)) (looser at high I), and
    # J_cs = Phi_init/Phi_c = 1/(1 - chi I/(I+I0)) >= 1 (dilates with shear rate).
    I0 = model.muI_I0[mat]
    chi = model.muI_phi_chi[mat]
    denom = wp.max(1.0 - chi * I / (I + I0), 1e-3)
    return 1.0 / denom


@wp.func
def mu_i_phi_pressure(model: MPMModelStruct, mat: int, J: float, Jp_ref: float):
    # compaction pressure from the elastic compression relative to the (slowly
    # relaxing) stress-free reference volume Jp_ref: p = K (Jp_ref/J - 1)_+ . This is
    # the elastic EOS shifted by the critical-state reference, so it is smooth and
    # monotone in J (stable, like the mu(I) elastic EOS) and the rate dependence
    # enters only through the slow drift of Jp_ref, never an instantaneous I feedback.
    K = model.E[mat] / (3.0 * (1.0 - 2.0 * model.nu[mat]))
    p = K * (Jp_ref / wp.max(J, 1e-6) - 1.0)
    return wp.max(p, 0.0)


@wp.func
def mu_i_phi_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int, mat: int, dt: float
):
    # Compressible mu(I)-Phi(I). The scalar mu(I) law is Jop-Forterre-Pouliquen;
    # the compaction EOS, lagged reference volume, and dilatancy rate beta form
    # the closure. Deviatoric mu(I) yield acts against the COMPACTION pressure 
    # p = K(Phi/Phi_c(I)-1)_+ (not the elastic EOS), so the solid fraction relaxes 
    # toward the rate-dependent critical state Phi_c(I).
    # The volumetric strain is preserved in the return (the volume is physical, set by the
    # flow); the pressure comes from Phi via the compaction law, applied in the stress.
    # The realized inertial number I is stored in particle_Jp for the stress and the
    # next step's Phi_c lag.
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)
    sig_c = wp.vec3(
        wp.max(wp.abs(sig[0]), 1e-6),
        wp.max(wp.abs(sig[1]), 1e-6),
        wp.max(wp.abs(sig[2]), 1e-6),
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    J = sig_c[0] * sig_c[1] * sig_c[2]
    tr_eps = eps[0] + eps[1] + eps[2]
    mean_eps = tr_eps / 3.0
    dev0 = eps[0] - mean_eps
    dev1 = eps[1] - mean_eps
    dev2 = eps[2] - mean_eps
    dev_norm = wp.sqrt(dev0 * dev0 + dev1 * dev1 + dev2 * dev2)

    G = model.mu[p]
    # stress-free reference volume Jp_ref (particle_Jp); the solver inits it to 0,
    # which we read as the natural value 1.0 (undeformed reference).
    Jp_ref = state.particle_Jp[p]
    if Jp_ref < 0.5:
        Jp_ref = 1.0
    p_comp = mu_i_phi_pressure(model, mat, J, Jp_ref)     # compaction pressure (Cauchy)
    beta = 8.0                                            # dilatancy rate PER UNIT PLASTIC STRAIN

    F_elastic = F_trial
    state.particle_Jp[p] = Jp_ref                         # default: reference frozen (no plastic strain)

    if p_comp <= 0.0:
        # at or below the reference volume: loose / stress free, only relax deviatoric;
        # preserve the volume (do not inject volume by resetting J to 1).
        s_iso = wp.exp(mean_eps)
        F_elastic = U * wp.mat33(s_iso, 0.0, 0.0, 0.0, s_iso, 0.0, 0.0, 0.0, s_iso) * wp.transpose(V)
    else:
        p_K = p_comp * J                                  # Kirchhoff pressure scale
        tau_bar_K = 2.0 * G * dev_norm / wp.sqrt(2.0)
        mu_s = model.muI_mu_s[mat]
        dmu = model.muI_delta_mu[mat]
        I0 = model.muI_I0[mat]
        if tau_bar_K > mu_s * p_K:
            I_coef = model.muI_d[mat] * wp.sqrt(model.muI_rho_s[mat] / p_comp)
            lo = float(0.0)
            hi = tau_bar_K / (G * dt)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                I_mid = mid * I_coef
                mu_mid = mu_s + dmu * I_mid / (I_mid + I0)
                g_mid = tau_bar_K - G * dt * mid - mu_mid * p_K
                if g_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
            gdp = 0.5 * (lo + hi)
            I_real = gdp * I_coef
            # DILATANCY (critical state) tied to PLASTIC STRAIN, not wall-clock: the
            # stress-free reference volume relaxes toward J_cs(I) by an amount
            # proportional to the plastic shear increment gdp*dt. Static / jittering
            # material (gdp~0) does NOT drift, so it compacts and holds hydrostatic
            # pressure; only sustained flow dilates. dgamma capped at 1 for stability.
            dgamma = wp.min(beta * gdp * dt, 1.0)
            j_cs = j_cs_of_I(model, mat, I_real)
            state.particle_Jp[p] = Jp_ref + dgamma * (j_cs - Jp_ref)
            tau_bar_new = tau_bar_K - G * dt * gdp
            scale = tau_bar_new / wp.max(tau_bar_K, 1e-20)
            e0 = dev0 * scale + mean_eps
            e1 = dev1 * scale + mean_eps
            e2 = dev2 * scale + mean_eps
            sig_new = wp.mat33(
                wp.exp(e0), 0.0, 0.0, 0.0, wp.exp(e1), 0.0, 0.0, 0.0, wp.exp(e2)
            )
            F_elastic = U * sig_new * wp.transpose(V)
    return F_elastic


@wp.func
def kirchoff_stress_mu_i_phi(
    U: wp.mat33, sig: wp.vec3, G: float, p_comp: float
):
    # deviatoric Hencky (mu(I)-limited via the returned F) + compaction isotropic
    # tau = U diag(2 G dev_eps_i) U^T  -  p_comp * J * I   (Cauchy pressure = p_comp)
    sig_c = wp.vec3(wp.max(sig[0], 1e-6), wp.max(sig[1], 1e-6), wp.max(sig[2], 1e-6))
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    J = sig_c[0] * sig_c[1] * sig_c[2]
    mean_eps = (eps[0] + eps[1] + eps[2]) / 3.0
    iso = -p_comp * J
    tau0 = 2.0 * G * (eps[0] - mean_eps) + iso
    tau1 = 2.0 * G * (eps[1] - mean_eps) + iso
    tau2 = 2.0 * G * (eps[2] - mean_eps) + iso
    tau_diag = wp.mat33(tau0, 0.0, 0.0, 0.0, tau1, 0.0, 0.0, 0.0, tau2)
    return U * tau_diag * wp.transpose(U)


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int, mat: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha[mat]
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.func
def compose_dp_return_mapping(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int
):
    """Drucker-Prager return with cohesion.

    The principal stretches are clamped at a 0.05 threshold, and the deviatoric
    norm is floored at 1e-12 to avoid NaN. At cohesion 0 this
    agrees with sand_return_mapping (material 2) up to that clamp.
    """
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    sig_c = wp.vec3(wp.max(sig[0], 0.05), wp.max(sig[1], 0.05),
                    wp.max(sig[2], 0.05))
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    tr = eps[0] + eps[1] + eps[2]
    eps_hat = eps - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    eps_hat_norm = wp.max(wp.length(eps_hat), 1.0e-12)

    coh = model.compose_cohesion[mat]
    shifted_tr = tr - coh * 3.0
    delta_gamma = eps_hat_norm + (3.0 * model.lam[p] + 2.0 * model.mu[p]) / (
        2.0 * model.mu[p]
    ) * shifted_tr * model.alpha[mat]

    eps_out = wp.vec3(coh, coh, coh)     # tensile side: stretch forgotten
    if shifted_tr < 0.0:
        eps_out = eps - eps_hat * (wp.max(delta_gamma, 0.0) / eps_hat_norm)
    s_new = wp.vec3(wp.exp(eps_out[0]), wp.exp(eps_out[1]), wp.exp(eps_out[2]))
    return U * wp.diag(s_new) * wp.transpose(V)


@wp.func
def compose_return_mapping(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int
):
    """Plasticity of the composed material (14), one PLASTICITY_KIND per type."""
    kind = model.compose_plastic[mat]
    F_out = F_trial                                   # 0: identity
    if kind == 1:
        F_out = von_mises_return_mapping(F_trial, model, p, mat)
    if kind == 2:
        F_out = compose_dp_return_mapping(F_trial, model, p, mat)
    if kind == 3:
        # shape forgotten, volume kept: their SigmaPlasticity, which is what our
        # fluid materials do too. J is floored where their cube root would be NaN.
        Jcbr = wp.pow(wp.max(wp.determinant(F_trial), 1.0e-9), 1.0 / 3.0)
        F_out = wp.mat33(Jcbr, 0.0, 0.0, 0.0, Jcbr, 0.0, 0.0, 0.0, Jcbr)
    return F_out


@wp.func
def compose_stress(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, J: float,
    model: MPMModelStruct, p: int, mat: int
):
    """Elasticity of the composed material (14), one ELASTICITY_KIND per type."""
    kind = model.compose_elastic[mat]
    stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if kind == 0:
        stress = kirchoff_stress_FCR(F, U, V, J, model.mu[p], model.lam[p])
    if kind == 1:
        stress = kirchoff_stress_hencky(U, sig, model.mu[p], model.lam[p])
    if kind == 2:
        stress = kirchoff_stress_stvk(F, J, model.mu[p], model.lam[p])
    if kind == 3:
        stress = kirchoff_stress_volume_linear(J, model.lam[p])
    if kind == 4:
        stress = kirchoff_stress_volume_ziran(
            J, model.mu[p], model.lam[p], model.compose_gamma[mat])
    return stress


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    mat = state.particle_material[p]
    model.mu[p] = model.E[mat] / (2.0 * (1.0 + model.nu[mat]))
    model.lam[p] = model.E[mat] * model.nu[mat] / ((1.0 + model.nu[mat]) * (1.0 - 2.0 * model.nu[mat]))

@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct, lo: wp.vec3i):
    grid_x, grid_y, grid_z = wp.tid()
    grid_x = grid_x + lo[0]
    grid_y = grid_y + lo[1]
    grid_z = grid_z + lo[2]
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


# ---------------------------------------------------------------------------
# CPIC thin-boundary support (Hu et al., SIGGRAPH 2018, Section 5; MIT reference
# implementation yuanming-hu/taichi_mpm; design notes in the paper at
# literature/mls_mpm_cpic_2018.pdf). Particle tags are reconstructed from node
# tags inside the transfer funcs rather than stored: the fused kernel advects the
# position between its g2p and p2g halves, so a stored tag would be one substep
# stale (and reconstruction keeps permute_particles untouched). One deviation
# from the paper: the rigid impulse accumulates during G2P from the ghost
# substitution, m w (v_p - ghost), the same quantity Section 5.4 accumulates on
# the P2G side; G2P is where the ghost is already computed.
# ---------------------------------------------------------------------------

@wp.func
def cdf_pair_incompatible(pc: int, nt: int) -> bool:
    """Particle-node pair incompatible when any lane has valid bits on BOTH and
    differing side bits (paper 5.3.4). 0x55 selects the four lanes' valid bits."""
    both_valid = pc & nt & 0x55
    sides_differ = (pc ^ nt) >> 1
    return (both_valid & sides_differ) != 0


@wp.func
def cdf_stencil_near(state: MPMStateStruct, base_x: int, base_y: int,
                     base_z: int) -> bool:
    """Coarse pretest for the tag vote: true when any CDF_BLOCK^3 block touched by
    the particle's 3x3x3 stencil carries a tag (cur or prev grid; the mask kernel
    marks both). The early-out is exact, not approximate: an all-empty stencil
    yields pc == 0 through the full vote and persistence path too, so skipping
    them changes nothing. At most 8 block reads replace the vote's ~108 loads."""
    bx0 = base_x >> CDF_BLOCK_SHIFT_C
    bx1 = (base_x + 2) >> CDF_BLOCK_SHIFT_C
    by0 = base_y >> CDF_BLOCK_SHIFT_C
    by1 = (base_y + 2) >> CDF_BLOCK_SHIFT_C
    bz0 = base_z >> CDF_BLOCK_SHIFT_C
    bz1 = (base_z + 2) >> CDF_BLOCK_SHIFT_C
    for bi in range(bx0, bx1 + 1):
        for bj in range(by0, by1 + 1):
            for bk in range(bz0, bz1 + 1):
                if state.grid_cdf_block[bi, bj, bk] != 0:
                    return True
    return False


@wp.func
def cdf_particle_tag(state: MPMStateStruct, base_x: int, base_y: int, base_z: int,
                     w: wp.mat33, use_prev: int):
    """Reconstruct a particle's tag from node tags: affinity = any valid node in the
    kernel; side = sign of the distance-weighted vote sum w * d * side (paper Eq. 21;
    the d factor de-weights near-surface nodes whose side is ambiguous)."""
    votes = wp.vec4(0.0, 0.0, 0.0, 0.0)
    seen = int(0)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_x + i
                iy = base_y + j
                iz = base_z + k
                t = int(0)
                d = float(0.0)
                if use_prev == 1:
                    t = state.grid_cdf_tag_prev[ix, iy, iz]
                    d = state.grid_cdf_d_prev[ix, iy, iz]
                else:
                    t = state.grid_cdf_tag[ix, iy, iz]
                    d = state.grid_cdf_d[ix, iy, iz]
                if t == 0:
                    continue
                weight = w[0, i] * w[1, j] * w[2, k]
                for lane in range(0, 4):
                    if ((t >> (2 * lane)) & 1) != 0:
                        seen = seen | (1 << (2 * lane))
                        s = -1.0
                        if ((t >> (2 * lane + 1)) & 1) != 0:
                            s = 1.0
                        votes[lane] = votes[lane] + weight * d * s
    pc = int(0)
    for lane in range(0, 4):
        if ((seen >> (2 * lane)) & 1) != 0:
            pc = pc | (1 << (2 * lane))
            if votes[lane] >= 0.0:
                pc = pc | (1 << (2 * lane + 1))
    return pc


@wp.func
def cdf_persist_tag(old: int, fresh: int) -> int:
    """Color persistence (paper 5.3.3): while a lane's affinity remains anywhere in
    the kernel, the particle keeps its previous SIDE for that lane; the fresh vote
    decides only for newly acquired affinities. Without this, a particle nudged
    across the surface by advection error flips allegiance and falls through."""
    pc = int(0)
    for lane in range(0, 4):
        if ((fresh >> (2 * lane)) & 1) != 0:
            pc = pc | (1 << (2 * lane))
            if ((old >> (2 * lane)) & 1) != 0:
                pc = pc | (old & (1 << (2 * lane + 1)))
            else:
                pc = pc | (fresh & (1 << (2 * lane + 1)))
    return pc


@wp.func
def cdf_particle_ghost(state: MPMStateStruct, model: MPMModelStruct, dt: float,
                       p: int, pc: int, base_x: int, base_y: int, base_z: int,
                       w: wp.mat33, dw: wp.mat33):
    """The collided particle velocity used at incompatible nodes (paper 5.5): the
    particle's velocity projected against its primary surface (sticky/slip/
    separable + Coulomb, the SDF collide kernel's projection), plus a separation
    push when the reconstructed signed distance is at or below a small epsilon
    (the paper's Delta-t c n term). Geometry comes from an MLS-style reconstruction
    over the prev tag grids: d_hat = sum w s d, n_hat = normalize(sum dweight s d),
    primary lane = owner lane of the heaviest valid node."""
    d_hat = float(0.0)
    n_acc = wp.vec3(0.0, 0.0, 0.0)
    best_w = float(0.0)
    lane_p = int(0)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_x + i
                iy = base_y + j
                iz = base_z + k
                t = state.grid_cdf_tag_prev[ix, iy, iz]
                if t == 0:
                    continue
                own = (t >> CDF_OWNER_SHIFT_C) & 3
                if ((t >> (2 * own)) & 1) == 0:
                    continue
                weight = w[0, i] * w[1, j] * w[2, k]
                s = -1.0
                if ((t >> (2 * own + 1)) & 1) != 0:
                    s = 1.0
                d = state.grid_cdf_d_prev[ix, iy, iz]
                d_hat = d_hat + weight * s * d
                dweight = compute_dweight(model, w, dw, i, j, k)
                n_acc = n_acc + dweight * s * d
                if weight > best_w:
                    best_w = weight
                    lane_p = own
    # project the particle's FREE velocity for this substep (old velocity plus the
    # body-force increment): grid nodes receive gravity before their BCs, and
    # without the same increment here a resting contact has no normal approach and
    # the Coulomb term has no budget (friction would silently do nothing)
    v_free_p = state.particle_v[p] + dt * model.gravitational_accelaration
    ghost = v_free_p
    if wp.length(n_acc) > 1.0e-12:
        n_hat = wp.normalize(n_acc)                 # points toward the + side
        side = 1.0
        if ((pc >> (2 * lane_p + 1)) & 1) == 0:
            side = -1.0
        n_par = n_hat * side                        # away from the surface, particle side
        rel = state.particle_x[p] - state.cdf_lane_center_prev[lane_p]
        v_surf = (state.cdf_lane_velocity_prev[lane_p]
                  + wp.cross(state.cdf_lane_omega_prev[lane_p], rel))
        v_rel = v_free_p - v_surf
        vn = wp.dot(v_rel, n_par)
        v_tan = v_rel - vn * n_par
        # separation push: signed distance on the particle's side at or below
        # epsilon means penetration by advection error; push out along n_par,
        # capped at half a cell per substep. The push is part of the NORMAL
        # impulse, so it feeds the Coulomb budget below; capping friction at
        # mu * (approach removed) alone would reference only this substep's
        # ~g dt approach and friction could never anchor a resting body.
        c = float(0.0)
        d_side = d_hat * side
        eps = 0.05 * model.dx
        if d_side < eps:
            c = wp.min((eps - d_side) / dt, 0.5 * model.dx / dt)
        stype = state.cdf_lane_type[lane_p]
        if stype == 0:                              # sticky
            ghost = v_surf + c * n_par
        elif stype == 1:                            # slip (frictionless, bilateral)
            ghost = v_surf + v_tan + c * n_par
        else:                                       # separable + Coulomb friction
            dvn = wp.max(-vn, 0.0) + c              # normal impulse per unit mass
            tlen = wp.length(v_tan)
            if tlen > 1.0e-12 and dvn > 0.0:
                scale = wp.max(0.0, tlen - state.cdf_lane_friction[lane_p] * dvn) / tlen
                v_tan = v_tan * scale
            ghost = v_surf + v_tan + wp.max(vn, 0.0) * n_par + c * n_par
    return ghost


@wp.func
def p2g_particle(state: MPMStateStruct, model: MPMModelStruct, dt: float, p: int):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    stress = state.particle_stress[p]
    grid_pos = state.particle_x[p] * model.inv_dx
    # floor, not toward-zero truncation: with periodic x a particle can sit at
    # grid_pos < 0.5 where int() picks base 0 instead of -1 and the B-spline
    # weights go invalid; for grid_pos - 0.5 >= 0 (every non-periodic scene, by
    # the edge guard) floor and int agree bitwise
    base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
    base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
    base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
    fx = grid_pos - wp.vec3(
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.matrix_from_cols(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.matrix_from_cols(
        fx - wp.vec3(1.5),
        -2.0 * (fx - wp.vec3(1.0)),
        fx - wp.vec3(0.5),
    )

    pc = int(0)
    if model.n_cdf > 0:
        # block-mask pretest first: far from any tagged region the vote is a
        # guaranteed pc == 0, so skip its 27-node reads (both operands of `or`
        # evaluate in warp; the mask read is safe whenever n_cdf > 0)
        if model.cdf_mask_off != 0 or cdf_stencil_near(state, base_pos_x,
                                                       base_pos_y, base_pos_z):
            pc = cdf_persist_tag(
                state.particle_cdf_tag[p],
                cdf_particle_tag(state, base_pos_x, base_pos_y, base_pos_z, w, 0))

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (
                    wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                ) * model.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                if model.periodic_x != 0:  # chute flows: x wraps onto the grid
                    ix = ((ix % model.grid_dim_x) + model.grid_dim_x) % model.grid_dim_x
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                dweight = compute_dweight(model, w, dw, i, j, k)
                C = state.particle_C[p]
                # if model.rpic = 0, standard apic
                C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                    C - wp.transpose(C)
                )
                if model.rpic_damping < -0.001:
                    # standard pic
                    C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                elastic_force = -state.particle_vol[p] * stress * dweight
                v_in_add = (
                    weight
                    * state.particle_mass[p]
                    * (state.particle_v[p] + C * dpos)
                    + dt * elastic_force
                )
                if model.mls_transfer != 0:
                    # MLS-MPM internal force (Hu et al. 2018):
                    # the stress rides the SAME affine channel as C,
                    # f_i = -4 inv_dx^2 V_p w_i sigma (x_i - x_p), instead of
                    # -V_p sigma grad w_i. Both sum to zero net force and the
                    # same net torque per particle; they differ per node by a
                    # second-difference term.
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + weight
                        * (
                            (-dt * state.particle_vol[p] * 4.0
                             * model.inv_dx * model.inv_dx)
                            * stress
                        )
                        * dpos
                    )
                # Nested on purpose. Warp does not short-circuit `and`. When
                # n_cdf == 0 the tag grids are (1,1,1) placeholders and must not
                # be indexed; pc != 0 implies the real grids exist.
                if pc != 0:
                    nt = state.grid_cdf_tag[ix, iy, iz]
                    if cdf_pair_incompatible(pc, nt):
                        # CPIC: no transfer across the thin boundary. The blocked
                        # deposit IS the momentum flux the sheet interrupts, so it
                        # is the reaction impulse on the sheet: a resting column's
                        # weight arrives through the stress term and an impact's
                        # m*v through the momentum term. (The ghost projection on
                        # the gather side corrects only the contact layer's
                        # kinematics; accounting there under-reads a deep column
                        # by the layer-to-column mass ratio.)
                        own = (nt >> CDF_OWNER_SHIFT_C) & 3
                        xw = wp.vec3(wp.float(ix), wp.float(iy),
                                     wp.float(iz)) * model.dx
                        wp.atomic_add(state.cdf_reaction_force, own, v_in_add)
                        wp.atomic_add(
                            state.cdf_reaction_torque, own,
                            wp.cross(xw - state.cdf_lane_center[own], v_in_add))
                        continue
                wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                wp.atomic_add(
                    state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                )


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        p2g_particle(state, model, dt, p)


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float, lo: wp.vec3i
):
    grid_x, grid_y, grid_z = wp.tid()
    grid_x = grid_x + lo[0]
    grid_y = grid_y + lo[1]
    grid_z = grid_z + lo[2]
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


# Grid operator with freeslip wall semantics: mass division, gravity and the wall
# clamp in one kernel. Free-surface and wall behavior differ from the default
# path in three independent ways (approach-only wall clamp, eps-softened mass
# division, free-fall velocity on empty nodes), each separately switchable. Reached
# only once set_grid_semantics has asked for one of them; the default kernel
# above is untouched.
@wp.kernel
def grid_normalization_and_gravity_freeslip(
    state: MPMStateStruct, model: MPMModelStruct, dt: float, lo: wp.vec3i
):
    grid_x, grid_y, grid_z = wp.tid()
    grid_x = grid_x + lo[0]
    grid_y = grid_y + lo[1]
    grid_z = grid_z + lo[2]
    m = state.grid_m[grid_x, grid_y, grid_z]
    # a node left at zero stays zero through the clamp below (no component points
    # into a wall), so the empty-node branch needs no early exit
    v = wp.vec3(0.0, 0.0, 0.0)
    if m > 0.0:
        # eps-softened division: damps the fringe nodes whose mass is a rounding
        # error of the stencil rather than a physical share
        v = state.grid_v_in[grid_x, grid_y, grid_z] / (m + model.grid_mass_eps) \
            + model.gravitational_accelaration * dt
    else:
        # a zero-mass node carries free-fall velocity into g2p, so a free surface
        # is not gathered against a grid of resting nodes
        if model.empty_node_gravity != 0:
            v = model.gravitational_accelaration * dt

    if model.freeslip_bound > 0:
        b = model.freeslip_bound
        # Approach-only clamp: the wall-normal velocity is zeroed only where it
        # points INTO the wall, allowing free separation.
        # Asymmetry matches external reference (low face clamps b layers, high face b-1).
        if grid_x < b and v[0] < 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        if grid_y < b and v[1] < 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        if grid_z < b and v[2] < 0.0:
            v = wp.vec3(v[0], v[1], 0.0)
        if grid_x > model.grid_dim_x - b and v[0] > 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        if grid_y > model.grid_dim_y - b and v[1] > 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        if grid_z > model.grid_dim_z - b and v[2] > 0.0:
            v = wp.vec3(v[0], v[1], 0.0)

    state.grid_v_out[grid_x, grid_y, grid_z] = v


@wp.func
def g2p_particle(state: MPMStateStruct, model: MPMModelStruct, dt: float, p: int):
    if True:
        grid_pos = state.particle_x[p] * model.inv_dx
        # floor, not toward-zero truncation: with periodic x a particle can sit at
        # grid_pos < 0.5 where int() picks base 0 instead of -1 and the B-spline
        # weights go invalid; for grid_pos - 0.5 >= 0 (every non-periodic scene, by
        # the edge guard) floor and int agree bitwise
        base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.matrix_from_cols(
            fx - wp.vec3(1.5),
            -2.0 * (fx - wp.vec3(1.0)),
            fx - wp.vec3(0.5),
        )
        pc = int(0)
        ghost = wp.vec3(0.0, 0.0, 0.0)
        if model.n_cdf > 0:
            # gather-side tags come from the PREV stamp (the pose the scattered
            # grid state was built under); the split pipeline copies cur -> prev
            # after stamping so both reads see the same pose there. The block-mask
            # pretest covers cur AND prev, so skipping the vote here is exact; the
            # tag store still runs so a particle leaving all tagged regions drops
            # its persisted tag exactly as the full path would.
            if model.cdf_mask_off != 0 or cdf_stencil_near(state, base_pos_x,
                                                           base_pos_y, base_pos_z):
                pc = cdf_persist_tag(
                    state.particle_cdf_tag[p],
                    cdf_particle_tag(state, base_pos_x, base_pos_y, base_pos_z, w, 1))
            state.particle_cdf_tag[p] = pc
            if pc != 0:
                ghost = cdf_particle_ghost(state, model, dt, p, pc,
                                           base_pos_x, base_pos_y, base_pos_z, w, dw)

        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    if model.periodic_x != 0:  # chute flows: x wraps onto the grid
                        ix = ((ix % model.grid_dim_x) + model.grid_dim_x) % model.grid_dim_x
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    # Nested on purpose: warp does not short-circuit `and`; see p2g_particle.
                    incompat = False
                    if pc != 0:
                        incompat = cdf_pair_incompatible(
                            pc, state.grid_cdf_tag_prev[ix, iy, iz])
                    if incompat:
                        # CPIC ghost fill: the particle's collided velocity stands
                        # in for the incompatible node with the node's own weight
                        # (partition of unity preserved for v, C, and L). The
                        # reaction wrench is accounted on the SCATTER side (the
                        # blocked p2g deposits); accumulating the projection's
                        # momentum change here as well would double-count the
                        # m*v flux of an impact.
                        grid_v = ghost
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        # A slow substep can be smaller than one float32 ULP of world position.
        # Retain the lost increment rather than freezing or systematically
        # rounding the motion. This changes arithmetic, not the integration law.
        x_old = state.particle_x[p]
        dx = dt * new_v - state.particle_x_roundoff[p]
        x_new = x_old + dx
        x_error = (x_new - x_old) - dx
        if model.particle_clip_cells > 0.0:
            # the advected position is clamped into
            # [clip * dx, grid_lim - clip * dx], a hard backstop behind the wall
            # clamp that keeps a particle's stencil inside the grid
            cb = model.particle_clip_cells * model.dx
            hi = model.grid_lim - cb
            for axis in range(3):
                if x_new[axis] < cb or x_new[axis] > hi:
                    x_error[axis] = 0.0
            x_new = wp.vec3(
                wp.clamp(x_new[0], cb, hi),
                wp.clamp(x_new[1], cb, hi),
                wp.clamp(x_new[2], cb, hi),
            )
        if model.periodic_x != 0:
            lx = wp.float(model.grid_dim_x) * model.dx
            if x_new[0] < 0.0:
                x_new = wp.vec3(x_new[0] + lx, x_new[1], x_new[2])
                x_error[0] = 0.0
            if x_new[0] >= lx:
                x_new = wp.vec3(x_new[0] - lx, x_new[1], x_new[2])
                x_error[0] = 0.0
        state.particle_x[p] = x_new
        state.particle_x_roundoff[p] = x_error
        state.particle_C[p] = new_C
        # new_F is the discrete velocity gradient L with L_ij = dv_i/dx_j
        # (sum_node v_node[i] * d w_node/dx_j). Stored for the TrackEUCLID dump.
        state.particle_L[p] = new_F
        F_rate = new_F
        if model.mls_transfer != 0:
            # MLS-MPM advances F with the APIC affine matrix C rather than with
            # the gradient-weight velocity gradient L; the two agree in the
            # summed sense but differ per particle. particle_L keeps L either
            # way, which is what the TrackEUCLID dump documents.
            F_rate = new_C
        # Form the small increment before adding it to F: rounding I + dt L
        # first can erase slow strain completely. Compensate the final addition.
        F_old = state.particle_F[p]
        dF = (F_rate * F_old) * dt - state.particle_F_roundoff[p]
        F_tmp = F_old + dF
        state.particle_F_roundoff[p] = (F_tmp - F_old) - dF
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, F_rate, dt)


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        if state.particle_material[p] == 7 or state.particle_material[p] == 8:  # stationary/rigid handled separately
            return
        g2p_particle(state, model, dt, p)


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.func
def stress_update_particle(state: MPMStateStruct, model: MPMModelStruct, dt: float, p: int):
    if True:
        mat = state.particle_material[p]

        # apply return mapping
        if mat == 1:  # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p, mat
            )
        elif mat == 5:  # plasticine
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 9:  # local mu(I) rheology (TrackEUCLID)
            state.particle_F[p] = mu_i_return_mapping(
                state.particle_F_trial[p], model, p, mat, dt
            )
        elif mat == 13:  # tabulated mu(I) (NN-EUCLID / FE-recovered curve, TrackEUCLID)
            state.particle_F[p] = mu_i_tabulated_return_mapping(
                state.particle_F_trial[p], model, p, mat, dt
            )
        elif mat == 15:  # tabulated yield surface h(p) (FE-recovered plasticity)
            state.particle_F[p] = yield_table_return_mapping(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 11:  # compressible mu(I)-Phi(I) dilatancy (TrackEUCLID)
            state.particle_F[p] = mu_i_phi_return_mapping(
                state.particle_F_trial[p], state, model, p, mat, dt
            )
        elif mat == 6 or mat == 10 or mat == 12:  # fluid / newtonian / tabulated (viscous)
            J = wp.determinant(state.particle_F_trial[p])
            Jcbr = J**(1.0 / 3.0)
            state.particle_F[p] = wp.mat33(Jcbr, 0.0, 0.0, 0.0, Jcbr, 0.0, 0.0, 0.0, Jcbr)
        elif mat == 14:  # composed: elasticity kind x plasticity kind
            state.particle_F[p] = compose_return_mapping(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 7 or mat == 8:  # stationary / rigid, no deformation
            state.particle_F[p] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        else:  # jelly (0)
            state.particle_F[p] = state.particle_F_trial[p]

        # A constitutive projection replaces the elastic state; the residual
        # from its pre-projection integration no longer belongs to that state.
        projected = False
        for row in range(3):
            for col in range(3):
                if state.particle_F[p][row, col] != state.particle_F_trial[p][row, col]:
                    projected = True
        if projected:
            state.particle_F_roundoff[p] = wp.mat33(0.0)

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # the SVD feeds the solid stress paths only; fluids (6), newtonian/Bingham (10),
        # tabulated viscous (12) and stationary/rigid (7, 8) never read U/V/sig, and the
        # per-particle svd3 dominates their stress cost
        if mat != 6 and mat != 7 and mat != 8 and mat != 10 and mat != 12:
            wp.svd3(state.particle_F[p], U, sig, V)
        if mat == 0 or mat == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if mat == 1 or mat == 15:
            # Coaxial Hencky elasticity, tau = U diag(2 mu eps + lam tr eps) U^T,
            # consistent with the log-space von-Mises return above.
            stress = kirchoff_stress_hencky(U, sig, model.mu[p], model.lam[p])
        if mat == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if mat == 6:  # fluid
            stress = kirchoff_stress_water(
                J, model.bulk[mat]
            )
        if mat == 10:  # newtonian / Bingham / power-law fluid (EOS + 2 eta_app dev D)
            stress = kirchoff_stress_newtonian(
                J, model.bulk[mat], state.particle_L[p], model.plastic_viscosity[mat],
                model.yield_stress[p], model.hardening[mat], model.softening[mat]
            )
        if mat == 12:  # tabulated eta_app(gd) fluid (FE-recovered curve, EOS + 2 eta_app dev D)
            stress = kirchoff_stress_tabulated(
                J, model.bulk[mat], state.particle_L[p], model.eta_table,
                model.eta_table_smin, model.eta_table_smax, model.eta_table_n
            )
        if mat == 9 or mat == 13:  # local mu(I) (parametric or tabulated): Hencky stress of returned F
            stress = kirchoff_stress_hencky(
                U, sig, model.mu[p], model.lam[p]
            )
        if mat == 14:  # composed: one elasticity kind per material type
            stress = compose_stress(state.particle_F[p], U, V, sig, J, model, p, mat)
        if mat == 11:  # compressible mu(I)-Phi(I): deviatoric Hencky + compaction pressure
            Jp_ref = state.particle_Jp[p]
            if Jp_ref < 0.5:
                Jp_ref = 1.0
            p_comp = mu_i_phi_pressure(model, mat, J, Jp_ref)
            stress = kirchoff_stress_mu_i_phi(U, sig, model.mu[p], p_comp)

        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        state.particle_stress[p] = stress


@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress_update_particle(state, model, dt, p)


@wp.kernel
def g2p_stress_p2g(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    """Claymore-style fused particle pass (Wang et al., ACM TOG 2020, MIT; see
    docs/performance.md and AUTHORS.md): gather from grid state n (grid_v_out),
    advect and update F_trial, return-map to stress, and scatter substep n+1's
    momentum/mass into the freshly zeroed grid_v_in/grid_m. The particle arrays
    are read and written once per substep instead of three times. Reads (grid_v_out)
    and writes (grid_v_in, grid_m) are disjoint arrays, and all three stages for a
    particle touch only that particle's state, so per-particle arithmetic order is
    identical to the separate kernels (bitwise on CPU). Gating mirrors the originals:
    stationary/rigid (7, 8) skip the gather but still return-map and scatter."""
    p = wp.tid()
    if state.particle_selection[p] == 0:
        mat = state.particle_material[p]
        if mat != 7 and mat != 8:
            g2p_particle(state, model, dt, p)
        stress_update_particle(state, model, dt, p)
        p2g_particle(state, model, dt, p)


@wp.kernel
def zero_grid_m_vin(state: MPMStateStruct, model: MPMModelStruct, lo: wp.vec3i):
    """Split zero, part 1 (fused pipeline): clear the scatter targets BEFORE the fused
    kernel; grid_v_out must survive it (the fused gather reads state n from there)."""
    grid_x, grid_y, grid_z = wp.tid()
    grid_x = grid_x + lo[0]
    grid_y = grid_y + lo[1]
    grid_z = grid_z + lo[2]
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_grid_vout(state: MPMStateStruct, model: MPMModelStruct, lo: wp.vec3i):
    """Split zero, part 2: AFTER the fused gather, clear grid_v_out so sub-threshold
    stencil nodes (mass <= 1e-15, skipped by normalization) read exactly zero, matching
    the unfused pipeline bitwise."""
    grid_x, grid_y, grid_z = wp.tid()
    grid_x = grid_x + lo[0]
    grid_y = grid_y + lo[1]
    grid_z = grid_z + lo[2]
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


# gather kernels for the block sort (claymore 5a): dst[i] = src[perm[i]]
@wp.kernel
def gather_vec3(src: wp.array(dtype=wp.vec3), perm: wp.array(dtype=int),
                dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[perm[i]]


@wp.kernel
def gather_mat33(src: wp.array(dtype=wp.mat33), perm: wp.array(dtype=int),
                 dst: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    dst[i] = src[perm[i]]


@wp.kernel
def gather_float(src: wp.array(dtype=float), perm: wp.array(dtype=int),
                 dst: wp.array(dtype=float)):
    i = wp.tid()
    dst[i] = src[perm[i]]


@wp.kernel
def gather_int(src: wp.array(dtype=int), perm: wp.array(dtype=int),
               dst: wp.array(dtype=int)):
    i = wp.tid()
    dst[i] = src[perm[i]]


@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    state.particle_cov[p * 6] = cov[0, 0]
    state.particle_cov[p * 6 + 1] = cov[0, 1]
    state.particle_cov[p * 6 + 2] = cov[0, 2]
    state.particle_cov[p * 6 + 3] = cov[1, 1]
    state.particle_cov[p * 6 + 4] = cov[1, 2]
    state.particle_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    state.particle_R[p] = wp.transpose(R)

@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float, lo: wp.vec3i):
    grid_x, grid_y, grid_z = wp.tid()
    grid_x = grid_x + lo[0]
    grid_y = grid_y + lo[1]
    grid_z = grid_z + lo[2]
    state.grid_v_out[grid_x, grid_y, grid_z] = (
        state.grid_v_out[grid_x, grid_y, grid_z] * scale
    )


# NOTE: E and nu are now per-type (indexed by material type, not particle).
# To use different E/nu for a region, define a new material type and use
# set_parameters_for_particles() instead.
@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(state: MPMStateStruct, impulse_modifier: Impulse_modifier):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
                ):
        impulse_modifier.mask[p] = 1 
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
                ):
        velocity_modifier.mask[p] = 1 
    else:
        velocity_modifier.mask[p] = 0

        
@wp.kernel
def selection_enforce_particle_velocity_cylinder(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal)
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
                ):
        velocity_modifier.mask[p] = 1 
    else:
        velocity_modifier.mask[p] = 0


# ---------------------------------------------------------------------------
# Rigid body kernels (material == 8)
# ---------------------------------------------------------------------------

@wp.kernel
def rigid_g2p_accumulate(
    state: MPMStateStruct,
    model: MPMModelStruct,
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_linear_mom: wp.array(dtype=wp.vec3),
    rigid_angular_mom: wp.array(dtype=wp.vec3),
):
    """For each rigid particle gather grid momentum and accumulate per-body
    linear and angular momentum contributions."""
    p = wp.tid()
    if state.particle_selection[p] == 0 and state.particle_material[p] == 8:
        bid = state.particle_rigid_id[p]

        # standard quadratic B-spline weights (same as g2p)
        grid_pos = state.particle_x[p] * model.inv_dx
        # floor, not toward-zero truncation: with periodic x a particle can sit at
        # grid_pos < 0.5 where int() picks base 0 instead of -1 and the B-spline
        # weights go invalid; for grid_pos - 0.5 >= 0 (every non-periodic scene, by
        # the edge guard) floor and int agree bitwise
        base_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )

        v_interp = wp.vec3(0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    v_interp = v_interp + state.grid_v_out[base_x + i, base_y + j, base_z + k] * weight

        mass_p = state.particle_mass[p]
        r = state.particle_x[p] - rigid_x_cm[bid]
        wp.atomic_add(rigid_linear_mom, bid, v_interp * mass_p)
        wp.atomic_add(rigid_angular_mom, bid, wp.cross(r, v_interp * mass_p))


@wp.kernel
def rigid_body_integrate(
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_v_cm: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_orientation: wp.array(dtype=wp.mat33),
    rigid_mass: wp.array(dtype=float),
    rigid_inv_inertia_body: wp.array(dtype=wp.mat33),
    rigid_linear_mom: wp.array(dtype=wp.vec3),
    rigid_angular_mom: wp.array(dtype=wp.vec3),
    dt: float,
):
    """Integrate rigid body EOM for body b (one thread per body).
    Updates v_cm, omega, x_cm, and orientation."""
    b = wp.tid()
    R = rigid_orientation[b]
    M = rigid_mass[b]

    # --- linear ---
    v_cm_new = rigid_linear_mom[b] / M

    # --- angular ---
    # I_world = R * I_body * R^T,  I_world_inv = R * I_body_inv * R^T
    I_world_inv = R * rigid_inv_inertia_body[b] * wp.transpose(R)
    omega_new = I_world_inv * rigid_angular_mom[b]

    # --- position ---
    x_cm_new = rigid_x_cm[b] + v_cm_new * dt

    # --- orientation: first-order update then polar-decomp re-orthogonalisation ---
    ox = omega_new[0]
    oy = omega_new[1]
    oz = omega_new[2]
    skew_w = wp.mat33(0.0, -oz, oy,  oz, 0.0, -ox,  -oy, ox, 0.0)
    R_new = R + skew_w * R * dt
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(R_new, U, sig, V)
    R_orth = U * wp.transpose(V)

    rigid_v_cm[b] = v_cm_new
    rigid_omega[b] = omega_new
    rigid_x_cm[b] = x_cm_new
    rigid_orientation[b] = R_orth


@wp.kernel
def rigid_particle_update(
    state: MPMStateStruct,
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_v_cm: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_orientation: wp.array(dtype=wp.mat33),
):
    """Set per-particle position, velocity, C, and F from rigid body state."""
    p = wp.tid()
    if state.particle_selection[p] == 0 and state.particle_material[p] == 8:
        bid = state.particle_rigid_id[p]
        R = rigid_orientation[bid]
        x_cm = rigid_x_cm[bid]
        v_cm = rigid_v_cm[bid]
        omega = rigid_omega[bid]

        x_p = x_cm + R * state.particle_x_ref[p]
        r = x_p - x_cm

        # velocity: v_cm + omega x r
        state.particle_v[p] = v_cm + wp.cross(omega, r)
        state.particle_x[p] = x_p
        state.particle_x_roundoff[p] = wp.vec3(0.0)

        # C = skew(omega) so APIC scatter uses the correct linear velocity field
        ox = omega[0]
        oy = omega[1]
        oz = omega[2]
        state.particle_C[p] = wp.mat33(0.0, -oz, oy,  oz, 0.0, -ox,  -oy, ox, 0.0)

        # rigid, no deformation
        state.particle_F[p] = wp.mat33(1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0)
        state.particle_F_roundoff[p] = wp.mat33(0.0)


# ---- active-block sparse compute (docs/performance.md): two-level block sparsity ---
# 4^3 node blocks. Per control tick: mark the block of each particle's stencil base,
# dilate by one block in every direction (covers the quadratic B-spline stencil
# crossing block faces and intra-tick particle motion), and compact into an active
# list. Grid STORAGE stays dense (fine through 256^3); grid COMPUTE (zero, normalize,
# damping) runs over active blocks only. Zeroing runs over the union with the previous
# tick's active set so nodes leaving the set are cleared exactly once (the same
# staleness rule as the particle-box path). Storage sparsity is a later stage.


@wp.kernel
def blocks_mark(particle_x: wp.array(dtype=wp.vec3), inv_dx: float, bd: wp.vec3i,
                mask: wp.array(dtype=int, ndim=3)):
    p = wp.tid()
    bx = wp.clamp(int(particle_x[p][0] * inv_dx - 0.5) >> 2, 0, bd[0] - 1)
    by = wp.clamp(int(particle_x[p][1] * inv_dx - 0.5) >> 2, 0, bd[1] - 1)
    bz = wp.clamp(int(particle_x[p][2] * inv_dx - 0.5) >> 2, 0, bd[2] - 1)
    mask[bx, by, bz] = 1


@wp.kernel
def blocks_dilate(src: wp.array(dtype=int, ndim=3), bd: wp.vec3i,
                  dst: wp.array(dtype=int, ndim=3)):
    x, y, z = wp.tid()
    on = int(0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                xi = wp.clamp(x + i, 0, bd[0] - 1)
                yj = wp.clamp(y + j, 0, bd[1] - 1)
                zk = wp.clamp(z + k, 0, bd[2] - 1)
                if src[xi, yj, zk] == 1:
                    on = 1
    dst[x, y, z] = on


@wp.kernel
def blocks_compact(cur: wp.array(dtype=int, ndim=3), prev: wp.array(dtype=int, ndim=3),
                   bd: wp.vec3i, cur_list: wp.array(dtype=int),
                   union_list: wp.array(dtype=int), counts: wp.array(dtype=int)):
    x, y, z = wp.tid()
    code = (x * bd[1] + y) * bd[2] + z
    c = cur[x, y, z]
    if c == 1:
        i = wp.atomic_add(counts, 0, 1)
        cur_list[i] = code
    if c == 1 or prev[x, y, z] == 1:
        j = wp.atomic_add(counts, 1, 1)
        union_list[j] = code


@wp.func
def _block_node(code: int, cell: int, bd: wp.vec3i):
    bz = code % bd[2]
    r = code // bd[2]
    by = r % bd[1]
    bx = r // bd[1]
    return wp.vec3i(bx * 4 + (cell >> 4), by * 4 + ((cell >> 2) & 3), bz * 4 + (cell & 3))


@wp.kernel
def zero_grid_blocks(state: MPMStateStruct, model: MPMModelStruct,
                     blist: wp.array(dtype=int), bd: wp.vec3i):
    t = wp.tid()
    n = _block_node(blist[t >> 6], t & 63, bd)
    if n[0] < model.grid_dim_x and n[1] < model.grid_dim_y and n[2] < model.grid_dim_z:
        state.grid_m[n[0], n[1], n[2]] = 0.0
        state.grid_v_in[n[0], n[1], n[2]] = wp.vec3(0.0, 0.0, 0.0)
        state.grid_v_out[n[0], n[1], n[2]] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def grid_normalization_and_gravity_blocks(state: MPMStateStruct, model: MPMModelStruct,
                                          dt: float, blist: wp.array(dtype=int),
                                          bd: wp.vec3i):
    t = wp.tid()
    n = _block_node(blist[t >> 6], t & 63, bd)
    if n[0] < model.grid_dim_x and n[1] < model.grid_dim_y and n[2] < model.grid_dim_z:
        if state.grid_m[n[0], n[1], n[2]] > 1e-15:
            v_out = state.grid_v_in[n[0], n[1], n[2]] * (
                1.0 / state.grid_m[n[0], n[1], n[2]]
            )
            v_out = v_out + dt * model.gravitational_accelaration
            state.grid_v_out[n[0], n[1], n[2]] = v_out


@wp.kernel
def add_damping_via_grid_blocks(state: MPMStateStruct, model: MPMModelStruct,
                                scale: float, blist: wp.array(dtype=int), bd: wp.vec3i):
    t = wp.tid()
    n = _block_node(blist[t >> 6], t & 63, bd)
    if n[0] < model.grid_dim_x and n[1] < model.grid_dim_y and n[2] < model.grid_dim_z:
        state.grid_v_out[n[0], n[1], n[2]] = state.grid_v_out[n[0], n[1], n[2]] * scale
