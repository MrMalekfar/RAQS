"""
Adaptive Maxwell-Boltzmann velocity integration for the Doppler ensemble.

The ensemble probe coherence is

    rho_ens(Delta_p) = integral dv f_MB(v) * rho21(v, Delta_p)

with the one-dimensional Maxwell-Boltzmann density

    f_MB(v) = exp(-v^2 / (2*sigma_v^2)) / (sqrt(2*pi)*sigma_v),
    sigma_v  = sqrt(k_B * T / m).

The COMPLEX coherence is integrated first (C-002 ordering); transmission
is applied once afterwards by the caller.  This module is purely a
numerical integrator: it does not know about QuTiP, Hamiltonians, or
Beer-Lambert propagation.  This lets the unit tests drive it with cheap
analytic/synthetic response functions.

DOP-001B established that fixed-order Gauss-Hermite quadrature is severely
under-resolved for the narrow EIT/AT features (relative complex-rho errors
of 40-478% even at N=129).  The adaptive scheme below therefore:

- integrates Re and Im components together with scipy.integrate.quad_vec;
- accepts optional physics-informed breakpoint HINTS (they improve initial
  sampling but never define/truncate the integral);
- explicitly normalizes the finite [-n_sigma*sigma_v, +n_sigma*sigma_v]
  domain (the omitted Maxwell tail is quantified by the caller).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad_vec

from ...config.constants import k_B


def get_velocity_std(temperature_k: float, atom_mass_kg: float) -> float:
    """RMS (standard deviation) of the 1-D Maxwell-Boltzmann velocity distribution.

    sigma_v = sqrt(k_B * T / m).
    """
    return float(math.sqrt(k_B * temperature_k / atom_mass_kg))


def _maxwell_density(v_ms, sigma_v):
    """Unnormalized Maxwell weight exp(-v^2/(2 sigma_v^2))."""
    return np.exp(-np.asarray(v_ms) ** 2 / (2.0 * sigma_v ** 2))
def integrate_maxwell_response(
    response_fn,
    temperature_k: float,
    atom_mass_kg: float,
    n_sigma: float = 5.0,
    epsrel: float = 1e-6,
    epsabs: float = 1e-12,
    breakpoints=None,
    return_info: bool = False,
    require_convergence: bool = False,
):
    """Adaptively integrate response_fn(v) over the truncated Maxwell domain.

    Returns
    -------
    complex or tuple
        The normalized Maxwell average of the complex response::

            rho_ens = numerator / Z,
            numerator = Integral_{-Vmax}^{+Vmax} f_MB(v) response_fn(v) dv,
            Z         = Integral_{-Vmax}^{+Vmax} f_MB(v) dv.

        If ``return_info`` is True, returns ``(rho_ens, info_dict)``.

    Parameters
    ----------
    response_fn : callable
        ``response_fn(v_ms)`` returning a complex number for a velocity in m/s.
    temperature_k, atom_mass_kg : float
        Physical inputs for sigma_v.
    n_sigma : float
        Symmetric integration half-width in units of sigma_v.
    epsrel, epsabs : float
        Relative/absolute tolerances passed to ``quad_vec``.
    breakpoints : optional iterable of float
        Velocity points (m/s) that must lie on the initial sampling grid --
        SAMPLING AIDS only; they do not restrict the integral.
    return_info : bool
        Return diagnostic information.
    require_convergence : bool
        Reject a failed/nonfinite quad_vec result instead of consuming it.
        Enabled by explicit profiles; the legacy default API is preserved.

    Notes
    -----
    Workers are locked to 1: the response wraps a QuTiP steady-state solve
    that is not proven thread-safe.
    """
    sigma_v = get_velocity_std(temperature_k, atom_mass_kg)
    v_max = n_sigma * sigma_v

    def integrand(v_ms):
        v_arr = np.atleast_1d(np.asarray(v_ms, dtype=float))
        w = _maxwell_density(v_arr, sigma_v)
        out = np.empty(v_arr.shape, dtype=complex)
        for i, vi in enumerate(v_arr):
            out[i] = response_fn(float(vi)) * w[i]
        if out.shape == (1,):
            return out[0]
        return out

    # Breakpoint hints: strictly interior, deduplicated, sorted.
    pts = []
    if breakpoints is not None:
        for bp in breakpoints:
            b = float(bp)
            if -v_max < b < v_max:
                pts.append(b)
    pts = sorted(set(round(p, 10) for p in pts))
    pts = [p for p in pts if -v_max < p < v_max]

    def real_imag(v):
        c = integrand(v)
        if np.ndim(c) == 0:
            return np.array([c.real, c.imag])
        c = np.asarray(c)
        return np.vstack([c.real, c.imag])

    integration_options = {'full_output': True} if require_convergence else {}
    if pts:
        integration = quad_vec(
            real_imag, -v_max, v_max, epsabs=epsabs, epsrel=epsrel,
            points=pts, workers=1, limit=500,
            **integration_options,
        )
    else:
        integration = quad_vec(
            real_imag, -v_max, v_max, epsabs=epsabs, epsrel=epsrel,
            workers=1, limit=500,
            **integration_options,
        )
    if require_convergence:
        res, err, status = integration
        if not status.success or not np.all(np.isfinite(res)) or not np.all(np.isfinite(err)):
            from ...exceptions import PhysicsCalculationError
            raise PhysicsCalculationError(f'LAB-007: Maxwell quadrature failed to converge: {status.message}; error={err!r}. No fixed-class retry.')
    else:
        res, err = integration

    numerator = complex(res[0], res[1])
    # quad_vec returns err as a length-2 array for vector integrands, but a
    # plain float when it falls back to a scalar path; normalize both.
    if np.ndim(err) == 0:
        err_re = err_im = float(err)
    else:
        err_re, err_im = float(err[0]), float(err[1])

    # Explicit truncation normalization (do NOT assume Z=1).
    z, z_err = _maxwell_normalization(sigma_v, v_max)

    rho_ens = numerator / z

    if return_info:
        info = {
            "numerator": numerator,
            "normalization": z,
            "norm_err": float(z_err),
            "neval": None,  # quad_vec does not expose neval directly
            "err_re": err_re,
            "err_im": err_im,
            "v_max": v_max,
            "sigma_v": sigma_v,
        }
        return rho_ens, info
    return rho_ens


def _maxwell_normalization(sigma_v: float, v_max: float):
    """Z = Integral exp(-v^2/(2 sigma^2)) dv = sqrt(2 pi) sigma erf(v_max/(sqrt2 sigma))."""
    import scipy.special as sp

    t = v_max / (math.sqrt(2.0) * sigma_v)
    z = math.sqrt(2.0 * math.pi) * sigma_v * sp.erf(t)
    return z, z * 1e-15


def maxwell_tail_mass(n_sigma: float) -> float:
    """Probability mass outside +/- n_sigma * sigma_v: P(|v|>n*sigma) = erfc(n/sqrt(2))."""
    import scipy.special as sp

    return float(sp.erfc(n_sigma / math.sqrt(2.0)))
