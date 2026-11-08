"""
Velocity-class generation for Doppler broadening of thermal atomic vapours.

Physics
-------
Atoms in a vapour cell at temperature T move with a one-dimensional
Maxwell-Boltzmann velocity distribution along the laser beam axis:

    P(v) dv = 1/(σ_v √(2π)) · exp(-v² / (2σ_v²)) dv,
    σ_v = √(k_B T / m),

where m is the atomic mass. The ensemble-averaged response of the medium is

    ⟨X⟩ = ∫ X(v) P(v) dv,

which is evaluated numerically as a weighted discrete sum over velocity
classes (Gauss-Hermite quadrature or a truncated uniform grid).

Methodology follows the discrete velocity-class approach of MaxwellBloch
(Ogden 2020, MIT License, https://github.com/tpogden/maxwellbloch).

Units: velocities are returned in m/s (SI); conversion to spectral shifts
happens in `doppler_shifts.py`.
"""

import numpy as np  # np.polynomial.hermite.hermgauss for quadrature nodes

from ...config.constants import k_B  # Boltzmann constant [J/K]


def get_velocity_std(temperature_k: float, atom_mass_kg: float) -> float:
    """
    RMS (standard deviation) of the 1-D Maxwell-Boltzmann velocity distribution.

    σ_v = √(k_B T / m)

    Parameters
    ----------
    temperature_k : float
        Vapour cell temperature T [K].
    atom_mass_kg : float
        Mass of one atom m [kg] (e.g. Rb-85: 1.41e-25 kg).

    Returns
    -------
    float
        Velocity standard deviation σ_v [m/s]. For Rb-85 at 300 K this is
        ≈ 171 m/s, producing a ~220 MHz rms Doppler spread on the 780 nm line.
    """
    return float(np.sqrt(k_B * temperature_k / atom_mass_kg))


def generate_velocity_classes(
    n_classes: int,
    temperature_k: float,
    atom_mass_kg: float,
    method: str = "gauss_hermite",
    n_sigma: float = 4.0,
    velocity_scale_ms: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete velocity classes approximating the Maxwell-Boltzmann distribution.

    Three sampling schemes are provided:

    - 'gauss_hermite': Gauss-Hermite quadrature. Exact for polynomials up to
      degree 2n-1 under the Gaussian weight. Narrow line shapes still require
      explicit convergence checks. Nodes implicitly span several σ_v.
    - 'uniform': n_classes uniformly-spaced velocities on [-n_sigma·σ_v,
      +n_sigma·σ_v] weighted by the Gaussian density P(v) (MaxwellBloch-style
      discrete class sum); useful for cross-validation and plotting.
    - 'sinh': uniform transformed coordinates with v=scale*sinh(y).
      Includes Gaussian density, Jacobian and trapezoidal endpoint factors;
      represents the same conditional truncated Maxwell distribution.

    Parameters
    ----------
    n_classes : int
        Number of velocity classes (quadrature nodes). Odd values place a node
        exactly at v = 0 (the cold-atom limit is recovered as T→0).
    temperature_k : float
        Vapour cell temperature T [K].
    atom_mass_kg : float
        Atomic mass m [kg].
    method : str
        'gauss_hermite' (default), 'uniform', or 'sinh'.
    n_sigma : float
        Truncation half-width in units of σ_v ('uniform' and 'sinh').
    velocity_scale_ms : float or None
        Positive declared scale [m/s], required only for 'sinh'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (velocities_ms, weights) — velocities in m/s, weights strictly
        positive and normalized so that weights.sum() == 1.
    """
    if n_classes < 1:
        raise ValueError("n_classes must be >= 1")
    sigma_v = get_velocity_std(temperature_k, atom_mass_kg)

    if velocity_scale_ms is not None and method != 'sinh':
        raise ValueError('velocity_scale_ms is only valid for sinh quadrature')
    if method == 'sinh':
        from ...config.laboratory import _real, _integer
        from ...exceptions import PhysicsCalculationError
        count = _integer(n_classes,'sinh velocity count',minimum=3)
        if count % 2 != 1:
            raise ValueError('sinh quadrature requires an odd velocity count')
        scale = _real(velocity_scale_ms,'sinh velocity scale [m/s]',positive=True)
        limit = _real(n_sigma*sigma_v,'sinh velocity domain [m/s]',positive=True)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            extent = np.arcsinh(limit/scale)
            half = count//2
            y = np.arange(-half,half+1,dtype=float)*(extent/half)
            velocities = scale*np.sinh(y)
            velocities[[0,-1]] = [-limit,limit]
            weights = np.exp(-.5*(velocities/sigma_v)**2)*scale*np.cosh(y)*(extent/half)
            weights[[0,-1]] *= .5
        if (not np.isfinite(velocities).all() or not np.isfinite(weights).all()
                or np.any(np.diff(velocities) <= 0) or np.any(weights <= 0)
                or not np.isfinite(weights.sum()) or weights.sum() <= 0):
            raise PhysicsCalculationError('DOP-CONV-001: invalid transformed Maxwell grid; no repair.')
    elif method == "gauss_hermite":
        # Gauss-Hermite nodes x_i, weights w_i for ∫ e^{-x²} f(x) dx ≈ Σ w_i f(x_i)
        nodes, quad_weights = np.polynomial.hermite.hermgauss(n_classes)
        # Substitution x = v/(√2 σ_v): E[f(v)] = (1/√π) Σ w_i f(√2 σ_v x_i)
        velocities = np.sqrt(2.0) * sigma_v * nodes
        weights = quad_weights / np.sqrt(np.pi)
    elif method == "uniform":
        # Uniform grid truncated symmetrically around v = 0
        velocities = np.linspace(-n_sigma * sigma_v, n_sigma * sigma_v, n_classes)
        # Gaussian density P(v_i) as class weight (unnormalized)
        p_v = np.exp(-velocities**2 / (2.0 * sigma_v**2))
        weights = p_v / p_v.sum()
    else:
        raise ValueError(f"Unknown velocity sampling method: {method!r}")

    # Force exact normalization against floating-point round-off
    weights = weights / weights.sum()
    return velocities.astype(float), weights.astype(float)
