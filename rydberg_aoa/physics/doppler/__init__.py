"""
Doppler broadening of the EIT response for thermal vapour cells.

Public API
----------
- generate_velocity_classes : discrete velocity classes + Maxwell-Boltzmann weights
- get_velocity_std          : σ_v = √(k_B T / m)
- get_doppler_shifts        : per-class probe/coupling detuning shifts (rad/μs)
- integrate_maxwell_response: adaptive Maxwell integration of a complex response
- resonance_velocity_hints  : parameter-derived velocity breakpoint hints

Methodology follows the discrete velocity-class approach of MaxwellBloch
(Ogden 2020, MIT License); the production integrator is the adaptive
Maxwell quadrature in ``adaptive_maxwell.py`` (DOP-002I).
"""

from .velocity_classes import generate_velocity_classes, get_velocity_std
from .doppler_shifts import get_doppler_shifts
from .adaptive_maxwell import (
    integrate_maxwell_response,
    maxwell_tail_mass,
    get_velocity_std as _adaptive_get_velocity_std,
)
from .resonance_hints import (
    resonance_velocity_hints,
    velocity_scale_hint,
    two_photon_slope_rad_per_us_per_ms,
)

__all__ = [
    "generate_velocity_classes",
    "get_velocity_std",
    "get_doppler_shifts",
    "integrate_maxwell_response",
    "maxwell_tail_mass",
    "resonance_velocity_hints",
    "velocity_scale_hint",
    "two_photon_slope_rad_per_us_per_ms",
]
