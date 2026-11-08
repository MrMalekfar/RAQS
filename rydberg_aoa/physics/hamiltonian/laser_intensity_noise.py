"""RIN correction of the original optical Hamiltonian; no second model."""
import numpy as np
import qutip as qt
from ...exceptions import PhysicsCalculationError


def laser_intensity_terms(realization, omega_p, omega_c):
    """Add only (sqrt(r)-1) times nominal optical drive, P/P0=r>0.

    Interpolate INTENSITY, then take sqrt for Rabi; interpolating Rabi instead
    would not match the incident power used by PIN between realization nodes.
    All operators act in the existing instantaneous-laser-phase frame.
    """
    terms = []
    for which, omega in enumerate((omega_p, omega_c)):
        transition = qt.basis(4, which+1)*qt.basis(4, which).dag()
        operator = -.5*omega*(transition+transition.dag())
        def correction(t, args=None, which=which):
            r = realization.laser_intensity_at(t)[which]
            if not np.isfinite(r) or r <= 0:
                raise PhysicsCalculationError('NOISE-007: invalid intensity in Hamiltonian; no clipping.')
            return np.sqrt(r)-1.
        terms.append([operator, correction])
    return terms
