"""Laser-following-frame frequency offsets for the canonical four-level basis."""

import numpy as np
import qutip as qt
from ...config.laboratory import _real
from ...exceptions import PhysicsCalculationError


def laser_frequency_terms(realization):
    """H_noise = 2*pi/1e6 * (df_p diag(0,1,1,1)+df_c diag(0,0,1,1)).

    df = instantaneous applied laser frequency minus its nominal carrier [Hz].
    Positive df increases Delta=omega_laser-omega_atom in the existing model.
    Both Rydberg levels share the optical frame: RF phase/direction is unchanged.
    The observable is coherence referred to the instantaneous probe phase, as
    required for susceptibility; Rabi magnitudes and incident optical power stay
    constant. No extra collapse operator or separate phase diffusion is added.
    """
    def coefficient(which):
        def evaluate(t, args=None):
            hz = _real(realization.laser_at(t)[which], 'NOISE-006 laser frequency [Hz]')
            value = (hz/1e6)*(2*np.pi)
            if not np.isfinite(value):
                raise PhysicsCalculationError('NOISE-006: nonfinite laser detuning; no substitute.')
            return value
        return evaluate
    return [[qt.Qobj(np.diag([0., 1., 1., 1.])), coefficient(0)],
            [qt.Qobj(np.diag([0., 0., 1., 1.])), coefficient(1)]]
