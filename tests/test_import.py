"""Import-contract regression: calculate_transmission_from_rho21.

Historically the susceptibility layer exposed a module object where a function
was expected, producing "module object is not callable" downstream. These
tests pin the repaired public contract: the symbol resolves to a callable
that maps a complex coherence rho_21 to a finite, real, positive transmission.
"""
import numpy as np

from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 import (
    calculate_transmission_from_rho21,
)


def test_transmission_symbol_is_callable():
    """The historical 'module object is not callable' regression stays fixed."""
    assert callable(calculate_transmission_from_rho21)


def test_transmission_of_small_coherence_is_finite_positive_real(atomic_physics):
    """T(rho_21) must be a finite, real, positive scalar for a small coherence.

    Physics: T follows a Beer-Lambert exponential in the susceptibility, so
    for any finite argument the result must be strictly positive and real;
    a complex or non-finite return would silently poison the spectra.
    """
    result = calculate_transmission_from_rho21(0.1 + 0.1j)
    assert np.isfinite(float(result))
    assert float(result) > 0.0

