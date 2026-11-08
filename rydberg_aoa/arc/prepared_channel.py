"""AUTO-001B atomic calculations for already-declared prepared channels.

No manifold or magnetic search lives here. The selection layer supplies
validated states and q; exact angular zeros are domain evidence, not a
substitute for a failed designated ARC dipole calculation.
"""

from dataclasses import dataclass
import math

from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j

from ._scan_validation import arc_real_scalar
from .dipole import EA0_TO_CM
from ..config.constants import C_LIGHT, EPSILON_0, HBAR
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError


def _angular_zero(kind, source, target, q, nuclear_spin):
    """Test exact vanishing for one validated channel; no floating cutoff.

    DME signs/normalizations are left to ARC. These small angular expressions
    enumerate no mJ, q or F and compute no substitute radial element.
    """
    j1, j2, i, q = map(Rational, (source.j, target.j, nuclear_spin, q))
    if kind == 'probe':
        f1, f2, m1, m2 = map(Rational, (source.f, target.f, source.mf, target.mf))
        return (wigner_3j(f1, 1, f2, -m1, -q, m2) == 0 or
                wigner_6j(f1, 1, f2, j2, i, j1) == 0)
    m2 = Rational(target.mj)
    if kind == 'coupling':
        m1 = m2 - q
        mf = Rational(source.mf)
        if clebsch_gordan(j1, i, Rational(source.f), m1, mf - m1, mf) == 0:
            return True
    else:
        m1 = Rational(source.mj)
    return wigner_3j(j1, 1, j2, -m1, -q, m2) == 0


def _dipole(atom, method, arguments, context):
    """Signed ARC DME [e·a0], permitting valid zero for domain screening."""
    try:
        value = getattr(atom, method)(*arguments)
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(f'AUTO-001B {context} failed; no substitute calculation.') from exc
    # Validate BEFORE float coercion: complex/bool/string/array are not DMEs.
    return arc_real_scalar(value, f'AUTO-001B {context} signed dipole [e·a0]')


@dataclass(frozen=True)
class PreparedRabiFrequencies:
    """Nonnegative angular magnitudes [rad/µs]; signed DMEs stay separate."""
    probe: float
    coupling: float
    lo: float
    signal: float


def _peak_field(laser):
    power_w, waist_m = laser.power_mw * 1e-3, laser.beam_waist_mm * 1e-3
    if waist_m <= 0 or (laser.power_mw > 0 and power_w == 0):
        raise PhysicsConfigurationError('AUTO-001B: optical power/waist conversion underflow; no replacement.')
    # E_peak=sqrt(2 I_peak/(eps0 c)), I_peak=2P/(pi w²).
    # This form avoids squaring a very small waist before the square root.
    field = 2 * math.sqrt(power_w / (math.pi * EPSILON_0 * C_LIGHT)) / waist_m
    return arc_real_scalar(field, 'AUTO-001B Gaussian peak field [V/m]')


def _omega(dipole_ea0, field):
    dipole_si = dipole_ea0 * EA0_TO_CM
    if dipole_ea0 != 0 and dipole_si == 0:
        raise PhysicsCalculationError('AUTO-001B: dipole SI conversion underflow; no floor.')
    omega = arc_real_scalar(abs(dipole_si) * field / HBAR / 1e6,
                            'AUTO-001B angular Rabi [rad/µs]')
    if field > 0 and dipole_ea0 != 0 and omega == 0:
        raise PhysicsCalculationError('AUTO-001B: angular Rabi underflow; no zero substitution.')
    return omega


def _prepared_rabi(probe_ea0, coupling_ea0, rf_ea0, request):
    """One RF dipole shared by the explicitly applied LO and signal."""
    return PreparedRabiFrequencies(_omega(probe_ea0, _peak_field(request.probe)),
                                   _omega(coupling_ea0, _peak_field(request.coupling)),
                                   _omega(rf_ea0, request.lo.electric_field_vm),
                                   _omega(rf_ea0, request.signal.electric_field_vm))
