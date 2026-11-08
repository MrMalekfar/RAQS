"""
Ground and excited state finding for ARC Interface.

Contains functions for determining ground states based on atomic species
and finding excited states from probe laser wavelengths.
"""

import logging
from typing import Optional
from .types import AtomicLevel
from ._scan_validation import real_scalar, integer_scalar, arc_real_scalar
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError, PhysicsError

logger = logging.getLogger(__name__)


def find_ground_state(atom) -> Optional[AtomicLevel]:
    """
    Determine the ground state for the given atom.
    The principal quantum number comes from the supplied ARC atom's
    groundStateN. The existing upper-F, mF=0 preparation convention is kept.

    Args:
        atom: ARC atom object (e.g., Rubidium85())

    Returns:
        AtomicLevel for ground state
    """
    if atom is None:
        raise PhysicsConfigurationError("LAB-001: ground-state selection requires an ARC atom. F05: no default species.")
    n_ground = integer_scalar(getattr(atom, 'groundStateN', None), "ARC groundStateN", minimum=1)
    nuclear_spin = real_scalar(getattr(atom, 'I', None), "ARC nuclear spin I")
    if nuclear_spin < 0 or not (2 * nuclear_spin).is_integer():
        raise PhysicsConfigurationError("LAB-001: ARC nuclear spin I must be a nonnegative integer or half-integer. F05.")

    # For alkali atoms, ground state is n, l=0, j=0.5
    l_ground = 0
    j_ground = 0.5    
    f_ground = nuclear_spin + 0.5  # Retained upper-F preparation convention

    return AtomicLevel(
        n=n_ground, l=l_ground, j=j_ground, f=f_ground, mf=0,
        label=f"{n_ground}S1/2 F={int(f_ground)}"
    )


def find_excited_state(atom, probe_wavelength_nm: float, tolerance_nm: float = 1.0,
                      ground_state: Optional[AtomicLevel] = None) -> Optional[AtomicLevel]:
    """
    Find the excited state corresponding to the probe wavelength by searching through possible transitions.

    Args:
        atom: ARC atom object
        probe_wavelength_nm: Probe laser wavelength in nm
        tolerance_nm: Wavelength tolerance in nm
        ground_state: Pre-determined ground state (will find if None)

    Returns:
        AtomicLevel for excited state, or None if not found
    """
    if atom is None:
        raise PhysicsConfigurationError("LAB-001: excited-state selection requires an ARC atom. F05: no default state.")
    probe_wavelength_nm = real_scalar(probe_wavelength_nm, "probe wavelength [nm]")
    tolerance_nm = real_scalar(tolerance_nm, "probe wavelength tolerance [nm]")
    if probe_wavelength_nm <= 0 or tolerance_nm < 0:
        raise PhysicsConfigurationError("LAB-001: probe wavelength must be positive and tolerance nonnegative. F05.")

    if ground_state is None:
        ground_state = find_ground_state(atom)

    # Search through possible excited states from ground state
    n_ground = ground_state.n
    l_ground = ground_state.l
    j_ground = ground_state.j

    # Common excited states for alkali D lines
    possible_excited = [
        (n_ground, 1, 1.5),  # P3/2 (D2 line)
        (n_ground, 1, 0.5),  # P1/2 (D1 line)
    ]

    for n_ex, l_ex, j_ex in possible_excited:
        try:
            wavelength = arc_real_scalar(atom.getTransitionWavelength(
                n_ground, l_ground, j_ground, n_ex, l_ex, j_ex
            ), f"ARC probe wavelength ({n_ground},{l_ground},{j_ground})->({n_ex},{l_ex},{j_ex}) [m]",
                positive=True) * 1e9  # Convert to nm
            wavelength = arc_real_scalar(wavelength, "ARC probe wavelength [nm]", positive=True)

            if abs(wavelength - probe_wavelength_nm) <= tolerance_nm:
                # Calculate hyperfine structure
                f_ex = j_ex + atom.I
                J_str = f"{int(2*j_ex)}/{2 if j_ex%1==0.5 else 1}"
                return AtomicLevel(
                    n=n_ex, l=l_ex, j=j_ex, f=f_ex, mf=0,
                    label=f"{n_ex}P{J_str} F={int(f_ex)}"
                )
        except PhysicsError:
            raise
        except Exception as exc:
            raise PhysicsCalculationError(
                f"LAB-001: ARC probe wavelength failed for "
                f"({n_ground},{l_ground},{j_ground})->({n_ex},{l_ex},{j_ex}). "
                "F05: calculation failure is not a domain exclusion.") from exc

    logger.warning(f"No excited state found for probe wavelength {probe_wavelength_nm} nm within tolerance {tolerance_nm} nm")
    return None
