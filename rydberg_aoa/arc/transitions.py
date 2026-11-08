"""
Transition search and analysis for ARC Interface.

Contains functions for finding laser and RF transitions between atomic levels
using systematic ARC-based searches and analytical fallbacks.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

# Try to import analytical modules
try:
    from ..analytical.energy import RydbergEnergyCalculator
    from ..analytical.dipoles import AnalyticalDipoleCalculator
    ANALYTICAL_AVAILABLE = True
except ImportError:
    ANALYTICAL_AVAILABLE = False
    print("Warning: Analytical modules not available, falling back to ARC-only methods")

from .types import AtomicLevel, Transition
from .dipole import calculate_dipole_matrix_element
from ..exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
    PhysicsError,
)

logger = logging.getLogger(__name__)


def find_rydberg_transitions(atom, from_level: AtomicLevel,
                            target_wavelength_nm: float,
                            wavelength_tolerance_nm: float = 1.0,
                            n_range: Tuple[int, int] = None) -> List[Transition]:
    """
    Find Rydberg transitions from a given level at target wavelength by systematic search.

    Args:
        atom: ARC atom object
        from_level: Starting atomic level
        target_wavelength_nm: Target transition wavelength in nm
        wavelength_tolerance_nm: Wavelength tolerance in nm
        n_range: Range of principal quantum numbers to search

    Returns:
        List of possible transitions sorted by dipole moment strength
    """
    transitions = []

    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the Rydberg transition search cannot "
            "run. F05: no empty-result substitution."
        )

    logger.debug(f"Searching for Rydberg transitions from {from_level.label} at {target_wavelength_nm} nm")

    # Systematic search through quantum numbers
    for n in range(n_range[0], n_range[1] + 1):
        for l in range(min(5, n)):  # Limit l to reduce search space, focus on low l
            for j in [l - 0.5, l + 0.5] if l > 0 else [0.5]:
                if j < 0:
                    continue

                try:
                    wavelength = atom.getTransitionWavelength(
                        from_level.n, from_level.l, from_level.j,
                        n, l, j
                    ) * 1e9  # Convert to nm

                    if abs(wavelength - target_wavelength_nm) <= wavelength_tolerance_nm:
                        # Calculate dipole moment to verify transition strength
                        dipole = calculate_dipole_matrix_element(atom, from_level, n, l, j)

                        if dipole > 1e-35:  # Only include significant transitions
                            # QN-001A: high-n Rydberg levels carry NO hyperfine
                            # labels. The historical "F ≈ J" approximation
                            # (f = j) combined with mf = 0 assigned an invalid
                            # magnetic sublevel (mF = 0 is not a sublevel of a
                            # half-integer F) and is rejected by the
                            # AtomicLevel validation contract; the level label
                            # carries the state identity instead.
                            L_letters = 'SPDFGHIJKLMNOP'
                            L_letter = L_letters[l] if l < len(L_letters) else 'X'
                            J_str = f"{int(2*j)}/{2 if j%1==0.5 else 1}"
                            rydberg_level = AtomicLevel(
                                n=n, l=l, j=j,
                                label=f"{n}{L_letter}{J_str}"
                            )

                            transition = Transition(
                                level1=from_level,
                                level2=rydberg_level,
                                wavelength_nm=wavelength,
                                dipole_moment=dipole,
                                rabi_frequency=0.0,  # Will be calculated later
                                detuning=0.0,
                                linewidth=0.001  # Typical laser linewidth in MHz
                            )

                            transitions.append(transition)
                            logger.debug(f"Found transition: {from_level.label} -> {rydberg_level.label} at {wavelength:.2f} nm, dipole={dipole:.2e}")

                except PhysicsError:
                    # F05: a physics fail-fast error must not be swallowed into
                    # "transition not found".
                    raise
                except Exception as e:
                    logger.debug(f"ARC error for Rydberg transition {from_level.n}{from_level.l}{from_level.j} -> {n}{l}{j}: {e}")
                    continue

    # Sort by dipole moment strength (strongest first)
    transitions.sort(key=lambda t: t.dipole_moment, reverse=True)
    # print(transitions)
    logger.info(f"Found {len(transitions)} Rydberg transitions for wavelength {target_wavelength_nm} nm")
    return transitions


def find_rf_transitions(atom, from_level: AtomicLevel,
                       rf_frequency_hz: float,
                       frequency_tolerance_hz: float = 1e7,
                       n_range: Tuple[int, int] = None) -> List[Transition]:
    """
    Find RF transitions from a Rydberg level to another Rydberg level at target frequency.

    Args:
        atom: ARC atom object
        from_level: Starting Rydberg level
        rf_frequency_hz: Target RF transition frequency in Hz
        frequency_tolerance_hz: Frequency tolerance in Hz
        n_range: Range of n to search (default: from_level.n ± 5)

    Returns:
        List of possible RF transitions sorted by dipole moment strength
    """
    transitions = []

    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the RF transition search cannot run. "
            "F05: no empty-result substitution."
        )
    if type(atom).__name__ in ('Rubidium87', 'Caesium'):
        raise PhysicsConfigurationError(
            'ATOM-001: this legacy RF finder can invoke Rb85 analytical tables and maximum-channel selection. '
            'Use the prepared RF-first experimental profile for this isotope; no cross-species fallback.')

    if n_range is None:
        n_range = (max(45, from_level.n - 2), min(85, from_level.n + 2))

    logger.debug(f"Searching for RF transitions from {from_level.label} at {rf_frequency_hz/1e6:.1f} MHz")

    # Get energy of starting level
    try:
        energy1 = atom.getEnergy(from_level.n, from_level.l, from_level.j)
    except Exception as exc:
        # F05: the starting level's energy is REQUIRED. The historical
        # "return []" made the caller believe that no RF transition existed
        # (which then triggered the fabricated n+1 fallback in interface.py).
        raise PhysicsCalculationError(
            f"ARC getEnergy failed for the RF starting level "
            f"{from_level.label}: {type(exc).__name__}: {exc}. "
            "F05: no empty-result substitution."
        ) from exc

    # Search through possible final states
    for n2 in range(n_range[0], n_range[1] + 1):
        # Allow same n for RF transitions between different l,j states

        for l2 in range(min(5, n2)):  # Limit l for computational efficiency
            for j2 in [l2 - 0.5, l2 + 0.5] if l2 > 0 else [0.5]:
                if j2 < 0:
                    continue

                try:
                    # Get energy of final level
                    energy2 = atom.getEnergy(n2, l2, j2)

                    # Calculate transition frequency
                    energy_diff = abs(energy1 - energy2)  # cm^-1
                    transition_freq = energy_diff * 3e10  # Hz (cm^-1 * cm/s)

                    if abs(transition_freq - rf_frequency_hz) <= frequency_tolerance_hz:
                        # Check dipole moment for allowed transition (Δl = ±1)
                        if abs(l2 - from_level.l) == 1:  # Electric dipole allowed
                            dipole = calculate_dipole_matrix_element(atom, from_level, n2, l2, j2)

                            if dipole > 0:  # Significant transition
                                # QN-001A: no F/mF labels on a Rydberg level
                                # (see the matching note in
                                # find_rydberg_transitions above).
                                L_letters = 'SPDFGHIJKLMNOP'
                                L_letter = L_letters[l2] if l2 < len(L_letters) else 'X'
                                J_str = f"{int(2*j2)}/{2 if j2%1==0.5 else 1}"
                                to_level = AtomicLevel(
                                    n=n2, l=l2, j=j2,
                                    label=f"{n2}{L_letter}{J_str}"
                                )

                                wavelength_nm = (3e8 / transition_freq) * 1e9  # Approximate

                                transition = Transition(
                                    level1=from_level,
                                    level2=to_level,
                                    wavelength_nm=wavelength_nm,
                                    dipole_moment=dipole,
                                    rabi_frequency=0.0,  # Will be calculated later
                                    detuning=0.0,
                                    linewidth=0.0,  # RF linewidth
                                    frequency_diff_hz=abs(transition_freq - rf_frequency_hz)
                                )

                                transitions.append(transition)
                                logger.debug(f"Found RF transition: {from_level.label} -> {to_level.label} at {transition_freq/1e6:.1f} MHz, dipole={dipole:.2e}")

                except PhysicsError:
                    # F05: a physics fail-fast error must not be swallowed into
                    # "transition not found".
                    raise
                except Exception as e:
                    logger.debug(f"ARC error for RF transition {from_level.n}{from_level.l}{from_level.j} -> {n2}{l2}{j2}: {e}")
                    continue

    # Sort by dipole moment strength
    transitions.sort(key=lambda t: t.dipole_moment, reverse=True)

    # If no transitions found with ARC, try analytical method
    if not transitions and ANALYTICAL_AVAILABLE:
        logger.info("No ARC RF transitions found, trying analytical method...")
        transitions = _find_rf_transitions_analytical(from_level, rf_frequency_hz, frequency_tolerance_hz, n_range)

    logger.info(f"Found {len(transitions)} RF transitions for frequency {rf_frequency_hz/1e6:.1f} MHz")
    return transitions


def _find_rf_transitions_analytical(from_level: AtomicLevel,
                                   rf_frequency_hz: float,
                                   frequency_tolerance_hz: float = 1e7,
                                   n_range: Tuple[int, int] = None) -> List[Transition]:
    """
    Find RF transitions using analytical energy calculations when ARC fails.

    Args:
        from_level: Starting Rydberg level
        rf_frequency_hz: Target RF frequency in Hz
        frequency_tolerance_hz: Frequency tolerance in Hz
        n_range: Range of n to search

    Returns:
        List of possible RF transitions
    """
    if not ANALYTICAL_AVAILABLE:
        logger.warning("Analytical modules not available for RF transition search")
        return []

    transitions = []

    # Initialize analytical calculator
    energy_calc = RydbergEnergyCalculator()
    dipole_calc = AnalyticalDipoleCalculator()

    # Find analytical transitions
    analytical_transitions = energy_calc.find_rf_transitions_analytical(
        from_level.n, rf_frequency_hz, frequency_tolerance_hz, n_range
    )

    # Convert to Transition objects
    for trans_dict in analytical_transitions:
        # Create AtomicLevel objects
        level1 = AtomicLevel(
            n=trans_dict['n1'], l=trans_dict['l1'], j=trans_dict['j1'],
            label=f"{trans_dict['n1']}D{int(2*trans_dict['j1']+1)//2}"
        )
        level2 = AtomicLevel(
            n=trans_dict['n2'], l=trans_dict['l2'], j=trans_dict['j2'],
            label=f"{trans_dict['n2']}{_l_to_letter(trans_dict['l2'])}{int(2*trans_dict['j2']+1)//2}"
        )

        # F05: the analytical RF search cannot supply a real dipole matrix
        # element — the historical fabricated placeholder dipole travelled into
        # level selection and the RF Rabi calculation.
        raise PhysicsCalculationError(
            f"Analytical RF transition {level1.label} -> {level2.label} at "
            f"{trans_dict['frequency_hz']/1e6:.1f} MHz cannot be assigned a "
            "real dipole matrix element: the analytical dipole calculation is "
            "not wired into this search path. "
            "F05: no placeholder dipole."
        )

        # Create Transition object
        transition = Transition(
            level1=level1,
            level2=level2,
            wavelength_nm=(3e8 / trans_dict['frequency_hz']) * 1e9,
            dipole_moment=dipole_si,
            rabi_frequency=0.0,  # Will be calculated later
            detuning=0.0,
            linewidth=0.0  # RF linewidth
        )

        transitions.append(transition)
        logger.debug(f"Analytical RF transition: {level1.label} -> {level2.label} at {trans_dict['frequency_hz']/1e6:.1f} MHz, dipole={dipole_si:.2e}")

    logger.info(f"Found {len(transitions)} analytical RF transitions")
    return transitions


def _l_to_letter(l: int) -> str:
    """Convert orbital angular momentum to letter notation."""
    letters = 'SPDFGHIJKLMNOP'
    return letters[l] if l < len(letters) else 'X'
