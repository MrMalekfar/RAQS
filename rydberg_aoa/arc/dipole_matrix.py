"""
Dipole matrix calculations for EIT system using ARC.

Provides dipole matrix elements for the optimized energy levels.
Calculates dipole moments between relevant transitions in the 4-level system.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from ..exceptions import PhysicsCalculationError, PhysicsError

logger = logging.getLogger(__name__)


def calculate_dipole_matrix(levels_dict: Dict[str, Any], arc_atom: Optional[Any] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate dipole matrix elements for the EIT system given optimized levels.

    Args:
        levels_dict: Dictionary with 'ground', 'excited', 'rydberg1', 'rydberg2' AtomicLevel objects
        arc_atom: ARC atom instance for dipole calculations

    Returns:
        Dictionary with transition dipoles:
        {
            'probe': {'dipole_si': dipole_for_ground_to_excited},
            'coupling': {'dipole_si': dipole_for_excited_to_rydberg1},
            'rf': {'dipole_si': dipole_for_rydberg1_to_rydberg2}
        }

    Raises:
        ValueError: If required levels are missing.
        PhysicsCalculationError: If the dipole calculation fails or returns an
            unusable (None / zero / non-finite) value for a driven transition
            (F05 fail-fast contract).
    """
    if not arc_atom:
        raise ValueError("ARC atom instance required for dipole calculations")

    required_levels = ['ground', 'excited', 'rydberg1', 'rydberg2']
    for level_name in required_levels:
        if level_name not in levels_dict:
            raise ValueError(f"Missing required level: {level_name}")

    logger.debug("Calculating dipole matrix for EIT transitions")

    # Import here to avoid circular imports
    from .dipole import calculate_dipole_matrix_element

    # Extract levels
    g = levels_dict['ground']
    e = levels_dict['excited']
    r1 = levels_dict['rydberg1']
    r2 = levels_dict['rydberg2']

    # Define the key EIT transitions
    transitions = {
        'probe': (g, e, 'ground_to_excited'),
        'coupling': (e, r1, 'excited_to_rydberg1'),
        'rf': (r1, r2, 'rydberg1_to_rydberg2')
    }

    dipole_matrix = {}

    try:
        for transition_name, (level1, level2, description) in transitions.items():
            logger.debug(f"Calculating dipole for {transition_name} transition: {description}")

            # Use the existing dipole calculation function
            dipole_result = calculate_dipole_matrix_element(arc_atom, level1, level2.n, level2.l, level2.j)

            # Handle both dict return (if standardized) and numeric return
            if isinstance(dipole_result, dict):
                if 'dipole_si' not in dipole_result:
                    # F05: a mapping without 'dipole_si' must not be read as 0.0.
                    raise PhysicsCalculationError(
                        f"Dipole result for the {transition_name} transition "
                        f"({description}) has no 'dipole_si' entry "
                        f"(keys={sorted(dipole_result)}). F05: no fallback."
                    )
                dipole_si = dipole_result['dipole_si']
            else:
                dipole_si = dipole_result

            if dipole_si is None:
                raise PhysicsCalculationError(
                    f"Dipole moment for the {transition_name} transition "
                    f"({description}) is None. F05: no fallback is permitted."
                )
            try:
                dipole_si = float(dipole_si)
            except (TypeError, ValueError) as exc:
                raise PhysicsCalculationError(
                    f"Dipole moment for the {transition_name} transition "
                    f"({description}) is not numeric ({dipole_si!r}). "
                    "F05: no fallback."
                ) from exc
            if not np.isfinite(dipole_si) or dipole_si <= 0.0:
                # F05: the probe, coupling and RF transitions are all DRIVEN
                # transitions of the model - a zero, negative or non-finite
                # matrix element is a calculation failure and must never be
                # carried silently into the Hamiltonian.
                raise PhysicsCalculationError(
                    f"Unusable dipole matrix element {dipole_si!r} C*m for the "
                    f"driven {transition_name} transition ({description}) "
                    f"({level1.label} -> {level2.label}). "
                    "F05: no fallback is permitted."
                )

            dipole_matrix[transition_name] = {
                'dipole_si': dipole_si,
                'transition': f"{level1.label} -> {level2.label}",
                'level1': level1,
                'level2': level2
            }

            logger.debug(f"{transition_name.capitalize()} dipole: {dipole_si:.2e} C*m ({level1.label} -> {level2.label})")

    except PhysicsError:
        raise
    except Exception as exc:
        logger.error(f"Failed to calculate dipole matrix: {exc}")
        # F05: fail loudly with the dedicated hierarchy (previously re-wrapped
        # as a bare ValueError).
        raise PhysicsCalculationError(
            f"Dipole matrix calculation failed: {type(exc).__name__}: {exc}. "
            "F05: no fallback is permitted."
        ) from exc

    return dipole_matrix


def get_dipole_for_transition(dipole_matrix: Dict[str, Dict[str, float]], transition_type: str) -> float:
    """
    Extract dipole value for a specific transition from the dipole matrix.

    Args:
        dipole_matrix: Output from calculate_dipole_matrix
        transition_type: 'probe', 'coupling', or 'rf'

    Returns:
        Dipole moment in SI units (C*m)
    """
    if transition_type not in dipole_matrix:
        raise ValueError(f"Unknown transition type: {transition_type}")

    return dipole_matrix[transition_type]['dipole_si']

def get_probe_dipole():
    """Retrieve the probe transition dipole moment.

    Returns:
        float: Dipole moment in SI units (C*m).

    Raises:
        PhysicsCalculationError: If no usable probe dipole moment can be
            obtained (F05 fail-fast contract). The historical behaviour
            returned a hard-coded dipole constant, which silently replaced the
            physical probe matrix element.
    """
    from ..config.atomic_config import get_cached_dipole_matrix

    dipole_matrix = get_cached_dipole_matrix()
    if dipole_matrix and 'probe' in dipole_matrix:
        probe_entry = dipole_matrix['probe']
        if 'dipole_si' not in probe_entry:
            raise PhysicsCalculationError(
                "Cached dipole matrix entry 'probe' has no 'dipole_si' "
                "key. F05: no fallback is permitted."
            )
        return probe_entry['dipole_si']

    raise PhysicsCalculationError(
        "No cached probe dipole moment is available for the current "
        "laser/RF parameters. F05: no hard-coded fallback dipole."
    )

def validate_dipole_matrix(dipole_matrix: Dict[str, Dict[str, float]]) -> bool:
    """
    Validate that dipole values are reasonable for EIT system.

    Args:
        dipole_matrix: Output from calculate_dipole_matrix

    Returns:
        True if validation passes
    """
    required_transitions = ['probe', 'coupling', 'rf']

    for transition in required_transitions:
        if transition not in dipole_matrix:
            logger.error(f"Missing dipole for {transition} transition")
            return False

        dipole_si = dipole_matrix[transition]['dipole_si']
        if dipole_si <= 0:
            logger.warning(f"Zero or negative dipole for {transition}: {dipole_si}")
            # Don't fail for zero dipole, just warn (some transitions may have negligible dipole)

    logger.info("Dipole matrix validation passed")
    return True
