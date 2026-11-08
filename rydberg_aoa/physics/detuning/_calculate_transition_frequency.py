"""
Calculate transition frequency between two levels using ARC getTransitionFrequency.
"""

import logging
from ...arc.types import AtomicLevel

logger = logging.getLogger(__name__)


def _calculate_transition_frequency(atom, level1: AtomicLevel, level2: AtomicLevel) -> float:
    """
    Calculate transition frequency between two levels using ARC's getTransitionFrequency method.

    Args:
        atom: ARC atom object
        level1, level2: AtomicLevel objects

    Returns:
        float: Transition frequency in Hz
    """
    frequency_hz = atom.getTransitionFrequency(
        level1.n, level1.l, level1.j,
        level2.n, level2.l, level2.j
    )

    logger.debug(f"Transition {level1.label}→{level2.label}: ν={frequency_hz/1e9:.3f}GHz")

    return frequency_hz
