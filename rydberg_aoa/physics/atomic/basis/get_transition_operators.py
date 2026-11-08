"""
Get the atomic transition operators for the 4-level system.
"""

import qutip as qt
from .get_four_level_basis import get_four_level_basis


def get_transition_operators():
    """
    Get the atomic transition operators for the 4-level system.

    Returns:
        dict: Dictionary of transition operators
    """
    s1, s2, s3, s4 = get_four_level_basis()

    s12 = s1 * s2.dag()
    s23 = s2 * s3.dag()
    s34 = s3 * s4.dag()

    return {
        's12': s12,
        's21': s12.dag(),
        's23': s23,
        's32': s23.dag(),
        's34': s34,
        's43': s34.dag(),
    }
