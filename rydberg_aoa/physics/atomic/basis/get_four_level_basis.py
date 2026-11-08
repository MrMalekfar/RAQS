"""
Get the four-level atomic basis states.
"""

import qutip as qt


def get_four_level_basis():
    """
    Get the four-level atomic basis states.

    Returns:
        tuple: (s1, s2, s3, s4) basis states
    """
    return [qt.basis(4, i) for i in range(4)]
