"""
Antenna array geometry and phase calculations.

DOA-ARCH-001: the legacy 2-D phase helper now delegates to the shared,
geometry-agnostic steering module (rydberg_aoa.doa.steering) so the arrival
phase convention has a single source of truth.  Results are bit-identical
to the historical implementation for 2-D positions at elevation 0.
"""

from ..doa.steering import arrival_phases


def get_array_phases(array_pos, rf_freq, aoa_deg):
    """
    Calculates the RF phase shift at each element of the array.

    Args:
        array_pos: Array positions (N_elements, 2)
        rf_freq: RF frequency in Hz
        aoa_deg: Angle of arrival in degrees

    Returns:
        Phase shifts in radians
    """
    return arrival_phases(array_pos, rf_freq, aoa_deg, elevation_deg=0.0)
