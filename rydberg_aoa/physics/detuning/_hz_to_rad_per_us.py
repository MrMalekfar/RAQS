"""
Convert frequency in Hz to angular frequency in rad/μs.
"""

import numpy as np


def _hz_to_rad_per_us(frequency_hz: float) -> float:
    """
    Convert frequency in Hz to angular frequency in rad/μs.

    Args:
        frequency_hz: Frequency in Hz

    Returns:
        float: Angular frequency in rad/μs
    """
    # ω (rad/s) = 2π × ν (Hz)
    # ω (rad/μs) = ω (rad/s) × 10⁻⁶
    angular_freq_rad_us = 2 * np.pi * frequency_hz * 1e-6
    return angular_freq_rad_us
