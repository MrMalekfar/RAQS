"""
Convert wavelength in nm to frequency in Hz.
"""

from rydberg_aoa.config import constants as const


def _wavelength_to_frequency(wavelength_nm: float) -> float:
    """
    Convert wavelength in nm to frequency in Hz.

    Args:
        wavelength_nm: Wavelength in nanometers

    Returns:
        float: Frequency in Hz
    """
    wavelength_m = wavelength_nm * 1e-9  # Convert to meters
    frequency_hz = const.C_LIGHT / wavelength_m
    return frequency_hz
