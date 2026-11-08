"""
Calculate RF detuning for Rydberg-Rydberg transition.
"""

import logging
from ._calculate_transition_frequency import _calculate_transition_frequency
from ._hz_to_rad_per_us import _hz_to_rad_per_us

logger = logging.getLogger(__name__)


def calculate_rf_detuning(atom, rydberg1_level, rydberg2_level, rf_frequency_hz: float) -> float:
    """
    Calculate RF field detuning: Δ_RF = ω_RF_field - ω_(rydberg1 → rydberg2)

    Args:
        atom: ARC atom object
        rydberg1_level, rydberg2_level: AtomicLevel objects for Rydberg-Rydberg transition
        rf_frequency_hz: Actual RF field frequency (Hz)

    Returns:
        float: Detuning in rad/μs for Hamiltonian (Δ_RF in rotating frame)
    """
    # Calculate resonant transition frequency from ARC energies
    omega_resonant_hz = _calculate_transition_frequency(atom, rydberg1_level, rydberg2_level)

    # Detuning: Δ_RF = ω_RF - ω_resonant
    delta_freq_hz = abs(rf_frequency_hz) - abs(omega_resonant_hz)

    # Convert to rad/μs for Hamiltonian
    delta_rad_us = _hz_to_rad_per_us(delta_freq_hz)

    logger.debug(f"RF detuning: ω_RF={rf_frequency_hz/1e9:.3f}GHz, "
                f"ω_resonant={omega_resonant_hz/1e9:.3f}GHz, "
                f"Δ={delta_rad_us:.2f}rad/μs ({delta_freq_hz/1e3:.1f}kHz)")

    return delta_rad_us
