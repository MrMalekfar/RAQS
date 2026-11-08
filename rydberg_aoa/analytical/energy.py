"""
Analytical Rydberg Energy Level Calculator.

This module provides analytical calculations of Rydberg energy levels using quantum defect theory,
eliminating dependency on ARC database limitations for high-n states.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class QuantumDefects:
    """Quantum defects for Rb85 Rydberg states."""
    # Rb85 quantum defects from Gallagher review (1994) and recent measurements
    # Format: {l: (μ0, μ1, μ2, μ3, μ4)} where μi are polynomial coefficients
    rb85_defects = {
        0: [3.1311804, -0.1784, 0.0, 0.0, 0.0],      # S states
        1: [2.654488, -0.2595, 0.0, 0.0, 0.0],       # P states
        2: [1.347539, -0.602, 0.0, 0.0, 0.0],        # D states
        3: [0.016543, -0.085, 0.0, 0.0, 0.0],        # F states
        4: [0.0, 0.0, 0.0, 0.0, 0.0],                # G states (minimal defect)
    }

    @classmethod
    def get_defect(cls, l: int, n: int) -> float:
        """
        Calculate quantum defect for given l and n using polynomial expansion.

        Args:
            l: Orbital angular momentum
            n: Principal quantum number

        Returns:
            Quantum defect μ_l
        """
        if l not in cls.rb85_defects:
            # For high l, use minimal defect approximation
            return 0.0

        coeffs = cls.rb85_defects[l]
        mu = coeffs[0]
        for i, coeff in enumerate(coeffs[1:], 1):
            mu += coeff / (n**(2*i))

        return mu

class RydbergEnergyCalculator:
    """
    Analytical calculator for Rydberg energy levels using quantum defect theory.
    """

    # Physical constants
    RYDBERG_CONSTANT = 13.605693122994  # eV
    RYDBERG_CONSTANT_CM = 10967758.34    # cm^-1
    HARTREE_TO_CM = 219474.6313708      # cm^-1 per Hartree
    PLANCK_H = 6.62607015e-34          # J*s
    SPEED_OF_LIGHT = 2.99792458e8       # m/s

    def __init__(self, atom='Rb85'):
        """
        Initialize calculator for specific atom.

        Args:
            atom: Atomic species ('Rb85', 'Rb87', 'Cs', etc.)
        """
        from ..exceptions import PhysicsConfigurationError
        if atom != 'Rb85':
            raise PhysicsConfigurationError('ATOM-001: legacy analytical quantum-defect table supports Rb85 only; use ARC for other isotopes.')
        self.atom = atom
        self.quantum_defects = QuantumDefects()

    def calculate_energy_ev(self, n: int, l: int, j: float) -> float:
        """
        Calculate Rydberg energy level in eV using quantum defect theory.

        E_n = -R / (n - μ_l)^2

        Args:
            n: Principal quantum number
            l: Orbital angular momentum
            j: Total angular momentum

        Returns:
            Energy in eV (negative for bound states)
        """
        mu_l = self.quantum_defects.get_defect(l, n)
        effective_n = n - mu_l

        if effective_n <= 0:
            raise ValueError(f"Effective principal quantum number {effective_n} <= 0 for n={n}, l={l}")

        energy_ev = -self.RYDBERG_CONSTANT / (effective_n ** 2)
        return energy_ev

    def calculate_energy_cm(self, n: int, l: int, j: float) -> float:
        """
        Calculate Rydberg energy level in cm^-1.

        Args:
            n: Principal quantum number
            l: Orbital angular momentum
            j: Total angular momentum

        Returns:
            Energy in cm^-1
        """
        energy_ev = self.calculate_energy_ev(n, l, j)
        return energy_ev * 8065.544  # eV to cm^-1 conversion

    def calculate_transition_frequency(self, n1: int, l1: int, j1: float,
                                     n2: int, l2: int, j2: float) -> float:
        """
        Calculate transition frequency between two Rydberg states.

        ν = |E1 - E2| / h

        Args:
            n1, l1, j1: Initial state quantum numbers
            n2, l2, j2: Final state quantum numbers

        Returns:
            Transition frequency in Hz
        """
        energy1_cm = self.calculate_energy_cm(n1, l1, j1)
        energy2_cm = self.calculate_energy_cm(n2, l2, j2)

        delta_e_cm = abs(energy1_cm - energy2_cm)
        frequency_hz = delta_e_cm * self.SPEED_OF_LIGHT * 100  # cm^-1 * cm/s = Hz

        return frequency_hz

    def calculate_transition_wavelength(self, n1: int, l1: int, j1: float,
                                      n2: int, l2: int, j2: float) -> float:
        """
        Calculate transition wavelength between two states.

        λ = c / ν, where ν is the transition frequency.

        Args:
            n1, l1, j1: Initial state quantum numbers
            n2, l2, j2: Final state quantum numbers

        Returns:
            Transition wavelength in nm
        """
        frequency_hz = self.calculate_transition_frequency(n1, l1, j1, n2, l2, j2)
        wavelength_m = self.SPEED_OF_LIGHT / frequency_hz  # meters
        wavelength_nm = wavelength_m * 1e9  # convert to nm

        return wavelength_nm

    def find_rf_transitions_analytical(self, n_start: int, rf_freq_hz: float,
                                     freq_tolerance_hz: float = 1e7,
                                     n_range: Tuple[int, int] = None) -> List[Dict]:
        """
        Find Rydberg-Rydberg transitions at target RF frequency using analytical calculations.

        Uses exact quantum defect calculation for proper energy differences.

        Args:
            n_start: Starting principal quantum number
            rf_freq_hz: Target RF frequency in Hz
            freq_tolerance_hz: Frequency tolerance in Hz
            n_range: Range of n to search

        Returns:
            List of transition dictionaries with quantum numbers and frequency differences
        """
        if n_range is None:
            n_range = (max(10, n_start - 10), n_start + 10)

        transitions = []

        logger.debug(f"Searching RF transitions from n={n_start} at {rf_freq_hz/1e6:.1f} MHz")

        # Search through possible final states using exact energy calculation
        for n_final in range(n_range[0], n_range[1] + 1):
            if n_final == n_start:
                continue  # Skip same n transitions

            # Try different l states (Δl = ±1 for electric dipole transitions)
            for l_final in [1, 3]:  # P and F states (Δl = ±1 from D state l=2)
                for j_final in [l_final - 0.5, l_final + 0.5] if l_final > 0 else [0.5]:
                    if j_final < 0:
                        continue

                    try:
                        # Calculate exact transition frequency using quantum defects
                        transition_freq = self.calculate_transition_frequency(
                            n_start, 2, 2.5, n_final, l_final, j_final
                        )

                        frequency_diff = abs(transition_freq - rf_freq_hz)

                        if frequency_diff <= freq_tolerance_hz:
                            transition = {
                                'n1': n_start, 'l1': 2, 'j1': 2.5,
                                'n2': n_final, 'l2': l_final, 'j2': j_final,
                                'frequency_hz': transition_freq,
                                'delta_frequency_hz': transition_freq - rf_freq_hz,
                                'energy_diff_cm': transition_freq / (self.SPEED_OF_LIGHT * 100)
                            }
                            transitions.append(transition)
                            logger.debug(f"Found analytical transition: {n_start}D5/2 -> {n_final}{self._l_to_letter(l_final)}{int(2*j_final+1)//2}/{int(2*j_final+1)%2} at {transition_freq/1e6:.1f} MHz, diff={frequency_diff/1e6:.1f} MHz")

                    except Exception as e:
                        logger.debug(f"Error calculating transition to n={n_final}, l={l_final}, j={j_final}: {e}")
                        continue

        # Sort by frequency difference (closest matches first)
        transitions.sort(key=lambda t: abs(t['delta_frequency_hz']))

        logger.info(f"Found {len(transitions)} analytical RF transitions for frequency {rf_freq_hz/1e6:.1f} MHz")
        return transitions

    def _l_to_letter(self, l: int) -> str:
        """Convert orbital angular momentum to letter notation."""
        letters = 'SPDFGHIJKLMNOP'
        return letters[l] if l < len(letters) else 'X'

    def validate_against_known_transitions(self) -> Dict[str, float]:
        """
        Validate analytical calculations against known Rb Rydberg transitions.

        Returns:
            Dictionary of validation results with frequency differences
        """
        # Known Rb transitions for validation (from literature)
        known_transitions = [
            # (n1, l1, j1, n2, l2, j2, expected_freq_mhz)
            (50, 2, 2.5, 51, 1, 1.5, 24000),  # Approximate 50D5/2 -> 51P3/2 at 24 GHz
            (49, 2, 2.5, 50, 1, 1.5, 25200),  # Approximate 49D5/2 -> 50P3/2
        ]

        validation_results = {}

        for n1, l1, j1, n2, l2, j2, expected_freq_mhz in known_transitions:
            try:
                calculated_freq_hz = self.calculate_transition_frequency(n1, l1, j1, n2, l2, j2)
                calculated_freq_mhz = calculated_freq_hz / 1e6
                diff_mhz = calculated_freq_mhz - expected_freq_mhz
                rel_error = abs(diff_mhz) / expected_freq_mhz * 100

                transition_name = f"{n1}{self._l_to_letter(l1)}{int(2*j1+1)//2} -> {n2}{self._l_to_letter(l2)}{int(2*j2+1)//2}"
                validation_results[transition_name] = {
                    'calculated_mhz': calculated_freq_mhz,
                    'expected_mhz': expected_freq_mhz,
                    'difference_mhz': diff_mhz,
                    'relative_error_percent': rel_error
                }

                logger.info(f"Validation {transition_name}: {calculated_freq_mhz:.1f} MHz (expected {expected_freq_mhz:.1f} MHz, error {rel_error:.2f}%)")

            except Exception as e:
                logger.error(f"Validation failed for {n1}->{n2} transition: {e}")

        return validation_results
