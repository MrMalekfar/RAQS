"""
Main ARC Interface module.

Provides the high-level ARCInterface class that orchestrates
ground state finding, transition searches, and level selection.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .types import LaserParams, RFParams, AtomicLevel, Transition
from .ground_states import find_ground_state, find_excited_state
from .transitions import find_rydberg_transitions, find_rf_transitions
from .rabi_frequency import calculate_rabi_frequency, calculate_rf_rabi_frequency
from ..config import parameters as params
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError

logger = logging.getLogger(__name__)


class ARCInterface:
    """
    Interface for querying ARC and selecting atomic levels dynamically.
    Uses systematic searches through quantum number combinations to find exact matches.
    """

    def __init__(self, atom=None):
        """
        Initialize the ARC interface.

        Args:
            atom: ARC atom object (e.g., Rubidium85())
        """
        self.atom = atom
        # Ground state depends on atom - will be determined dynamically
        self.ground_state = None

    def select_optimal_levels(self, probe_params: LaserParams,
                            coupling_params: LaserParams,
                            rf_params: RFParams,
                            selection_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Select optimal atomic levels based on user parameters using systematic ARC searches.
        Finds exact level matches for the given laser and RF field properties.

        Args:
            probe_params: Probe laser parameters
            coupling_params: Coupling laser parameters
            rf_params: RF parameters
            selection_criteria: Dictionary of selection criteria (uses config.LEVEL_SELECTION_TOLERANCES if None)

        Returns:
            Dictionary containing selected levels, transitions, and calculated parameters
        """
        if selection_criteria is None:
            selection_criteria = params.LEVEL_SELECTION_TOLERANCES.copy()

        # Initialize ground state
        if self.ground_state is None:
            self.ground_state = find_ground_state(self.atom)

        logger.info("Starting level selection based on laser and RF parameters...")

        # Step 1: Find excited state from probe wavelength
        logger.info(f"Finding excited state for probe wavelength {probe_params.wavelength_nm} nm")
        excited_level = find_excited_state(self.atom, probe_params.wavelength_nm)
        if excited_level is None:
            # F05: no silent empty-result substitution.
            raise PhysicsCalculationError(
                "No excited state was found for probe wavelength "
                f"{probe_params.wavelength_nm} nm. "
                "F05: no empty-result substitution."
            )

        logger.info(f"Selected excited state: {excited_level.label}")

        # Step 2: Find Rydberg transitions from excited state at coupling wavelength
        logger.info(f"Finding Rydberg transitions from {excited_level.label} at coupling wavelength {coupling_params.wavelength_nm} nm")
        n_range = (selection_criteria.get('n_range_min', 40), selection_criteria.get('n_range_max', 80))
        rydberg_transitions = find_rydberg_transitions(
            self.atom, excited_level, coupling_params.wavelength_nm,
            wavelength_tolerance_nm=selection_criteria.get('wavelength_tolerance_nm', 1.0),
            n_range=n_range
        )

        if not rydberg_transitions:
            # F05: no silent empty-result substitution.
            raise PhysicsCalculationError(
                "No Rydberg transition was found from "
                f"{excited_level.label} at coupling wavelength "
                f"{coupling_params.wavelength_nm} nm "
                f"(n={n_range[0]}-{n_range[1]}). "
                "F05: no empty-result substitution."
            )

        # Select the strongest Rydberg transition
        best_rydberg_transition = rydberg_transitions[0]
        rydberg1_level = best_rydberg_transition.level2
        logger.info(f"Selected first Rydberg state: {rydberg1_level.label}")

        # Step 3: Find RF transitions using ARC energy calculations
        logger.info(f"Finding RF transitions from {rydberg1_level.label} at frequency {rf_params.frequency_hz/1e6:.1f} MHz")
        rf_n_range = (max(10, rydberg1_level.n - 5), rydberg1_level.n + 5)  # Search around Rydberg level
        rf_transitions = find_rf_transitions(
            self.atom, rydberg1_level, rf_params.frequency_hz,
            frequency_tolerance_hz=selection_criteria.get('frequency_tolerance_hz', 1e7),
            n_range=rf_n_range
        )

        if not rf_transitions:
            # F05: the historical fallback fabricated an (n+1)P3/2 Rydberg
            # level together with a placeholder dipole (1e-30 C*m), a zero
            # wavelength and a zero frequency error. That is a fabricated
            # physics object, not a calculation. Fail loudly instead.
            raise PhysicsCalculationError(
                "No RF transition was found from "
                f"{rydberg1_level.label} at f_RF="
                f"{rf_params.frequency_hz/1e6:.1f} MHz within the "
                f"{selection_criteria.get('frequency_tolerance_hz', 1e7)/1e6:.1f} MHz "
                f"tolerance (searched n={rf_n_range[0]}-{rf_n_range[1]}). "
                "F05: no fabricated n+1 Rydberg level and no placeholder dipole."
            )
        else:
            # Use the best found transition
            best_rf_transition = rf_transitions[0]
            rydberg2_level = best_rf_transition.level2

        logger.info(f"Selected second Rydberg state: {rydberg2_level.label}")

        # Step 4: Calculate Rabi frequencies and dipole moments
        levels = [self.ground_state, excited_level, rydberg1_level, rydberg2_level]

        # Calculate probe transition dipole (ground to excited)
        if self.ground_state is None:
            # F05: no silent 0.0 substitution for a missing ground state.
            raise PhysicsConfigurationError(
                "self.ground_state is None; the probe transition for the "
                "selected cascade cannot be calculated. "
                "F05: no fallback is permitted."
            )
        probe_dipole = calculate_rabi_frequency(
            self.atom, excited_level, probe_params, from_level=self.ground_state
        )

        # Calculate coupling transition dipole (excited to rydberg1)
        coupling_dipole = best_rydberg_transition.dipole_moment

        # Calculate RF transition dipole (rydberg1 to rydberg2)
        rf_dipole = best_rf_transition.dipole_moment

        # Calculate Rabi frequencies
        omega_p = calculate_rabi_frequency(
            self.atom, excited_level, probe_params, from_level=self.ground_state, ground_state=self.ground_state
        )
        omega_c = calculate_rabi_frequency(
            self.atom, rydberg1_level, coupling_params, from_level=excited_level, ground_state=self.ground_state
        )
        omega_rf = calculate_rf_rabi_frequency(self.atom, rydberg1_level, rydberg2_level, rf_params)

        # Calculate detunings
        delta_p = probe_params.detuning_mhz * 2 * np.pi * 1e6  # Convert MHz to rad/s
        delta_c = coupling_params.detuning_mhz * 2 * np.pi * 1e6
        delta_rf = 0.0  # Assume on resonance

        logger.info("Level selection completed successfully")
        logger.info(f"Selected cascade: {self.ground_state.label} -> {excited_level.label} -> {rydberg1_level.label} -> {rydberg2_level.label}")

        return {
            'levels': levels,
            'transitions': {
                'probe': {
                    'level1': self.ground_state,
                    'level2': excited_level,
                    'omega': omega_p,
                    'delta': delta_p,
                    'dipole': probe_dipole,
                    'wavelength': probe_params.wavelength_nm
                },
                'coupling': {
                    'level1': excited_level,
                    'level2': rydberg1_level,
                    'omega': omega_c,
                    'delta': delta_c,
                    'dipole': coupling_dipole,
                    'wavelength': coupling_params.wavelength_nm
                },
                'rf': {
                    'level1': rydberg1_level,
                    'level2': rydberg2_level,
                    'omega': omega_rf,
                    'delta': delta_rf,
                    'dipole': rf_dipole,
                    'frequency': rf_params.frequency_hz
                }
            },
            'selection_info': {
                'probe_transitions_found': 1,  # Only one possible for D lines
                'rydberg_transitions_found': len(rydberg_transitions),
                'rf_transitions_found': len(rf_transitions),
                'selection_criteria': selection_criteria
            }
        }

    def find_rf_transitions(self, from_level: AtomicLevel,
                           rf_frequency_hz: float,
                           frequency_tolerance_hz: float = 1e7,
                           n_range: Tuple[int, int] = None) -> List[Transition]:
        """
        Find RF transitions from a Rydberg level.

        Args:
            from_level: Starting Rydberg level
            rf_frequency_hz: Target RF transition frequency in Hz
            frequency_tolerance_hz: Frequency tolerance in Hz
            n_range: Range of principal quantum numbers to search

        Returns:
            List of possible RF transitions sorted by dipole moment strength
        """
        return find_rf_transitions(self.atom, from_level, rf_frequency_hz, frequency_tolerance_hz, n_range)

    def find_ground_state(self) -> Optional[AtomicLevel]:
        """
        Find the optimal ground state for the given atom.

        Returns:
            AtomicLevel object for the optimal ground state
        """
        return find_ground_state(self.atom)
