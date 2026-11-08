"""
Analytical Range Finder - Stage 1 Optimization.

Uses quantum defect theory for rapid scanning over broad parameter space
to identify promising regions for more expensive ARC calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from ..analytical.energy import RydbergEnergyCalculator
from .types import NarrowedRanges, LevelTransitionCandidate
from .criteria import calculate_analytical_figure_of_merit, calculate_rf_figure_of_merit

logger = logging.getLogger(__name__)


class LevelRangeOptimizer:
    """
    Stage 1: Fast analytical optimization to narrow search ranges.

    Uses quantum defect theory for rapid scanning over broad parameter space
    to identify promising regions for more expensive ARC calculations.
    """

    def __init__(self, atom_symbol: str = 'Rb85'):
        """
        Initialize the range optimizer.

        Args:
            atom_symbol: Atomic species ('Rb85', 'Rb87', 'Cs', etc.)
        """
        self.atom_symbol = atom_symbol
        self.energy_calc = RydbergEnergyCalculator(atom=atom_symbol)

    def optimize_ranges(self, probe_params: Any,
                       coupling_params: Any,
                       rf_params: Any,
                       broad_n_range: Tuple[int, int] = (25, 80),
                       frequency_steps: int = 50) -> NarrowedRanges:
        """
        Perform analytical optimization to narrow search ranges.

        Args:
            probe_params: Probe laser parameters
            coupling_params: Coupling laser parameters
            rf_params: RF field parameters
            broad_n_range: Initial broad range to scan analytically
            frequency_steps: Number of frequency steps for prediction

        Returns:
            NarrowedRanges with optimized search bounds
        """

        logger.info("Stage 1: Analytical range optimization starting...")

        # Initialize broadest possible ranges
        narrowed = NarrowedRanges()

        # Store analytical predictions
        predictions = {}

        # Step 1: Find optimal coupling levels using wavelength matching
        logger.debug("Finding optimal coupling levels analytically...")

        coupling_candidates = self._optimize_coupling_levels_analytically(
            probe_params, coupling_params, broad_n_range, frequency_steps
        )

        if coupling_candidates:
            best_coupling = coupling_candidates[0]
            predictions['coupling'] = best_coupling

            # Set narrow ranges around analytical predictions
            coupling_n1 = best_coupling.n_from
            coupling_n2 = best_coupling.n_to

            narrowed.coupling_n1_range = (max(broad_n_range[0], coupling_n1 - 3),
                                        min(broad_n_range[1], coupling_n1 + 3))
            narrowed.coupling_n2_range = (max(broad_n_range[0], coupling_n2 - 3),
                                        min(broad_n_range[1], coupling_n2 + 3))

            narrowed.rf_n_range = (max(broad_n_range[0], coupling_n1 - 5),
                                 min(broad_n_range[1], coupling_n2 + 5))

            logger.info(f"Analytical coupling prediction: n={coupling_n1} -> n={coupling_n2} "
                       f"at {best_coupling.wavelength_nm:.1f} nm")

        # Step 2: Find optimal RF transitions using frequency matching
        logger.debug("Finding optimal RF transitions analytically...")

        if coupling_candidates:
            rf_candidates = self._optimize_rf_transitions_analytically(
                coupling_candidates[0], rf_params, broad_n_range, frequency_steps
            )

            if rf_candidates:
                predictions['rf'] = rf_candidates[0]
                rf_n = rf_candidates[0].n1
                narrowed.rf_n_range = (max(broad_n_range[0], rf_n - 2),
                                     min(broad_n_range[1], rf_n + 10))  # RF can go n+1, n+2, etc.

                logger.info(f"Analytical RF prediction: n={rf_n} -> n={rf_candidates[0].n2} "
                           f"at {rf_candidates[0].frequency_hz/1e9:.2f} GHz")

        # Set conservative defaults if prediction fails
        if not coupling_candidates:
            logger.warning("No coupling level predictions found analytically, using defaults")
            narrowed.coupling_n1_range = (45, 55)
            narrowed.coupling_n2_range = (46, 56)
            narrowed.rf_n_range = (45, 65)

        narrowed.analytical_predictions = predictions
        narrowed.probe_n_range = (35, 45)  # Probe levels are typically in ground manifold

        logger.info("Stage 1: Analytical optimization complete")
        logger.info(f"Narrowed ranges - Coupling: n={narrowed.coupling_n1_range[0]}-{narrowed.coupling_n2_range[1]}, "
                   f"RF: n={narrowed.rf_n_range[0]}-{narrowed.rf_n_range[1]}")

        return narrowed

    def _optimize_coupling_levels_analytically(self, probe_params: Any,
                                              coupling_params: Any,
                                              n_range: Tuple[int, int],
                                              steps: int) -> List[LevelTransitionCandidate]:
        """
        Find optimal coupling levels using analytical wavelength predictions.

        Args:
            probe_params: Probe laser parameters (for ground->excited info)
            coupling_params: Coupling laser parameters
            n_range: Broad n range to scan
            steps: Number of steps for wavelength scanning

        Returns:
            List of candidate transitions sorted by figure of merit
        """

        candidates = []
        target_wavelength = coupling_params.wavelength_nm

        # For Rydberg transitions, scan excited state to Rydberg manifold
        # Assume D2 line excited state (5P3/2 for Rb)
        n_excited = 5  # Rb ground state
        l_excited = 1  # P state
        j_excited = 1.5  # 3/2

        logger.debug(f"Scanning coupling transitions from {n_excited}P{j_excited} at {target_wavelength:.1f} nm")

        # Estimate the n range that could produce ~480 nm transitions
        # For Rydberg transitions from 5P3/2, wavelength decreases rapidly with n
        # Focus scan around n=45-65 where 480 nm transitions are likely
        n_scan_range = (40, 70)  # More focused range for 480 nm coupling

        # Scan through possible Rydberg levels
        for n_ryd in range(max(n_range[0], n_scan_range[0]), min(n_range[1], n_scan_range[1]) + 1):
            for l_ryd in [0, 1, 2, 3]:  # S, P, D, F states (Δl = ±1 from excited P state)
                for j_ryd in [0.5, 1.5] if l_ryd > 0 else [0.5]:

                    try:
                        # Calculate transition wavelength using quantum defect theory
                        wavelength = self.energy_calc.calculate_transition_wavelength(
                            n_excited, l_excited, j_excited,
                            n_ryd, l_ryd, j_ryd
                        )

                        wavelength_diff = abs(wavelength - target_wavelength)

                        # Debug: log some wavelength calculations
                        if n_ryd % 5 == 0 and l_ryd == 2:  # Log every 5th n for D states
                            logger.debug(f"Calculated: 5P3/2 -> {n_ryd}D{j_ryd}: {wavelength:.1f} nm (diff: {wavelength_diff:.1f} nm)")

                        if wavelength_diff < 100:  # Within 100 nm, promising candidate
                            # Estimate transition strength using analytical dipole
                            fig_of_merit = calculate_analytical_figure_of_merit(
                                wavelength_diff, n_ryd, l_ryd, j_ryd
                            )

                            candidate = LevelTransitionCandidate(
                                wavelength_nm=wavelength,
                                wavelength_diff_nm=wavelength_diff,
                                n_from=n_excited,
                                l_from=l_excited,
                                j_from=j_excited,
                                n_to=n_ryd,
                                l_to=l_ryd,
                                j_to=j_ryd,
                                fig_of_merit=fig_of_merit
                            )
                            candidates.append(candidate)

                    except Exception as e:
                        logger.debug(f"Error calculating wavelength for n={n_ryd}, l={l_ryd}: {e}")
                        continue

        # Sort by figure of merit (closer wavelength = better)
        candidates.sort(key=lambda x: x.fig_of_merit, reverse=True)

        logger.debug(f"Found {len(candidates)} analytical coupling candidates within 100 nm")
        for i, cand in enumerate(candidates[:3]):  # Show top 3
            logger.debug(f"  Candidate {i+1}: {cand.label} "
                        f"at {cand.wavelength_nm:.1f} nm (diff: {cand.wavelength_diff_nm:.1f} nm)")

        return candidates[:5]  # Return top 5 candidates

    def _optimize_rf_transitions_analytically(self, coupling_level: LevelTransitionCandidate,
                                             rf_params: Any,
                                             n_range: Tuple[int, int],
                                             frequency_steps: int) -> List[LevelTransitionCandidate]:
        """
        Find optimal RF transitions using analytical frequency predictions.

        Args:
            coupling_level: Best coupling level from previous stage
            rf_params: RF parameters
            n_range: Broad n range to scan
            frequency_steps: Number of frequency steps

        Returns:
            List of candidate RF transitions
        """

        candidates = []
        target_freq = rf_params.frequency_hz

        # Start from the coupling level's Rydberg state
        n_start = coupling_level.n_to
        l_start = coupling_level.l_to  # Typically D state
        j_start = coupling_level.j_to

        logger.debug(f"Scanning RF transitions from n={n_start}, l={l_start}, j={j_start} at {target_freq/1e9:.2f} GHz")

        # For RF transitions, scan n_start to various n_final = n_start + delta_n
        for delta_n in range(1, 8):  # n+1, n+2, ..., n+7 typical for microwave range
            n_final = n_start + delta_n

            if n_final > n_range[1]:
                continue

            # Try different final l states (Δl = ±1 for electric dipole)
            for delta_l in [-1, 1]:
                l_final = l_start + delta_l
                if l_final < 0:
                    continue

                for j_final in [l_final - 0.5, l_final + 0.5] if l_final > 0 else [0.5]:
                    if j_final < 0:
                        continue

                    try:
                        # Calculate transition frequency
                        freq = self.energy_calc.calculate_transition_frequency(
                            n_start, l_start, j_start,
                            n_final, l_final, j_final
                        )

                        freq_diff = abs(freq - target_freq)

                        if freq_diff < 5e9:  # Within 5 GHz, promising candidate
                            fig_of_merit = calculate_rf_figure_of_merit(
                                freq_diff, n_start, n_final, abs(delta_l)
                            )

                            candidate = LevelTransitionCandidate(
                                wavelength_nm=0.0,  # RF transition, no optical wavelength
                                wavelength_diff_nm=0.0,
                                frequency_hz=freq,
                                frequency_diff_hz=freq_diff,
                                n_from=n_start,
                                l_from=l_start,
                                j_from=j_start,
                                n_to=n_final,
                                l_to=l_final,
                                j_to=j_final,
                                fig_of_merit=fig_of_merit
                            )
                            candidates.append(candidate)

                    except Exception as e:
                        continue

        # Sort by figure of merit (closer frequency = better)
        candidates.sort(key=lambda x: x.fig_of_merit, reverse=True)

        logger.debug(f"Found {len(candidates)} analytical RF candidates")
        return candidates[:3]  # Return top 3 candidates
