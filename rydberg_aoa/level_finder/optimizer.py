"""
Single-Stage ARC Level Optimizer - Main Orchestrator.

Uses ARC database directly for atomic level optimization with broader ranges.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from .types import NarrowedRanges, OptimizationCriteria, AtomicLevel
from .arc_level_finder import ARCOptimizer
from ..arc._scan_validation import integer_scalar, real_scalar
from ..config.parameters import LEVEL_SELECTION_TOLERANCES
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError

logger = logging.getLogger(__name__)


class SingleStageLevelOptimizer:
    """
    Main optimizer using only ARC for level optimization.
    """

    def __init__(self, arc_atom=None, atom_symbol: str = 'Rb85'):
        """
        Initialize the single-stage optimizer.

        Args:
            arc_atom: ARC atom object (loads from atomic_config if None)
            atom_symbol: Symbol for analytical calculations (no longer used)
        """
        # Load global ARC atom if none provided
        if arc_atom is None:
            try:
                from ..config.atomic_config import ATOM
                arc_atom = ATOM
            except ImportError as exc:
                raise PhysicsConfigurationError("LAB-001: configured ARC atom cannot be loaded. F05.") from exc

        self.arc_optimizer = ARCOptimizer(arc_atom)

    def optimize_laboratory(self, request):
        """LAB-003: explicit lab FS selection with applied fields preserved.

        Returns LabSelection. It does not invoke the legacy optimize() path,
        mutate global wavelengths or publish a magnetic-channel runtime.
        """
        from .laboratory import select_laboratory_manifolds
        return select_laboratory_manifolds(self.arc_optimizer.arc_interface.atom, request)

    def optimize_prepared_laboratory(self, request):
        """AUTO-001B: RF-first selection in the fixed prepared channel.

        Returns PreparedLabSelection without publishing an initialized runtime.
        """
        from .prepared_channel import select_prepared_laboratory_channel
        return select_prepared_laboratory_channel(self.arc_optimizer.arc_interface.atom, request)

    def optimize(self, probe_params: Any,
                coupling_params: Any,
                rf_params: Any,
                criteria: OptimizationCriteria = None) -> Dict[str, Any]:
        """
        Perform single-stage ARC optimization of atomic levels.

        Configuration n_range_min/n_range_max include both endpoints;
        n1 and n2 are scanned independently inside that domain. An empty
        selection or ARC failure raises a project physics error without
        widening tolerances, retrying or returning a substitute selection.

        Args:
            probe_params: Probe laser parameters
            coupling_params: Coupling laser parameters
            rf_params: RF field parameters
            criteria: Optimization criteria

        Returns:
            Optimized level selection with performance metrics
        """

        logger.info("=== Starting Single-Stage ARC Level Optimization ===")
        carrier = real_scalar(rf_params.frequency_hz, "rf_params.frequency_hz")
        if carrier <= 0:
            raise PhysicsConfigurationError("LAB-001: applied RF carrier must be positive. F05.")
        for name, field in (("probe", probe_params), ("coupling", coupling_params)):
            wavelength = real_scalar(field.wavelength_nm, f"{name} wavelength [nm]")
            if wavelength <= 0:
                raise PhysicsConfigurationError(f"LAB-001: {name} wavelength must be positive. F05.")
        # Operate log: f_RF [GHz] and lambda_c [nm] of this optimization pass
        # (repairs stale literal ".1f" ".0f" format-paste left by RF-005 class)
        logger.info(f"RF frequency: {rf_params.frequency_hz/1e9:.1f} GHz, "
                    f"coupling wavelength: {coupling_params.wavelength_nm:.0f} nm")

        # Set up detailed logging for debugging
        import logging
        logging.getLogger('rydberg_aoa.level_finder.optimizer').setLevel(logging.DEBUG)

        # LAB-001: configuration min/max are INCLUSIVE. The FS scanner's
        # existing stop-exclusive interface receives max + 1 explicitly.
        try:
            n_min = integer_scalar(LEVEL_SELECTION_TOLERANCES['n_range_min'], 'n_range_min', minimum=1)
            n_max = integer_scalar(LEVEL_SELECTION_TOLERANCES['n_range_max'], 'n_range_max', minimum=1)
        except KeyError as exc:
            raise PhysicsConfigurationError("LAB-001: required RF n bounds are missing. F05.") from exc
        if n_max < n_min:
            raise PhysicsConfigurationError("LAB-001: n_range_max must be >= n_range_min. F05.")
        broad_ranges = NarrowedRanges(
            coupling_n1_range=(n_min, n_max + 1),
            coupling_n2_range=(n_min, n_max + 1),
            rf_n_range=(n_min, n_max + 1)
        )

        # Single stage: ARC optimization with broad ranges
        optimal_result = self.arc_optimizer.optimize_levels(
            broad_ranges, probe_params, coupling_params, rf_params, criteria
        )

        if not optimal_result:
            raise PhysicsCalculationError(
                f"LAB-001: No RF manifold candidate returned for n={n_min}..{n_max}, "
                f"carrier={carrier:g} Hz. F05: no fallback result or retry.")

        if optimal_result:
            optimal_result['optimization_info'] = {
                'method': 'arc_direct_broad_range',
                'success': True
            }

            # Calculate final performance metrics
            optimal_result['performance_metrics'] = self._calculate_performance_metrics(
                optimal_result, probe_params, coupling_params, rf_params
            )

            logger.info("=== Single-Stage ARC Optimization Complete ===")
            logger.info(f"Optimal cascade: Ground -> Excited -> {optimal_result['levels']['rydberg1'].label} -> {optimal_result['levels']['rydberg2'].label}")

        return optimal_result

    def _calculate_performance_metrics(self, result: Dict,
                                     probe_params: Any,
                                     coupling_params: Any,
                                     rf_params: Any) -> Dict[str, float]:
        """
        Calculate final performance metrics for the optimized levels.
        """

        metrics = {}

        # Frequency detunings
        rf_metrics = result.get('optimization_metrics', {})
        metrics['rf_detuning_hz'] = rf_metrics.get('rf_frequency_error_hz', 0)

        # Wavelength accuracies (would need transition calculations)
        # For now, just store target values
        metrics['target_probe_wavelength_nm'] = probe_params.wavelength_nm
        metrics['target_coupling_wavelength_nm'] = coupling_params.wavelength_nm
        metrics['target_rf_frequency_hz'] = rf_params.frequency_hz

        # Dipole strengths
        metrics['rf_dipole_coupling'] = rf_metrics.get('dipole_rf', 0)

        return metrics
