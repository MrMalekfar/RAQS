"""
Dynamic Parameter Manager for Rydberg AoA Simulation.

Provides centralized management of all simulation parameters with dynamic recalculation,
dependency tracking, and invalidation mechanisms. Ensures all parameters are physics-based
and recalculated when dependencies change.
"""

import logging
from typing import Dict, Any, Optional, Set, Tuple
import numpy as np

# F05 — every physics-derived quantity in this module fails loudly: no
# placeholder constants, no zero-detuning tuples, no None substitutions.
from ..exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
    PhysicsError,
)

logger = logging.getLogger(__name__)


class ParameterManager:
    """
    Centralized parameter management with dynamic updates and dependency tracking.
    
    Handles laser parameters, RF parameters, atomic levels, and derived quantities
    like detunings and Rabi frequencies. Automatically invalidates dependent parameters
    when base parameters change.
    """

    def __init__(self):
        """Initialize parameter manager with empty parameter store."""
        self._params: Dict[str, Any] = {}
        self._is_valid: Dict[str, bool] = {}
        self._dependencies: Dict[str, Set[str]] = self._build_dependency_graph()
        self._reverse_dependencies: Dict[str, Set[str]] = self._build_reverse_dependencies()

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Define which parameters depend on which others.
        
        Returns:
            Dict mapping parameter names to sets of parameters they depend on
        """
        return {
            'coupling_detuning_rad_us': {'coupling_wavelength_nm', 'levels'},
            'rf_detuning_rad_us': {'rf_frequency_hz', 'levels'},
            'probe_rabi_frequency_rad_us': {'probe_wavelength_nm', 'levels'},
            'coupling_rabi_frequency_rad_us': {'coupling_wavelength_nm', 'levels'},
            'rf_rabi_frequency_rad_us': {'rf_frequency_hz', 'levels'},
            'all_detunings': {'coupling_wavelength_nm', 'rf_frequency_hz', 'levels'},
            'arc_result_cache': {'coupling_wavelength_nm', 'rf_frequency_hz'},
        }

    def _build_reverse_dependencies(self) -> Dict[str, Set[str]]:
        """
        Build reverse dependency mapping: which parameters to invalidate when a dependency changes.
        
        Returns:
            Dict mapping parameter names to sets of parameters that depend on them
        """
        reverse = {}
        for dependent, deps in self._dependencies.items():
            for dep in deps:
                reverse.setdefault(dep, set()).add(dependent)
        return reverse

    def load_defaults(self):
        """Load default parameters with hardcoded defaults to avoid circular imports."""
        # Hardcoded defaults to avoid circular imports
        LASER_PARAMS = {
            'probe': {
                'power_mw': 0.001,
                'beam_waist_mm': 0.5,
                'wavelength_nm': 780.2412,
                'detuning_mhz': 0.0,
                'polarization': 0
            },
            'coupling': {
                'power_mw': 10.0,
                'beam_waist_mm': 0.5,
                'wavelength_nm': None,  # Dynamically calculated
                'detuning_mhz': 0.0,
                'polarization': 0
            }
        }
        RF_PARAMS = {
            'frequency_hz': 10e9,  # RF_FREQUENCY
            'electric_field_vm': 0.5,
            'polarization': 0
        }
        TARGET_AOA_DEG = -30.0
        N_ELEMENTS = 3
        SPACING = 0.66e-2

        # Store all parameters
        self._params.update({
            # Laser parameters
            'probe_wavelength_nm': LASER_PARAMS['probe']['wavelength_nm'],
            'coupling_wavelength_nm': LASER_PARAMS['coupling']['wavelength_nm'],
            'probe_power_mw': LASER_PARAMS['probe']['power_mw'],
            'coupling_power_mw': LASER_PARAMS['coupling']['power_mw'],
            'probe_beam_waist_mm': LASER_PARAMS['probe']['beam_waist_mm'],
            'coupling_beam_waist_mm': LASER_PARAMS['coupling']['beam_waist_mm'],
            'probe_detuning_mhz': LASER_PARAMS['probe']['detuning_mhz'],
            'coupling_detuning_mhz': LASER_PARAMS['coupling']['detuning_mhz'],
            'probe_polarization': LASER_PARAMS['probe']['polarization'],
            'coupling_polarization': LASER_PARAMS['coupling']['polarization'],

            # RF parameters
            'rf_frequency_hz': RF_PARAMS['frequency_hz'],
            'rf_electric_field_vm': RF_PARAMS['electric_field_vm'],
            'rf_polarization': RF_PARAMS['polarization'],

            # Array parameters
            'target_aoa_deg': TARGET_AOA_DEG,
            'n_elements': N_ELEMENTS,
            'element_spacing_m': SPACING,

            # Atomic levels (will be set by optimization)
            'levels': None,
        })

        # Mark all as valid initially
        for key in self._params:
            self._is_valid[key] = True

        # Mark derived parameters as invalid initially
        self._is_valid.update({
            'coupling_detuning_rad_us': False,
            'rf_detuning_rad_us': False,
            'probe_rabi_frequency_rad_us': False,
            'coupling_rabi_frequency_rad_us': False,
            'rf_rabi_frequency_rad_us': False,
            'all_detunings': False,
            'arc_result_cache': False,
        })

    def get(self, key: str, recalculate_if_invalid: bool = True) -> Any:
        """
        Get a parameter value, recalculating if necessary.
        
        Args:
            key: Parameter name
            recalculate_if_invalid: Whether to recalculate invalid parameters
            
        Returns:
            Parameter value
        """
        if recalculate_if_invalid and not self._is_valid.get(key, True):
            self._recalculate(key)
            
        if key not in self._params:
            logger.warning(f"Parameter '{key}' not found, returning None")
            return None
            
        return self._params[key]

    def set(self, key: str, value: Any, invalidate_dependents: bool = True):
        """
        Set a parameter value and invalidate dependent parameters.
        
        Args:
            key: Parameter name
            value: New value
            invalidate_dependents: Whether to invalidate dependent parameters
        """
        old_value = self._params.get(key, None)
        self._params[key] = value
        self._is_valid[key] = True
        
        if old_value != value and invalidate_dependents:
            self._invalidate_dependents(key)
            
        logger.debug(f"Parameter updated: {key} = {value}")

    def _invalidate_dependents(self, changed_key: str):
        """Invalidate all parameters that depend on the changed parameter."""
        dependents = self._reverse_dependencies.get(changed_key, set())
        for dep in dependents:
            self._is_valid[dep] = False
            logger.debug(f"Invalidated dependent parameter: {dep} due to change in {changed_key}")

    def _recalculate(self, key: str):
        """Recalculate a specific parameter."""
        try:
            if key == 'all_detunings':
                self._params[key] = self._calculate_detunings()
            elif key == 'probe_rabi_frequency_rad_us':
                self._params[key] = self._calculate_probe_rabi()
            elif key == 'coupling_rabi_frequency_rad_us':
                self._params[key] = self._calculate_coupling_rabi()
            elif key == 'rf_rabi_frequency_rad_us':
                self._params[key] = self._calculate_rf_rabi()
            elif key == 'coupling_detuning_rad_us':
                self._params[key] = self._calculate_coupling_detuning()
            elif key == 'rf_detuning_rad_us':
                self._params[key] = self._calculate_rf_detuning()
            else:
                logger.warning(f"No recalculation method for parameter: {key}")
                return
                
            self._is_valid[key] = True
            logger.debug(f"Recalculated parameter: {key} = {self._params[key]}")
            
        except PhysicsError:
            # F05: a physics fail-fast must reach the caller unchanged;
            # storing None here would re-create the forbidden substitution.
            raise
        except Exception as e:
            logger.error(f"Failed to recalculate parameter {key}: {e}")
            raise PhysicsCalculationError(
                f"Failed to recalculate parameter {key!r}: "
                f"{type(e).__name__}: {e}. F05: no silent None substitution."
            ) from e

    def update_from_optimization(self, optimization_result: Dict[str, Any]):
        """
        Update parameters based on optimization results.
        
        Args:
            optimization_result: Dictionary from level optimization
        """
        if 'levels' in optimization_result:
            self.set('levels', optimization_result['levels'])
            
        if 'optimization_metrics' in optimization_result:
            metrics = optimization_result['optimization_metrics']
            if 'calculated_coupling_wavelength_nm' in metrics:
                self.set('coupling_wavelength_nm', metrics['calculated_coupling_wavelength_nm'])

        logger.info("Parameters updated from optimization results")

    def is_valid(self, key: str) -> bool:
        """Check if a parameter is currently valid."""
        return self._is_valid.get(key, False)

    def invalidate_all(self):
        """Mark all derived parameters as invalid."""
        for key in self._is_valid:
            if key in self._dependencies:
                self._is_valid[key] = False

    def get_laser_params_for_optimization(self) -> Dict[str, Any]:
        """Get laser parameters formatted for level optimization."""
        return {
            'probe': {
                'wavelength_nm': self.get('probe_wavelength_nm'),
                'power_mw': self.get('probe_power_mw'),
                'beam_waist_mm': self.get('probe_beam_waist_mm'),
                'detuning_mhz': self.get('probe_detuning_mhz'),
                'polarization': self.get('probe_polarization'),
            },
            'coupling': {
                'wavelength_nm': self.get('coupling_wavelength_nm'),
                'power_mw': self.get('coupling_power_mw'),
                'beam_waist_mm': self.get('coupling_beam_waist_mm'),
                'detuning_mhz': self.get('coupling_detuning_mhz'),
                'polarization': self.get('coupling_polarization'),
            }
        }

    def get_rf_params_for_optimization(self) -> Dict[str, Any]:
        """Get RF parameters formatted for level optimization."""
        return {
            'frequency_hz': self.get('rf_frequency_hz'),
            'electric_field_vm': self.get('rf_electric_field_vm'),
            'polarization': self.get('rf_polarization'),
        }

    def _calculate_detunings(self) -> Tuple[float, float, float]:
        """Calculate all detunings using current levels and laser parameters."""
        from .atomic_config import get_calculated_detunings_direct
        
        levels = self.get('levels')
        coupling_wavelength = self.get('coupling_wavelength_nm')
        rf_frequency = self.get('rf_frequency_hz')
        
        if levels is None or coupling_wavelength is None or rf_frequency is None:
            # F05: (0.0, 0.0, 0.0) is a resonance substitution — forbidden.
            raise PhysicsConfigurationError(
                "Cannot calculate detunings: missing required input(s) "
                f"(levels={'present' if levels is not None else 'missing'}, "
                "coupling_wavelength_nm="
                f"{'present' if coupling_wavelength is not None else 'missing'}, "
                f"rf_frequency_hz="
                f"{'present' if rf_frequency is not None else 'missing'}). "
                "F05: no zero-detuning fallback."
            )
            
        # Temporarily update the params for calculation
        from . import laser_rf_config
        
        # This is a bit of a hack - we need to refactor atomic_config to take parameters
        # For now, return cached calculation
        return get_calculated_detunings_direct(
            self.get_laser_params_for_optimization(),
            self.get_rf_params_for_optimization(),
            levels
        )

    def _calculate_probe_rabi(self) -> float:
        """Calculate probe laser Rabi frequency."""
        raise PhysicsCalculationError(
            "ParameterManager has no ARC-backed probe Rabi calculation path. "
            "F05: the historical hard-coded placeholder Rabi constant is "
            "removed — no fabricated Rabi frequency."
        )

    def _calculate_coupling_rabi(self) -> float:
        """Calculate coupling laser Rabi frequency."""
        raise PhysicsCalculationError(
            "ParameterManager has no ARC-backed coupling Rabi calculation "
            "path. F05: the historical hard-coded placeholder Rabi constant "
            "is removed — no fabricated Rabi frequency."
        )

    def _calculate_rf_rabi(self) -> float:
        """Calculate RF Rabi frequency."""
        raise PhysicsCalculationError(
            "ParameterManager has no ARC-backed RF Rabi calculation path. "
            "F05: the historical hard-coded placeholder Rabi constant is "
            "removed — no fabricated Rabi frequency."
        )

    def _calculate_coupling_detuning(self) -> float:
        """Calculate coupling laser detuning."""
        raise PhysicsCalculationError(
            "ParameterManager has no coupling-detuning calculation path. "
            "F05: no resonance (0.0) substitution."
        )

    def _calculate_rf_detuning(self) -> float:
        """Calculate RF detuning."""
        raise PhysicsCalculationError(
            "ParameterManager has no RF-detuning calculation path. "
            "F05: no resonance (0.0) substitution."
        )


# Global parameter manager instance
parameter_manager = ParameterManager()
