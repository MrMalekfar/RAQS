"""
Support Functions - Convenience Interfaces.

Contains convenience functions and helper utilities for easy integration.
"""

from typing import Dict, Any
from .optimizer import SingleStageLevelOptimizer


def optimize_atomic_levels(probe_params: Any,
                          coupling_params: Any,
                          rf_params: Any,
                          arc_atom=None) -> Dict[str, Any]:
    """
    Convenience function to optimize atomic levels using the single-stage
    ARC-based optimizer.

    Args:
        probe_params: Probe laser parameters
        coupling_params: Coupling laser parameters
        rf_params: RF field parameters
        arc_atom: ARC atom object

    Returns:
        Optimized level configuration
    """

    # Fix latent NameError: previously referenced TwoStageLevelOptimizer,
    # which was neither imported here nor exported by level_finder anymore.
    optimizer = SingleStageLevelOptimizer(arc_atom=arc_atom)
    return optimizer.optimize(probe_params, coupling_params, rf_params)


def validate_optimization_result(result: Dict[str, Any]) -> bool:
    """
    Validate that an optimization result has all required fields.

    Args:
        result: Optimization result to validate

    Returns:
        True if result is valid
    """
    # AUTO-001A-R2: the FS-only optimization result carries no 'transitions'
    # object (the fabricated legacy RF Transition was removed from selection).
    required_keys = ['levels', 'optimization_metrics']

    for key in required_keys:
        if key not in result:
            return False

    # Check required level keys
    required_levels = ['ground', 'excited', 'rydberg1', 'rydberg2']
    if 'levels' in result:
        for level_key in required_levels:
            if level_key not in result['levels']:
                return False

    return True


def get_optimization_summary(result: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of the optimization result.

    Args:
        result: Optimization result

    Returns:
        Summary string
    """
    if not result or 'levels' not in result:
        return "No optimization result available"

    levels = result['levels']
    metrics = result.get('optimization_metrics', {})

    cascade = f"{levels['ground'].label} -> {levels['excited'].label} -> {levels['rydberg1'].label} -> {levels['rydberg2'].label}"

    rf_freq_error = metrics.get('rf_frequency_error_hz', 0)
    dipole = metrics.get('dipole_rf', 0)
    stage = metrics.get('stage', 'unknown')

    summary = f"Optimization: {stage}\n"
    summary += f"Cascade: {cascade}\n"
    summary += f"RF Error: {rf_freq_error:.0f} Hz\n"
    summary += f"RF Dipole: {dipole:.2e}"

    return summary
