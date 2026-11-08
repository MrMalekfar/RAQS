"""
Test suite for two-stage level optimization.
"""

import pytest
import numpy as np
from rydberg_aoa.arc import LaserParams, RFParams

# Audit B.3 / Part B: the deprecated two-stage optimizer backend
# (rydberg_aoa.optimization.level_optimizer) was removed; only a stub package remains.
pytest.skip(
    "Stale test: rydberg_aoa.optimization.level_optimizer removed (audit Part B / B.3)",
    allow_module_level=True,
)

from rydberg_aoa.optimization import TwoStageLevelOptimizer, OptimizationCriteria, optimize_atomic_levels


class TestTwoStageOptimization:
    """Test the two-stage optimization approach."""

    @pytest.fixture
    def sample_params(self):
        """Create sample laser and RF parameters for testing."""
        probe_params = LaserParams(
            wavelength_nm=780.0,
            power_mw=0.001,
            beam_waist_mm=0.5,
            detuning_mhz=0.0,
            polarization=0
        )

        coupling_params = LaserParams(
            wavelength_nm=480.0,
            power_mw=10.0,
            beam_waist_mm=0.5,
            detuning_mhz=0.0,
            polarization=0
        )

        rf_params = RFParams(
            frequency_hz=24e9,  # 24 GHz
            electric_field_vm=1000.0,
            polarization=0
        )

        return probe_params, coupling_params, rf_params

    def test_optimizer_initialization(self):
        """Test that optimizer can be initialized."""
        optimizer = TwoStageLevelOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'range_optimizer')
        assert hasattr(optimizer, 'arc_optimizer')

    def test_convience_function(self, sample_params):
        """Test the convenience function for optimization."""
        probe_params, coupling_params, rf_params = sample_params

        result = optimize_atomic_levels(probe_params, coupling_params, rf_params)

        # Check that result has expected structure
        assert isinstance(result, dict)
        if result:  # Only check if optimization succeeded
            assert 'levels' in result
            assert 'optimization_info' in result
            assert 'performance_metrics' in result

    def test_optimization_criteria(self):
        """Test optimization criteria class."""
        criteria = OptimizationCriteria()
        assert criteria.max_dipole_strength == 1e-25
        assert criteria.max_detuning_tolerance == 5.0
        assert criteria.prioritize_electric_dipole is True
        assert criteria.prefer_low_n is True

        # Test custom criteria
        custom_criteria = OptimizationCriteria(
            max_dipole_strength=1e-24,
            max_detuning_tolerance=10.0,
            prioritize_electric_dipole=False
        )
        assert custom_criteria.max_dipole_strength == 1e-24
        assert custom_criteria.max_detuning_tolerance == 10.0
        assert custom_criteria.prioritize_electric_dipole is False

    @pytest.mark.slow
    def test_full_optimization_integration(self, sample_params):
        """Test full optimization with the integration."""
        probe_params, coupling_params, rf_params = sample_params

        # This test requires ARC to be installed
        try:
            from arc import Rubidium85
            atom = Rubidium85()
        except ImportError:
            pytest.skip("ARC library not available for full integration test")
            return

        optimizer = TwoStageLevelOptimizer(arc_atom=atom)
        criteria = OptimizationCriteria()

        result = optimizer.optimize(probe_params, coupling_params, rf_params, criteria)

        assert isinstance(result, dict)

        if result:  # Only run these checks if optimization succeeded
            # Check level structure
            assert 'levels' in result
            levels = result['levels']
            assert len(levels) == 4  # ground, excited, rydberg1, rydberg2

            # Check that levels have required attributes
            for level_key in ['ground', 'excited', 'rydberg1', 'rydberg2']:
                level = levels[level_key]
                assert hasattr(level, 'n')
                assert hasattr(level, 'l')
                assert hasattr(level, 'j')
                assert hasattr(level, 'label')

            # Check optimization info
            assert 'optimization_info' in result
            opt_info = result['optimization_info']
            assert opt_info['method'] == 'analytical_narrowing + arc_refinement'
            assert opt_info['success'] is True

            # Check performance metrics
            assert 'performance_metrics' in result
            metrics = result['performance_metrics']
            assert 'rf_detuning_hz' in metrics
            assert 'target_probe_wavelength_nm' in metrics
            assert 'target_coupling_wavelength_nm' in metrics
            assert 'target_rf_frequency_hz' in metrics

            print(f"Successfully optimized levels: Ground -> Excited -> {levels['rydberg1'].label} -> {levels['rydberg2'].label}")


if __name__ == "__main__":
    pytest.main([__file__])
