"""
Unit tests for the simulation_runner module.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import qutip as qt

# Import modules to test
from rydberg_aoa.config import parameters as cfg
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.simulation.runner import SimulationRunner
from rydberg_aoa.utils.validation import check_conservation_laws
from tests.atomic_init_support import ensure_resolved_atomic_physics


class TestSimulationRunner(unittest.TestCase):
    """Test cases for SimulationRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = SimulationRunner()

    def test_initialization(self):
        """Test that SimulationRunner initializes correctly."""
        self.assertIsNone(self.runner.array_positions)
        self.assertIsNone(self.runner.element_phases_rad)
        self.assertIsNone(self.runner.results)

    def test_setup_array_geometry(self):
        """Test array geometry setup."""
        positions = self.runner.setup_array_geometry()

        # Check shape and values
        self.assertEqual(positions.shape, (cfg.N_ELEMENTS, 2))
        np.testing.assert_array_equal(positions[:, 0], np.arange(cfg.N_ELEMENTS) * cfg.SPACING)
        np.testing.assert_array_equal(positions[:, 1], np.zeros(cfg.N_ELEMENTS))

        # Check that it's stored
        self.assertIsNotNone(self.runner.array_positions)

    @patch('rydberg_aoa.simulation.runner.get_array_phases')
    def test_get_element_phases(self, mock_get_phases):
        """Test element phase calculation."""
        # Mock the phase calculation
        mock_phases = np.array([0.1, 0.2, 0.3])
        mock_get_phases.return_value = mock_phases

        # Set up array positions first
        self.runner.setup_array_geometry()

        # Get phases
        phases = self.runner.get_element_phases()

        # Check that the function was called correctly
        mock_get_phases.assert_called_once_with(
            self.runner.array_positions,
            cfg.RF_FREQUENCY,
            cfg.TARGET_AOA_DEG
        )

        # Check return value
        np.testing.assert_array_equal(phases, mock_phases)
        self.assertIsNotNone(self.runner.element_phases_rad)

    @patch('rydberg_aoa.simulation.runner.simulate_spectrum')
    def test_run_steady_state_spectrum(self, mock_simulate):
        """Test steady-state spectrum simulation."""
        # ARCH-001T: even though the spectrum call itself is mocked, the
        # runner body resolves the full atomic bundle (Rabi frequencies,
        # decay rates, cached detunings, dephasing, absorption constant), so
        # explicit initialization is required. unittest.TestCase cannot take
        # pytest fixtures; the shared opt-in helper is called directly.
        ensure_resolved_atomic_physics()

        # Mock spectrum results
        dp_scan = np.linspace(-10, 10, 100)
        trans_no_rf = np.random.random(100)
        trans_no_rf_analytic = np.random.random(100)
        trans_with_rf = np.random.random(100)
        trans_with_rf_analytic = np.random.random(100)

        mock_simulate.side_effect = [(dp_scan, trans_no_rf, trans_no_rf_analytic),
                                    (dp_scan, trans_with_rf, trans_with_rf_analytic)]

        # Run the method
        result_dp, result_no_rf, result_no_rf_analytic, result_with_rf, result_with_rf_analytic = self.runner.run_steady_state_spectrum()

        # Check that simulate_spectrum was called twice
        self.assertEqual(mock_simulate.call_count, 2)

        # Runner passes the full physics bundle positionally, so compare the
        # RF-drive ω (first positional): no-RF baseline uses ω_rf = 0.0 rad/μs,
        # while the spectra-with-RF run uses the configured base ω_RF [rad/μs].
        first_call_args = mock_simulate.call_args_list[0].args
        second_call_args = mock_simulate.call_args_list[1].args
        self.assertEqual(first_call_args[0], 0.0)
        # OMEGA_RF_BASE lives in atomic_config (RF-005 finding), not parameters
        self.assertEqual(second_call_args[0], acfg.OMEGA_RF_BASE)

        # Check return values
        np.testing.assert_array_equal(result_dp, dp_scan)
        np.testing.assert_array_equal(result_no_rf, trans_no_rf)
        np.testing.assert_array_equal(result_no_rf_analytic, trans_no_rf_analytic)
        np.testing.assert_array_equal(result_with_rf, trans_with_rf)
        np.testing.assert_array_equal(result_with_rf_analytic, trans_with_rf_analytic)

    def test_extract_phases(self):
        """Test phase extraction from complex responses."""
        # Create test complex responses
        complex_responses = np.array([
            1.0 + 0.0j,    # phase = 0
            0.0 + 1.0j,    # phase = π/2
            -1.0 + 0.0j,   # phase = π
            0.0 - 1.0j     # phase = -π/2
        ])

        expected_phases = np.unwrap(np.angle(complex_responses))

        # Extract phases
        extracted = self.runner.extract_phases(complex_responses)

        # Check result
        np.testing.assert_array_almost_equal(extracted, expected_phases)

    def test_process_simulation_results_no_results(self):
        """Test that processing fails when no results are available."""
        with self.assertRaises(ValueError):
            self.runner.process_simulation_results()

    def test_estimate_angle_of_arrival(self):
        """Test AoA estimation."""
        # Set up array positions
        self.runner.setup_array_geometry()

        # Mock complex responses
        complex_responses = np.array([1.0, 1.0, 1.0])

        with patch('rydberg_aoa.tools.analysis.estimate_aoa_beamforming') as mock_beamform:
            # Mock beamforming results
            angles = np.linspace(-90, 90, 181)
            power = np.random.random(181)
            mock_beamform.return_value = (angles, power)

            # Run estimation
            result_angles, result_power, estimated_aoa = self.runner.estimate_angle_of_arrival(complex_responses)

            # Check that beamforming was called
            mock_beamform.assert_called_once()

            # Check results
            np.testing.assert_array_equal(result_angles, angles)
            np.testing.assert_array_equal(result_power, power)
            self.assertEqual(estimated_aoa, angles[np.argmax(power)])

    def test_check_conservation_laws_valid(self):
        """Test conservation check with valid density matrix."""
        # Create a valid density matrix (identity normalized)
        rho = qt.fock_dm(4, 0)  # |0><0| state

        # Should not raise exception
        result = check_conservation_laws(rho, tolerance=1e-10)
        self.assertTrue(result)

    def test_check_conservation_laws_invalid_trace(self):
        """Test conservation check with invalid trace."""
        # Create invalid density matrix with trace != 1
        rho_array = np.eye(4) * 0.5  # trace = 2.0
        rho = qt.Qobj(rho_array)

        with self.assertRaises(AssertionError) as context:
            check_conservation_laws(rho, tolerance=1e-10)

        self.assertIn("Trace error", str(context.exception))

    def test_check_conservation_laws_invalid_populations(self):
        """Test conservation check with invalid population sum."""
        # Create density matrix with population sum != 1
        rho_array = np.eye(4)
        rho_array[0, 0] = 0.5  # Make populations sum to 1.5
        rho = qt.Qobj(rho_array)

        with self.assertRaises(AssertionError) as context:
            check_conservation_laws(rho, tolerance=1e-10)

        self.assertIn("Population sum error", str(context.exception))

    def test_check_conservation_laws_tolerance(self):
        """Test conservation check honors the requested numeric tolerance."""
        # Purity/trace constraint Σ_k ρ_kk = Tr(ρ) = 1 with a controlled
        # numerical defect δρ = 1e-12 applied to a valid pure-state seed:
        # 1e-14 < δρ < 1e-10 separates the loose-pass / strict-fail branch.
        perturbation = np.zeros((4, 4))
        perturbation[0, 0] = 1e-12
        rho = qt.fock_dm(4, 0) + qt.Qobj(perturbation)

        # Should pass with appropriate tolerance
        result = check_conservation_laws(rho, tolerance=1e-10)
        self.assertTrue(result)

        # Should fail with stricter tolerance
        with self.assertRaises(AssertionError):
            check_conservation_laws(rho, tolerance=1e-14)


if __name__ == '__main__':
    unittest.main()
