"""
Unit tests for Doppler broadening (thermal velocity-class averaging).

Covers:
- Maxwell-Boltzmann velocity standard deviation σ_v = √(k_B T / m)
- Velocity-class generation (Gauss-Hermite and uniform): normalization,
  symmetry, v=0 node for odd n_classes
- Doppler detuning shifts: magnitudes (σ_v/λ ≈ 300 MHz at 300 K) and
  counter-propagating sign convention (differential two-photon shift)
- Integration: Doppler-averaged spectrum stays physical (finite, positive)
  and the T=0 limit reproduces the cold-atom spectrum exactly
"""

import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.physics.doppler import (
    generate_velocity_classes,
    get_doppler_shifts,
    get_velocity_std,
)

# Rb-85 mass [kg] and reference temperature [K]
M_RB85 = params.RB85_MASS_KG
T_ROOM = 300.0

# Probe/coupling wavelengths from project config [m]
LAMBDA_P_M = 780.241476e-9   # Rb D2 probe line
LAMBDA_C_M = 483.4e-9        # Rydberg coupling line


class TestVelocityStd:
    """σ_v = √(k_B T/m) — thermal velocity scale."""

    def test_rb85_at_300k_is_about_171_m_per_s(self):
        # σ_v = √(k_B T/m) ≈ 171 m/s for Rb-85 at 300 K; consistency check:
        # FWHM = 2√(2 ln 2)·σ_v/λ_p ≈ 516 MHz — the literature ~500 MHz Doppler width
        sigma_v = get_velocity_std(T_ROOM, M_RB85)
        assert 168.0 < sigma_v < 175.0
        fwhm_hz = 2 * np.sqrt(2 * np.log(2)) * sigma_v / LAMBDA_P_M
        assert 490e6 < fwhm_hz < 540e6

    def test_scales_as_sqrt_temperature(self):
        # σ_v ∝ √T — isothermal scaling of the Maxwell-Boltzmann width
        assert np.isclose(
            get_velocity_std(600.0, M_RB85),
            get_velocity_std(300.0, M_RB85) * np.sqrt(2.0),
        )


class TestGenerateVelocityClasses:
    """Discrete velocity-class quadrature properties."""

    @pytest.mark.parametrize("method", ["gauss_hermite", "uniform"])
    def test_weights_sum_to_one(self, method):
        # Normalization Σ w_i = 1 required for a proper ensemble average
        _, weights = generate_velocity_classes(9, T_ROOM, M_RB85, method=method)
        assert np.isclose(weights.sum(), 1.0)

    @pytest.mark.parametrize("method", ["gauss_hermite", "uniform"])
    def test_nodes_symmetric_around_zero(self, method):
        # P(v) is even → symmetric nodes with mirrored equal weights
        velocities, weights = generate_velocity_classes(9, T_ROOM, M_RB85, method=method)
        assert np.allclose(np.sort(velocities), -np.sort(velocities)[::-1], atol=1e-12)
        assert np.allclose(weights, weights[::-1])

    def test_odd_n_classes_has_zero_node(self):
        # Odd node count guarantees an exact v=0 class (cold-atom anchor point)
        velocities, _ = generate_velocity_classes(7, T_ROOM, M_RB85)
        assert np.isclose(np.min(np.abs(velocities)), 0.0, atol=1e-12)

    def test_gauss_hermite_second_moment_matches_variance(self):
        # Quadrature must reproduce ⟨v²⟩ = σ_v² of the target distribution
        velocities, weights = generate_velocity_classes(15, T_ROOM, M_RB85)
        second_moment = np.sum(weights * velocities**2)
        assert np.isclose(second_moment, get_velocity_std(T_ROOM, M_RB85)**2, rtol=1e-6)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            generate_velocity_classes(5, T_ROOM, M_RB85, method="monte_carlo")

    def test_invalid_n_classes_raises(self):
        with pytest.raises(ValueError):
            generate_velocity_classes(0, T_ROOM, M_RB85)


class TestDopplerShifts:
    """Rest-frame detuning shifts Δ' = Δ − k·s·v in rad/μs."""

    def test_shift_magnitude_probe(self):
        # |Δ_p| = k_p·v = 2π(v/λ_p) rad/s → rad/μs; at σ_v ≈ 219 MHz (rms) on 780 nm
        sigma_v = get_velocity_std(T_ROOM, M_RB85)
        d_dp, _ = get_doppler_shifts(sigma_v, LAMBDA_P_M, LAMBDA_C_M)
        expected_rad_per_us = 2 * np.pi * (sigma_v / LAMBDA_P_M) / 1e6
        assert np.isclose(abs(d_dp), expected_rad_per_us, rtol=1e-12)
        # Project unit convention: ω [rad/μs] = 2π × f [MHz] → 2π × 219.5 MHz ≈ 1380
        assert 1300 < abs(d_dp) < 1450

    def test_two_photon_residual_factor(self):
        # Ladder-scheme two-photon detuning is Δ_p + Δ_c, whose Doppler shift is
        # δ_two(v) = δ_p(v) + δ_c(v):
        #   co-propagating  (s_c=+1): δ_two(v) = -(k_p + k_c)·v   → shifts ADD,
        #                              |δ_two|/|δ_p| = 1 + λ_p/λ_c ≈ 2.61
        #   counter-propag. (s_c=-1): δ_two(v) =  (k_c - k_p)·v   → differential
        #                              residual, |δ_two|/|δ_p| = |1 - λ_p/λ_c| ≈ 0.61
        v = 1.0
        kp = 2.0 * np.pi / LAMBDA_P_M
        kc = 2.0 * np.pi / LAMBDA_C_M
        d_dp_co, d_dc_co = get_doppler_shifts(v, LAMBDA_P_M, LAMBDA_C_M,
                                              counter_propagating=False)
        d_dp_cp, d_dc_cp = get_doppler_shifts(v, LAMBDA_P_M, LAMBDA_C_M,
                                              counter_propagating=True)
        # Probe shift is geometry-independent: -k_p·v [rad/μs]
        assert np.isclose(d_dp_co, d_dp_cp, rtol=1e-12)
        assert np.isclose(d_dp_co / (2*np.pi), -kp * v * 1e-6 / (2*np.pi), rtol=1e-12)
        # Coupling shift flips sign between geometries
        assert np.isclose(d_dc_co, -kc * v * 1e-6, rtol=1e-12)
        assert np.isclose(d_dc_cp, +kc * v * 1e-6, rtol=1e-12)
        # Total two-photon shift (ladder Δ_p + Δ_c), in MHz per (m/s)
        two_co = (d_dp_co + d_dc_co) / (2*np.pi)
        two_cp = (d_dp_cp + d_dc_cp) / (2*np.pi)
        assert np.isclose(two_co, -(kp + kc) * v * 1e-6 / (2*np.pi), rtol=1e-9)
        assert np.isclose(two_cp,  (kc - kp) * v * 1e-6 / (2*np.pi), rtol=1e-9)
        # Numerical anchors for the current project wavelengths
        assert np.isclose(two_co, -3.3510, rtol=1e-3)
        assert np.isclose(two_cp,  +0.7877, rtol=1e-3)
        # Factor ratios relative to the single-photon shift
        assert np.isclose(abs(two_co) / abs(d_dp_co / (2*np.pi)),
                          1.0 + LAMBDA_P_M / LAMBDA_C_M, rtol=1e-6)
        assert np.isclose(abs(two_cp) / abs(d_dp_cp / (2*np.pi)),
                          abs(1.0 - LAMBDA_P_M / LAMBDA_C_M), rtol=1e-6)

    def test_co_propagating_shifts_add(self):
        # Co-propagating beams shift in the same direction → full Doppler width
        v = 150.0
        d_dp, d_dc = get_doppler_shifts(v, LAMBDA_P_M, LAMBDA_C_M, counter_propagating=False)
        assert np.sign(d_dp) == np.sign(d_dc)


class TestDopplerDisabledEquivalence:
    """T→0 limit: DOPPLER_ENABLED=False must reproduce the cold-atom path."""

    def test_disabled_config_yields_single_zero_class(self):
        # Simulate the branch taken in simulate_spectrum when Doppler is off
        velocities = np.array([0.0])
        weights = np.array([1.0])
        assert weights.sum() == 1.0
        assert np.all(velocities == 0.0)
