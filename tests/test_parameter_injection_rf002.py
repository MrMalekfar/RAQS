"""RF-002: explicit parameter injection and isolation on a prepared channel."""

import copy

import numpy as np
import pytest

from rydberg_aoa.config import constants as const
from rydberg_aoa.config import parameters as params
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config.laser_rf_config import LASER_PARAMS
from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import \
    calculate_dephasing_rates
from rydberg_aoa.physics.susceptibility import absorption_const as absm
from rydberg_aoa.physics.susceptibility.absorption_const import \
    calculate_absorption_constant

from tests.test_physics_baseline_regression import prepared_atomic_physics, isolated

INJECTION_ATOL = 1e-12


class TestInjectionEquivalence:
    """Explicit args equal to config defaults => bit-identical outputs."""

    def test_dephasing_full_injection_is_bit_identical(self, prepared_atomic_physics):
        # Every implicit input pulled from its legacy source and supplied
        # explicitly; identical operands through the identical code path
        # MUST yield identical bits.
        default = calculate_dephasing_rates()
        injected = calculate_dephasing_rates(
            temperature_k=const.VAPOR_CELL_TEMPERATURE_K,
            beam_waist_mm=LASER_PARAMS['probe']['beam_waist_mm'],
            atom_mass_kg=acfg.ATOM.mass,
            n_density_m3=params.CALCULATED_N0,
            rydberg_n=acfg.ATOMIC_LEVELS['rydberg1']['n'],
            gamma_probe_linewidth=params.GAMMA_PROBE_LINEWIDTH,
            gamma_coupling_linewidth=params.GAMMA_COUPLING_LINEWIDTH,
            gamma_rf_linewidth=params.GAMMA_RF_LINEWIDTH,
        )
        np.testing.assert_array_equal(
            injected, default,
            err_msg="explicit dephasing injection altered the result")

    def test_absorption_full_injection_is_bit_identical(self, prepared_atomic_physics):
        # μ₂₁ from the same cache entry the default path consumes;
        # ω_p mirrored with the function's own conversion
        # (ω_p[rad/s] = Ω_P[rad/μs] × 10⁶).
        default = calculate_absorption_constant()
        mu_21 = acfg.get_cached_dipole_matrix()['probe']['dipole_si']
        injected = calculate_absorption_constant(
            mu_21_c_m=mu_21,
            omega_p_rad_per_s=acfg.OMEGA_P * params.US_TO_S_FACTOR,
            n0_m3=absm._CONST_SI.N0,
            cell_length_m=params.CELL_LENGTH,
            lambda_p_m=absm._CONST_SI.LAMBDA_P,
        )
        assert injected[0] == default[0]  # od_factor — exact
        assert injected[1] == default[1]  # K_chi — exact


# ---------------------------------------------------------------------------
# Test 3 — Dependency isolation
# ---------------------------------------------------------------------------
class TestDependencyIsolation:
    """Injected values steer the math; global state stays untouched."""

    _TRACKED = (
        ('const', 'VAPOR_CELL_TEMPERATURE_K'),
        ('params', 'CALCULATED_N0'),
        ('params', 'GAMMA_PROBE_LINEWIDTH'),
        ('params', 'GAMMA_COUPLING_LINEWIDTH'),
        ('params', 'GAMMA_RF_LINEWIDTH'),
        ('params', 'CELL_LENGTH'),
        ('acfg', 'OMEGA_P'),
        ('absm', '_CONST_SI'),
    )

    @classmethod
    def _snapshot(cls):
        mods = {'const': const, 'params': params,
                'acfg': acfg, 'absm': absm}
        snap = {('%s.%s' % (m, a)): getattr(mods[m], a)
                for m, a in cls._TRACKED}
        # Deep-copy mutable structures injection could corrupt via aliasing.
        snap['LASER_PARAMS'] = copy.deepcopy(LASER_PARAMS)
        snap['ATOMIC_LEVELS'] = copy.deepcopy(acfg.ATOMIC_LEVELS)
        return snap

    def test_dephasing_no_global_mutation_and_expected_response(
            self, prepared_atomic_physics):
        before = self._snapshot()
        default = np.asarray(calculate_dephasing_rates(), dtype=float)

        # (a) Determinism under unchanged inputs.
        assert calculate_dephasing_rates() == list(default)

        # (b) Only the injected channel moves: γ₂(Γ') − γ₂(Γ) ≡ Γ' − Γ
        #     (DPH-002B: γ₂ = Γ_probe_lw + γ_collision is a single sum; transit
        #     has moved to the reset channel and is no longer part of γ₂).
        injected = calculate_dephasing_rates(gamma_probe_linewidth=1000.0)
        delta = injected[0] - default[0]
        assert abs(delta - (1000.0 - params.GAMMA_PROBE_LINEWIDTH)) <= INJECTION_ATOL

        # (c) DPH-002B contract migration — transit left the pure-dephasing
        #     budget and is now the loss/reset channel.
        waist = LASER_PARAMS['probe']['beam_waist_mm']
        wider = calculate_dephasing_rates(beam_waist_mm=2.0 * waist)
        # (c-A) PURE DEPHASING: the probe beam waist must NOT alter the
        #     pure-dephasing output now that transit is excluded from it.
        assert wider[2] == default[2]
        assert wider[0] == default[0]
        assert wider[1] == default[1]
        # (c-B) TRANSIT RATE: independent first-principles oracle
        #     Gamma_t = sqrt(2/pi) * sqrt(k_B*T/m) / w0  — wider beam,
        #     smaller Gamma_t, exactly Gamma_t ∝ 1/w0.
        from rydberg_aoa.physics.lindblad.calculate_transit_rate import \
            calculate_transit_rate_per_us
        w0 = waist * 1e-3
        gamma_t_expected = (np.sqrt(2 / np.pi)
                            * np.sqrt(const.k_B * params.VAPOR_CELL_TEMPERATURE_K
                                      / acfg.ATOM.mass) / w0 * 1e-6)
        assert calculate_transit_rate_per_us() == pytest.approx(
            gamma_t_expected, rel=1e-12)
        assert calculate_transit_rate_per_us(beam_waist_mm=2.0 * waist) == \
            pytest.approx(gamma_t_expected / 2.0, rel=1e-12)

        # (d) No hidden writes anywhere in the tracked state surface.
        assert self._snapshot() == before

    def test_absorption_no_global_mutation_and_injection_bypasses_cache(
            self, prepared_atomic_physics):
        before = self._snapshot()
        default_od, default_kchi = calculate_absorption_constant()

        # (a) Explicit μ₂₁ fully replaces the cache lookup: μ→0 ⇒ K_chi→0
        #     (K_chi = −2N₀μ²/(ε₀ħω_p)); od_factor = πL/λ_p is μ-independent,
        #     hence bit-identical to the default run.
        od_zero_mu, kchi_zero_mu = calculate_absorption_constant(mu_21_c_m=0.0)
        assert od_zero_mu == default_od
        assert kchi_zero_mu == 0.0

        # (b) Cell-length injection drives od_factor exactly as πL/λ_p.
        od_double, _ = calculate_absorption_constant(
            cell_length_m=2.0 * params.CELL_LENGTH)
        expected_od = (np.pi * (2.0 * params.CELL_LENGTH)) / absm._CONST_SI.LAMBDA_P
        assert od_double == expected_od

        # (c) K_chi is quadratic in μ₂₁: μ → k·μ ⇒ K_chi → k²·K_chi
        #     (deviation bounded by a few double-precision ulps ⇒ rtol 1e-12).
        mu_21 = acfg.get_cached_dipole_matrix()['probe']['dipole_si']
        _, k_scaled = calculate_absorption_constant(mu_21_c_m=3.0 * mu_21)
        np.testing.assert_allclose(k_scaled, 9.0 * default_kchi,
                                   rtol=1e-12, atol=0)

        # (d) No hidden writes anywhere in the tracked state surface.
        assert self._snapshot() == before