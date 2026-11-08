"""DOP-002I regression tests: adaptive Maxwell velocity integration.

Layers covered here:

1. Synthetic integrator tests (fast, physics-independent) — the adaptive
   Maxwell averager must reproduce known analytic moments, integrate the
   complex response as a whole (Re and Im together), and resolve a narrow
   off-center feature when supplied with a breakpoint hint.  These guard
   against the failure mode that defeated fixed-order Gauss-Hermite (GH9).

2. Real-physics validation (slow, marked) — the production adaptive path
   at the RF-off EIT center must agree with an INDEPENDENT QUADPACK
   ``quad`` reference (separate Re/Im integrals, not quad_vec) on the
   complex ensemble coherence rho21.  Transmission-only agreement is not
   sufficient: the comparison is on complex rho.

C-002 ordering is exercised at the production level in
``test_c002_doppler_ensemble.py`` and is not duplicated here.
"""
import math

import numpy as np
import pytest
import scipy.special as sp
from scipy.integrate import quad

from rydberg_aoa.config import parameters as params
from rydberg_aoa.physics.doppler.adaptive_maxwell import (
    get_velocity_std,
    integrate_maxwell_response,
    maxwell_tail_mass,
)
from tests.atomic_init_support import ensure_resolved_atomic_physics

M_RB85 = params.RB85_MASS_KG
T_ROOM = 300.0
SIGMA_V = get_velocity_std(T_ROOM, M_RB85)  # ~171.4 m/s


class TestSyntheticMoments:
    """Analytic Maxwell moments of simple responses (Phase 13, A-D)."""

    def test_constant_response_averages_to_itself(self):
        val = integrate_maxwell_response(lambda v: 1.0 + 0j, T_ROOM, M_RB85)
        assert abs(val - 1.0) < 1e-9

    def test_linear_response_averages_to_zero(self):
        val = integrate_maxwell_response(lambda v: v, T_ROOM, M_RB85)
        assert abs(val) < 1e-6 * SIGMA_V

    def test_quadratic_response_matches_truncated_moment(self):
        val = integrate_maxwell_response(lambda v: v * v, T_ROOM, M_RB85)
        # Truncated second moment: sigma^2 * [1 - 2*n*phi(n)/erf(n/sqrt2)]
        n = 5.0
        phi = math.exp(-0.5 * n * n) / math.sqrt(2 * math.pi)
        erf = sp.erf(n / math.sqrt(2))
        expected = SIGMA_V ** 2 * (1.0 - 2.0 * n * phi / erf)
        assert abs(val - expected) < 1e-6 * SIGMA_V ** 2

    def test_complex_response_integrated_as_whole(self):
        # response = 1 + i*v/sigma  ->  mean = 1 + i*0 exactly
        val = integrate_maxwell_response(
            lambda v: 1.0 + 1j * v / SIGMA_V, T_ROOM, M_RB85)
        assert abs(val.real - 1.0) < 1e-9
        assert abs(val.imag) < 1e-9 * SIGMA_V


class TestNarrowFeatureResolution:
    """The failure mode that defeated GH9: narrow feature, few samples."""

    def test_narrow_gaussian_with_production_stencil(self):
        # Narrow Gaussian (width 1e-3 sigma) centered OFF any default node.
        # Hint DENSITY must match the feature width: a single-point hint
        # provably returns HALF the true value (quad_vec cannot refine the
        # flanks), so the production multi-scale SYMMETRIC stencil is the
        # contract under test.
        from rydberg_aoa.simulation.steady_state import (
            _breakpoint_stencil_for)

        center = 0.37 * SIGMA_V
        width = 1e-3 * SIGMA_V
        height = 100.0

        def response(v):
            return height * math.exp(-0.5 * ((v - center) / width) ** 2)

        val = integrate_maxwell_response(
            response, T_ROOM, M_RB85,
            breakpoints=_breakpoint_stencil_for(center, width))
        # Analytic: the sqrt(2*pi) of the Gaussian integral cancels the
        # sqrt(2*pi) of the Maxwell normalization:
        #   expected = height*width*exp(-0.5 (center/sigma)^2)/sigma
        expected = (height * width / SIGMA_V
                    * math.exp(-0.5 * (center / SIGMA_V) ** 2))
        assert abs(val.real - expected) < 1e-3 * abs(expected)

    def test_breakpoint_stencil_is_symmetric(self):
        # Regression guard: a ONE-SIDED stencil silently misses the
        # negative-velocity flank of a resonance feature (measured 50%
        # error on this synthetic case).
        from rydberg_aoa.simulation.steady_state import (
            _breakpoint_stencil_for)

        bps = _breakpoint_stencil_for(10.0, 2.0)
        assert (10.0 - 2.0) in bps and (10.0 + 2.0) in bps
        assert (10.0 - 8.0) in bps and (10.0 + 8.0) in bps
        assert any(b < 10.0 for b in bps) and any(b > 10.0 for b in bps)

    def test_feature_at_production_scale_without_hint(self):
        # At the production velocity scale (~0.01 sigma, comparable to the
        # homogeneous-linewidth velocity width) the adaptive integrator
        # must converge WITHOUT hints.  Ultra-narrow (1e-3 sigma)
        # off-center features require hints by construction.
        center = 0.37 * SIGMA_V
        width = 0.01 * SIGMA_V

        def response(v):
            return math.exp(-0.5 * ((v - center) / width) ** 2)

        val = integrate_maxwell_response(response, T_ROOM, M_RB85)
        expected = (width / SIGMA_V
                    * math.exp(-0.5 * (center / SIGMA_V) ** 2))
        assert abs(val.real - expected) < 5e-3 * abs(expected)


class TestRealPhysicsOracle:
    """Slow real-physics validation against an INDEPENDENT quad oracle.

    The production adaptive result at the RF-off EIT center (dp=0) is
    compared on COMPLEX rho21 against separate scipy.integrate.quad
    integrals of Re/Im (a different QUADPACK algorithm than quad_vec).
    Runs one cold-model ensemble per RF state; ~3-4 minutes total.
    """

    @pytest.fixture(scope="class")
    def oracle_data(self):
        pytest.importorskip("scipy.integrate")
        # ARCH-001T: the real-physics oracle needs RESOLVED quantities;
        # opt into the explicit ARCH-001 seam (idempotent; skipped with a
        # ``BLOCKED: QN-001A`` reason while the coupling mJ contract is
        # open). The synthetic integrator tests above need no atomic physics.
        ensure_resolved_atomic_physics()
        import rydberg_aoa.config.atomic_config as acfg
        from rydberg_aoa.config.laser_rf_config import LASER_PARAMS
        from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
            calculate_dephasing_rates)
        from rydberg_aoa.physics.lindblad.calculate_transit_rate import (
            calculate_transit_rate_per_us)
        from rydberg_aoa.physics.doppler import (
            get_doppler_shifts, resonance_velocity_hints, velocity_scale_hint)
        from rydberg_aoa.simulation.steady_state import (
            _breakpoint_stencil_for, _clean_breakpoints,
            _homogeneous_gamma_scale, _solve_probe_coherence)

        transit = calculate_transit_rate_per_us()
        omega_p, omega_c, omega_rf = acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE
        dp0, dc, drf = acfg.get_cached_detunings()
        gammas = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
        deph = calculate_dephasing_rates()
        lp = LASER_PARAMS['probe']['wavelength_nm'] * 1e-9
        lc = LASER_PARAMS['coupling']['wavelength_nm'] * 1e-9
        counter = params.DOPPLER_COUNTER_PROPAGATING

        def make_point(dp_mhz, omega_rf_val):
            scan_rad = 2 * np.pi * dp_mhz
            dpc = dp0 + scan_rad
            cache = {}

            def rho_of_v(v):
                key = round(float(v), 10)
                if key in cache:
                    return cache[key]
                sdp, sdc = get_doppler_shifts(float(v), lp, lc,
                                              counter_propagating=counter)
                val = _solve_probe_coherence(
                    dpc + sdp, dc + sdc, omega_p, omega_c,
                    omega_rf_val, drf, gammas, deph,
                    transit_rate_per_us=transit)
                cache[key] = val
                return val

            hints = resonance_velocity_hints(dp0, dc, scan_rad, drf,
                                             omega_rf_val, lp, lc,
                                             counter_propagating=counter)
            dv = velocity_scale_hint(
                _homogeneous_gamma_scale(gammas, deph, transit), lp, lc,
                counter_propagating=counter)
            bps = _clean_breakpoints(
                [c for hc in hints for c in _breakpoint_stencil_for(hc, dv)],
                n_sigma=params.DOPPLER_N_SIGMA)
            return rho_of_v, bps

        def adaptive(rho_of_v, bps):
            return integrate_maxwell_response(
                rho_of_v, params.VAPOR_CELL_TEMPERATURE_K, M_RB85,
                n_sigma=params.DOPPLER_N_SIGMA,
                epsrel=params.DOPPLER_ADAPTIVE_RTOL,
                epsabs=params.DOPPLER_ADAPTIVE_ATOL,
                breakpoints=bps)

        def oracle(rho_of_v, bps):
            sv = get_velocity_std(params.VAPOR_CELL_TEMPERATURE_K, M_RB85)
            vmax = params.DOPPLER_N_SIGMA * sv
            pts = sorted(set(float(p) for p in (bps or [])
                             if -vmax < float(p) < vmax))
            Z = sp.erf(vmax / (math.sqrt(2) * sv))

            def w(v):
                return math.exp(-0.5 * (v / sv) ** 2) / (
                    math.sqrt(2 * math.pi) * sv)

            kw = dict(limit=800, epsabs=1e-15, epsrel=1e-10)
            if pts:
                kw["points"] = pts
            re, _ = quad(lambda v: (w(v) * rho_of_v(v)).real, -vmax, vmax, **kw)
            im, _ = quad(lambda v: (w(v) * rho_of_v(v)).imag, -vmax, vmax, **kw)
            return complex(re, im) / Z

        data = {}
        for tag, dp_mhz, wrf in (("rf_off", 0.0, 0.0), ("rf_on", 0.0, omega_rf)):
            rov, bps = make_point(dp_mhz, wrf)
            rho_a = adaptive(rov, bps)
            rho_o = oracle(rov, bps)
            data[tag] = (rho_a, rho_o)
        return data

    @pytest.mark.slow
    def test_rf_off_center_matches_independent_oracle(self, oracle_data):
        rho_a, rho_o = oracle_data["rf_off"]
        scale = max(abs(rho_o), 1e-30)
        assert abs(rho_a - rho_o) / scale < 0.01

    @pytest.mark.slow
    def test_rf_on_center_matches_independent_oracle(self, oracle_data):
        rho_a, rho_o = oracle_data["rf_on"]
        scale = max(abs(rho_o), 1e-30)
        assert abs(rho_a - rho_o) / scale < 0.01

    @pytest.mark.slow
    def test_rf_on_center_imag_part_matches(self, oracle_data):
        # Im(rho21) drives absorption; verify it independently of |rho|
        rho_a, rho_o = oracle_data["rf_on"]
        scale = max(abs(rho_o.imag), 1e-30)
        assert abs(rho_a.imag - rho_o.imag) / scale < 0.01

class TestTruncationNormalization:
    """Finite-domain normalization and tail accounting (Phase 4/15)."""

    def test_tail_mass_5sigma_negligible(self):
        assert maxwell_tail_mass(5.0) < 6e-7
        assert maxwell_tail_mass(4.0) > maxwell_tail_mass(5.0)
        assert maxwell_tail_mass(6.0) < maxwell_tail_mass(5.0)

    def test_normalization_divides_by_truncated_z(self):
        # With n_sigma=1 the truncated average of v^2 is measurably below
        # sigma^2; a missing Z normalization would overshoot.  The helper
        # returns COMPLEX by contract (C-002), so compare .real.
        val1 = integrate_maxwell_response(lambda v: v * v, T_ROOM, M_RB85,
                                          n_sigma=1.0).real
        val5 = integrate_maxwell_response(lambda v: v * v, T_ROOM, M_RB85,
                                          n_sigma=5.0).real
        assert val1 < 0.5 * SIGMA_V ** 2  # strongly truncated
        assert 0.98 * SIGMA_V ** 2 < val5 < SIGMA_V ** 2
