"""C-002 regression tests: Doppler ensemble aggregation level.

The production Doppler path must ensemble velocity classes at the
coherence/susceptibility level and apply Beer-Lambert propagation ONCE:

    rho_ens = sum_i w_i * rho_i        (aggregation)
    T       = transmission(rho_ens)    (ONE propagation step)

and must NEVER average per-class transmissions. Tests 1-3/5 drive the real
``simulate_spectrum`` aggregation loop with synthetic class coherences
(the solver oracle is mocked; the ensemble contract is what is tested).
Test 6/7 execute the real physics end-to-end.
"""
import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.physics.doppler import generate_velocity_classes
from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 import (
    calculate_transmission_from_rho21)
from rydberg_aoa.simulation import steady_state as ss
from tests.atomic_init_support import ensure_resolved_atomic_physics


@pytest.fixture(scope="module")
def acfg():
    """Resolved atomic runtime for this module's production-level tests.

    ARCH-001T: opts into the explicit ARCH-001 seam (idempotent; skipped
    with a ``BLOCKED: QN-001A`` reason while the coupling mJ contract is
    open).
    """
    ensure_resolved_atomic_physics()
    from rydberg_aoa.config import atomic_config as ac
    return ac


def _run_aggregation(monkeypatch, velocities, weights, rhos):
    """Drive the production ``simulate_spectrum`` aggregation loop with
    synthetic velocity classes and synthetic per-class coherences.

    The legacy fixed-class quadrature ("gauss_hermite") is pinned so the
    synthetic classes drive the aggregation loop. The adaptive-default
    ensemble path (DOP-002I) is validated separately against an independent
    quad oracle in tests/test_adaptive_maxwell.py."""
    monkeypatch.setattr(params, "DOPPLER_ENABLED", True)
    monkeypatch.setattr(params, "DOPPLER_METHOD", "gauss_hermite")
    monkeypatch.setattr(params, "N_SCAN", 1)
    monkeypatch.setattr(params, "SCAN_RANGE_MHZ", 1.0)
    monkeypatch.setattr(
        ss, "generate_velocity_classes",
        lambda **kw: (np.asarray(velocities, float), np.asarray(weights, float)))
    seq = iter(rhos)
    monkeypatch.setattr(ss, "_solve_probe_coherence",
                        lambda *a, **kw: complex(next(seq)))
    _, t_num, _ = ss.simulate_spectrum(
        0.0, np.array([0.0]),
        (1.0, 1.0, 0.0),          # p_omegas (values irrelevant to mocked solver)
        (0.0, 0.0, 0.0),          # p_deltas
        [1.0, 0.0, 0.0],          # p_gammas
        [0.0, 0.0, 0.0],          # p_dephasing
        absorption_const=None)
    return t_num[0]


# ---------------------------------------------------------------------------
# TEST 1 — ensemble linearity oracle
# ---------------------------------------------------------------------------
def test_aggregation_matches_independent_weighted_coherence(
        atomic_physics, monkeypatch):
    v = [-1.0, 1.0]
    w = [0.25, 0.75]
    rhos = [0.02 - 0.04j, -0.01 - 0.02j]
    rho_ens_expected = sum(wi * r for wi, r in zip(w, rhos))
    t_expected = calculate_transmission_from_rho21(rho_ens_expected)
    t_prod = _run_aggregation(monkeypatch, v, w, rhos)
    assert t_prod == pytest.approx(t_expected, rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# TEST 2 — prohibit transmission averaging
# ---------------------------------------------------------------------------
def test_aggregation_is_not_intensity_averaging(atomic_physics, monkeypatch):
    v = [-1.0, 1.0]
    w = [0.5, 0.5]
    rhos = [-0.05j, -1e-9j]   # one opaque class, one transparent (absorption = Im)
    t_ensemble = calculate_transmission_from_rho21(
        sum(wi * r for wi, r in zip(w, rhos)))
    t_intensity = sum(wi * calculate_transmission_from_rho21(r)
                      for wi, r in zip(w, rhos))
    assert abs(t_ensemble - t_intensity) > 0.1, "oracle classes must discriminate"
    t_prod = _run_aggregation(monkeypatch, v, w, rhos)
    assert t_prod == pytest.approx(t_ensemble, rel=1e-12, abs=1e-15)
    assert abs(t_prod - t_intensity) > 0.1, (
        "production aggregation reproduced intensity averaging — C-002 regression")


# ---------------------------------------------------------------------------
# TEST 3 — zero-Doppler limit
# ---------------------------------------------------------------------------
def test_single_class_equals_direct_transmission(atomic_physics, monkeypatch):
    rho = -0.03 + 0.01j
    t_prod = _run_aggregation(monkeypatch, [0.0], [1.0], [rho])
    assert t_prod == pytest.approx(calculate_transmission_from_rho21(rho),
                                   rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# TEST 4 — weight normalization (quadrature untouched)
# ---------------------------------------------------------------------------
def test_velocity_weights_normalized():
    # Weight normalization is a property of the legacy fixed-class
    # quadratures; the adaptive method (DOP-002I default) does not use
    # discrete classes and is validated separately in test_adaptive_maxwell.py
    for method in ("gauss_hermite", "uniform"):
        v, w = generate_velocity_classes(
            params.DOPPLER_N_CLASSES, params.VAPOR_CELL_TEMPERATURE_K,
            params.RB85_MASS_KG, method, params.DOPPLER_N_SIGMA)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TEST 5 — aggregation introduces no asymmetry
# ---------------------------------------------------------------------------
def test_aggregation_is_order_and_symmetry_safe(atomic_physics, monkeypatch):
    rho = -0.02 - 0.03j
    t_ab = _run_aggregation(monkeypatch, [-1.0, 1.0], [0.3, 0.7], [rho, rho])
    t_ba = _run_aggregation(monkeypatch, [1.0, -1.0], [0.7, 0.3], [rho, rho])
    assert t_ab == pytest.approx(t_ba, rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# TEST 6 — real production center point vs independent ensemble oracle
# ---------------------------------------------------------------------------
def test_production_center_matches_independent_ensemble(acfg, monkeypatch):
    """At the C-001 operating point the production Doppler path must equal
    an independently constructed susceptibility-level weighted sum, and
    must NOT reproduce the old wrong intensity-averaged T~0.57."""
    from qutip import steadystate
    from rydberg_aoa.config.laser_rf_config import LASER_PARAMS
    from rydberg_aoa.physics.doppler import get_doppler_shifts
    from rydberg_aoa.physics.hamiltonian.create_sensor_model import (
        create_sensor_model)
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates)

    gam = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    deph = calculate_dephasing_rates()
    from rydberg_aoa.physics.lindblad import calculate_transit_rate_per_us
    transit = calculate_transit_rate_per_us()
    dp0, dc0, drf0 = acfg.get_cached_detunings()
    # Pin the legacy fixed-class quadrature on BOTH sides (independent oracle
    # and production call) so they solve identical ensemble sums; the
    # adaptive-default production path is oracle-validated separately in
    # tests/test_adaptive_maxwell.py::TestRealPhysicsOracle (DOP-002I).
    # BEAM-002: the oracle must include the resolved transit reset rate —
    # the production path threads it (DPH-002B) and the corrected radii made
    # its influence exceed the old 1e-9 tolerance.
    monkeypatch.setattr(params, "DOPPLER_METHOD", "gauss_hermite")
    vel, w = generate_velocity_classes(
        params.DOPPLER_N_CLASSES, params.VAPOR_CELL_TEMPERATURE_K,
        params.RB85_MASS_KG, params.DOPPLER_METHOD, params.DOPPLER_N_SIGMA)
    lp = LASER_PARAMS['probe']['wavelength_nm'] * 1e-9
    lc = LASER_PARAMS['coupling']['wavelength_nm'] * 1e-9

    rho_ens = 0.0 + 0.0j
    for v_ms, w_i in zip(vel, w):
        s_p, s_c = get_doppler_shifts(float(v_ms), lp, lc,
                                      params.DOPPLER_COUNTER_PROPAGATING)
        H, c_ops, obs = create_sensor_model(
            acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE,
            dp0 + s_p, dc0 + s_c, drf0, gam, deph,
            transit_rate_per_us=transit)
        rho_ss = steadystate(H, c_ops)
        rho_ens += w_i * complex((obs * rho_ss).tr())
    t_indep = calculate_transmission_from_rho21(rho_ens)

    monkeypatch.setattr(params, "N_SCAN", 1)
    monkeypatch.setattr(params, "SCAN_RANGE_MHZ", 0.0)
    _, t_num, _ = ss.simulate_spectrum(
        acfg.OMEGA_RF_BASE, np.array([0.0]),
        (acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE),
        (dp0, dc0, drf0), gam, deph, absorption_const=None)
    assert t_num[0] == pytest.approx(t_indep, rel=1e-6)
    # REF-002L: with the 75 mm reference cell the operating point is deeply
    # opaque (T ~ 3e-5), so the Beer-Lambert exponent amplifies the DOP-003S
    # |drho21| <= 1e-12 ABSOLUTE solver parity into a relative T difference of
    # order 1e-7; rel=1e-6 still requires 6-digit agreement between the two
    # independent solvers.
    assert abs(t_num[0] - 0.5704) > 0.1, (
        "production still reproduces the old intensity-averaged T~0.57")


# ---------------------------------------------------------------------------
# TEST 7 — RF-off / RF-on distinction survives the ensemble
# ---------------------------------------------------------------------------
def test_rf_on_off_distinct_after_ensemble(acfg, monkeypatch):
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates)
    gam = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    deph = calculate_dephasing_rates()
    dp0, dc0, drf0 = acfg.get_cached_detunings()
    monkeypatch.setattr(params, "N_SCAN", 1)
    monkeypatch.setattr(params, "SCAN_RANGE_MHZ", 0.0)
    res = {}
    for tag, orf in [("off", 0.0), ("on", acfg.OMEGA_RF_BASE)]:
        _, t_num, _ = ss.simulate_spectrum(
            orf, np.array([0.0]),
            (acfg.OMEGA_P, acfg.OMEGA_C, orf),
            (dp0, dc0, drf0), gam, deph, absorption_const=None)
        res[tag] = t_num[0]
    assert np.isfinite(res["off"]) and np.isfinite(res["on"])
    assert abs(res["on"] - res["off"]) > 1e-12, (
        "RF-on and RF-off ensembles are numerically identical")

