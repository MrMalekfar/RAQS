"""DOP-003S — exact direct steady-state solver: contract and parity tests.

The direct solver solves the SAME Lindblad master equation as
``qutip.steadystate`` (same H, same c_ops, same noise seam); it is a
performance seam only. These tests pin the vectorization convention,
the trace-constraint construction, backend dispatch contracts, and
direct-vs-qutip numerical parity on the production sensor model.
"""

import math

import numpy as np
import pytest
import qutip

import rydberg_aoa.simulation.steady_state as ss
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.physics.lindblad.solve_steady_state_direct import (
    _trace_functional,
    _unvec,
    _vec,
    solve_steady_state_direct,
)
from tests.atomic_init_support import ensure_resolved_atomic_physics


# ---------------------------------------------------------------------------
# Fixtures: real production physics (single operating point)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_physics():
    # ARCH-001T: opts into the explicit ARCH-001 seam (idempotent; skipped
    # with a ``BLOCKED: QN-001A`` reason while the coupling mJ contract is
    # open).
    ensure_resolved_atomic_physics()
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates)
    gammas = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    dephasing = [float(x) for x in calculate_dephasing_rates()]
    delta_p, delta_c, delta_rf = acfg.get_cached_detunings()
    return {
        "omega_p": acfg.OMEGA_P,
        "omega_c": acfg.OMEGA_C,
        "omega_rf": acfg.OMEGA_RF_BASE,
        "delta_p": delta_p,
        "delta_c": delta_c,
        "delta_rf": delta_rf,
        "gammas": gammas,
        "dephasing": dephasing,
    }


def _sensor(real_physics, scan_mhz=0.0):
    from rydberg_aoa.physics.hamiltonian.create_sensor_model import (
        create_sensor_model)
    H, c_ops, obs = create_sensor_model(
        real_physics["omega_p"], real_physics["omega_c"],
        real_physics["omega_rf"],
        real_physics["delta_p"] + 2 * math.pi * scan_mhz,
        real_physics["delta_c"], real_physics["delta_rf"],
        real_physics["gammas"], real_physics["dephasing"],
    )
    return H, c_ops, obs


# ---------------------------------------------------------------------------
# 1. Vectorization convention (HARD GATE)
# ---------------------------------------------------------------------------
def test_vectorization_is_column_stacking():
    """QuTiP vec() is column stacking; round-trip must be exact."""
    rng = np.random.default_rng(7)
    n = 5
    m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    v = _vec(m)
    # column k of M occupies contiguous block k of the vector
    for k in range(n):
        np.testing.assert_allclose(v[k * n:(k + 1) * n], m[:, k], rtol=0, atol=0)
    np.testing.assert_allclose(_unvec(v, n), m, rtol=0, atol=0)


def test_vectorization_matches_qutip_operator_to_vector():
    rng = np.random.default_rng(11)
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q = qutip.Qobj(m)
    np.testing.assert_allclose(
        np.asarray(qutip.operator_to_vector(q).full()).ravel(),
        _vec(m), rtol=0, atol=0)


def test_liouvillian_action_matches_explicit_lindblad(real_physics):
    """L @ vec(rho) must equal the explicit Lindblad generator (Hermitian H)."""
    H, c_ops, _ = _sensor(real_physics)
    liou = qutip.liouvillian(H, list(c_ops)).full()
    rng = np.random.default_rng(3)
    rho = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = rho @ rho.conj().T          # Hermitian test state
    rho = rho / np.trace(rho)
    # explicit: -i[H, rho] + sum L rho L^dag - 0.5{L^dag L, rho}
    explicit = -1j * (H.full() @ rho - rho @ H.full())
    for L in c_ops:
        Lm = L.full()
        explicit = explicit + Lm @ rho @ Lm.conj().T \
            - 0.5 * (Lm.conj().T @ Lm @ rho + rho @ Lm.conj().T @ Lm)
    np.testing.assert_allclose(liou @ _vec(rho), _vec(explicit),
                               rtol=1e-12, atol=1e-14)


def test_trace_functional_matches_identity_trace():
    for n in (2, 4, 6):
        e = _trace_functional(n)
        rng = np.random.default_rng(n)
        m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        np.testing.assert_allclose(complex(e @ _vec(m)), np.trace(m),
                                   rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Trace constraint: row fully zeroed before insertion (DOP-003P lesson)
# ---------------------------------------------------------------------------
def test_trace_constraint_row_zeroing_is_load_bearing(real_physics):
    """A trace row written WITHOUT zeroing first must NOT reproduce qutip.

    This pins the DOP-003P prototype bug: overwriting only the diagonal of
    an unmodified Liouvillian row mixes population-rate equations into the
    constraint and corrupts the solution.
    """
    H, c_ops, _ = _sensor(real_physics)
    rho_ref = qutip.steadystate(H, c_ops).full()

    def _solve_with_bad_trace(ldense, n):
        a = ldense.copy()
        b = np.zeros(n * n, dtype=complex)
        for i in range(n):
            a[-1, i * n + i] = 1.0          # diagonal overwritten, row NOT zeroed
        b[-1] = 1.0
        return _unvec(np.linalg.solve(a, b), n)

    liou = qutip.liouvillian(H, list(c_ops)).full()
    rho_bad = _solve_with_bad_trace(liou, H.shape[0])
    # At this operating point the corruption is ~1e-7 (larger at other
    # operating points) — negligible only against nothing: it is 5+ orders
    # of magnitude above the DOP-003S parity target of 1e-12.
    assert np.max(np.abs(rho_bad - rho_ref)) > 1e-8, (
        "un-zeroed trace row should corrupt the solution "
        "(DOP-003P prototype bug regression guard)")

    rho_good = solve_steady_state_direct(H, c_ops).full()
    np.testing.assert_allclose(rho_good, rho_ref, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3-7. Direct-vs-qutip parity + physicality on the production sensor model
# ---------------------------------------------------------------------------
def test_direct_vs_qutip_parity_real_sensor_model(real_physics):
    H, c_ops, obs = _sensor(real_physics)
    rho_q = qutip.steadystate(H, c_ops)
    rho_d, diag = solve_steady_state_direct(H, c_ops, return_diagnostics=True)
    # 5. zero/near-zero Liouvillian residual
    assert diag["liouvillian_residual"] < 1e-12
    # 6. trace = 1
    assert diag["trace_error"] < 1e-12
    assert abs(rho_d.tr() - 1.0) < 1e-12
    # 7. Hermiticity
    assert diag["hermiticity_error"] < 1e-12
    # parity of the full density matrix and of the extracted rho21
    np.testing.assert_allclose(rho_d.full(), rho_q.full(), rtol=0, atol=1e-12)
    rho21_q = complex((obs * rho_q).tr())
    rho21_d = complex((obs * rho_d).tr())
    assert abs(rho21_d - rho21_q) < 1e-12
    # positivity of the Hermitian part
    assert np.min(np.linalg.eigvalsh(
        (rho_d.full() + rho_d.full().conj().T) / 2)) > -1e-10


def test_return_contract_is_qobj(real_physics):
    """Default return is a Qobj (branch-independent contract)."""
    H, c_ops, _ = _sensor(real_physics)
    r = solve_steady_state_direct(H, c_ops)
    assert isinstance(r, qutip.Qobj)
    r2, diag = solve_steady_state_direct(H, c_ops, return_diagnostics=True)
    assert isinstance(r2, qutip.Qobj)
    assert isinstance(diag, dict)
    np.testing.assert_allclose(r.full(), r2.full(), rtol=0, atol=0)


def test_invalid_inputs_fail_loudly(real_physics):
    H, c_ops, _ = _sensor(real_physics)
    with pytest.raises(ValueError):
        solve_steady_state_direct(H, c_ops, trace_weight="bogus")
    bad_H = qutip.Qobj(np.array([[1.0, 1.0], [0.0, 1.0]]))  # non-Hermitian
    with pytest.raises(ValueError):
        ss._steady_state_solve(H, c_ops, "warp9")


def test_randomized_parameter_parity(real_physics):
    """25 randomized cases within physically reasonable ranges.

    Varies detunings, RF Rabi, probe/coupling Rabi and transit rate; does
    NOT alter production constants. Protects parameter sweeps, not just
    the 11 audit operating points.
    """
    from rydberg_aoa.physics.hamiltonian.create_sensor_model import (
        create_sensor_model)
    rng = np.random.default_rng(20260906)
    om_rf = real_physics["omega_rf"]
    om_p, om_c = real_physics["omega_p"], real_physics["omega_c"]
    worst_21 = 0.0
    worst_rho = 0.0
    for _ in range(25):
        dp = rng.uniform(-2 * math.pi * 40, 2 * math.pi * 40)
        dc = rng.uniform(-2 * math.pi * 20, 2 * math.pi * 20)
        drf = rng.uniform(-2 * math.pi * 5, 2 * math.pi * 5)
        wrf = rng.uniform(0.0, om_rf)
        wp = rng.uniform(0.2 * om_p, 2.0 * om_p)
        wc = rng.uniform(0.2 * om_c, 1.5 * om_c)
        tr = rng.uniform(0.0, 5.0)
        H, c_ops, obs = create_sensor_model(
            wp, wc, wrf, dp, dc, drf,
            real_physics["gammas"], real_physics["dephasing"],
            transit_rate_per_us=tr)
        rho_q = qutip.steadystate(H, c_ops)
        rho_d = solve_steady_state_direct(H, c_ops)
        worst_rho = max(worst_rho,
                        float(np.max(np.abs(rho_d.full() - rho_q.full()))))
        worst_21 = max(worst_21,
                       abs(complex((obs * rho_d).tr()) - complex((obs * rho_q).tr())))
    assert worst_rho < 1e-12
    assert worst_21 < 1e-12


# ---------------------------------------------------------------------------
# 8. Adaptive integral parity (expensive — slow marker)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_adaptive_integral_direct_vs_qutip_parity(real_physics):
    """Full adaptive Maxwell ensemble, both backends, identical integrator.

    Backend is the ONLY difference; the adaptive subdivision path must be
    bit-identical (identical evaluation counts) and the ensembles must
    agree to <= 1e-9.
    """
    from rydberg_aoa.config import parameters as params
    from rydberg_aoa.config.laser_rf_config import LASER_PARAMS
    from rydberg_aoa.physics.doppler import (
        get_doppler_shifts, integrate_maxwell_response,
        resonance_velocity_hints, velocity_scale_hint)
    from rydberg_aoa.physics.lindblad.calculate_transit_rate import (
        calculate_transit_rate_per_us)

    transit = calculate_transit_rate_per_us()
    lp = LASER_PARAMS['probe']['wavelength_nm'] * 1e-9
    lc = LASER_PARAMS['coupling']['wavelength_nm'] * 1e-9
    at = real_physics["omega_rf"] / (4 * math.pi)

    def ensemble(backend, scan_mhz):
        dpc = real_physics["delta_p"] + 2 * math.pi * scan_mhz

        def rho_of_v(v_ms):
            ddp, ddc = get_doppler_shifts(
                float(v_ms), lp, lc,
                counter_propagating=params.DOPPLER_COUNTER_PROPAGATING)
            return ss._solve_probe_coherence(
                dpc + ddp, real_physics["delta_c"] + ddc,
                real_physics["omega_p"], real_physics["omega_c"],
                real_physics["omega_rf"], real_physics["delta_rf"],
                real_physics["gammas"], real_physics["dephasing"],
                transit_rate_per_us=transit,
                steady_state_backend=backend)

        hints = resonance_velocity_hints(
            0.0, real_physics["delta_c"], 2 * math.pi * scan_mhz,
            real_physics["delta_rf"], real_physics["omega_rf"], lp, lc,
            counter_propagating=params.DOPPLER_COUNTER_PROPAGATING)
        dvs = velocity_scale_hint(
            ss._homogeneous_gamma_scale(
                real_physics["gammas"], real_physics["dephasing"], transit),
            lp, lc, counter_propagating=params.DOPPLER_COUNTER_PROPAGATING)
        bps = []
        for vc in hints:
            bps.extend(ss._breakpoint_stencil_for(vc, dvs))
        bps = ss._clean_breakpoints(bps, n_sigma=params.DOPPLER_N_SIGMA)
        return integrate_maxwell_response(
            rho_of_v, temperature_k=params.VAPOR_CELL_TEMPERATURE_K,
            atom_mass_kg=params.RB85_MASS_KG,
            n_sigma=params.DOPPLER_N_SIGMA,
            epsrel=params.DOPPLER_ADAPTIVE_RTOL,
            epsabs=params.DOPPLER_ADAPTIVE_ATOL,
            breakpoints=bps)

    for scan in (0.0, -at):     # EIT center and AT-minus, RF on
        rho_q = ensemble("qutip", scan)
        rho_d = ensemble("direct", scan)
        assert abs(rho_d - rho_q) <= 1e-9


# ---------------------------------------------------------------------------
# 9-10. Production dispatch: cold path stays qutip, adaptive path uses direct
# ---------------------------------------------------------------------------
def test_solve_probe_coherence_returns_complex_by_default(real_physics):
    """Phase 12: return contract unchanged — complex rho21, float T."""
    val = ss._solve_probe_coherence(
        real_physics["delta_p"], real_physics["delta_c"],
        real_physics["omega_p"], real_physics["omega_c"], 0.0,
        real_physics["delta_rf"], real_physics["gammas"],
        real_physics["dephasing"])
    assert isinstance(val, complex)
    val_q = ss._solve_probe_coherence(
        real_physics["delta_p"], real_physics["delta_c"],
        real_physics["omega_p"], real_physics["omega_c"], 0.0,
        real_physics["delta_rf"], real_physics["gammas"],
        real_physics["dephasing"], steady_state_backend="qutip")
    assert val == val_q            # default IS the qutip reference backend
    t = ss._solve_probe_transmission(
        real_physics["delta_p"], real_physics["delta_c"],
        real_physics["omega_p"], real_physics["omega_c"], 0.0,
        real_physics["delta_rf"], real_physics["gammas"],
        real_physics["dephasing"])
    assert isinstance(t, float) and 0.0 < t <= 1.0 + 1e-9


def test_adaptive_hot_path_dispatches_direct_backend(monkeypatch, real_physics):
    """The production adaptive Doppler path must route to the direct solver.

    Exercises the REAL simulate_spectrum body (N_SCAN=1) with a spy around
    the direct solver and a (never-called) guard around qutip.steadystate.
    """
    calls = {"direct": 0}

    real_direct = ss.solve_steady_state_direct

    def spy_direct(H, c_ops, *a, **kw):
        calls["direct"] += 1
        return real_direct(H, c_ops, *a, **kw)

    def forbidden_qutip(H, c_ops, *a, **kw):
        raise AssertionError("adaptive hot path must not call qutip.steadystate")

    monkeypatch.setattr(ss, "solve_steady_state_direct", spy_direct)
    monkeypatch.setattr(ss, "steadystate", forbidden_qutip)
    monkeypatch.setattr(ss.params, "N_SCAN", 1)
    monkeypatch.setattr(ss.params, "SCAN_RANGE_MHZ", 0.0)
    scan, t_num, t_ana = ss.simulate_spectrum(
        0.0, np.array([0.0]),
        (real_physics["omega_p"], real_physics["omega_c"], 0.0),
        (real_physics["delta_p"], real_physics["delta_c"], real_physics["delta_rf"]),
        real_physics["gammas"], real_physics["dephasing"], absorption_const=None)
    assert calls["direct"] > 0
    assert len(scan) == 1 and np.isfinite(t_num[0])


def test_explicit_qutip_backend_overrides_adaptive_default(monkeypatch, real_physics):
    """An explicit steady_state_backend='qutip' must override 'direct'."""
    calls = {"qutip": 0}
    real_ss = ss.steadystate

    def spy_qutip(H, c_ops, *a, **kw):
        calls["qutip"] += 1
        return real_ss(H, c_ops, *a, **kw)

    monkeypatch.setattr(ss, "steadystate", spy_qutip)
    monkeypatch.setattr(ss.params, "N_SCAN", 1)
    monkeypatch.setattr(ss.params, "SCAN_RANGE_MHZ", 0.0)
    ss.simulate_spectrum(
        0.0, np.array([0.0]),
        (real_physics["omega_p"], real_physics["omega_c"], 0.0),
        (real_physics["delta_p"], real_physics["delta_c"], real_physics["delta_rf"]),
        real_physics["gammas"], real_physics["dephasing"], absorption_const=None,
        steady_state_backend="qutip")
    assert calls["qutip"] > 0
