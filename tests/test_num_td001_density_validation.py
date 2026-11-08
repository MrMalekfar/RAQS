"""NUM-TD-001 — fail-loud certification of temporal density matrices.

A time-dependent solver *returning* a trajectory (or expectation values) is
not proof that each sampled state is a valid density matrix.  These tests
pin the temporal gate ``rydberg_aoa.utils.density_validation.validate_
density_matrix`` and its integration into the worker's e_ops callback:

* valid pure/mixed states are certified with honest metrics;
* non-STATIONARY states are accepted (unlike the NUM-SS steady-state gate);
* every structural/physical violation is rejected with
  ``PhysicsCalculationError`` and the input is never repaired;
* the worker certifies each state BEFORE ``qt.expect`` consumes it, and a
  solver returning finite expectations without invoking the callback cannot
  bypass certification (coverage gate);
* every certified state is delivered at its INTENDED output-grid time: a
  duplicated, reordered, off-grid, non-finite or excess callback timestamp
  is rejected before ``qt.expect`` consumes it (grid-identity gate).

Physics is SYNTHETIC (small four-level model); nothing here claims EIT,
convergence or laboratory validation.
"""

import numpy as np
import pytest
import qutip

from rydberg_aoa.exceptions import PhysicsCalculationError
from rydberg_aoa.simulation import time_domain as td
from rydberg_aoa.utils.density_validation import validate_density_matrix
from rydberg_aoa.utils.validation import validate_steady_state
from tests.test_f05_worker001_error_propagation import (
    _invalid_state,
    _mock_mesolve_callback_then_return,
    _mock_mesolve_driving_callback,
    _nonstationary,
    _synthetic_model,
    _valid_state,
    _worker_args,
    _worker_kwargs,
)

CTX = "num-td-001-test"


# ---------------------------------------------------------------------------
# 1. Validator — acceptance (valid states certify with honest metrics)
# ---------------------------------------------------------------------------
def test_valid_pure_state_certified_with_honest_metrics():
    metrics = validate_density_matrix(
        qutip.ket2dm(qutip.basis(4, 0)), context=CTX, expected_dim=4)
    assert metrics["dimension"] == 4
    assert metrics["trace_error"] < 1e-12
    assert metrics["hermiticity_error"] < 1e-12
    assert metrics["min_eigenvalue"] >= -1e-12


def test_valid_mixed_state_certified():
    mixed = np.diag([0.4, 0.3, 0.2, 0.1]).astype(complex)
    metrics = validate_density_matrix(mixed, context=CTX, expected_dim=4)
    assert metrics["trace_error"] < 1e-12
    assert metrics["min_eigenvalue"] >= 0.0


def test_nonstationary_state_accepted_by_temporal_gate():
    """An evolving (non-stationary) state is legitimate here.

    The SAME matrix is rejected by the NUM-SS steady-state gate, proving the
    temporal gate is its strict sibling without the stationarity demand.
    """
    rho = qutip.Qobj(_nonstationary())
    metrics = validate_density_matrix(rho, context=CTX, expected_dim=4)
    assert metrics["trace_error"] < 1e-8

    H, c_ops, _ = _synthetic_model()
    liouvillian = qutip.liouvillian(H, c_ops)
    with pytest.raises(PhysicsCalculationError):
        validate_steady_state(rho, liouvillian, context=CTX)


# ---------------------------------------------------------------------------
# 2. Validator — structural rejections
# ---------------------------------------------------------------------------
def test_none_rejected():
    with pytest.raises(PhysicsCalculationError, match="received None"):
        validate_density_matrix(None, context=CTX)


def test_non_square_rejected():
    with pytest.raises(PhysicsCalculationError, match="square"):
        validate_density_matrix(np.zeros((2, 3)), context=CTX)


def test_zero_size_rejected():
    with pytest.raises(PhysicsCalculationError, match="square"):
        validate_density_matrix(np.zeros((0, 0)), context=CTX)


def test_scalar_rejected():
    with pytest.raises(PhysicsCalculationError, match="2-D"):
        validate_density_matrix(np.array(1.0), context=CTX)


def test_boolean_matrix_rejected_before_conversion():
    bad = np.array([[True, False], [False, True]])
    with pytest.raises(PhysicsCalculationError, match="non-numeric/boolean"):
        validate_density_matrix(bad, context=CTX)


def test_string_matrix_rejected():
    bad = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(PhysicsCalculationError, match="non-numeric"):
        validate_density_matrix(bad, context=CTX)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_entries_rejected(bad_value):
    bad = np.eye(4, dtype=complex)
    bad[2, 2] = bad_value
    with pytest.raises(PhysicsCalculationError, match="NaN/Inf"):
        validate_density_matrix(bad, context=CTX)


def test_expected_dim_mismatch_rejected():
    with pytest.raises(PhysicsCalculationError, match="expected"):
        validate_density_matrix(np.eye(4), context=CTX, expected_dim=3)


# ---------------------------------------------------------------------------
# 3. Validator — physical rejections (trace / Hermiticity / positivity)
# ---------------------------------------------------------------------------
def test_trace_violation_rejected():
    bad = np.diag([0.5, 0.5, 0.5, 0.0]).astype(complex)   # Tr = 1.5
    with pytest.raises(PhysicsCalculationError, match="Tr\\(rho\\)-1"):
        validate_density_matrix(bad, context=CTX, expected_dim=4)


def test_hermiticity_violation_rejected():
    bad = np.array([[1.0, 0.5], [0.25, 0.0]], dtype=complex)
    with pytest.raises(PhysicsCalculationError, match="Hermiticity"):
        validate_density_matrix(bad, context=CTX)


def test_positivity_violation_rejected():
    bad = qutip.Qobj(np.diag([1.2, -0.2, 0.0, 0.0]).astype(complex))
    with pytest.raises(PhysicsCalculationError, match="min eigenvalue"):
        validate_density_matrix(bad, context=CTX, expected_dim=4)


# ---------------------------------------------------------------------------
# 4. Validator — tolerance contract (a mis-set tolerance must fail loud)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_tol", [
    True, "1e-9", [1e-9], np.nan, np.inf, -1.0,
])
def test_invalid_tolerance_rejected(bad_tol):
    with pytest.raises(PhysicsCalculationError, match="tolerance"):
        validate_density_matrix(
            np.eye(4), context=CTX, trace_tolerance=bad_tol)


def test_zero_tolerance_is_strictest_and_legitimate():
    metrics = validate_density_matrix(
        qutip.ket2dm(qutip.basis(4, 0)), context=CTX, expected_dim=4,
        trace_tolerance=0.0, hermiticity_tolerance=0.0,
        positivity_tolerance=0.0)
    assert metrics["trace_error"] == 0.0


# ---------------------------------------------------------------------------
# 5. Validator — non-mutation and input-representation equivalence
# ---------------------------------------------------------------------------
def test_qobj_input_is_not_mutated():
    rho = qutip.ket2dm(qutip.basis(4, 0))
    before = rho.full().copy()
    validate_density_matrix(rho, context=CTX, expected_dim=4)
    assert isinstance(rho, qutip.Qobj)
    assert np.array_equal(rho.full(), before)


def test_array_input_is_not_mutated():
    arr = np.diag([0.5, 0.5, 0.0, 0.0]).astype(complex)
    before = arr.copy()
    validate_density_matrix(arr, context=CTX, expected_dim=4)
    assert np.array_equal(arr, before)


def test_qobj_and_array_inputs_agree():
    """The same matrix must certify identically through both interfaces."""
    from_qobj = validate_density_matrix(
        _valid_state(0.0), context=CTX, expected_dim=4)
    from_array = validate_density_matrix(
        _valid_state(0.0).full(), context=CTX, expected_dim=4)
    assert from_qobj == from_array


def test_transpose_orientation_certified_symmetrically():
    """rho and its transpose (conjugate orientation) certify equivalently."""
    rho = np.array([[0.6, 0.2 - 0.1j], [0.2 + 0.1j, 0.4]], dtype=complex)
    padded = np.zeros((4, 4), dtype=complex)
    padded[:2, :2] = rho
    transposed = padded.T
    m1 = validate_density_matrix(padded, context=CTX, expected_dim=4)
    m2 = validate_density_matrix(transposed, context=CTX, expected_dim=4)
    assert m1["trace_error"] == pytest.approx(m2["trace_error"])
    assert m1["min_eigenvalue"] == pytest.approx(m2["min_eigenvalue"])


# ---------------------------------------------------------------------------
# 6. Worker integration — certification happens BEFORE qt.expect
# ---------------------------------------------------------------------------
def test_certification_precedes_qt_expect_and_aborts_worker(monkeypatch):
    """An invalid temporal state must abort BEFORE it becomes an observable.

    A spy on ``qutip.expect`` proves the observable was never computed for
    any sampled time: certification ran first and failed loud.
    """
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve",
                        _mock_mesolve_driving_callback(_invalid_state))
    expect_calls = []
    real_expect = qutip.expect

    def _spy(oper, state):
        expect_calls.append(state)
        return real_expect(oper, state)

    monkeypatch.setattr(td.qt, "expect", _spy)
    with pytest.raises(PhysicsCalculationError, match="NUM-TD-001"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    assert expect_calls == []


def test_finite_fabricated_expectations_cannot_bypass_certification(
        monkeypatch):
    """A solver returning finite expects WITHOUT driving the callback fails.

    The values are perfectly plausible (all zeros); only the coverage gate
    reveals that no temporal state was ever certified.
    """
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(
        td, "mesolve",
        lambda *a, **k: type("R", (), {"expect": [np.zeros(3)]})())
    with pytest.raises(PhysicsCalculationError) as excinfo:
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    message = str(excinfo.value)
    assert "NUM-TD-001" in message
    assert "0 of 3" in message


def test_partially_driven_grid_rejected(monkeypatch):
    """Certifying only a subset of the output grid is NOT certification."""
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))

    def _mock(H_t, rho0, tlist, c_ops_, e_ops, options=None, **kwargs):
        op = e_ops[0]
        op(float(tlist[0]), _valid_state(tlist[0]))   # certify ONLY t0
        return type("R", (), {"expect": [np.zeros(len(tlist))]})()

    monkeypatch.setattr(td, "mesolve", _mock)
    with pytest.raises(PhysicsCalculationError, match="1 of 3"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())


def test_valid_temporal_states_reach_extraction_uncorrupted(monkeypatch):
    """Certification must not alter the observable: the extracted trace
    equals direct ``qt.expect`` of the same certified states."""
    H, c_ops, obs = _synthetic_model()

    def _coherent(_t):
        plus = (qutip.basis(4, 1) + qutip.basis(4, 2)).unit()
        return qutip.ket2dm(plus)

    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(
        td, "mesolve", _mock_mesolve_driving_callback(_coherent))
    monkeypatch.setattr(
        td, "extract_if_component",
        lambda *a, **k: (0.5 + 0.25j, np.array([0.0]), np.array([1.0])))
    result = td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    assert len(result) == 5
    amp, _xf, _yf, idx, signal_trace = result
    control = np.imag(np.array(
        [qutip.expect(obs, _coherent(t)) for t in (0.0, 1.0, 2.0)]))
    assert np.allclose(signal_trace, control)
    assert amp == pytest.approx(0.5 + 0.25j)
    assert idx == 0


# ---------------------------------------------------------------------------
# 6b. Temporal grid identity — every certified state on its intended tlist
#     point (duplicated / reordered / off-grid / non-finite / excess times)
# ---------------------------------------------------------------------------
def _drive_at_times(times, state_factory=_valid_state):
    """Mock mesolve driving the worker's e_ops callback at ARBITRARY times.

    The times are deliberately decoupled from ``tlist`` so a regression test
    can emulate a solver that re-delivers t0, reorders the grid, lands
    off-grid, passes NaN/Inf or calls back after the grid is exhausted.
    ``state_factory`` alone decides whether each sampled state is valid, so
    any rejection below can only come from the grid-identity gate.
    """
    def _mock(H_t, rho0, tlist, c_ops, e_ops, options=None, **kwargs):
        op = e_ops[0] if isinstance(e_ops, (list, tuple)) else e_ops
        for t in times:
            op(float(t), state_factory(t))
        return type("R", (), {"expect": [np.zeros(len(tlist))]})()

    return _mock


def _install_spies(monkeypatch):
    """Spy on ``qt.expect`` and ``extract_if_component`` to prove whether a
    rejected timestamp was ever consumed downstream (expect counts states
    actually handed to the observable; extract counts IF-extraction calls).
    """
    expect_calls = []
    real_expect = qutip.expect

    def _expect_spy(oper, state):
        expect_calls.append(state)
        return real_expect(oper, state)

    monkeypatch.setattr(td.qt, "expect", _expect_spy)
    extract_calls = []
    monkeypatch.setattr(
        td, "extract_if_component",
        lambda *a, **k: extract_calls.append(a)
        or (0j, np.array([0.0]), np.array([1.0])))
    return expect_calls, extract_calls


def test_repeated_t0_with_full_count_rejected(monkeypatch):
    """Callbacks [t0, t0, t2] match the grid LENGTH but not its IDENTITY.

    A count-only gate would certify 3 of 3 and pass; the grid-identity gate
    must reject at output index 1 (expected t=1.0us, received t=0.0us) BEFORE
    the duplicated-t0 state reaches ``qt.expect`` or IF extraction.
    """
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve", _drive_at_times([0.0, 0.0, 2.0]))
    expect_calls, extract_calls = _install_spies(monkeypatch)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    message = str(excinfo.value)
    assert "NUM-TD-001" in message
    assert "output index 1" in message
    assert "expected t=1.0us" in message
    assert "received t=0.0us" in message
    assert len(expect_calls) == 1       # only the FIRST t0 was consumed
    assert extract_calls == []          # IF extraction never reached


def test_reordered_timestamps_rejected(monkeypatch):
    """Callbacks [t2, t1, t0] certify the wrong state at every index.

    The trace handed to IF extraction would be time-reversed; the gate must
    reject at output index 0 (expected t=0.0us, received t=2.0us) before any
    state reaches ``qt.expect``.
    """
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve", _drive_at_times([2.0, 1.0, 0.0]))
    expect_calls, extract_calls = _install_spies(monkeypatch)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    message = str(excinfo.value)
    assert "NUM-TD-001" in message
    assert "output index 0" in message
    assert "expected t=0.0us" in message
    assert "received t=2.0us" in message
    assert expect_calls == []
    assert extract_calls == []


def test_off_grid_timestamp_rejected(monkeypatch):
    """t=1.5us lies strictly between t1 and t2: a solver landing off the
    output grid would shift phase sampling; rejected at output index 1."""
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve", _drive_at_times([0.0, 1.5, 2.0]))
    expect_calls, extract_calls = _install_spies(monkeypatch)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    message = str(excinfo.value)
    assert "NUM-TD-001" in message
    assert "output index 1" in message
    assert "expected t=1.0us" in message
    assert "received t=1.5us" in message
    assert len(expect_calls) == 1
    assert extract_calls == []


@pytest.mark.parametrize("bad_t", [float("nan"), float("inf")])
def test_nonfinite_timestamp_rejected(monkeypatch, bad_t):
    """NaN or Inf callback times cannot identify an output-grid point and
    would corrupt every downstream phase computation; rejected at the gate.
    """
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve",
                        _drive_at_times([0.0, bad_t, 2.0]))
    expect_calls, extract_calls = _install_spies(monkeypatch)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    message = str(excinfo.value)
    assert "NUM-TD-001" in message
    assert "output index 1" in message
    assert "finite" in message
    assert len(expect_calls) == 1       # only the finite t0 was consumed
    assert extract_calls == []


def test_excess_callback_after_full_grid_rejected(monkeypatch):
    """A fourth callback AFTER the full 3-point grid was certified is excess:
    the coverage count would still read 3 of 3, but the extra timestamp proves
    the solver no longer tracks the dispatched grid, so the run must abort
    before IF extraction.
    """
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve",
                        _drive_at_times([0.0, 1.0, 2.0, 1.0]))
    expect_calls, extract_calls = _install_spies(monkeypatch)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    message = str(excinfo.value)
    assert "NUM-TD-001" in message
    assert "EXCESS" in message
    assert "output index 3" in message
    assert len(expect_calls) == 3       # grid points certified; excess not
    assert extract_calls == []          # ...and extraction never ran


def test_incomplete_grid_after_valid_prefix_rejected(monkeypatch):
    """Callbacks [t0, t1] certify a valid PREFIX of the grid; stopping early
    leaves t2 uncertified and must fail the final coverage gate (2 of 3)."""
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve", _drive_at_times([0.0, 1.0]))
    expect_calls, extract_calls = _install_spies(monkeypatch)
    with pytest.raises(PhysicsCalculationError, match="2 of 3"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    assert len(expect_calls) == 2
    assert extract_calls == []


def test_valid_grid_passes_identity_gate_uncorrupted(monkeypatch):
    """The honest path must be untouched: driving the callback at exactly
    [t0, t1, t2] certifies 3 of 3, reaches IF extraction once and returns
    the five-element success contract.
    """
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "mesolve", _drive_at_times([0.0, 1.0, 2.0]))
    expect_calls, extract_calls = _install_spies(monkeypatch)
    result = td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    assert len(result) == 5
    assert len(expect_calls) == 3
    assert len(extract_calls) == 1


def test_worker_options_force_raw_solver_output(monkeypatch):
    """``normalize_output=False`` is mandatory: certification must see the
    RAW solver trace, not a silently renormalized one (QuTiP default True)."""
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    captured = {}
    inner = _mock_mesolve_driving_callback(_valid_state)

    def _capture(*a, **k):
        captured.update(k)
        return inner(*a, **k)

    monkeypatch.setattr(td, "mesolve", _capture)
    monkeypatch.setattr(
        td, "extract_if_component",
        lambda *a, **k: (0j, np.array([0.0]), np.array([1.0])))
    td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    options = captured["options"]
    if isinstance(options, dict):        # QuTiP 5.2.1 returns a plain dict
        assert options["normalize_output"] is False
        assert options["method"] == "bdf"
        assert options["nsteps"] == 5000
    else:
        assert options.normalize_output is False


def test_real_mesolve_certifies_full_grid_and_returns_five_tuple():
    """Bounded real BDF run: certification covers every output time.

    Success here proves the coverage gate holds for the REAL solver path
    (mocks cannot produce this evidence).
    """
    tlist = np.linspace(0.0, 60.0, 600)
    args = (0, 1.0 + 0.5j, tlist, 1.0,
            5.0, 3.0, 2.0, 0.3, -0.2, 0.1,
            [1.0, 1.0, 1.0], [0.2, 0.2, 0.2])
    result = td.run_single_heterodyne_worker(
        *args, noise_model=None, transit_rate_per_us=0.0, rf_orientation=None)
    assert isinstance(result, tuple) and len(result) == 5
    amp, _xf, _yf, idx, trace = result
    assert idx == 0
    assert np.isfinite(amp)
    assert trace.shape == (600,)
    assert np.all(np.isfinite(trace))
