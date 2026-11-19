"""Affine-detuning kernel of the adaptive Doppler integral versus the reference path.

The kernel (``DetuningAffineSteadyState``) is a performance path only: it
must reproduce the reference ``_solve_probe_coherence`` / direct-solver path
and must hand the whole spectrum back to that path on any disagreement.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.physics.hamiltonian import create_sensor_model
from rydberg_aoa.physics.lindblad.solve_steady_state_direct import (
    DetuningAffineSteadyState,
    solve_steady_state_direct,
)
from rydberg_aoa.simulation import steady_state as ss
from tests.test_lab007_profile_integration import isolated  # noqa: F401 - explicit full state isolation

ROOT = Path(__file__).resolve().parents[1]

# Explicit numerical operating point (rad/us and 1/us); no atomic lookup.
OMEGAS = (15.7585, 57.3175, 48.8196)
DELTA_RF = -0.67688
GAMMAS = [38.113, 0.034588, 4.79e-5]
DEPHASING = [0.02, 0.03, 0.03]
TRANSIT = 0.5

# Unwrapped originals: a test runs the profile twice and must not nest spies.
_REFERENCE_SOLVE = ss._solve_probe_coherence
_KERNEL_COHERENCE = DetuningAffineSteadyState.coherence


def _model(delta_p, delta_c, omega_rf=OMEGAS[2]):
    return create_sensor_model(OMEGAS[0], OMEGAS[1], omega_rf, delta_p, delta_c, DELTA_RF,
                               GAMMAS, DEPHASING, transit_rate_per_us=TRANSIT)


@pytest.mark.parametrize('omega_rf', [0.0, OMEGAS[2]])
def test_kernel_matches_direct_solver_over_the_doppler_detuning_range(omega_rf):
    H0, c_ops0, observable = _model(0.0, 0.0, omega_rf)
    kernel = DetuningAffineSteadyState(H0, c_ops0, observable)
    rng = np.random.default_rng(3)
    velocity = rng.normal(0.0, 170.0, 200)                       # m/s, room-temperature Rb
    delta_p = 0.7 - 2 * np.pi / 780.24e-9 * 1e-6 * velocity      # rad/us
    delta_c = -0.3 + 2 * np.pi / 483.24e-9 * 1e-6 * velocity
    worst = 0.0
    for dp, dc in zip(delta_p, delta_c):
        H, c_ops, obs = _model(dp, dc, omega_rf)
        reference = qt.expect(obs, solve_steady_state_direct(H, c_ops))
        worst = max(worst, abs(kernel.coherence(dp, dc) - reference))
    assert worst <= 1e-13


def test_kernel_rejects_non_ladder_inputs():
    H0, c_ops0, observable = _model(0.0, 0.0)
    with pytest.raises(ValueError):
        DetuningAffineSteadyState(H0 + 1j * qt.basis(4, 0) * qt.basis(4, 1).dag(), c_ops0, observable)


def _run_lab007(monkeypatch, *, kernel_enabled, corrupt_kernel=False):
    from rydberg_aoa.simulation.profile_execution import execute_profile
    monkeypatch.setattr(ss, '_FAST_KERNEL_ENABLED', kernel_enabled)
    counts = {'reference': 0, 'kernel': 0}
    reference, kernel_coherence = _REFERENCE_SOLVE, _KERNEL_COHERENCE

    def spy_reference(*args, **kwargs):
        counts['reference'] += 1
        return reference(*args, **kwargs)

    def spy_kernel(self, delta_p, delta_c):
        counts['kernel'] += 1
        value = kernel_coherence(self, delta_p, delta_c)
        return value + 1e-6 if corrupt_kernel else value

    monkeypatch.setattr(ss, '_solve_probe_coherence', spy_reference)
    monkeypatch.setattr(DetuningAffineSteadyState, 'coherence', spy_kernel)
    from rydberg_aoa.config import atomic_config
    atomic_config.reset_atomic_physics()      # explicit reset between the two runs of one test
    data = json.loads((ROOT / 'examples/lab007/prescribed_full.json').read_text())
    data['spectrum']['probe_offsets_mhz'] = [-3.0, -0.5, 0.0, 0.5, 3.0]
    report, output = execute_profile(parse_run_profile(data))
    assert report['spectrum_computed'] and report['doppler_ensemble_included']
    spectrum = output['spectrum']
    return np.asarray(spectrum['rf_off']), np.asarray(spectrum['rf_on']), counts


def test_profile_spectrum_with_kernel_equals_reference_path(monkeypatch, isolated):  # noqa: F811
    off_ref, on_ref, reference_counts = _run_lab007(monkeypatch, kernel_enabled=False)
    off_fast, on_fast, fast_counts = _run_lab007(monkeypatch, kernel_enabled=True)
    assert reference_counts['kernel'] == 0
    assert np.max(np.abs(off_fast - off_ref)) <= 1e-12
    assert np.max(np.abs(on_fast - on_ref)) <= 1e-12
    # same velocity classes in total; only two reference solves per scan
    # point (5 offsets x RF off/on) remain on the reference path
    scan_points = 2 * 5
    assert fast_counts['reference'] == ss._FAST_PARITY_POINTS * scan_points
    assert fast_counts['kernel'] >= reference_counts['reference'] - fast_counts['reference']
    assert reference_counts['reference'] > 50 * fast_counts['reference']


def test_disagreeing_kernel_returns_the_whole_spectrum_to_the_reference_path(monkeypatch, isolated):  # noqa: F811
    off_ref, on_ref, reference_counts = _run_lab007(monkeypatch, kernel_enabled=False)
    off_bad, on_bad, bad_counts = _run_lab007(monkeypatch, kernel_enabled=True, corrupt_kernel=True)
    assert np.array_equal(off_bad, off_ref) and np.array_equal(on_bad, on_ref)
    assert bad_counts['reference'] == reference_counts['reference']
    assert bad_counts['kernel'] == 2          # one rejected comparison per spectrum (RF off, RF on)
