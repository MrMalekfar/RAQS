"""Explicit backend selection preserves the four-state model and legacy defaults."""
import copy
from dataclasses import replace
import numpy as np
import pytest
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.exceptions import PhysicsConfigurationError
from tests.test_lab007_profile_integration import isolated, document


@pytest.mark.parametrize('backend', [None, 'direct', 'qutip'])
def test_backend_roundtrip_preserves_legacy_document(backend):
    data = document()
    if backend is not None:
        data['spectrum']['steady_state_backend'] = backend
    original = copy.deepcopy(data)
    p = parse_run_profile(data)
    assert data == original
    exported = profile_document(p)
    assert exported['spectrum'].get('steady_state_backend') == backend
    assert ('steady_state_backend' in exported['spectrum']) == (backend is not None)
    assert parse_run_profile(exported).spectrum == p.spectrum


@pytest.mark.parametrize('bad', ['auto', 'resolvent', '', True, 1, [], {}])
def test_invalid_backend_rejected_before_initialization(bad):
    data = document(); data['spectrum']['steady_state_backend'] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


@pytest.mark.parametrize('backend', [None, 'direct', 'qutip'])
def test_profile_forwards_same_backend_to_rf_off_and_on(isolated, monkeypatch, backend):
    from rydberg_aoa.config import atomic_config as ac
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation.profile_spectrum import resolve_profile_medium, run_profile_spectrum
    from rydberg_aoa.simulation import steady_state as ss
    data = document(); data['spectrum']['probe_offsets_mhz'] = [-.2, .3]
    if backend is not None:data['spectrum']['steady_state_backend'] = backend
    p = parse_run_profile(data); execute_profile(p, diagnostics_only=True)
    medium, config = resolve_profile_medium(p, ac)
    calls = []
    def solve(*args, **kwargs):
        calls.append((args[0], kwargs))
        return np.array(args[1]), np.ones(2), None
    monkeypatch.setattr(ss, 'simulate_spectrum', solve)
    result = run_profile_spectrum(p, config, medium, ac)
    assert len(calls) == 2 and calls[0][0] == 0 and calls[1][0] == ac.OMEGA_RF_BASE
    for _, kwargs in calls:
        assert kwargs.get('steady_state_backend') == backend
        assert ('steady_state_backend' in kwargs) == (backend is not None)
    assert result.get('steady_state_backend') == backend


@pytest.mark.parametrize('rf', [0., .8])
def test_selected_backends_solve_identical_four_level_density(rf):
    from rydberg_aoa.physics.hamiltonian import create_sensor_model
    from rydberg_aoa.simulation.steady_state import _steady_state_solve
    h, jumps, _ = create_sensor_model(1.3, 2.1, rf, .2, -.1, .05,
                                     [6., .01, .02], [.1, .03, .04], transit_rate_per_us=.2)
    direct = _steady_state_solve(h, jumps, 'direct')
    reference = _steady_state_solve(h, jumps, 'qutip')
    np.testing.assert_allclose(direct.full(), reference.full(), atol=1e-10, rtol=0)


def test_actual_profile_spectra_agree_between_backends(isolated):
    from rydberg_aoa.config import atomic_config as ac
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation.profile_spectrum import resolve_profile_medium, run_profile_spectrum
    data = document()
    data['spectrum'].update(probe_offsets_mhz=[-.2, .3], doppler_enabled=False,
                            analytical_enabled=False, steady_state_backend='direct')
    profile = parse_run_profile(data); execute_profile(profile, diagnostics_only=True)
    medium, config = resolve_profile_medium(profile, ac)
    direct = run_profile_spectrum(profile, config, medium, ac)
    other = replace(profile, spectrum=replace(profile.spectrum, steady_state_backend='qutip'))
    reference = run_profile_spectrum(other, config, medium, ac)
    for key in ('rf_off', 'rf_on'):
        np.testing.assert_allclose(direct[key], reference[key], atol=1e-10, rtol=1e-10)
