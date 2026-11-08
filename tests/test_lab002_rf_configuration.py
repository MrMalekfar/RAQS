"""LAB-002: coherent RF publication, explicit overrides and no stale runtime."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rydberg_aoa.config import atomic_config as ac, laser_rf_config as lr, parameters as params
from rydberg_aoa.config.laboratory import AppliedRF
from rydberg_aoa.config.rf_configuration import (
    configure_rf_fields, read_configured_rf_fields, build_configured_lab_request,
)
from rydberg_aoa.exceptions import PhysicsConfigurationError
from tests.test_lab002_laboratory_request import reference_request


SCALARS = ('RF_LO_FREQUENCY', 'RF_SIGNAL_FREQUENCY', 'RF_LO_E_FIELD', 'RF_SIGNAL_E_FIELD',
           'RF_FREQUENCY', 'IF_FREQ', 'SPACING', 'T_MAX', 'N_ELEMENTS', 'TARGET_AOA_DEG', 'CELL_LENGTH')
CACHES = ('_ATOMIC_LEVELS_CACHE', '_DETUNING_CACHE', '_DIPOLE_MATRIX_CACHE', '_OPTIMIZATION_RESULTS_CACHE')


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    scalars = {key: getattr(params, key) for key in SCALARS}
    mappings = [(mapping, copy.deepcopy(mapping)) for mapping in
                (lr.LO_RF_PARAMS, lr.RF_PARAMS, lr.SIGNAL_RF_PARAMS)]
    caches = {key: dict(getattr(ac, key)) for key in CACHES}
    monkeypatch.setattr(ac, '_ATOMIC_PHYSICS_INITIALIZED', False)
    try:
        yield
    finally:
        for key, value in scalars.items():
            setattr(params, key, value)
        for mapping, values in mappings:
            mapping.clear()
            mapping.update(values)
        for key, values in caches.items():
            getattr(ac, key).clear()
            getattr(ac, key).update(values)


def snapshot():
    return (copy.deepcopy((lr.LO_RF_PARAMS, lr.RF_PARAMS, lr.SIGNAL_RF_PARAMS)),
            {key: getattr(params, key) for key in SCALARS},
            {key: dict(getattr(ac, key)) for key in CACHES})


def test_explicit_pair_updates_all_rf_consumers_and_preserves_aliases():
    refs = lr.LO_RF_PARAMS, lr.RF_PARAMS, lr.SIGNAL_RF_PARAMS
    before_spacing, before_time = params.SPACING, params.T_MAX
    for key in CACHES:
        getattr(ac, key)['old'] = object()
    lo = AppliedRF(120e9, 2., 0)
    signal = AppliedRF(120e9 - 200e3, .01, 0)
    configure_rf_fields(lo=lo, signal=signal)
    assert refs[0] is lr.LO_RF_PARAMS and refs[1] is lr.RF_PARAMS and refs[2] is lr.SIGNAL_RF_PARAMS
    assert lr.RF_PARAMS == lr.LO_RF_PARAMS and lr.RF_PARAMS is not lr.LO_RF_PARAMS
    assert read_configured_rf_fields() == (lo, signal)
    assert params.RF_LO_FREQUENCY == 120e9
    assert params.RF_SIGNAL_FREQUENCY == params.RF_FREQUENCY == signal.frequency_hz
    assert params.IF_FREQ == -.2
    assert params.RF_LO_E_FIELD == 2. and params.RF_SIGNAL_E_FIELD == .01
    assert params.SPACING == before_spacing and params.T_MAX == before_time
    assert all(not getattr(ac, key) for key in CACHES)


@pytest.mark.parametrize('view,key,value', [('selection', 'frequency_hz', 120e9),
                                         ('selection', 'polarization', 1),
                                         ('source', 'RF_LO_FREQUENCY', 120e9),
                                         ('source', 'RF_SIGNAL_E_FIELD', 1.),
                                         ('source', 'RF_FREQUENCY', 120e9),
                                         ('source', 'IF_FREQ', -.1)])
def test_disagreement_is_rejected_without_reconciliation(monkeypatch, view, key, value):
    if view == 'selection':
        monkeypatch.setitem(lr.RF_PARAMS, key, value)
    else:
        monkeypatch.setattr(params, key, value)
    before = snapshot()
    with pytest.raises(PhysicsConfigurationError, match='disagree'):
        read_configured_rf_fields()
    assert snapshot() == before


@pytest.mark.parametrize('bad', [None, {}, 'LO'])
def test_invalid_pair_cannot_partially_publish(bad):
    before = snapshot()
    with pytest.raises(PhysicsConfigurationError):
        configure_rf_fields(lo=AppliedRF(120e9, 1., 0), signal=bad)
    assert snapshot() == before


def test_changed_pair_is_refused_when_physics_is_published(monkeypatch):
    lo, signal = read_configured_rf_fields()
    monkeypatch.setattr(ac, '_ATOMIC_PHYSICS_INITIALIZED', True)
    before = snapshot()
    with pytest.raises(PhysicsConfigurationError, match='published'):
        configure_rf_fields(lo=replace(lo, frequency_hz=120e9), signal=signal)
    assert snapshot() == before
    assert ac.is_atomic_physics_initialized()


def test_identical_pair_preserves_valid_caches_even_when_initialized(monkeypatch):
    lo, signal = read_configured_rf_fields()
    marker = object()
    ac._ATOMIC_LEVELS_CACHE['valid'] = marker
    monkeypatch.setattr(ac, '_ATOMIC_PHYSICS_INITIALIZED', True)
    configure_rf_fields(lo=lo, signal=signal)
    assert ac._ATOMIC_LEVELS_CACHE['valid'] is marker


def test_configured_lab_snapshot_is_detached_and_never_initializes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('LAB-002 construction must not initialize or scan')
    monkeypatch.setattr(ac, 'initialize_atomic_physics', forbidden)
    reference = reference_request()
    before = snapshot()
    request = build_configured_lab_request(preparation=reference.preparation,
                                           bounds=reference.bounds,
                                           atom=SimpleNamespace(I=2.5, groundStateN=5))
    assert snapshot() == before
    monkeypatch.setitem(lr.RF_PARAMS, 'frequency_hz', 120e9)
    assert request.selection_carrier_hz == 132.649e9
    assert request.coupling.wavelength_nm == 483.4


def test_cli_signal_override_changes_signal_geometry_and_signed_if(monkeypatch):
    import cli
    monkeypatch.setattr('sys.argv', ['cli.py', '--rf-frequency', '132648900000', '--no-plots'])
    args = cli.parse_arguments()
    cli.override_config(args)
    assert params.RF_SIGNAL_FREQUENCY == params.RF_FREQUENCY == 132648900000.
    assert lr.SIGNAL_RF_PARAMS['frequency_hz'] == 132648900000.
    assert params.RF_LO_FREQUENCY == lr.RF_PARAMS['frequency_hz'] == lr.LO_RF_PARAMS['frequency_hz'] == 132649000000.
    assert params.IF_FREQ == -.1


def test_cli_lo_override_controls_selection_carrier(monkeypatch):
    import cli
    monkeypatch.setattr('sys.argv', ['cli.py', '--lo-rf-frequency', '132649200000'])
    cli.override_config(cli.parse_arguments())
    assert lr.RF_PARAMS['frequency_hz'] == lr.LO_RF_PARAMS['frequency_hz'] == params.RF_LO_FREQUENCY == 132649200000.
    assert params.IF_FREQ == -.1
    assert params.RF_FREQUENCY == params.RF_SIGNAL_FREQUENCY == 132649100000.


@pytest.mark.parametrize('bad', ['nan', 'inf', '-1', '0'])
def test_invalid_cli_rf_cannot_partially_change_other_config(monkeypatch, bad):
    import cli
    monkeypatch.setattr('sys.argv', ['cli.py', '--rf-frequency', bad, '--n-elements', '99'])
    before = snapshot()
    with pytest.raises(PhysicsConfigurationError):
        cli.override_config(cli.parse_arguments())
    assert snapshot() == before


def test_cli_without_rf_overrides_preserves_original_configuration(monkeypatch):
    import cli
    monkeypatch.setattr('sys.argv', ['cli.py'])
    before = snapshot()
    cli.override_config(cli.parse_arguments())
    assert snapshot() == before


def test_nonreal_legacy_rf_value_is_rejected_before_float_warning(monkeypatch):
    monkeypatch.setitem(lr.RF_PARAMS, 'frequency_hz', np.complex128(132.649e9))
    with pytest.raises(PhysicsConfigurationError, match='real scalar'):
        read_configured_rf_fields()


def test_experimental_seam_rejects_conflicting_rf_before_search(monkeypatch):
    monkeypatch.setitem(lr.RF_PARAMS, 'frequency_hz', 120e9)
    calls = []
    monkeypatch.setattr(ac, '_initialize_atomic_levels', lambda: calls.append('search'))
    before = snapshot()
    with pytest.raises(PhysicsConfigurationError, match='selection carrier is ambiguous'):
        ac.initialize_atomic_physics(resolution_request='experimental')
    assert not calls
    assert snapshot() == before


def test_malformed_existing_views_fail_before_explicit_publication(monkeypatch):
    monkeypatch.setitem(lr.RF_PARAMS, 'frequency_hz', np.array([1., 2.]))
    with pytest.raises(PhysicsConfigurationError, match='malformed'):
        configure_rf_fields(lo=AppliedRF(120e9, 1., 0), signal=AppliedRF(120e9 + 1e5, .01, 0))
    assert lr.LO_RF_PARAMS['frequency_hz'] == 132.649e9
    assert np.array_equal(lr.RF_PARAMS['frequency_hz'], [1., 2.])


@pytest.mark.parametrize('arguments,exit_code', [(['--help'], 0), (['--unknown-option'], 2)])
def test_cli_help_and_parse_errors_never_open_run_log(monkeypatch, arguments, exit_code):
    import cli
    monkeypatch.setattr('sys.argv', ['cli.py', *arguments])
    def forbidden_open(*args, **kwargs):
        pytest.fail('CLI help/parse failure must not open or truncate the run log')
    monkeypatch.setattr('builtins.open', forbidden_open)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == exit_code
