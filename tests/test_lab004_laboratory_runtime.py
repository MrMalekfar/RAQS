"""LAB-004 canonical seam, full identities, stale guards and rollback."""

import copy
from dataclasses import replace

import pytest

from rydberg_aoa.config import atomic_config as ac, parameters as params, laboratory_runtime as lab
from rydberg_aoa.config.laboratory import OpticalEnergyConvention
from rydberg_aoa.config.rf_configuration import build_configured_lab_request, configure_rf_fields
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state, PUBLISHED_ATOMIC_GLOBALS
from tests.test_auto001b_prepared_channel import Atom
from tests.test_lab002_laboratory_request import reference_request


CACHES = ('_ATOMIC_LEVELS_CACHE', '_DETUNING_CACHE', '_DIPOLE_MATRIX_CACHE', '_OPTIMIZATION_RESULTS_CACHE')
RF_SCALARS = ('RF_LO_FREQUENCY', 'RF_SIGNAL_FREQUENCY', 'RF_LO_E_FIELD', 'RF_SIGNAL_E_FIELD', 'RF_FREQUENCY', 'IF_FREQ')


class RuntimeAtom(Atom):
    # Explicit synthetic SI mass for ATOM-001 worker receipts; not ARC evidence.
    mass = 1.41e-25

    def getTransitionRate(self, *args):
        return 38e6 if args[0] == 5 else .02e6


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    published = snapshot_published_atomic_state()
    pipeline = ac._snapshot_pipeline_transaction()
    scalars = {key: getattr(params, key) for key in RF_SCALARS}
    atom = RuntimeAtom()
    monkeypatch.setattr(ac, 'ATOM', atom)
    ac.reset_atomic_physics()
    try:
        yield atom
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        restore_published_atomic_state(published)
        for key, value in scalars.items():
            setattr(params, key, value)


def request(atom):
    template = reference_request()
    bounds = replace(template.bounds, n_min=26, n_max=27, rf_tolerance_hz=1000.)
    return build_configured_lab_request(preparation=template.preparation, bounds=bounds, atom=atom)


def applied_snapshot():
    return copy.deepcopy((ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS,
                          tuple(getattr(params, key) for key in RF_SCALARS), params.LEVEL_SELECTION_TOLERANCES))


def assert_unpublished():
    assert not ac.is_atomic_physics_initialized()
    assert all(getattr(ac, name) is None for name in PUBLISHED_ATOMIC_GLOBALS)
    assert ac._INITIALIZED_REQUEST is ac._INITIALIZED_RESOLUTION_IDENTITY is ac._RF_ORIENTATION is None


def test_canonical_seam_populates_all_caches_without_legacy_or_applied_mutation(isolation, monkeypatch):
    atom = isolation
    req = request(atom)
    before = applied_snapshot()
    refs = ac.LASER_PARAMS, ac.LASER_PARAMS['probe'], ac.LASER_PARAMS['coupling'], ac.RF_PARAMS
    def forbidden(*args, **kwargs):
        pytest.fail('LAB-004 must not use legacy or PRESCRIBED initialization paths')
    for name in ('_initialize_atomic_levels', '_initialize_prescribed_levels', '_declared_dipole_matrix',
                 'calculate_laser_rabi_frequency', 'calculate_rf_rabi_frequency', 'get_signal_rf_rabi_frequency'):
        monkeypatch.setattr(ac, name, forbidden)
    assert ac.initialize_atomic_physics(resolution_request=req)
    runtime = ac.get_laboratory_runtime()
    assert ac._INITIALIZED_REQUEST is req and ac._INITIALIZED_RESOLUTION_IDENTITY == runtime.identity
    assert all(runtime.identity in getattr(ac, key) for key in CACHES)
    assert ac.ATOMIC_LEVELS['rydberg1']['mj'] == ac.ATOMIC_LEVELS['rydberg2']['mj'] == -.5
    assert ac.ATOMIC_LEVELS['excited']['mj'] is None and ac.ATOMIC_LEVELS['excited']['mf'] == 0
    assert ac.get_cached_detunings() == runtime.detunings
    assert ac.get_arc_calculated_rf_dipole() == runtime.selection.winner.rf_ea0
    assert ac.get_rf_orientation().direction == -1
    assert (ac.OMEGA_P, ac.OMEGA_C, ac.OMEGA_RF_BASE, ac.OMEGA_SIGNAL_RF_BASE) == tuple(vars(runtime.selection.rabi).values())
    assert params.OMEGA_S_BASE == ac.OMEGA_SIGNAL_RF_BASE and params.OMEGA_LO_BASE == ac.OMEGA_RF_BASE
    assert params.SI_Constants.LAMBDA_P == req.probe.wavelength_nm * 1e-9
    assert applied_snapshot() == before
    assert refs[0] is ac.LASER_PARAMS and refs[1] is ac.LASER_PARAMS['probe']
    assert refs[2] is ac.LASER_PARAMS['coupling'] and refs[3] is ac.RF_PARAMS
    matrix = ac.get_cached_dipole_matrix()
    matrix['probe']['dipole_si'] = 99
    matrix['rf']['level1'].mj = 1.5
    assert ac.get_cached_dipole_matrix()['probe']['dipole_si'] != 99
    assert ac.get_cached_dipole_matrix()['rf']['level1'].mj == -.5


def test_idempotency_does_not_recalculate_but_force_does(isolation, monkeypatch):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    original = lab.build_laboratory_runtime
    calls = []
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(lab, 'build_laboratory_runtime', counted)
    assert ac.initialize_atomic_physics(resolution_request=req)
    assert not calls
    assert ac.initialize_atomic_physics(resolution_request=req, force=True)
    assert calls == [True]


@pytest.mark.parametrize('kind', ['bounds', 'preparation', 'axis', 'energy_policy', 'power', 'waist', 'offset', 'signal', 'lo_field'])
def test_full_identity_does_not_round_or_omit_request_fields(isolation, kind):
    req = request(isolation)
    changes = {
        'bounds': dict(bounds=replace(req.bounds, n_max=28)),
        'preparation': dict(preparation=replace(req.preparation, ground=replace(req.preparation.ground, mf=1),
                                             excited=replace(req.preparation.excited, mf=1))),
        'axis': dict(preparation=replace(req.preparation, quantization_axis=(1., 0., 0.))),
        'energy_policy': dict(optical_energy_convention=OpticalEnergyConvention.LOW_LEVEL_HFS),
        'power': dict(coupling=replace(req.coupling, power_mw=req.coupling.power_mw + 1e-5)),
        'waist': dict(probe=replace(req.probe, beam_waist_mm=req.probe.beam_waist_mm + 1e-8)),
        'offset': dict(coupling=replace(req.coupling, detuning_mhz=1e-6)),
        'signal': dict(signal=replace(req.signal, frequency_hz=req.signal.frequency_hz + 1)),
        'lo_field': dict(lo=replace(req.lo, electric_field_vm=req.lo.electric_field_vm + 1e-5)),
    }
    changed = replace(req, **changes[kind])
    assert ac._resolution_cache_key(req) != ac._resolution_cache_key(changed)


def test_different_request_cannot_reuse_published_cache_without_force(isolation):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    changed = replace(req, bounds=replace(req.bounds, n_max=28))
    calls = len(isolation.dipole_calls)
    with pytest.raises(PhysicsConfigurationError, match='different request/atom'):
        ac.initialize_atomic_physics(resolution_request=changed)
    assert len(isolation.dipole_calls) == calls
    assert ac.get_laboratory_runtime().request == req
    ac.initialize_atomic_physics(resolution_request=changed, force=True)
    assert ac.get_laboratory_runtime().request == changed
    assert all(ac._resolution_cache_key(req) in getattr(ac, key) and ac._resolution_cache_key(changed) in getattr(ac, key) for key in CACHES)


@pytest.mark.parametrize('field', ['probe_power', 'coupling_offset', 'signal_frequency', 'lo_field'])
def test_stale_applied_inputs_fail_before_idempotency_or_cache_reads(isolation, field):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    if field == 'probe_power':
        ac.LASER_PARAMS['probe']['power_mw'] += 1e-6
    elif field == 'coupling_offset':
        ac.LASER_PARAMS['coupling']['detuning_mhz'] += 1
    else:
        ac.reset_atomic_physics()
        if field == 'signal_frequency':
            configure_rf_fields(lo=req.lo, signal=replace(req.signal, frequency_hz=req.signal.frequency_hz + 1))
        else:
            configure_rf_fields(lo=replace(req.lo, electric_field_vm=req.lo.electric_field_vm + 1e-5), signal=req.signal)
        # Recreate the scenario of a direct coherent global edit after publication.
        ac.initialize_atomic_physics(resolution_request=build_configured_lab_request(preparation=req.preparation, bounds=req.bounds, atom=isolation))
        if field == 'signal_frequency':
            ac.SIGNAL_RF_PARAMS['frequency_hz'] -= 1
            params.RF_SIGNAL_FREQUENCY -= 1
            params.RF_FREQUENCY -= 1
            params.IF_FREQ = (params.RF_SIGNAL_FREQUENCY - params.RF_LO_FREQUENCY) / 1e6
        else:
            ac.LO_RF_PARAMS['electric_field_vm'] -= 1e-5
            ac.RF_PARAMS['electric_field_vm'] -= 1e-5
            params.RF_LO_E_FIELD -= 1e-5
        req = ac._INITIALIZED_REQUEST
    count = len(isolation.dipole_calls)
    for reader in (ac.get_cached_detunings, ac.get_cached_dipole_matrix, ac.get_arc_calculated_rf_dipole, ac.get_rf_orientation):
        with pytest.raises(PhysicsConfigurationError):
            reader()
    with pytest.raises(PhysicsConfigurationError):
        ac.initialize_atomic_physics(resolution_request=req)
    assert len(isolation.dipole_calls) == count


@pytest.mark.parametrize('cache', CACHES)
def test_missing_cache_cannot_self_heal_or_return_zero(isolation, cache):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    getattr(ac, cache).clear()
    with pytest.raises(PhysicsCalculationError, match='missing cache'):
        ac.get_cached_dipole_matrix()


@pytest.mark.parametrize('target', ['dipole', 'detuning', 'rate', 'levels'])
def test_corrupted_runtime_quantities_fail_loud(isolation, target):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    key = ac._resolution_cache_key(req)
    if target == 'dipole':
        ac._DIPOLE_MATRIX_CACHE[key]['rf']['dipole_si'] *= 2
    elif target == 'detuning':
        ac._DETUNING_CACHE[key]['delta_rf_rad_us'] += 1
    elif target == 'rate':
        ac.GAMMA_43 *= 2
    else:
        ac.ATOMIC_LEVELS['rydberg1']['mj'] = .5
    with pytest.raises(PhysicsCalculationError, match='differ'):
        ac.get_cached_dipole_matrix()


def test_force_failure_rolls_back_caches_but_leaves_runtime_unpublished(isolation, monkeypatch):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    cached = {name: dict(getattr(ac, name)) for name in CACHES}
    before = applied_snapshot()
    lambda_before, s_before, lo_before = params.SI_Constants.LAMBDA_P, params.OMEGA_S_BASE, params.OMEGA_LO_BASE
    failure = PhysicsCalculationError('injected cache-publication failure')
    def fail(module, runtime):
        module._ATOMIC_LEVELS_CACHE['partial'] = {'bad': True}
        module.LASER_PARAMS['coupling']['wavelength_nm'] = 499
        module.params.SI_Constants.LAMBDA_P = 1
        module.params.OMEGA_S_BASE = 2
        raise failure
    monkeypatch.setattr(lab, 'populate_laboratory_caches', fail)
    with pytest.raises(PhysicsCalculationError) as error:
        ac.initialize_atomic_physics(resolution_request=req, force=True)
    assert error.value is failure
    assert_unpublished()
    assert applied_snapshot() == before
    assert (params.SI_Constants.LAMBDA_P, params.OMEGA_S_BASE, params.OMEGA_LO_BASE) == (lambda_before, s_before, lo_before)
    assert all(getattr(ac, name) == values for name, values in cached.items())


def test_arc_rate_failure_is_not_published_and_preserves_cause(isolation, monkeypatch):
    req = request(isolation)
    failure = ValueError('rate data unavailable')
    def fail(*args):
        raise failure
    monkeypatch.setattr(isolation, 'getTransitionRate', fail)
    with pytest.raises(PhysicsCalculationError) as error:
        ac.initialize_atomic_physics(resolution_request=req)
    assert error.value.__cause__ is failure
    assert_unpublished()
    assert all(not getattr(ac, name) for name in CACHES)


def test_atom_replacement_invalidates_reads_and_requires_explicit_force(isolation, monkeypatch):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    monkeypatch.setattr(ac, 'ATOM', RuntimeAtom())
    with pytest.raises(PhysicsConfigurationError, match='identity changed'):
        ac.get_cached_detunings()
    ac.initialize_atomic_physics(resolution_request=req, force=True)
    assert ac.get_cached_detunings() == ac.get_laboratory_runtime().detunings


def test_reset_preserves_applied_inputs_and_requires_explicit_reinitialization(isolation):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    before = applied_snapshot()
    ac.reset_atomic_physics()
    assert_unpublished()
    assert applied_snapshot() == before and all(not getattr(ac, name) for name in CACHES)
    with pytest.raises(PhysicsConfigurationError, match='not been initialized'):
        ac.get_laboratory_runtime()


@pytest.mark.parametrize('target', ['metadata', 'derived'])
def test_malformed_runtime_views_raise_project_error(isolation, target):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    if target == 'metadata':
        ac._OPTIMIZATION_RESULTS_CACHE[ac._resolution_cache_key(req)] = []
    else:
        params.OMEGA_S_BASE = True
    with pytest.raises(PhysicsCalculationError):
        ac.get_cached_detunings()


def test_request_mismatch_is_rejected_before_any_atomic_work(isolation):
    req = request(isolation)
    changed = replace(req, coupling=replace(req.coupling, power_mw=req.coupling.power_mw + 1))
    with pytest.raises(PhysicsConfigurationError, match='coupling input differs'):
        ac.initialize_atomic_physics(resolution_request=changed)
    assert not isolation.dipole_calls
    assert_unpublished()


def test_tiny_configured_power_change_recomputes_rabi_without_rounded_cache_reuse(isolation):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    old_key, old_rabi = ac._INITIALIZED_RESOLUTION_IDENTITY, ac.OMEGA_C
    ac.LASER_PARAMS['coupling']['power_mw'] += 1e-5
    changed = request(isolation)
    ac.initialize_atomic_physics(resolution_request=changed, force=True)
    assert ac._INITIALIZED_RESOLUTION_IDENTITY != old_key
    assert ac.OMEGA_C != old_rabi
    assert ac.OMEGA_C / old_rabi == pytest.approx((changed.coupling.power_mw / req.coupling.power_mw)**.5, rel=2e-15)


def test_final_rf_check_failure_has_project_context_and_no_publication(isolation, monkeypatch):
    req = request(isolation)
    cause = ValueError('final RF data failure')
    def fail(*args):
        raise cause
    monkeypatch.setattr(lab, 'resolve_rydberg_orientation', fail)
    with pytest.raises(PhysicsCalculationError, match='final ARC RF orientation') as error:
        ac.initialize_atomic_physics(resolution_request=req)
    assert error.value.__cause__ is cause
    assert_unpublished()
    assert all(not getattr(ac, name) for name in CACHES)
