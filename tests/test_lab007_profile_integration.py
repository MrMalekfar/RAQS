"""Profile declarations, independent thermal oracles and actual model consumers."""

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rydberg_aoa.config import atomic_config as ac, parameters as params
from rydberg_aoa.config.constants import k_B
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.prescribed_runtime import capture_prescribed_runtime
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.profile_execution import execute_profile, plot_profile_spectrum, plot_profile_rf_levels
from rydberg_aoa.simulation.profile_spectrum import resolve_profile_medium, run_profile_spectrum
from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state

ROOT = Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT / 'examples/lab007' / f'{mode}_full.json').read_text())


@pytest.fixture
def isolated():
    state, pipeline = snapshot_published_atomic_state(), ac._snapshot_pipeline_transaction()
    names = ('RF_LO_FREQUENCY', 'RF_LO_E_FIELD', 'RF_SIGNAL_FREQUENCY',
             'RF_SIGNAL_E_FIELD', 'RF_FREQUENCY', 'IF_FREQ')
    scalars = {key: getattr(params, key) for key in names}
    ac.reset_atomic_physics()
    try:
        yield
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        for key, value in scalars.items():
            setattr(params, key, value)
        restore_published_atomic_state(state)


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_thermal_declarations_have_no_invented_rates(mode):
    data = document(mode)
    before = copy.deepcopy(data)
    profile = parse_run_profile(data)
    assert data == before
    assert not hasattr(profile.run_config, 'dephasing_rates_per_us')
    assert profile.run_config.tlist_us[-1] == 60.
    assert profile.signed_if_mhz == .1


@pytest.mark.parametrize('change', ['duplicate_rates', 'bad_temperature', 'bad_flag', 'bad_scan', 'bad_method',
                                   'negative_linewidth', 'bad_window', 'one_uniform_class', 'extra_environment'])
def test_invalid_thermal_and_spectrum_contracts_rejected(change):
    data = document()
    if change == 'duplicate_rates':
        data['execution']['transit_rate_per_us'] = 1.
    elif change == 'bad_temperature':
        data['environment']['temperature_k'] = True
    elif change == 'bad_flag':
        data['environment']['collision_enabled'] = 'yes'
    elif change == 'bad_scan':
        data['spectrum']['probe_offsets_mhz'] = [False]
    elif change == 'bad_method':
        data['spectrum']['method'] = 'auto'
    elif change == 'negative_linewidth':
        data['environment']['linewidth_rates_per_us'][1] = -1
    elif change == 'bad_window':
        data['execution']['duration_us'] = 21.
    elif change == 'one_uniform_class':
        data['spectrum'].update(method='uniform', n_classes=1)
    else:
        data['environment']['mass_kg_fallback'] = 1.
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


def test_explicit_rate_source_is_exact_and_rejects_component_overlap(isolated):
    data = document()
    data['environment'].update(rate_source='explicit', linewidth_rates_per_us=[0., 0., 0.],
                               linewidth_enabled=False, collision_enabled=False, transit_enabled=False)
    data['execution'].update(dephasing_rates_per_us=[.02, .03, .03], transit_rate_per_us=.5)
    profile = parse_run_profile(data)
    report, _ = execute_profile(profile, diagnostics_only=True)
    assert report['resolved_rate_budget_per_us'] == {'pure_dephasing': (.02, .03, .03), 'transit_reset': .5}
    data['environment']['collision_enabled'] = True
    with pytest.raises(PhysicsConfigurationError, match='cannot be combined'):
        parse_run_profile(data)


def test_actual_prescribed_budget_and_offsets_without_search(isolated, monkeypatch):
    from rydberg_aoa.level_finder.arc_level_finder import ARCOptimizer
    def forbidden(*args, **kwargs):
        pytest.fail('prescribed initialization must never search levels')
    monkeypatch.setattr(ARCOptimizer, 'optimize_levels', forbidden)
    profile = parse_run_profile(document())
    report, result = execute_profile(profile, diagnostics_only=True)
    medium, config = resolve_profile_medium(profile, ac)
    n = ac.ATOMIC_LEVELS['rydberg1']['n']
    density = ac.ATOM.getNumberDensity(profile.environment.temperature_k)
    collision = density * 1e-18*(n/50)**4 * math.sqrt(8*k_B*medium.temperature_k/(math.pi*medium.atom_mass_kg))/1e6
    transit = math.sqrt(math.pi*k_B*medium.temperature_k/(2*medium.atom_mass_kg)) / (
        math.pi*profile.probe.beam_waist_mm*1e-3/2)/1e6
    assert config.dephasing_rates_per_us == pytest.approx((.006283185307179587+collision,)*2+(collision,), rel=2e-15)
    assert config.transit_rate_per_us == pytest.approx(transit, rel=2e-15)
    assert report['resolved_rate_budget_per_us']['transit_reset'] == config.transit_rate_per_us
    assert report['detunings_rad_per_us'][1] == ac.get_cached_detunings()[1] + 2*math.pi*.25
    assert not report['spectrum_computed'] and result is None
    warmer = replace(profile, environment=replace(profile.environment, temperature_k=320.15))
    warm_medium, warm_config = resolve_profile_medium(warmer, ac)
    assert warm_medium.number_density_m3 > medium.number_density_m3
    assert warm_config.transit_rate_per_us > config.transit_rate_per_us
    assert warm_config.dephasing_rates_per_us[2] > config.dephasing_rates_per_us[2]


def test_missing_prescribed_cache_is_not_repaired(isolated):
    profile = parse_run_profile(document())
    execute_profile(profile, diagnostics_only=True)
    before = capture_prescribed_runtime(ac, profile)
    ac._DIPOLE_MATRIX_CACHE.clear()
    with pytest.raises(PhysicsCalculationError, match='no repair'):
        capture_prescribed_runtime(ac, profile)
    assert not ac._DIPOLE_MATRIX_CACHE
    assert before.rf_orientation.direction == -1


def test_actual_spectrum_uses_applied_medium_and_same_rates(isolated):
    data = document()
    data['spectrum'].update(probe_offsets_mhz=[-.5, .5], doppler_enabled=False, analytical_enabled=False)
    profile = parse_run_profile(data)
    execute_profile(profile, diagnostics_only=True)
    medium, config = resolve_profile_medium(profile, ac)
    spectrum = run_profile_spectrum(profile, config, medium, ac)
    assert np.isfinite(spectrum['rf_off']).all() and np.isfinite(spectrum['rf_on']).all()
    assert spectrum['rf_off_analytical'] is None
    assert not np.array_equal(spectrum['rf_off'], spectrum['rf_on'])
    assert spectrum['base_detunings_rad_per_us'] == capture_prescribed_runtime(ac, profile).detunings
    assert spectrum['dephasing_rates_per_us'] == config.dephasing_rates_per_us
    assert spectrum['transit_rate_per_us'] == config.transit_rate_per_us
    od, chi = spectrum['transmission_prefactors']
    assert od == math.pi*medium.cell_length_m/(profile.probe.wavelength_nm*1e-9)
    dipole = ac.get_cached_dipole_matrix()['probe']['dipole_si']
    si = params.SI_Constants()
    assert chi == -2*medium.number_density_m3*dipole**2/(si.EPSILON_0*si.HBAR*ac.OMEGA_P*1e6)
    longer = run_profile_spectrum(profile, config, replace(medium, cell_length_m=2*medium.cell_length_m), ac)
    np.testing.assert_allclose(longer['rf_on'], spectrum['rf_on']**2, rtol=2e-15)


def test_doppler_settings_reach_coherence_average_before_transmission(isolated, monkeypatch):
    from rydberg_aoa.simulation import steady_state as ss
    data = document()
    data['spectrum'].update(probe_offsets_mhz=[.125], analytical_enabled=False, method='uniform', n_classes=3)
    profile = parse_run_profile(data)
    execute_profile(profile, diagnostics_only=True)
    medium, config = resolve_profile_medium(profile, ac)
    seen = []
    def nodes(**kwargs):
        seen.append(kwargs)
        return np.array([-1., 1.]), np.array([.25, .75])
    monkeypatch.setattr(ss, 'generate_velocity_classes', nodes)
    monkeypatch.setattr(ss, '_solve_probe_coherence', lambda dp, *args, **kwargs: 1j*dp*1e-4)
    spectrum = run_profile_spectrum(profile, config, medium, ac)
    assert all(x['temperature_k'] == medium.temperature_k and x['atom_mass_kg'] == medium.atom_mass_kg for x in seen)
    from rydberg_aoa.physics.doppler import get_doppler_shifts
    center = spectrum['base_detunings_rad_per_us'][0] + 2*math.pi*.125
    shifts = [get_doppler_shifts(v, profile.probe.wavelength_nm*1e-9,
                               profile.coupling.wavelength_nm*1e-9, counter_propagating=True)[0] for v in (-1., 1.)]
    coherence = 1j*1e-4*(center + .25*shifts[0] + .75*shifts[1])
    od, chi = spectrum['transmission_prefactors']
    assert spectrum['rf_on'][0] == pytest.approx(np.exp(-2*od*np.imag(chi*coherence)), rel=2e-15)


def test_failed_quadrature_is_visible_without_retry(monkeypatch):
    from rydberg_aoa.physics.doppler import adaptive_maxwell as module
    monkeypatch.setattr(module, 'quad_vec', lambda *args, **kwargs:
                        (np.array([1., 0.]), 1., SimpleNamespace(success=False, message='limit reached')))
    with pytest.raises(PhysicsCalculationError, match='failed to converge'):
        module.integrate_maxwell_response(lambda v: 1., 300., 1.4e-25, require_convergence=True)


@pytest.mark.parametrize('coherence,prefactors', [(True, (1., -1.)), ('1j', (1., -1.)),
    (np.array(1j), (1., -1.)), (complex(float('nan'), 0.), (1., -1.)),
    (1j, (10**1000, -1.)), (1j, (float('inf'), -1.)), (1j, (True, -1.)),
    (1j, (-1., -1.)), (1j, (1.,)), (1000j, (1., -1.))])
def test_explicit_transmission_rejects_invalid_inputs_and_overflow(coherence, prefactors):
    from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 import calculate_transmission_from_rho21
    with pytest.raises(PhysicsCalculationError):
        calculate_transmission_from_rho21(coherence, absorption_prefactors=prefactors)


def test_injected_transmission_never_reads_global_medium(monkeypatch):
    import importlib
    module = importlib.import_module('rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21')
    def forbidden(*args, **kwargs):
        pytest.fail('explicit prefactors must bypass global/cache lookup')
    monkeypatch.setattr(module, 'calculate_absorption_constant', forbidden)
    assert module.calculate_transmission_from_rho21(0.j, absorption_prefactors=(1., -1.)) == 1.
    assert module.calculate_transmission_from_rho21(np.complex128(-.2j), absorption_prefactors=(1., -1.)) == np.exp(-.4)


@pytest.mark.parametrize('value', [True, '1e16', -1., float('nan'), float('inf')])
def test_invalid_arc_density_is_not_coerced_or_substituted(isolated, monkeypatch, value):
    profile = parse_run_profile(document())
    execute_profile(profile, diagnostics_only=True)
    monkeypatch.setattr(ac.ATOM, 'getNumberDensity', lambda temperature: value)
    with pytest.raises(PhysicsCalculationError, match='invalid ARC density') as error:
        resolve_profile_medium(profile, ac)
    assert isinstance(error.value.__cause__, PhysicsConfigurationError)


def test_thermal_component_switches_give_exact_zero_without_rate_floor(isolated):
    profile = parse_run_profile(document())
    execute_profile(profile, diagnostics_only=True)
    off = replace(profile, environment=replace(profile.environment, linewidth_enabled=False,
                                               collision_enabled=False, transit_enabled=False))
    _, config = resolve_profile_medium(off, ac)
    assert config.dephasing_rates_per_us == (0., 0., 0.)
    assert config.transit_rate_per_us == 0.
    assert ac.GAMMA_43 > 0.  # Disabling driven/thermal effects does not disable spontaneous rates.


def test_analytical_and_numerical_branches_receive_same_effective_offsets(isolated, monkeypatch):
    from rydberg_aoa.simulation import steady_state as ss
    data = document()
    data['spectrum'].update(probe_offsets_mhz=[.125], doppler_enabled=False)
    data['execution']['sensor_probe_offset_mhz'] = .375
    profile = parse_run_profile(data)
    execute_profile(profile, diagnostics_only=True)
    medium, config = resolve_profile_medium(profile, ac)
    numerical, analytical = [], []
    def coherence(dp, dc, *args, **kwargs):
        numerical.append((dp, dc, kwargs['transit_rate_per_us']))
        return 0.j
    def reference(scan, cfg, omega, **kwargs):
        analytical.append((scan, kwargs['detunings_rad_per_us']))
        return 0.j
    monkeypatch.setattr(ss, '_solve_probe_coherence', coherence)
    monkeypatch.setattr(ss.solutions, 'calculate_analytical_rho21', reference)
    run_profile_spectrum(profile, config, medium, ac)
    center = capture_prescribed_runtime(ac, profile).detunings
    assert numerical == [(center[0]+2*math.pi*.125, center[1], config.transit_rate_per_us)]*2
    assert analytical == [(2*math.pi*.125, center)]*2
    assert center[1] == ac.get_cached_detunings()[1]+2*math.pi*.25


def test_zero_driven_field_is_rejected_before_input_writes(isolated):
    data = document()
    data['fields']['signal']['electric_field_vm'] = 0.
    before = copy.deepcopy(ac.LASER_PARAMS)
    with pytest.raises(PhysicsConfigurationError, match='nonzero applied fields'):
        execute_profile(parse_run_profile(data))
    assert ac.LASER_PARAMS == before and not ac.is_atomic_physics_initialized()


def test_failure_after_initialization_restores_inputs(isolated, monkeypatch):
    from rydberg_aoa.simulation import profile_spectrum as module
    values = copy.deepcopy((ac.LASER_PARAMS, ac.RF_PARAMS, ac.SIGNAL_RF_PARAMS))
    original = PhysicsCalculationError('spectrum failure sentinel')
    def fail(*args, **kwargs):
        assert ac.is_atomic_physics_initialized()
        raise original
    monkeypatch.setattr(module, 'run_profile_spectrum', fail)
    with pytest.raises(PhysicsCalculationError) as error:
        execute_profile(parse_run_profile(document()))
    assert error.value is original
    assert (ac.LASER_PARAMS, ac.RF_PARAMS, ac.SIGNAL_RF_PARAMS) == values
    assert not ac.is_atomic_physics_initialized()
    assert not ac._DIPOLE_MATRIX_CACHE


def test_plots_show_only_computed_curves_and_signed_energy():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spectrum = {'probe_offsets_mhz': [-1., 0., 1.], 'rf_off': [.2, .3, .2], 'rf_on': [.3, .5, .3],
                'rf_off_analytical': None, 'rf_on_analytical': None}
    figure = plot_profile_spectrum({'spectrum': spectrum})
    assert len(figure.axes[0].lines) == 2
    np.testing.assert_array_equal(figure.axes[0].lines[1].get_ydata(), spectrum['rf_on'])
    report = {'rf_orientation': {'signed_frequency_hz': -132e9},
              'levels': {'rydberg1': {'n': 26, 'l': 2, 'j': 2.5, 'mj': -.5},
                         'rydberg2': {'n': 27, 'l': 1, 'j': 1.5, 'mj': -.5}}}
    energies = plot_profile_rf_levels(report)
    assert energies.axes[0].collections[1].get_segments()[0][0, 1] == -132.
    assert plot_profile_spectrum({'spectrum': None}) is None
    plt.close('all')
