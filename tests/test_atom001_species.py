"""Explicit isotope selection, ARC oracles, rollback and spawn-safe receipts."""
import copy
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from rydberg_aoa.config import atomic_config as ac, parameters as params
from rydberg_aoa.config.atom_species import (AtomSpec, LEGACY_ATOM, get_arc_atom,
    species_capabilities, arc_density, resolved_atom)
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.constants import k_B
from rydberg_aoa.simulation.profile_execution import execute_profile
from rydberg_aoa.exceptions import PhysicsConfigurationError, PhysicsCalculationError
from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state

ROOT = Path(__file__).resolve().parents[1]


def document(species='rb85', mode='prescribed'):
    return json.loads((ROOT/f'examples/atom001/{species}_{mode}.json').read_text())


@pytest.fixture
def isolated():
    state = snapshot_published_atomic_state()
    pipeline = ac._snapshot_pipeline_transaction()
    names = ('RF_LO_FREQUENCY', 'RF_LO_E_FIELD', 'RF_SIGNAL_FREQUENCY', 'RF_SIGNAL_E_FIELD', 'RF_FREQUENCY', 'IF_FREQ')
    scalars = {name: getattr(params, name) for name in names}
    ac.reset_atomic_physics()
    try:
        yield
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        for name, value in scalars.items():
            setattr(params, name, value)
        restore_published_atomic_state(state)


@pytest.mark.parametrize('species', ['Cs', 'Rb', 'rb85', None, True, 'Cs135'])
def test_unknown_selection_is_rejected(species):
    with pytest.raises(PhysicsConfigurationError):
        AtomSpec(species, 1.)


@pytest.mark.parametrize('fraction', [0., -1., 1.1, True, np.bool_(True), '1', np.nan, np.inf, 10**1000, np.array(1.)])
def test_fraction_requires_finite_real_probability(fraction):
    with pytest.raises(PhysicsConfigurationError):
        AtomSpec('Rb85', fraction)


@pytest.mark.parametrize('species', ['rb85', 'rb87', 'cs133'])
@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_profile_round_trip_does_not_query_arc(species, mode, monkeypatch):
    import rydberg_aoa.config.atom_species as api
    monkeypatch.setattr(api, 'get_arc_atom', lambda *args: pytest.fail('parsing must not construct ARC'))
    d = document(species, mode)
    p = parse_run_profile(d)
    assert profile_document(parse_run_profile(profile_document(p))) == profile_document(p)
    assert profile_document(p)['atom'] == d['atom']
    assert set(species_capabilities()['species']) == {'Rb85', 'Rb87', 'Cs133'}


@pytest.mark.parametrize('species', ['rb85', 'rb87', 'cs133'])
@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_real_arc_initialization_matches_independent_isotope_oracles(isolated, species, mode):
    from arc import Rubidium85, Rubidium87, Caesium
    oracle = {'rb85': Rubidium85, 'rb87': Rubidium87, 'cs133': Caesium}[species]()
    d = document(species, mode)
    before = copy.deepcopy(d)
    p = parse_run_profile(d)
    report, result = execute_profile(p, diagnostics_only=True)
    assert d == before and result is None
    assert report['atom']['mass_kg'] == oracle.mass
    assert report['atom']['nuclear_spin'] == oracle.I
    assert report['resolved_medium']['number_density_m3'] == oracle.getNumberDensity(300.)
    assert ac.ATOM_SPEC == p.atom
    assert report['atom']['selection_source'] == 'profile'
    assert report['applied_fields']['lo']['frequency_hz'] == p.lo.frequency_hz
    levels = report['levels']
    a, b = levels['rydberg1'], levels['rydberg2']
    signed = oracle.getTransitionFrequency(a['n'], a['l'], a['j'], b['n'], b['l'], b['j'])
    assert report['rf_orientation']['signed_frequency_hz'] == signed
    assert a['mj'] == b['mj'] == -.5
    from scipy.constants import hbar
    rf_ea0 = oracle.getDipoleMatrixElement(a['n'], a['l'], a['j'], a['mj'],
                                          b['n'], b['l'], b['j'], b['mj'], 0)
    assert ac.ARC_RF_DIPOLE_EA0 == pytest.approx(rf_ea0, rel=2e-13, abs=0)
    # Preserve the existing project's rounded e*a0 conversion contract.
    # Updating this universal constant is a separate physical calibration.
    mu_rf = abs(rf_ea0) * 8.478e-30
    assert ac.get_cached_dipole_matrix()['rf']['dipole_si'] == pytest.approx(mu_rf, rel=2e-13, abs=0)
    assert ac.OMEGA_RF_BASE == pytest.approx(mu_rf*p.lo.electric_field_vm/hbar/1e6, rel=2e-13, abs=0)
    if mode == 'prescribed':
        for role, declared in d['ladder'].items():
            for key, value in declared.items():
                assert levels[role][key] == value
    g, e = levels['ground'], levels['excited']
    assert ac.GAMMA_21 == oracle.getTransitionRate(e['n'], e['l'], e['j'], g['n'], g['l'], g['j'])/1e6
    from rydberg_aoa.physics.lindblad.calculate_transit_rate import calculate_transit_rate_per_us
    transit = math.sqrt(math.pi*k_B*300./(2*oracle.mass))/(math.pi*.1e-3/2)/1e6
    assert calculate_transit_rate_per_us(temperature_k=300., beam_waist_mm=.1, atom_mass_kg=oracle.mass) == pytest.approx(transit)
    assert pickle.loads(pickle.dumps(resolved_atom(ac.ATOM, p.atom))) == resolved_atom(ac.ATOM, p.atom)


def test_fraction_changes_density_not_atom_or_transition(isolated):
    d = document('rb87')
    d['atom']['isotope_fraction'] = .2783
    p = parse_run_profile(d)
    report, _ = execute_profile(p, diagnostics_only=True)
    assert report['resolved_medium']['number_density_m3'] == ac.ATOM.getNumberDensity(300.)*.2783
    assert params.calculate_atomic_density_with_arc() == ac.ATOM.getNumberDensity(params.VAPOR_CELL_TEMPERATURE_K)*.2783


def test_switch_away_and_back_clears_caches_and_preserves_rb85_values(isolated):
    reports = []
    for species in ('rb85', 'cs133', 'rb85'):
        ac.reset_atomic_physics()
        p = parse_run_profile(document(species))
        report, _ = execute_profile(p, diagnostics_only=True)
        reports.append(report)
        assert len(ac._ATOMIC_LEVELS_CACHE) == 1
        with pytest.raises(PhysicsConfigurationError, match='reset'):
            ac.configure_atom(LEGACY_ATOM)
    assert reports[0]['rabi_rad_per_us'] == reports[2]['rabi_rad_per_us']
    assert reports[0]['detunings_rad_per_us'] == reports[2]['detunings_rad_per_us']


def test_failure_after_real_cs_initialization_restores_atom_density_and_caches(isolated, monkeypatch):
    import rydberg_aoa.simulation.profile_spectrum as spectrum
    entry = ac._snapshot_pipeline_transaction()
    real = spectrum.resolve_profile_medium
    error = PhysicsCalculationError('injected after successful Cs initialization')
    def fail(profile, config):
        assert ac.is_atomic_physics_initialized() and type(ac.ATOM).__name__ == 'Caesium'
        real(profile, config)
        raise error
    monkeypatch.setattr(spectrum, 'resolve_profile_medium', fail)
    with pytest.raises(PhysicsCalculationError) as exc:
        execute_profile(parse_run_profile(document('cs133')), diagnostics_only=True)
    assert exc.value is error
    assert ac.ATOM is entry['atom'] and ac.ATOM_SPEC == entry['atom_spec']
    assert params.CALCULATED_N0 == entry['number_density']
    assert params.SI_Constants.N0 == entry['si_number_density']
    assert ac._ATOMIC_LEVELS_CACHE == entry['atomic_levels_cache']
    assert not ac.is_atomic_physics_initialized()


@pytest.mark.parametrize('species', ['rb87', 'cs133'])
def test_unvalidated_collision_is_rejected_without_publishing(isolated, species):
    d = document(species)
    d['environment']['collision_enabled'] = True
    with pytest.raises(PhysicsConfigurationError, match='collision'):
        execute_profile(parse_run_profile(d), diagnostics_only=True)
    assert not ac.is_atomic_physics_initialized()


def test_wrong_hyperfine_state_is_rejected_before_seam(isolated, monkeypatch):
    d = document('rb87')
    d['ladder']['ground']['f'] = 3
    d['ladder']['excited']['f'] = 4
    monkeypatch.setattr(ac, 'initialize_atomic_physics', lambda **kwargs: pytest.fail('invalid isotope channel reached seam'))
    with pytest.raises(PhysicsConfigurationError, match='F/mF'):
        execute_profile(parse_run_profile(d), diagnostics_only=True)


def test_legacy_profile_retains_rb85_contract_after_explicit_cs_run(isolated):
    execute_profile(parse_run_profile(document('cs133')), diagnostics_only=True)
    ac.reset_atomic_physics()
    d = document()
    d.pop('atom')
    p = parse_run_profile(d)
    assert 'atom' not in profile_document(p)
    report, _ = execute_profile(p, diagnostics_only=True)
    assert ac.ATOM_SPEC == LEGACY_ATOM
    assert report['atom']['selection_source'] == 'legacy_Rb85_contract'


def test_direct_atom_rebind_cannot_reuse_published_prescribed_values(isolated, monkeypatch):
    execute_profile(parse_run_profile(document()), diagnostics_only=True)
    monkeypatch.setattr(ac, 'ATOM', get_arc_atom(AtomSpec('Cs133', 1.)))
    with pytest.raises(PhysicsConfigurationError, match='changed'):
        ac.get_cached_detunings()


@pytest.mark.parametrize('value', [True, '1', complex(1), np.array(1.), np.nan, -1])
def test_invalid_arc_density_never_becomes_an_isotope_density(value):
    from types import SimpleNamespace
    with pytest.raises(PhysicsCalculationError):
        arc_density(SimpleNamespace(getNumberDensity=lambda _: value), LEGACY_ATOM, 300.)


@pytest.mark.parametrize('species', ['rb85', 'rb87', 'cs133'])
def test_real_temporal_solver_and_spawn_transport_use_selected_species(isolated, species):
    d = document(species)
    # Actual spawn for Cs, serial adapter for both Rb isotopes. Same solver.
    d['execution']['processes'] = 2 if species == 'cs133' else 1
    report, result = execute_profile(parse_run_profile(d))
    assert report['sensing']['doa_status'] == 'not executed: IF-only receiver'
    assert result['estimated_aoa'] is None
    assert np.all(np.isfinite(result['complex_responses']))
    assert np.all(np.isfinite(result['time_traces']))
    assert report['atom']['species'].lower() == species


def test_worker_rejects_mismatched_isotope_doppler_mass_before_solver(monkeypatch):
    from dataclasses import replace
    from tests.test_dop_td001_temporal_doppler import task, plan
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    from rydberg_aoa.simulation import time_domain
    receipt = resolved_atom(get_arc_atom(AtomSpec('Cs133', 1.)), AtomSpec('Cs133', 1.))
    t = replace(task(plan()), atom=receipt)
    monkeypatch.setattr(time_domain, 'run_single_heterodyne_worker', lambda *a, **k: pytest.fail('mismatched mass reached solver'))
    with pytest.raises(PhysicsConfigurationError, match='mass differs'):
        run_laboratory_worker(t)


def test_experimental_cache_identity_contains_composition(isolated):
    from rydberg_aoa.config.laboratory_runtime import laboratory_identity
    p = parse_run_profile(document('rb87', 'experimental'))
    ac.configure_atom(p.atom)
    first = laboratory_identity(ac.ATOM, p.request)
    ac.configure_atom(AtomSpec('Rb87', .2783))
    second = laboratory_identity(ac.ATOM, p.request)
    assert first != second


@pytest.mark.parametrize('name', ['Rb87', 'Cs133'])
def test_legacy_analytical_tables_reject_other_species(name):
    from rydberg_aoa.analytical.energy import RydbergEnergyCalculator
    from rydberg_aoa.analytical.dipoles import AnalyticalDipoleCalculator
    for cls in (RydbergEnergyCalculator, AnalyticalDipoleCalculator):
        with pytest.raises(PhysicsConfigurationError, match='Rb85'):
            cls(name)


@pytest.mark.parametrize('species', ['Rb87', 'Cs133'])
def test_old_rf_search_cannot_fall_back_to_rb85_for_other_isotopes(species):
    from rydberg_aoa.arc.transitions import find_rf_transitions
    from rydberg_aoa.arc.types import AtomicLevel
    with pytest.raises(PhysicsConfigurationError, match='cross-species fallback'):
        find_rf_transitions(get_arc_atom(AtomSpec(species, 1.)), AtomicLevel(n=26, l=2, j=2.5), 1e10)


def test_failed_arc_constructor_retains_cause_without_replacement(monkeypatch):
    from types import SimpleNamespace
    import rydberg_aoa.config.atom_species as api
    error = RuntimeError('missing selected isotope database')
    def broken():
        raise error
    monkeypatch.delitem(api._ATOMS, 'Rb87', raising=False)
    monkeypatch.setattr(api, 'import_module', lambda _: SimpleNamespace(Rubidium87=broken))
    with pytest.raises(PhysicsCalculationError) as exc:
        get_arc_atom(AtomSpec('Rb87', 1.))
    assert exc.value.__cause__ is error
    assert 'Rb87' not in api._ATOMS


def test_programmatic_prescribed_seam_rejects_wrong_species_ladder(isolated):
    ac.configure_atom(AtomSpec('Cs133', 1.))
    rb_request = parse_run_profile(document()).request.resolution_request()
    with pytest.raises(PhysicsConfigurationError, match='ground state'):
        ac.initialize_atomic_physics(resolution_request=rb_request)
    assert not ac.is_atomic_physics_initialized()


def test_primary_schematic_uses_actual_cs_report_and_species_label(isolated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rydberg_aoa.visualization.plot_energy_levels import plot_energy_levels
    report, _ = execute_profile(parse_run_profile(document('cs133')), diagnostics_only=True)
    fig = plot_energy_levels(report=report)
    try:
        assert 'Cs133' in fig.axes[0].get_title(loc='left')
        assert any('6S' in text.get_text() for text in fig.axes[0].texts)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('species', ['rb87', 'cs133'])
def test_low_level_hfs_uses_selected_isotope_arc_data(isolated, species):
    from scipy.constants import c
    d = document(species, 'experimental')
    d['optical_energy_convention'] = 'low_level_hfs'
    p = parse_run_profile(d)
    report, _ = execute_profile(p, diagnostics_only=True)
    atom = ac.ATOM
    g, e = p.request.preparation.ground, p.request.preparation.excited
    sg = atom.getHFSEnergyShift(g.j, g.f, *atom.getHFSCoefficients(g.n, g.l, g.j))
    se = atom.getHFSEnergyShift(e.j, e.f, *atom.getHFSCoefficients(e.n, e.l, e.j))
    fp = atom.getTransitionFrequency(g.n, g.l, g.j, e.n, e.l, e.j) + se - sg
    expected = (c/(p.probe.wavelength_nm*1e-9)-fp)*2*math.pi/1e6
    assert report['detunings_rad_per_us'][0] == pytest.approx(expected, rel=2e-13, abs=1e-6)
    assert report['optical_energy_convention'] == 'low_level_hfs'


def test_missing_required_hfs_never_switches_to_fs_centroid(isolated, monkeypatch):
    atom = get_arc_atom(AtomSpec('Cs133', 1.))
    error = ValueError('no HFS data for selected isotope/state')
    def missing(*args):
        raise error
    monkeypatch.setattr(atom, 'getHFSCoefficients', missing)
    d = document('cs133', 'experimental')
    d['optical_energy_convention'] = 'low_level_hfs'
    with pytest.raises(PhysicsCalculationError, match='HFS') as exc:
        execute_profile(parse_run_profile(d), diagnostics_only=True)
    assert exc.value.__cause__ is error
    assert not ac.is_atomic_physics_initialized()


def test_arc_energy_preference_change_invalidates_published_values(isolated, monkeypatch):
    execute_profile(parse_run_profile(document()), diagnostics_only=True)
    monkeypatch.setattr(ac.ATOM, 'preferQuantumDefects', not ac.ATOM.preferQuantumDefects)
    with pytest.raises(PhysicsConfigurationError, match='identity changed'):
        ac.get_cached_dipole_matrix()
