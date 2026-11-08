"""Laser noise: independent algebra, OU statistics and the real certified worker."""
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.exceptions import PhysicsConfigurationError, PhysicsCalculationError
from rydberg_aoa.physics.hamiltonian import create_sensor_model
from rydberg_aoa.physics.hamiltonian.laser_frequency_noise import laser_frequency_terms
from rydberg_aoa.simulation.rf_drive_noise import build_rf_realization, RFDriveRealization
from rydberg_aoa.simulation import time_domain as td, runner as module
from tests.test_lab005_laboratory_execution import isolation, request, plan, fake_pool, oracle_record

ROOT = Path(__file__).resolve().parents[1]


def declaration():
    return json.loads((ROOT/'examples/noise006/prescribed_laser_frequency.json').read_text())['noise']


@pytest.mark.parametrize('field,bad', [('enabled', 1), ('stationary_std', True),
    ('stationary_std', np.bool_(True)), ('stationary_std', '1'), ('stationary_std', 1j),
    ('stationary_std', np.array(1.)), ('stationary_std', float('nan')),
    ('stationary_std', float('inf')), ('stationary_std', -1), ('stationary_std', 10**1000),
    ('correlation_time_us', 0), ('units', 'MHz'), ('budget', 'same_as_linewidth')])
def test_invalid_laser_contract_even_when_disabled(field, bad):
    data = declaration()
    data['enabled'] = False
    data['sources']['laser_probe_frequency'][field] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


def test_budget_cannot_be_omitted():
    data = declaration()
    del data['sources']['laser_probe_frequency']['budget']
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


def test_invalid_grid_precedes_atomic_initialization(isolation, monkeypatch):
    from rydberg_aoa.config import atomic_config as ac
    data = declaration()
    data['drive']['noise_step_us'] = 1.
    monkeypatch.setattr(ac, 'initialize_atomic_physics', lambda **kw: pytest.fail('invalid grid reached ARC'))
    with pytest.raises(PhysicsConfigurationError):
        module.SimulationRunner().run_laboratory_simulation(request(isolation), plan(), noise=parse_noise_config(data))


def test_node_covariance_and_integrated_linear_frequency():
    data = declaration()
    data['trials'] = 2000
    data['drive'].update(horizon_us=1., noise_step_us=.2)
    cfg = parse_noise_config(data)
    values = np.asarray([build_rf_realization(cfg, trial=i).probe_frequency_hz[:2] for i in range(cfg.trials)])
    sigma = 1000.
    a = np.exp(-.2/5.)
    assert abs(values[:, 0].mean()) < 6*sigma/np.sqrt(cfg.trials)
    assert abs(values[:, 0].var(ddof=1)/sigma**2-1) < 6*np.sqrt(2/(cfg.trials-1))
    assert abs(np.mean(values[:, 0]*values[:, 1])/sigma**2-a) < 6*np.sqrt((1+a*a)/cfg.trials)
    r = RFDriveRealization((0., 1., 2.), (1.,)*3, (1.,)*3, (0.,)*3,
                          (1000., 2000., 0.), (-1000., 0., 1000.))
    assert r.laser_at(.5) == (1500., -500.)
    assert r.laser_phase_at(.5) == pytest.approx(2*np.pi*np.array([625., -375.])/1e6)
    assert r.laser_phase_at(2.) == pytest.approx(2*np.pi*np.array([2500., 0.])/1e6)
    assert np.array_equal(r.laser_phase_nodes(), [r.laser_phase_at(t) for t in r.times_us])
    with pytest.raises(PhysicsCalculationError, match='no clamp'):
        r.laser_at(2.1)


def test_stable_streams_toggles_prefix_and_pickle():
    cfg = parse_noise_config(declaration())
    r = build_rf_realization(cfg, trial=0)
    assert pickle.loads(pickle.dumps(r)) == r
    for source in cfg.sources:
        other = replace(cfg, sources=tuple(replace(s, specification=replace(s.specification, enabled=False))
            if s.name == source.name else s for s in cfg.sources))
        changed = build_rf_realization(other, trial=0)
        attr = 'coupling_frequency_hz' if source.name == 'laser_probe_frequency' else 'probe_frequency_hz'
        assert getattr(changed, attr) == getattr(r, attr)
    extended = build_rf_realization(replace(cfg, trials=5, drive=replace(cfg.drive, horizon_us=240.)), trial=0)
    assert extended.probe_frequency_hz[:len(r.times_us)] == r.probe_frequency_hz
    # Appending laser sources cannot move the previously committed RF streams.
    rf = json.loads((ROOT/'examples/noise005/prescribed_rf.json').read_text())['noise']
    original = build_rf_realization(parse_noise_config(rf), trial=0)
    rf['sources'].update(declaration()['sources'])
    mixed = build_rf_realization(parse_noise_config(rf), trial=0)
    assert mixed.at(3.14) == original.at(3.14)


@pytest.mark.parametrize('kind', ['master', 'sources', 'zero'])
def test_disabled_and_zero_are_bit_identical_and_rng_free(isolation, monkeypatch, kind):
    fake_pool(monkeypatch, [])
    req, run = request(isolation), plan()
    baseline = module.SimulationRunner().run_laboratory_simulation(req, run)
    data = declaration()
    if kind == 'master':
        data['enabled'] = False
    else:
        for s in data['sources'].values():
            s['stationary_std' if kind == 'zero' else 'enabled'] = 0. if kind == 'zero' else False
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('disabled laser RNG'))
    result = module.SimulationRunner().run_laboratory_simulation(req, run, noise=parse_noise_config(data))
    assert np.array_equal(result['time_traces'], baseline['time_traces'])
    assert np.array_equal(result['complex_responses'], baseline['complex_responses'])
    if kind == 'zero':
        assert result['rf_drive_campaign']['atomic_evolutions_per_path'] == 1
        assert not result['rf_drive_campaign']['laser_frequency_noise_included']


def test_current_projector_dephasing_is_not_independent_laser_diffusion():
    rates = [.2, .6, .8]
    H, cops, _ = create_sensor_model(0, 0, 0, 0, 0, 0, [0]*3, rates, transit_rate_per_us=0)
    gamma = [0]+rates
    for i in range(4):
        for j in range(4):
            rho = qt.basis(4, i)*qt.basis(4, j).dag()
            actual = sum((c*rho*c.dag()-.5*(c.dag()*c*rho+rho*c.dag()*c) for c in cops), 0*H)
            expected = -(gamma[i]+gamma[j])/2*rho if i != j else 0*rho
            assert np.allclose(actual.full(), expected.full(), atol=1e-15)
    # Independent probe phase noise moves levels 2,3,4 together: it cannot
    # dephase the 3<->4 optical-common coherence. Legacy level-3 projector does.
    rho = qt.basis(4, 2)*qt.basis(4, 3).dag()
    g = qt.Qobj(np.diag([0., 1., 1., 1.]))
    assert np.array_equal((g*rho*g-.5*(g*g*rho+rho*g*g)).full(), np.zeros((4, 4)))


@pytest.mark.parametrize('direction', [1, -1])
def test_laser_frame_matrix_and_phase_gauge_independent_oracle(direction):
    r = RFDriveRealization((0., 10.), (1., 1.), (1., 1.), (0., 0.),
                          (1000., 2000.), (-3000., -4000.))
    t = 3.25
    p, c = 2*np.pi*np.array([1325., -3325.])/1e6
    expected = np.diag([0., p, p+c, p+c])
    terms = laser_frequency_terms(r)
    actual = qt.QobjEvo(terms)(t).full()
    assert np.array_equal(actual, actual.conj().T)
    assert np.allclose(actual, expected, atol=1e-16)
    phi_p, phi_c = r.laser_phase_at(t)
    U = np.diag(np.exp(1j*np.array([0, phi_p, phi_p+phi_c, phi_p+phi_c])))
    h0 = np.array([[0, -.1, 0, 0], [-.1, .03, -.3, 0],
                   [0, -.3, .01, -.2], [0, 0, -.2, .01+direction*.01]], complex)
    # Nominal-carrier frame has phase-bearing optical drives. U follows the
    # instantaneous laser phase: U^dag Hnom U - i U^dag dot(U)=H0+offsets.
    h_nom = np.array([[0, -.1*np.exp(-1j*phi_p), 0, 0],
        [-.1*np.exp(1j*phi_p), .03, -.3*np.exp(-1j*phi_c), 0],
        [0, -.3*np.exp(1j*phi_c), .01, -.2], [0, 0, -.2, .01+direction*.01]], complex)
    dotU = 1j*expected@U
    transformed = U.conj().T@h_nom@U-1j*U.conj().T@dotU
    assert np.allclose(transformed, h0+expected, atol=1e-15)
    # Susceptibility uses the probe-referred coherence, not its nominal-frame
    # imaginary part; no extra phase factor is needed in the worker's frame.
    state = np.outer([.5]*4, [.5]*4).astype(complex)
    s21 = np.zeros((4,4)); s21[1,0] = 1
    nominal_coherence = np.trace(s21@U@state@U.conj().T)
    assert np.allclose(nominal_coherence*np.exp(1j*phi_p), np.trace(s21@state))
    # Each spontaneous-emission jump acquires only a scalar unit phase in
    # this frame; its dissipator and all diagonal dephasing/reset terms stay.
    for i, j in ((0,1), (1,2), (2,3), (3,2), (0,3)):
        jump = np.zeros((4,4), complex); jump[i,j] = 1
        changed = U.conj().T@jump@U
        def dissipator(c):
            cc = c.conj().T@c
            return c@state@c.conj().T-.5*(cc@state+state@cc)
        assert np.allclose(dissipator(changed), dissipator(jump), atol=1e-15)


@pytest.mark.parametrize('direction', [1, -1])
def test_real_certified_worker_gets_correct_laser_diagonal(monkeypatch, direction):
    from rydberg_aoa.config import parameters as params
    monkeypatch.setattr(params, 'ATOMIC_TRANSIENT_DELAY_US', 0.)
    r = build_rf_realization(parse_noise_config(declaration()), trial=0)
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('RNG in RHS'))
    real = td.mesolve
    def capture(H, *args, **kw):
        evo = qt.QobjEvo(H)
        for t in (.1, 3.14159, 8.35):
            p, c = 2*np.pi*np.array(r.laser_at(t))/1e6
            signal = .05*np.exp(1j*(.3+direction*2*np.pi*.1*t))
            expected = np.array([[0, -.1, 0, 0], [-.1, .03+p, -.3, 0],
                [0, -.3, .01+p+c, -.5*np.conj(.4+signal)],
                [0, 0, -.5*(.4+signal), .01+direction*.01+p+c]], complex)
            assert np.allclose(evo(t).full(), expected, atol=1e-15)
        assert not kw['options']['normalize_output']
        return real(H, *args, **kw)
    monkeypatch.setattr(td, 'mesolve', capture)
    orientation = SimpleNamespace(direction=direction, is_downward=direction<0,
        upper_ladder_index=4 if direction>0 else 3, lower_ladder_index=3 if direction>0 else 4)
    out = td.run_single_heterodyne_worker(0, .05*np.exp(.3j), np.linspace(0,10,101), .1,
        .2, .6, .4, .03, -.02, .01, (.8,.1,.05), (.01,.02,.02),
        transit_rate_per_us=.2, rf_orientation=orientation, rf_drive=r)
    assert len(out) == 5 and np.isfinite(out[4]).all()


def test_real_output_grid_refinement_with_same_laser_path(monkeypatch):
    from rydberg_aoa.config import parameters as params
    monkeypatch.setattr(params, 'ATOMIC_TRANSIENT_DELAY_US', 0.)
    r = build_rf_realization(parse_noise_config(declaration()), trial=0)
    def run(times, realization):
        return td.run_single_heterodyne_worker(0, .05*np.exp(.3j), times, .1,
            .2, .6, .4, .03, -.02, .01, (.8,.1,.05), (.01,.02,.02),
            transit_rate_per_us=.2, rf_drive=realization)
    coarse = run(np.linspace(0,20,201), r)
    fine = run(np.linspace(0,20,401), r)
    nominal = run(np.linspace(0,20,201), None)
    assert not np.array_equal(coarse[4], nominal[4])
    assert np.allclose(coarse[4], fine[4][::2], rtol=1e-4, atol=1e-6)
    assert np.allclose(coarse[0], fine[0], rtol=1e-3, atol=1e-5)


def test_common_laser_dispatch_and_single_readout_trial(isolation, monkeypatch):
    data = declaration()
    data['readout'] = json.loads((ROOT/'examples/noise003/prescribed_campaign.json').read_text())['noise']
    data['readout']['trials'] = data['trials']
    data['readout']['seed'] = data['seed']
    tasks = []
    fake_pool(monkeypatch, tasks)
    cfg = parse_noise_config(data)
    out = module.SimulationRunner().run_laboratory_simulation(request(isolation), plan(), noise=cfg)
    assert len(tasks) == 12  # nominal + 3 laser trials, 3 paths, no T^2
    for trial in range(3):
        group = tasks[3+3*trial:6+3*trial]
        assert all(task.rf_drive is group[0].rf_drive for task in group)
        assert group[0].rf_drive == build_rf_realization(cfg, trial=trial)
        assert out['rf_drive_trial_results'][trial]['readout_noise']['trial_index'] == trial
    assert out['rf_drive_campaign']['laser_frequency_noise_included']
    assert not out['rf_drive_campaign']['rf_drive_included']


def test_laser_worker_failure_aborts_without_nominal_fallback(isolation, monkeypatch):
    cause = PhysicsCalculationError('injected laser worker failure')
    def fail(task):
        if task.rf_drive is not None:
            raise cause
        return oracle_record(task)
    fake_pool(monkeypatch, [], worker=fail)
    runner = module.SimulationRunner()
    with pytest.raises(PhysicsCalculationError, match='entire RF campaign aborted') as error:
        runner.run_laboratory_simulation(request(isolation), plan(), noise=parse_noise_config(declaration()))
    assert error.value.__cause__ is cause
    assert not runner.results


@pytest.mark.parametrize('mode', ['prescribed', 'experimental', 'prescribed_pin'])
def test_real_cli_saved_laser_realization_and_fields(tmp_path, mode):
    filename = 'prescribed_laser_pin.json' if mode == 'prescribed_pin' else f'{mode}_laser_frequency.json'
    data = json.loads((ROOT/'examples/noise006'/filename).read_text())
    data['noise']['trials'] = 1
    if mode == 'prescribed_pin':
        data['noise']['readout']['trials'] = 1
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    path = tmp_path/'profile.json'; path.write_text(json.dumps(data))
    args = [sys.executable, '-B', 'main.py', '--profile', str(path), '--output-dir', str(tmp_path)]
    if mode != 'prescribed':
        args.append('--no-plots')
    proc = subprocess.run(args, cwd=ROOT, env=dict(os.environ, MPLBACKEND='Agg'),
        capture_output=True, text=True, errors='replace', timeout=240)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    report = json.loads((tmp_path/'run_report.json').read_text(encoding='utf-8'))
    assert report['laser_frequency_noise_included'] and report['noise_included']
    assert not report['rf_drive_included'] and not report['spectrum_drive_noise_included']
    assert report['applied_fields'] == data['fields']
    expected = build_rf_realization(parse_noise_config(data['noise']), trial=0)
    with np.load(tmp_path/'simulation_arrays.npz', allow_pickle=False) as a:
        assert np.array_equal(a['trial_laser_probe_frequency_hz'][0], expected.probe_frequency_hz)
        assert np.array_equal(a['trial_laser_integrated_phase_rad'][0],
                              [expected.laser_phase_at(t) for t in expected.times_us])
        assert not np.array_equal(a['nominal_time_traces'], a['time_traces'])
        if mode == 'prescribed_pin':
            assert report['sensing_trace_units'] == 'V'
            assert report['readout_noise']['noise_applied']
            assert np.array_equal(a['time_traces'], a['clean_time_traces']+a['additive_readout_noise'])
            assert a['trial_measured_current_a'].shape[0] == 1
    if mode == 'prescribed':
        assert (tmp_path/'fig_laser_frequency_noise.png').is_file()
