"""Public ODE wiring contracts; synthetic tests are not laboratory evidence."""
import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.run_numerics import TemporalODE, RunNumerics, ode_receipt
from rydberg_aoa.exceptions import PhysicsConfigurationError, PhysicsCalculationError
from tests.test_lab007_profile_integration import isolated

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = dict(method='bdf',rtol=1e-10,atol=1e-12,nsteps=20000)


def document(mode='prescribed'):
    return json.loads((ROOT/f'examples/atom002/rb85_{mode}.json').read_text())


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_roundtrip_and_spawn_safe_control(mode):
    d = document(mode); d['numerics'] = {'temporal_ode':CONTROLS}
    p = parse_run_profile(d)
    assert parse_run_profile(profile_document(p)) == p
    assert pickle.loads(pickle.dumps(p.run_config)).temporal_ode == TemporalODE(**CONTROLS)
    assert 'temporal_ode' not in profile_document(p)['execution']
    assert p.temporal_doppler.rtol == .01  # ODE tolerances do not overwrite quadrature.


def test_legacy_profile_has_no_new_solver_keywords():
    p = parse_run_profile(document())
    assert p.numerics is None and p.run_config.temporal_ode is None
    assert 'numerics' not in profile_document(p)
    r = ode_receipt(None)
    assert r['source'] == 'legacy_qutip_defaults'
    assert r['effective']['nsteps'] == 5000 and r['effective']['normalize_output'] is False


@pytest.mark.parametrize('field,value', [('method','adams'),('method',None),('rtol',True),
    ('rtol',0.),('rtol',1.),('atol','1e-12'),('atol',float('nan')),('atol',1+0j),
    ('nsteps',False),('nsteps',20e3),('nsteps',0),('nsteps',2**31)])
def test_invalid_controls_rejected_before_initialization(field,value):
    d = document(); d['numerics'] = {'temporal_ode':{**CONTROLS,field:value}}
    with pytest.raises(PhysicsConfigurationError): parse_run_profile(d)


@pytest.mark.parametrize('raw',[None,{}, {'temporal_ode':None}, {'temporal_ode':{}},
    {'temporal_ode':{**CONTROLS,'normalize_output':True}}])
def test_incomplete_or_repair_switch_not_allowed(raw):
    d = document(); d['numerics'] = raw
    with pytest.raises(PhysicsConfigurationError): parse_run_profile(d)


def test_hidden_direct_config_rejected():
    p = parse_run_profile(document())
    p = replace(p,run_config=replace(p.run_config,temporal_ode=TemporalODE(**CONTROLS)))
    with pytest.raises(PhysicsConfigurationError): profile_document(p)


def test_every_velocity_gets_same_explicit_ode_and_failure_propagates(monkeypatch):
    from rydberg_aoa.simulation import laboratory_execution as lab, time_domain as td
    from rydberg_aoa.config.temporal_doppler import TemporalDoppler,ResolvedTemporalDoppler
    controls = TemporalODE(**CONTROLS)
    plan = ResolvedTemporalDoppler(TemporalDoppler(True,'uniform',3,9,5.,.01,1e-10,True),
        300.,1.409e-25,780e-9,483e-9)
    seen = []
    def solve(*a,**kw):
        seen.append(kw['temporal_ode'])
        return np.zeros(len(a[2]),complex)
    monkeypatch.setattr(td,'run_single_heterodyne_worker',solve)
    task = lab.LaboratoryWorkerTask(0,1j,tuple(np.linspace(0.,20.,201)),.1,(1.,2.,3.),
        (0.,0.,0.),(1.,1.,1.),(0.,0.,0.),0.,None,0.,temporal_doppler=plan,temporal_ode=controls)
    result = lab.run_laboratory_worker(task)
    assert result.report['converged'] and len(seen) >= 9
    assert all(c is controls for c in seen)
    def fail(*a,**kw): raise PhysicsCalculationError('injected ODE failure')
    monkeypatch.setattr(td,'run_single_heterodyne_worker',fail)
    with pytest.raises(PhysicsCalculationError,match='no dropped class') as caught:
        lab.run_laboratory_worker(task)
    assert 'injected ODE failure' in str(caught.value.__cause__)


def test_real_mesolve_receives_declared_options_before_certification(monkeypatch):
    from rydberg_aoa.simulation import time_domain as td
    from rydberg_aoa.config import parameters
    monkeypatch.setattr(parameters,'ATOMIC_TRANSIENT_DELAY_US',0.)
    captured = []; original = td.mesolve
    def spy(*a,**kw):
        captured.append(dict(kw['options']))
        return original(*a,**kw)
    monkeypatch.setattr(td,'mesolve',spy)
    t = np.linspace(0.,20.,201)
    result = td.run_single_heterodyne_worker(0,1+.5j,t,1.,5.,3.,2.,.3,-.2,.1,
        [1.,1.,1.],[.2,.2,.2],transit_rate_per_us=0.,temporal_ode=TemporalODE(**CONTROLS))
    assert captured == [{**CONTROLS,'normalize_output':False}]
    assert len(result[4]) == 201 and np.isfinite(result[4]).all()


def test_declared_diagnostic_override_preserves_profile_baseline(monkeypatch):
    from types import SimpleNamespace
    from rydberg_aoa.config.detection_numerics import SolverDiagnostic
    from rydberg_aoa.simulation.temporal_diagnostic import run_diagnostic_worker
    from rydberg_aoa.simulation import time_domain as td
    base = TemporalODE(**CONTROLS); seen = []
    original = lambda **kw: seen.append(dict(kw['options']))
    monkeypatch.setattr(td,'mesolve',original)
    task = SimpleNamespace(temporal_ode=base,solver_diagnostic=SolverDiagnostic('check',1e-11,1e-13,1,30000),
        tlist_us=(0.,1.,2.))
    def invoke(times):
        td.mesolve(options=base.options())
        return 0j,[],[],0,np.zeros(3)
    run_diagnostic_worker(task,invoke)
    assert seen[0]['nsteps'] == 30000 and seen[0]['atol'] == 1e-13
    assert td.mesolve is original and base.nsteps == 20000
    assert ode_receipt(base,task.solver_diagnostic)['source'] == 'explicit_diagnostic_override'


def test_quantile_only_case_reports_profile_work_limit(monkeypatch):
    from types import SimpleNamespace
    from rydberg_aoa.simulation import detection_validation as validation
    base = TemporalODE(**CONTROLS)
    actual = dict(solver=None,poisson_quantiles=True,acquisition_grid_unchanged=True,
        solver_method='bdf',nsteps=20000,normalize_output=False,option_enforcement='not instrumented')
    point = SimpleNamespace(profile=SimpleNamespace(run_config=SimpleNamespace(temporal_ode=base)),
        report={'numerical_diagnostic':actual})
    monkeypatch.setattr(validation,'_jobs',lambda study:[point])
    validation.verify_diagnostic_evidence(object(),None,True)
    actual['nsteps'] = 5000
    with pytest.raises(PhysicsCalculationError): validation.verify_diagnostic_evidence(object(),None,True)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_public_profile_dispatch_receipt_and_all_class_controls(mode,isolated,monkeypatch):
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation import runner,time_domain as td
    d = document(mode); d['numerics'] = {'temporal_ode':CONTROLS}
    d['temporal_doppler'].update(initial_classes=3,max_classes=9)
    profile = parse_run_profile(d); seen = []
    class Pool:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def map(self,fn,tasks): return list(map(fn,tasks))
    class Context:
        def Pool(self,**kw): return Pool()
    monkeypatch.setattr(runner.multiprocessing,'get_context',lambda *a:Context())
    def oracle(*args,**kw):
        seen.append(kw['temporal_ode'])
        assert kw['temporal_ode'] == profile.numerics.temporal_ode
        return .001 + 1e-5*np.exp(1j*td.atomic_if_polarity(kw['rf_orientation'])*2*np.pi*args[3]*np.asarray(args[2]))
    monkeypatch.setattr(td,'run_single_heterodyne_worker',oracle)
    report,result = execute_profile(profile)
    assert len(seen) >= 9 and all(v == TemporalODE(**CONTROLS) for v in seen)
    assert report['temporal_ode']['source'] == 'profile' and report['temporal_ode']['executed']
    assert report['temporal_ode']['effective'] == {**CONTROLS,'normalize_output':False}
    assert result['temporal_doppler_reports'][0]['converged']


@pytest.mark.parametrize('species,mode',[('rb85','prescribed'),('rb87','prescribed'),
    ('cs133','prescribed'),('rb85','experimental')])
def test_real_public_profile_and_spawn_controls_without_doppler(species,mode,isolated):
    from rydberg_aoa.simulation.profile_execution import execute_profile
    d = json.loads((ROOT/f'examples/num_ode002/{species}_{mode}.json').read_text())
    d['temporal_doppler']['enabled'] = False
    report,result = execute_profile(parse_run_profile(d))
    assert report['temporal_ode']['source'] == 'profile' and report['temporal_ode']['executed']
    assert report['temporal_ode']['effective'] == {**CONTROLS,'normalize_output':False}
    assert np.asarray(result['time_traces']).shape == (1,401)
    assert np.isfinite(result['complex_responses']).all()
    assert not report['temporal_doppler_ensemble_included']


def test_explicit_legacy_controls_are_bit_identical_to_omission():
    from rydberg_aoa.simulation import time_domain as td
    defaults = ode_receipt(None)['effective']
    controls = TemporalODE(**{k:defaults[k] for k in CONTROLS})
    args = (0,1+.5j,np.linspace(0.,20.,201),1.,5.,3.,2.,.3,-.2,.1,
        [1.,1.,1.],[.2,.2,.2])
    old = td.run_single_heterodyne_worker(*args,transit_rate_per_us=0.,_coherence_only=True)
    explicit = td.run_single_heterodyne_worker(*args,transit_rate_per_us=0.,_coherence_only=True,temporal_ode=controls)
    assert np.array_equal(old,explicit)
