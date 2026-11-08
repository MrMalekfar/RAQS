"""Optical temporal ensemble contracts and real certified BDF consumers."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import numpy as np
import pytest

from rydberg_aoa.config.temporal_doppler import TemporalDoppler, ResolvedTemporalDoppler
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.temporal_doppler import ensemble_coherence, DopplerWorkerResult

ROOT = Path(__file__).resolve().parents[1]
TIMES = np.linspace(0., 20., 101)


def controls(**kwargs):
    return replace(TemporalDoppler(True, 'gauss_hermite', 3, 33, 5., 1e-5, 1e-9, True), **kwargs)


def plan(**kwargs):
    return ResolvedTemporalDoppler(controls(**kwargs), 300., 1.41e-25, 780e-9, 483e-9)


@pytest.mark.parametrize('field,value', [('enabled',1), ('method','adaptive'),
    ('initial_classes',4), ('max_classes',5), ('rtol',True), ('atol',np.nan),
    ('n_sigma',0), ('counter_propagating','true'), ('initial_classes',np.bool_(True)), ('rtol',1.)])
def test_invalid_declarations(field, value):
    with pytest.raises(PhysicsConfigurationError):controls(**{field:value})


@pytest.mark.parametrize('mode', ['prescribed','experimental'])
def test_profile_round_trip_and_off(mode):
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.config.profile_document import profile_document
    data = json.loads((ROOT/'examples/lab007'/f'{mode}_full.json').read_text())
    data['temporal_doppler'] = asdict(controls(enabled=False))
    p = parse_run_profile(data)
    exported = profile_document(p)
    assert exported['temporal_doppler'] == data['temporal_doppler']
    assert parse_run_profile(exported).temporal_doppler == p.temporal_doppler
    assert p.run_config.temporal_doppler is None


@pytest.mark.parametrize('method', ['uniform','gauss_hermite'])
def test_weighted_complex_response_and_shift_signs(method):
    seen = []
    base = (1+.4j)*np.exp(2j*np.pi*.1*TIMES)
    def response(dp, dc):
        seen.append((dp,dc))
        return base+1e-7*dp
    actual, report = ensemble_coherence(plan(method=method), TIMES,.1,0.,response)
    np.testing.assert_allclose(actual,base,rtol=0,atol=1e-15)
    assert report['classes'] == 9 and report['converged']
    assert all(dp*dc <= 0 for dp,dc in seen)
    assert any(dp<0 for dp,dc in seen) and any(dp>0 for dp,dc in seen)
    assert len(seen) == report['class_solves']
    assert report['rf_doppler_included'] is False


def test_quadrature_matches_independent_gaussian_moment():
    # E[(-k v 1e-6)^2] = k^2 (kBT/m) 1e-12, not a second helper oracle.
    from rydberg_aoa.config.constants import k_B as Boltzmann
    p = plan()
    scale = (2*np.pi/p.probe_wavelength_m*1e-6)**2*Boltzmann*p.temperature_k/p.atom_mass_kg
    actual, _ = ensemble_coherence(p,TIMES,.1,0.,lambda dp,dc:
        (dp**2/scale)*(1+.2j)*np.exp(2j*np.pi*.1*TIMES))
    np.testing.assert_allclose(actual,(1+.2j)*np.exp(2j*np.pi*.1*TIMES),rtol=1e-12)


def test_unconverged_quadrature_never_returns_v0():
    calls=[]
    def response(dp,dc):
        calls.append(dp)
        return np.exp(-(dp/10)**2)*np.exp(2j*np.pi*.1*TIMES)
    with pytest.raises(PhysicsCalculationError,match='did not converge'):
        ensemble_coherence(plan(max_classes=9),TIMES,.1,0.,response)
    assert len(calls)>1


def test_failed_class_keeps_cause_and_no_survivors():
    cause = PhysicsCalculationError('broken class')
    def fail(dp,dc):raise cause
    with pytest.raises(PhysicsCalculationError,match='velocity') as e:
        ensemble_coherence(plan(),TIMES,.1,0.,fail)
    assert e.value.__cause__ is cause


@pytest.mark.parametrize('bad', [np.full(101,np.nan), np.zeros(100),np.zeros(101,dtype=bool)])
def test_bad_class_trace_rejected(bad):
    with pytest.raises(PhysicsCalculationError,match='coherence'):
        ensemble_coherence(plan(),TIMES,.1,0.,lambda dp,dc:bad)


def task(resolved=None):
    from rydberg_aoa.simulation.laboratory_execution import LaboratoryWorkerTask
    return LaboratoryWorkerTask(0,1+.5j,tuple(np.linspace(0.,5.,101)),1.,
        (5.,3.,2.),(.3,-.2,.1),(1.,1.,1.),(.2,.2,.2),.1,None,0.,
        temporal_doppler=resolved)


def test_spawn_pickle_keeps_immutable_controls():
    t=task(plan())
    assert pickle.loads(pickle.dumps(t)) == t


def test_real_worker_zero_spread_limit_and_disabled_path(monkeypatch):
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    from rydberg_aoa.simulation import time_domain as td
    initializations=[]
    original=td.steadystate
    def spy(H, c_ops):
        initializations.append(H.full().copy())
        return original(H,c_ops)
    monkeypatch.setattr(td,'steadystate',spy)
    p=ResolvedTemporalDoppler(controls(atol=1e-8),1e-12,1.41e-25,780e-9,483e-9)
    baseline=run_laboratory_worker(task())
    result=run_laboratory_worker(task(p))
    assert isinstance(result,DopplerWorkerResult)
    np.testing.assert_allclose(result.record[4],baseline[4],rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(result.record[0],baseline[0],rtol=1e-6,atol=1e-8)
    assert result.report['converged']
    assert len(initializations)==1+result.report['class_solves']
    assert any(not np.array_equal(H,initializations[0]) for H in initializations[1:])


def test_each_class_owns_initial_state_and_same_drive(monkeypatch):
    from rydberg_aoa.simulation import time_domain as td
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    calls=[]
    drive=object()
    t=replace(task(plan()),rf_drive=drive)
    def fake(*args,**kwargs):
        calls.append((args[7:10],kwargs))
        assert kwargs['_coherence_only'] is True
        return np.exp(2j*np.pi*np.asarray(args[2]))
    monkeypatch.setattr(td,'run_single_heterodyne_worker',fake)
    result=run_laboratory_worker(t)
    assert all(k['rf_drive'] is drive for d,k in calls)
    assert len({d[0] for d,k in calls})>1
    assert all(d[2] == t.detunings_rad_per_us[2] for d,k in calls)
    assert len(result.record)==5


def test_coherence_only_certifies_before_return(monkeypatch):
    from rydberg_aoa.simulation import time_domain as td
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    def reject(*a,**k):raise PhysicsCalculationError('bad temporal state')
    monkeypatch.setattr(td,'validate_density_matrix',reject)
    with pytest.raises(PhysicsCalculationError,match='velocity'):
        run_laboratory_worker(task(plan()))


def test_diagnostic_ensemble_instruments_every_class():
    from rydberg_aoa.config.detection_numerics import SolverDiagnostic
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    p=ResolvedTemporalDoppler(controls(atol=1e-8),1e-12,1.41e-25,780e-9,483e-9)
    result=run_laboratory_worker(replace(task(p),solver_diagnostic=SolverDiagnostic('tight',1e-8,1e-10,2)))
    assert isinstance(result,DopplerWorkerResult)
    assert result.record[4].shape == (101,)


def test_real_warm_ensemble_cross_quadrature():
    """Broad-line controlled model, physical 300 K optical shifts, real BDF.

    This is an independent quadrature cross-check, not an experimental EIT
    point. Large declared rates make the whole Maxwell response resolvable.
    """
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    from rydberg_aoa.config.detection_numerics import SolverDiagnostic
    records=[]
    for method in ('gauss_hermite', 'uniform'):
        p=plan(method=method,initial_classes=17,max_classes=257,rtol=1e-4,atol=1e-8,n_sigma=6.)
        t=replace(task(p),tlist_us=tuple(np.linspace(0.,.05,101)),signed_if_mhz=100.,
                  omega_signal_complex=100.+50j, omegas_rad_per_us=(5000.,3000.,2000.),
                  decay_rates_per_us=(1000.,1000.,1000.),dephasing_rates_per_us=(1000.,1000.,1000.),
                  solver_diagnostic=SolverDiagnostic('warm_crosscheck',1e-10,1e-12,1))
        records.append(run_laboratory_worker(t))
    np.testing.assert_allclose(records[0].record[4],records[1].record[4],rtol=2e-4,atol=2e-8)
    np.testing.assert_allclose(records[0].record[0],records[1].record[0],rtol=5e-4,atol=2e-8)
    assert all(r.report['converged'] for r in records)
    assert abs(records[0].record[0])>1e-8


# Real ARC initialization and production wiring; only the temporal evolution
# is replaced by a controlled coherence oracle to inspect the dispatched data.
from tests.test_lab007_profile_integration import isolated


@pytest.mark.parametrize('mode', ['prescribed','experimental'])
def test_profile_dispatches_ensemble_and_reaches_aoa(mode, isolated, monkeypatch):
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation import runner, time_domain as td
    calls=[]
    class Pool:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def map(self,fn,tasks):return list(map(fn,tasks))
    class Context:
        def Pool(self,**k):return Pool()
    monkeypatch.setattr(runner.multiprocessing,'get_context',lambda *a:Context())
    def response(*args,**kwargs):
        assert kwargs['_coherence_only'] is True
        calls.append(args[7:10])
        times=np.asarray(args[2]); phase=np.angle(args[1])
        sigma=td.atomic_if_polarity(kwargs['rf_orientation'])
        return (.001+1e-12*args[7]+1e-5*np.exp(1j*(sigma*2*np.pi*args[3]*times+phase)))
    monkeypatch.setattr(td,'run_single_heterodyne_worker',response)
    data=json.loads((ROOT/'examples/lab007'/f'{mode}_full.json').read_text())
    data['spectrum']=None
    data['temporal_doppler']=asdict(controls())
    report,result=execute_profile(parse_run_profile(data))
    assert report['temporal_doppler_ensemble_included']
    reports=result['temporal_doppler_reports']
    assert len(reports)==len(data['execution']['positions_m'])
    assert all(r['converged'] for r in reports)
    assert len({d[0] for d in calls})>1
    assert abs(result['estimated_aoa']-data['execution']['target_azimuth_deg'])<.2


@pytest.mark.parametrize('mode', ['prescribed','experimental'])
def test_pin_examples_apply_forward_model_after_ensemble(mode, isolated, monkeypatch):
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation import runner, time_domain as td, detector_readout as pin
    class Pool:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def map(self,fn,tasks):return list(map(fn,tasks))
    class Context:
        def Pool(self,**k):return Pool()
    monkeypatch.setattr(runner.multiprocessing,'get_context',lambda *a:Context())
    calls=[]; seen=[]
    def response(*args,**kwargs):
        calls.append(args[7])
        sigma=td.atomic_if_polarity(kwargs['rf_orientation'])
        return 1e-12*args[7]-1e-4j*(1+.05*np.cos(sigma*2*np.pi*args[3]*np.asarray(args[2])+np.angle(args[1])))
    monkeypatch.setattr(td,'run_single_heterodyne_worker',response)
    forward=pin.detector_forward
    def spy(traces,*a,**k):
        seen.append(np.asarray(traces).copy())
        return forward(traces,*a,**k)
    monkeypatch.setattr(pin,'detector_forward',spy)
    data=json.loads((ROOT/'examples/dop_td001'/f'{mode}_optical_doppler.json').read_text())
    data['temporal_doppler']=asdict(controls())
    report,result=execute_profile(parse_run_profile(data))
    assert report['temporal_doppler_ensemble_included']
    assert seen and len(calls)>len(data['execution']['positions_m'])
    assert all(x.shape == (3,601) for x in seen)
    assert result['readout_noise']['units']=='V'
    assert np.all(result['additive_readout_noise']==0.)
    assert abs(result['estimated_aoa']+30.)<.2
