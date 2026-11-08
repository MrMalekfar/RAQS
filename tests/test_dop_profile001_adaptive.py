"""Production quadrature contracts; fabricated traces are labelled as such."""
from dataclasses import asdict, replace
import copy
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pytest

from rydberg_aoa.config.temporal_doppler import (
    AdaptiveTemporalDoppler, VelocityErrorBudget, ResolvedTemporalDoppler,
    parse_temporal_doppler, temporal_doppler_capabilities)
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.adaptive_temporal_doppler import adaptive_ensemble

ROOT = Path(__file__).resolve().parents[1]
TIMES = np.linspace(0., 20., 101)


def controls(**kwargs):
    return replace(AdaptiveTemporalDoppler(True, 'adaptive_simpson', True, 5., 1.,
        8, (-.5, 0., .5), 80, .001, 3, 500, 60.,
        VelocityErrorBudget(1e-9, .02, 1e-9, .02, .002, 1e-8)), **kwargs)


def plan(**kwargs):
    return ResolvedTemporalDoppler(controls(**kwargs), 300., 1.41e-25, 780e-9, 483e-9)


@pytest.mark.parametrize('name,value', [
    ('enabled',1), ('method','sinh'), ('counter_propagating',None), ('n_sigma',False),
    ('velocity_scale_ms',0), ('initial_y_intervals',3), ('max_panels',8),
    ('hint_offsets_ms',[]), ('hint_offsets_ms',[.1]), ('hint_offsets_ms',[0.,-.5]),
    ('mass_relative_tolerance',np.nan), ('independent_legendre_order',True),
    ('max_new_solves',0), ('max_wall_seconds',float('inf')), ('max_wall_seconds',10**1000),
    ('budgets',{}), ('budgets',None)])
def test_bad_controls_rejected(name,value):
    with pytest.raises(PhysicsConfigurationError):
        controls(**{name:value})


@pytest.mark.parametrize('name', [f for f in asdict(controls().budgets)])
@pytest.mark.parametrize('bad', [True, '0.1', complex(.1), np.nan, -1., 10**1000])
def test_bad_budget_scalars(name,bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(controls().budgets, **{name:bad})


def test_profile_round_trip_capabilities_and_pickle():
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.config.profile_document import profile_document
    data=json.loads((ROOT/'examples/dop_profile001/rb85_adaptive_if.json').read_text())
    before=copy.deepcopy(data)
    p=parse_run_profile(data)
    assert data==before
    assert parse_run_profile(profile_document(p)).temporal_doppler==p.temporal_doppler
    assert pickle.loads(pickle.dumps(p.temporal_doppler))==p.temporal_doppler
    assert 'adaptive_simpson' in temporal_doppler_capabilities()['methods']
    data['temporal_doppler']['unknown']=1
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(data)


@pytest.mark.parametrize('n_sigma',[3.,5.,6.])
def test_conditional_gaussian_moment_complex_mean_and_exact_cache(n_sigma):
    # Analytic conditional Gaussian second moment, independent of quadrature.
    p=plan(n_sigma=n_sigma)
    from rydberg_aoa.config.constants import k_B
    sigma=math.sqrt(k_B*p.temperature_k/p.atom_mass_kg)
    k=2*np.pi/p.probe_wavelength_m*1e-6
    seen=[]
    carrier=(1+.3j)*np.exp(2j*np.pi*.1*TIMES)
    def solve(dp,dc):
        v=-dp/k
        seen.append(v.hex())
        return (1+v*v/sigma**2)*carrier
    z,r=adaptive_ensemble(p,TIMES,.1,0.,solve,[0.])
    moment=1-2*n_sigma*np.exp(-n_sigma*n_sigma/2)/math.sqrt(2*np.pi)/math.erf(n_sigma/math.sqrt(2))
    np.testing.assert_allclose(z,(1+moment)*carrier,rtol=.002,atol=1e-9)
    assert len(seen)==len(set(seen))==r['class_solves']
    assert r['converged'] and r['independent']['comparison']['accepted']
    assert r['adaptive']['history'][-1]['consecutive_global_passes']==2
    assert min(r['weights'])>0 and abs(sum(r['weights'])-1)<1e-14
    assert not r['full_reference_qualified']


def test_failed_class_preserves_cause_no_partial_mean():
    cause=PhysicsCalculationError('certification rejected')
    def solve(*a):raise cause
    with pytest.raises(PhysicsCalculationError,match='velocity') as e:
        adaptive_ensemble(plan(),TIMES,.1,0.,solve,[])
    assert e.value.__cause__ is cause
    assert e.value.velocity_diagnostics['total_class_solves']==0


@pytest.mark.parametrize('trace',[np.zeros(100),np.zeros(101,dtype=bool),np.full(101,np.nan)])
def test_invalid_trace_cannot_reach_if(trace):
    with pytest.raises(PhysicsCalculationError,match='coherence'):
        adaptive_ensemble(plan(),TIMES,.1,0.,lambda *a:trace,[])


def test_solve_budget_exhaustion_stops_before_dispatch():
    seen=[]
    def solve(*a):seen.append(a); return np.exp(2j*np.pi*.1*TIMES)
    with pytest.raises(PhysicsCalculationError,match='solve budget'):
        adaptive_ensemble(plan(max_new_solves=2),TIMES,.1,0.,solve,[])
    assert len(seen)==2


def test_wall_budget_checked_after_native_call(monkeypatch):
    from rydberg_aoa.simulation import adaptive_temporal_doppler as module
    clock=[0.]
    monkeypatch.setattr(module,'perf_counter',lambda:clock[0])
    def solve(*a):clock[0]=100.;return np.exp(2j*np.pi*.1*TIMES)
    with pytest.raises(PhysicsCalculationError,match='wall budget'):
        adaptive_ensemble(plan(),TIMES,.1,0.,solve,[])


def test_zero_if_has_undefined_phase_and_cannot_converge():
    with pytest.raises(PhysicsCalculationError,match='did not converge'):
        adaptive_ensemble(plan(max_panels=12),TIMES,.1,0.,lambda *a:np.zeros(101,complex),[])


def test_independent_stage_is_mandatory(monkeypatch):
    from rydberg_aoa.simulation import adaptive_temporal_doppler as module
    real=module.legendre_rule
    def corrupted(*a):
        z,v,w,m=real(*a)
        return z+1j*np.sin(2*np.pi*.1*TIMES),v,w,m
    monkeypatch.setattr(module,'legendre_rule',corrupted)
    with pytest.raises(PhysicsCalculationError,match='independent'):
        adaptive_ensemble(plan(),TIMES,.1,0.,lambda *a:np.exp(2j*np.pi*.1*TIMES),[])


def test_independent_stage_consumes_same_solve_budget(monkeypatch):
    from rydberg_aoa.simulation import adaptive_temporal_doppler as module
    calls=[]; before=[]
    real=module.legendre_rule
    def solve(*a):calls.append(a);return np.exp(2j*np.pi*.1*TIMES)
    def spy(*a):before.append(len(calls));return real(*a)
    monkeypatch.setattr(module,'legendre_rule',spy)
    adaptive_ensemble(plan(),TIMES,.1,0.,solve,[])
    limit=before[0]
    assert len(calls)>limit
    calls.clear()
    with pytest.raises(PhysicsCalculationError,match='solve budget'):
        adaptive_ensemble(plan(max_new_solves=limit),TIMES,.1,0.,solve,[])
    assert len(calls)==limit and len(before)==2


def test_disabled_new_method_round_trips_without_resolved_ensemble():
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.config.profile_document import profile_document
    data=json.loads((ROOT/'examples/dop_profile001/rb85_adaptive_if.json').read_text())
    data['temporal_doppler']['enabled']=False
    p=parse_run_profile(data)
    assert p.run_config.temporal_doppler is None
    assert not p.temporal_doppler.enabled
    assert parse_run_profile(profile_document(p)).temporal_doppler==p.temporal_doppler


def test_pickled_worker_task_keeps_exact_plan_and_ode():
    from tests.test_dop_td001_temporal_doppler import task
    from rydberg_aoa.config.run_numerics import BDFOrderControls
    t=replace(task(plan()),temporal_ode=BDFOrderControls('bdf',1e-10,1e-12,20000,2))
    assert pickle.loads(pickle.dumps(t))==t


def test_legacy_helper_rejects_adaptive_without_task_context():
    from rydberg_aoa.simulation.temporal_doppler import ensemble_coherence
    with pytest.raises(PhysicsCalculationError,match='task-level resonance hints'):
        ensemble_coherence(plan(),TIMES,.1,0.,lambda *a:np.zeros(101,complex))


def test_production_numerics_has_no_diagnostic_tools_import():
    import ast
    for name in ('physics/doppler/temporal_quadrature.py','simulation/adaptive_temporal_doppler.py'):
        tree=ast.parse((ROOT/'rydberg_aoa'/name).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(m and m.startswith('tools') for m in imports)


def test_shifted_narrow_complex_response_against_independent_integral():
    from scipy.integrate import quad
    from rydberg_aoa.config.constants import k_B
    p=plan()
    sigma=math.sqrt(k_B*p.temperature_k/p.atom_mass_kg)
    k=2*np.pi/p.probe_wavelength_m*1e-6
    center,width=3.25,.3
    carrier=(1+.3j)*np.exp(2j*np.pi*.1*TIMES)
    def solve(dp,dc):return carrier/(1+((-dp/k-center)/width)**2)
    z,r=adaptive_ensemble(p,TIMES,.1,0.,solve,[center])
    def f(v):return math.exp(-.5*(v/sigma)**2)/(sigma*math.sqrt(2*np.pi))/(1+((v-center)/width)**2)
    expected=quad(f,-5*sigma,5*sigma,points=[center-width,center,center+width],
        epsabs=1e-12,epsrel=1e-10)[0]/math.erf(5/math.sqrt(2))
    np.testing.assert_allclose(z,expected*carrier,rtol=.002,atol=1e-9)
    assert r['independent']['comparison']['accepted']


def test_nonfinite_derived_arithmetic_is_typed_failure():
    with pytest.raises(PhysicsCalculationError):
        adaptive_ensemble(plan(),TIMES,.1,0.,lambda *a:np.full(101,1e308+1e308j),[])


def test_worker_preserves_nontrivial_shared_drive_and_rf_detuning(monkeypatch):
    from tests.test_dop_td001_temporal_doppler import task
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    from rydberg_aoa.simulation import time_domain as td
    drive=object(); seen=[]
    def response(*a,**k):
        seen.append((a[7:10],k['rf_drive']))
        return np.exp(2j*np.pi*np.asarray(a[2]))
    monkeypatch.setattr(td,'run_single_heterodyne_worker',response)
    t=replace(task(plan()),rf_drive=drive)
    result=run_laboratory_worker(t)
    assert result.report['converged']
    assert all(d[2]==t.detunings_rad_per_us[2] and r is drive for d,r in seen)
    assert len({d[0] for d,r in seen})>1


# Real ARC initialization, controlled temporal traces, actual profile/IF/AoA.
from tests.test_lab007_profile_integration import isolated


@pytest.mark.parametrize('mode',['prescribed','experimental'])
@pytest.mark.parametrize('angle',[-30.,30.])
def test_profile_to_if_aoa_and_common_drive(mode,angle,isolated,monkeypatch):
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation import runner, time_domain as td
    class Pool:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def map(self,fn,tasks):return list(map(fn,tasks))
    class Context:
        def Pool(self,**k):return Pool()
    monkeypatch.setattr(runner.multiprocessing,'get_context',lambda *a:Context())
    seen={}
    def response(*args,**kwargs):
        assert kwargs['_coherence_only'] is True
        seen.setdefault(args[0],[]).append(kwargs.get('rf_drive'))
        t=np.asarray(args[2]);phase=np.angle(args[1])
        direction=td.atomic_if_polarity(kwargs['rf_orientation'])
        return .001+1e-12*args[7]+1e-5*np.exp(1j*(direction*2*np.pi*args[3]*t+phase))
    monkeypatch.setattr(td,'run_single_heterodyne_worker',response)
    data=json.loads((ROOT/'examples/lab007'/f'{mode}_full.json').read_text())
    data['spectrum']=None
    data['execution']['target_azimuth_deg']=angle
    data['temporal_doppler']=asdict(controls())
    report,result=execute_profile(parse_run_profile(data))
    assert report['temporal_doppler_ensemble_included']
    assert abs(result['estimated_aoa']-angle)<.2
    assert all(r['independent']['comparison']['accepted'] for r in result['temporal_doppler_reports'])
    assert all(len(v)>1 and all(x is v[0] for x in v) for v in seen.values())
    from rydberg_aoa.visualization.profile_figures import create_profile_figures
    import matplotlib.pyplot as plt
    figures=create_profile_figures(report,result)
    try:
        assert {'fig_time_domain','fig_phase_progression','fig_aoa_spectrum'} <= set(figures)
        figure=figures['fig_time_domain']
        # Displayed values must come from this profile's actual ensemble.
        assert any(len(line.get_ydata())==len(result['tlist']) for line in figure.axes[0].lines)
    finally:
        for figure in figures.values():plt.close(figure)


def test_real_certified_bdf_small_spread_matches_doppler_off():
    from tests.test_dop_td001_temporal_doppler import task
    from rydberg_aoa.simulation.laboratory_execution import run_laboratory_worker
    # Existing controlled broad-line task; near-zero thermal spread is a
    # numerical limiting case, not a prepared-room-temperature benchmark.
    p=replace(plan(),temperature_k=1e-12)
    baseline=run_laboratory_worker(task())
    result=run_laboratory_worker(task(p))
    np.testing.assert_allclose(result.record[4],baseline[4],rtol=1e-5,atol=1e-8)
    np.testing.assert_allclose(result.record[0],baseline[0],rtol=1e-5,atol=1e-8)
    assert result.report['converged']
