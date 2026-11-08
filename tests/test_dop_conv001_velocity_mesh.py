"""Independent quadrature oracles; synthetic coherence is not an ARC reference."""
import json
import math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pytest
from scipy.integrate import quad
from rydberg_aoa.config.temporal_doppler import SinhTemporalDoppler,ResolvedTemporalDoppler,parse_temporal_doppler
from rydberg_aoa.physics.doppler.velocity_classes import generate_velocity_classes,get_velocity_std
from rydberg_aoa.simulation.temporal_doppler import ensemble_coherence
from rydberg_aoa.exceptions import PhysicsConfigurationError,PhysicsCalculationError

ROOT = Path(__file__).resolve().parents[1]


def controls(scale=1.,**extra):
    return SinhTemporalDoppler(**{**dict(enabled=True,method='sinh',initial_classes=17,
        max_classes=257,n_sigma=5.,rtol=.01,atol=1e-10,counter_propagating=True,
        velocity_scale_ms=scale),**extra})


def nodes(n=257,scale=1.):
    return generate_velocity_classes(n,300.,1.409993e-25,'sinh',5.,scale)


def test_geometry_normalization_and_exact_nested_cache_keys():
    v,w = nodes(33); u,z = nodes(65)
    assert np.array_equal(v,u[::2])
    assert np.all(np.diff(v)>0) and np.array_equal(v,-v[::-1])
    assert v[16] == 0. and np.all(w>0.)
    assert w.sum() == pytest.approx(1.,abs=1e-14)
    assert v[-1] == 5*get_velocity_std(300.,1.409993e-25)


def test_maxwell_variance_independent_closed_form():
    v,w = nodes(); sigma = get_velocity_std(300.,1.409993e-25)
    norm = math.erf(5/math.sqrt(2))
    expected = sigma**2*(1-2*5*math.exp(-.5*25)/math.sqrt(2*math.pi)/norm)
    assert np.dot(w,v*v) == pytest.approx(expected,rel=1e-5)


@pytest.mark.parametrize('scale',[.5,1.,2.])
@pytest.mark.parametrize('center',[0.,3.25])
def test_narrow_response_matches_independent_integral(scale,center):
    v,w = nodes(513,scale); sigma = get_velocity_std(300.,1.409993e-25)
    width = .3; limit=5*sigma
    def integrand(x):
        return math.exp(-.5*(x/sigma)**2)/(sigma*math.sqrt(2*math.pi)) / (1+((x-center)/width)**2)
    expected = quad(integrand,-limit,limit,points=[center-width,center,center+width],epsabs=1e-12,epsrel=1e-10)[0]/math.erf(5/math.sqrt(2))
    assert np.dot(w,1/(1+((v-center)/width)**2)) == pytest.approx(expected,rel=2e-6)


@pytest.mark.parametrize('value',[None,0.,-1.,True,'1',np.array(1.),float('nan'),float('inf')])
def test_invalid_scale_rejected(value):
    with pytest.raises(PhysicsConfigurationError): controls(value)


def test_method_specific_json_and_legacy_shape_preserved():
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.config.profile_document import profile_document
    d = json.loads((ROOT/'examples/num_ode002/rb85_prescribed.json').read_text())
    legacy = parse_run_profile(d)
    assert 'velocity_scale_ms' not in asdict(legacy.temporal_doppler)
    d['temporal_doppler'] = asdict(controls())
    p = parse_run_profile(d)
    assert parse_run_profile(profile_document(p)) == p
    missing = dict(d['temporal_doppler']); del missing['velocity_scale_ms']
    with pytest.raises(PhysicsConfigurationError): parse_temporal_doppler(missing)
    bad = dict(d['temporal_doppler'],method='uniform')
    with pytest.raises(PhysicsConfigurationError): parse_temporal_doppler(bad)


@pytest.mark.parametrize('method',[None,True,['sinh'],np.array(['sinh','uniform'])])
def test_non_string_method_is_configuration_error(method):
    document=asdict(controls()); document['method']=method
    with pytest.raises(PhysicsConfigurationError):parse_temporal_doppler(document)


def test_two_successive_refinements_and_cost_counts():
    plan = ResolvedTemporalDoppler(controls(),300.,1.409993e-25,780e-9,483e-9)
    times = np.linspace(0.,20.,101)
    def response(dp,dc):
        v = -dp*780e-9*1e6/(2*math.pi)
        return (1+.2j)*np.exp(2j*math.pi*.1*times)/(1+(v/.3)**2)
    average,report = ensemble_coherence(plan,times,.1,0.,response)
    assert report['converged'] and report['history'][-1]['passed'] and report['history'][-2]['passed']
    assert report['class_solves'] == report['classes']
    assert len(report['class_costs']) == report['class_solves']
    assert report['solve_wall_seconds'] >= 0.
    assert report['truncated_tail_mass'] == pytest.approx(math.erfc(5/math.sqrt(2)))
    assert np.isfinite(average).all()


def test_failed_class_not_dropped_and_cause_kept():
    plan = ResolvedTemporalDoppler(controls(),300.,1.409993e-25,780e-9,483e-9)
    cause = PhysicsCalculationError('injected real solver failure')
    def solve(*shifts): raise cause
    with pytest.raises(PhysicsCalculationError,match='no dropped class') as caught:
        ensemble_coherence(plan,np.linspace(0.,20.,101),.1,0.,solve)
    assert caught.value.__cause__ is cause
    assert caught.value.velocity_diagnostics['failed_class_wall_seconds'] >= 0.
    import pickle
    transported=pickle.loads(pickle.dumps(caught.value))
    assert transported.velocity_diagnostics==caught.value.velocity_diagnostics


def test_exhausted_mesh_preserves_measured_costs_without_response():
    from dataclasses import replace
    plan=ResolvedTemporalDoppler(replace(controls(),initial_classes=3,max_classes=9),
        300.,1.409993e-25,780e-9,483e-9)
    times=np.linspace(0.,20.,101)
    def narrow(dp,dc):
        return np.exp(-(dp/.01)**2)*np.exp(2j*np.pi*.1*times)
    with pytest.raises(PhysicsCalculationError,match='did not converge') as caught:
        ensemble_coherence(plan,times,.1,0.,narrow)
    evidence=caught.value.velocity_diagnostics
    assert evidence['total_class_solves']==9
    assert len(evidence['completed_class_costs'])==9
    assert [r['classes'] for r in evidence['history']]==[3,5,9]
    assert evidence['solve_wall_seconds']>=0.


def test_velocity_std_matches_declared_one_dimensional_formula():
    # Preserve the project's rounded constant; changing it is physical scope.
    expected = math.sqrt(1.38e-23*300./1.409993e-25)
    assert get_velocity_std(300.,1.409993e-25) == pytest.approx(expected,rel=1e-14)


@pytest.mark.parametrize('value',[0,6,True,2.5,'2'])
def test_invalid_bdf_order_rejected(value):
    from rydberg_aoa.config.run_numerics import parse_temporal_ode
    with pytest.raises(PhysicsConfigurationError):
        parse_temporal_ode(dict(method='bdf',rtol=1e-10,atol=1e-12,nsteps=20000,max_order=value))


def test_explicit_order_roundtrip_and_real_worker_options(monkeypatch):
    from rydberg_aoa.config.run_profile import parse_run_profile
    from rydberg_aoa.config.profile_document import profile_document
    from rydberg_aoa.simulation import time_domain as td
    d=json.loads((ROOT/'examples/dop_conv001/rb85_sinh_scale1_order2.json').read_text())
    profile=parse_run_profile(d)
    assert parse_run_profile(profile_document(profile)) == profile
    seen=[]; original=td.mesolve
    def spy(*a,**kw):
        seen.append(dict(kw['options'])); return original(*a,**kw)
    monkeypatch.setattr(td,'mesolve',spy)
    trace=td.run_single_heterodyne_worker(0,1+.5j,np.linspace(0.,20.,201),1.,
        5.,3.,2.,.3,-.2,.1,[1.,1.,1.],[.2,.2,.2],transit_rate_per_us=0.,_coherence_only=True,
        temporal_ode=profile.numerics.temporal_ode)
    assert seen[0]['order'] == 2 and seen[0]['normalize_output'] is False
    assert trace.shape==(201,) and np.isfinite(trace).all()
