"""Prepared radiative graphs: angular oracles, actual solvers, no closure claim."""
import copy
import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt
from sympy import Rational

from rydberg_aoa.arc.prepared_decay import (
    PreparedDecayEdge, _branch, calculate_prepared_radiative_graph,
)
from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.physics.hamiltonian import create_sensor_model
from rydberg_aoa.simulation.steady_state import (
    _solve_probe_coherence, _homogeneous_gamma_scale, simulate_spectrum,
)
from rydberg_aoa.simulation import time_domain as td


def levels(species='Cs133'):
    if species == 'Cs133':
        return dict(ground=dict(n=6,l=0,j=.5,f=4,mf=0),
                    excited=dict(n=6,l=1,j=1.5,f=5,mf=0),
                    rydberg1=dict(n=47,l=2,j=2.5,mj=-.5),
                    rydberg2=dict(n=48,l=1,j=1.5,mj=-.5))
    return dict(ground=dict(n=5,l=0,j=.5,f=3,mf=3),
                excited=dict(n=5,l=1,j=1.5,f=4,mf=4),
                rydberg1=dict(n=26,l=2,j=2.5,mj=2.5),
                rydberg2=dict(n=27,l=1,j=1.5,mj=1.5))


class Atom:
    """Synthetic FS energies/rates; unrelated to physical spectroscopy."""
    I = 3.5

    def __init__(self):
        self.calls = []
        self.energy = {(6,0,.5):0., (6,1,1.5):100., (47,2,2.5):200., (48,1,1.5):190.}

    def getTransitionFrequency(self, *args):
        return self.energy[args[3:]] - self.energy[args[:3]]

    def getTransitionRate(self, *args, temperature):
        assert temperature == 0.
        assert self.getTransitionFrequency(*args) < 0
        self.calls.append(args)
        return 1e6


@pytest.fixture
def graph():
    return calculate_prepared_radiative_graph(Atom(), levels(), nuclear_projection=.5)


def test_exact_branch_numbers_and_nonadjacent_internal_exit(graph):
    result = {(e.source,e.target):e for e in graph.edges}
    assert set(result) == {(1,0),(2,1),(2,3),(3,0)}
    for pair, branch in [((1,0),5/9),((2,1),1/4),((2,3),3/5),((3,0),1/3)]:
        assert result[pair].branch == branch
        assert result[pair].rate_per_us == branch
    assert graph.outgoing_rates_per_us == pytest.approx((0,5/9,1/4+3/5,1/3))


def test_exact_sum_over_all_hyperfine_destinations_recovers_one_fs_line():
    # Cs47D mJ=-1/2,mI=+1/2 -> every Cs6P3/2 F,mF, not just the driven F=5,mF=0.
    src=(47,2,2.5,None,None,-.5)
    total=sum((_branch(src,(6,1,1.5,float(f),float(m),None),Rational(7,2),Rational(1,2))
               for f in range(2,6) for m in range(-f,f+1)),Rational(0))
    assert total == 1


def test_no_mutation_magnetic_search_or_lifetime_substitution(graph):
    a=Atom(); before=copy.deepcopy(levels()); declared=copy.deepcopy(before)
    g=calculate_prepared_radiative_graph(a,declared,nuclear_projection=.5)
    assert declared==before and len(a.calls)==4
    assert g == graph and pickle.loads(pickle.dumps(g)) == g
    report=json.loads(json.dumps(g.report()))
    assert not report['closed_physical_space']
    assert report['outside_space_rates_per_us'] is None
    assert report['outside_return_model'] is None
    assert report['temperature_k']==0.


@pytest.mark.parametrize('value',[None,True,np.bool_(True),'0.5',.25,4.5,np.nan,np.inf,1+0j,np.array(.5),10**1000])
def test_missing_or_invalid_nuclear_projection_fails_loudly(value):
    with pytest.raises(PhysicsConfigurationError):
        calculate_prepared_radiative_graph(Atom(),levels(),nuclear_projection=value)


@pytest.mark.parametrize('role,field,value',[
    ('ground','n',True),('ground','l',6),('ground','f',1),('ground','mf',.5),
    ('ground','mj',.5),('excited','f',None),('rydberg1','mj',None),
    ('rydberg1','mj',3.5),('rydberg2','j',2.),('rydberg2','f',2.),
])
def test_incomplete_or_invalid_states_are_not_repaired(role,field,value):
    ls=levels(); ls[role][field]=value
    with pytest.raises(PhysicsConfigurationError):
        calculate_prepared_radiative_graph(Atom(),ls,nuclear_projection=.5)


@pytest.mark.parametrize('bad',[True,'1',1+0j,np.array(1.),-1.,np.nan,np.inf])
def test_invalid_arc_rates_fail_before_float_coercion(bad):
    a=Atom(); a.getTransitionRate=lambda *args,**kw:bad
    with pytest.raises(PhysicsCalculationError):
        calculate_prepared_radiative_graph(a,levels(),nuclear_projection=.5)


@pytest.mark.parametrize('operation',['getTransitionRate','getTransitionFrequency'])
def test_arc_exception_keeps_context_and_original_cause(operation):
    a=Atom(); cause=OSError('broken atomic database')
    def broken(*args,**kw): raise cause
    setattr(a,operation,broken)
    with pytest.raises(PhysicsCalculationError,match=operation) as e:
        calculate_prepared_radiative_graph(a,levels(),nuclear_projection=.5)
    assert e.value.__cause__ is cause


def test_existing_physics_exception_is_not_remasked():
    a=Atom(); cause=PhysicsCalculationError('typed atomic failure')
    def broken(*args,**kw): raise cause
    a.getTransitionRate=broken
    with pytest.raises(PhysicsCalculationError) as e:
        calculate_prepared_radiative_graph(a,levels(),nuclear_projection=.5)
    assert e.value is cause


def test_valid_zero_and_subfloor_rates_are_preserved():
    for raw in (0.,1e-5):
        a=Atom(); a.getTransitionRate=lambda *args,**kw:raw
        g=calculate_prepared_radiative_graph(a,levels(),nuclear_projection=.5)
        assert all(e.fs_rate_per_us==raw/1e6 for e in g.edges)


def test_unrepresentable_positive_rate_fails_without_becoming_zero():
    a=Atom();a.getTransitionRate=lambda *args,**kw:1e-320
    with pytest.raises(PhysicsCalculationError,match='underflow'):
        calculate_prepared_radiative_graph(a,levels(),nuclear_projection=.5)
    with pytest.raises(PhysicsCalculationError,match='underflow'):
        _=PreparedDecayEdge(1,0,1e-300,1e-50).rate_per_us


def test_angular_zero_remains_zero_without_another_channel():
    ls=levels();ls['ground']['mf']=3
    g=calculate_prepared_radiative_graph(Atom(),ls,nuclear_projection=.5)
    edge=next(e for e in g.edges if (e.source,e.target)==(1,0))
    assert edge.fs_rate_per_us==1. and edge.branch==0. and edge.rate_per_us==0.


def model(graph,**kw):
    return create_sensor_model(.7,.9,.6,.13,-.21,-.37,graph,[.05,.04,.03],
                               transit_rate_per_us=0.,**kw)


def test_graph_replaces_all_legacy_jumps_and_keeps_the_hamiltonian(graph):
    h,c,obs=model(graph)
    old_h,_,old_obs=model([0.,0.,0.],rf_orientation=graph.rf_orientation)
    assert np.array_equal(h.full(),old_h.full())
    assert np.array_equal(obs.full(),old_obs.full())
    assert len(c)==len(graph.edges)+3
    for edge,op in zip(graph.edges,c):
        oracle=np.zeros((4,4),complex)
        oracle[edge.target,edge.source]=np.sqrt(edge.fs_rate_per_us*edge.branch)
        assert np.array_equal(op.full(),oracle)
    assert _homogeneous_gamma_scale(graph,[0,0,0],0)==max(graph.outgoing_rates_per_us)


def test_population_ledger_and_lindblad_trace_use_partial_rates_exactly(graph):
    _,c,_=model(graph);L=qt.liouvillian(qt.qzero(4),c)
    for i in range(4):
        rho=qt.basis(4,i).proj()
        drho=qt.vector_to_operator(L*qt.operator_to_vector(rho)).full()
        expected=np.zeros(4)
        for e in graph.edges:
            if e.source==i:
                expected[e.target]+=e.rate_per_us
                expected[e.source]-=e.rate_per_us
        assert np.diag(drho)==pytest.approx(expected)
        assert np.trace(drho)==pytest.approx(0.,abs=1e-15)


def test_conflicting_orientation_is_rejected_before_solving(graph):
    with pytest.raises(PhysicsConfigurationError,match='orientation'):
        model(graph,rf_orientation=resolve_rf_orientation(10.))
    with pytest.raises(PhysicsConfigurationError,match='orientation'):
        td.run_single_heterodyne_worker(0,.01,[0,1,2],.1,.7,.9,.6,.13,-.21,-.37,
            graph,[.05,.04,.03],transit_rate_per_us=0.,rf_orientation=resolve_rf_orientation(10.))


def test_bad_duplicate_or_reverse_graph_edges_are_not_allowed(graph):
    for edges in (graph.edges+(graph.edges[0],),
                  graph.edges+(PreparedDecayEdge(0,1,1.,1.),)):
        with pytest.raises(PhysicsConfigurationError):replace(graph,edges=edges)


def test_real_steady_state_backends_use_the_same_graph(graph):
    h,c,obs=model(graph);reference=qt.expect(obs,qt.steadystate(h,c))
    for backend in ('qutip','direct'):
        value=_solve_probe_coherence(.13,-.21,.7,.9,.6,-.37,graph,[.05,.04,.03],
                                    transit_rate_per_us=0.,steady_state_backend=backend)
        assert value==pytest.approx(reference,abs=1e-13)


def test_real_temporal_worker_certifies_graph_and_uses_its_rf_phase(graph,monkeypatch):
    h,c,obs=model(graph);reference=qt.expect(obs,qt.steadystate(h,c))
    calls=[]; original=td.signal_coefficient
    def spy(t,omega,frequency,orientation=None):
        calls.append(orientation)
        return original(t,omega,frequency,orientation)
    monkeypatch.setattr(td,'signal_coefficient',spy)
    trace=td.run_single_heterodyne_worker(0,0j,np.linspace(0,3,31),.1,.7,.9,.6,.13,-.21,-.37,
            graph,[.05,.04,.03],transit_rate_per_us=0.,_coherence_only=True)
    assert np.asarray(trace).shape==(31,)
    assert np.asarray(trace)==pytest.approx(np.full(31,reference),abs=1e-10)
    assert calls and all(o==graph.rf_orientation for o in calls)


def test_nonzero_signal_matches_independent_downward_rwa_time_evolution(graph):
    from rydberg_aoa.config.run_numerics import TemporalODE
    times=np.linspace(0,30,301);signal=.01+.02j;frequency=.1
    h,c,obs=model(graph);rho0=qt.steadystate(h,c)
    raising=qt.basis(4,3)*qt.basis(4,2).dag()
    # Independent downward-energy RWA coefficients: H43 uses exp(-i omega_IF t).
    h_time=[h,[-.5*raising,lambda t:signal*np.exp(-2j*np.pi*frequency*t)],
            [-.5*raising.dag(),lambda t:np.conj(signal)*np.exp(2j*np.pi*frequency*t)]]
    reference=qt.mesolve(h_time,rho0,times,c,e_ops=[obs],
        options=dict(method='bdf',nsteps=5000,normalize_output=False,atol=1e-11,rtol=1e-9)).expect[0]
    actual=td.run_single_heterodyne_worker(0,signal,times,frequency,.7,.9,.6,.13,-.21,-.37,
            graph,[.05,.04,.03],transit_rate_per_us=0.,_coherence_only=True,
            temporal_ode=TemporalODE('bdf',1e-9,1e-11,5000))
    assert np.max(np.abs(np.asarray(actual)-reference)) < 1e-7
    assert np.ptp(np.imag(actual)) > 1e-4


def test_old_reduced_analytical_formula_cannot_claim_graph_parity(graph):
    with pytest.raises(PhysicsConfigurationError,match='analytical'):
        simulate_spectrum(.6,[-1,0,1],[.7,.9,.6],[.13,-.21,-.37],graph,[0,0,0],0.)


def test_upward_rf_pair_uses_the_opposite_spontaneous_edge():
    a=Atom();a.energy[(48,1,1.5)]=210.
    g=calculate_prepared_radiative_graph(a,levels(),nuclear_projection=.5)
    e=next(e for e in g.edges if {e.source,e.target}=={2,3})
    assert (e.source,e.target)==(3,2)
    assert g.rf_orientation.is_upward and e.branch==pytest.approx(.4)
    h,_,_=model(g)
    h_old,_,_=model([0,0,0],rf_orientation=g.rf_orientation)
    assert np.array_equal(h.full(),h_old.full())


def test_numerical_spectrum_accepts_graph_without_legacy_analytical_branch(graph):
    settings=SimpleNamespace(atom_mass_kg=2e-25,temperature_k=0.,method='gauss_hermite',
                             n_classes=1,n_sigma=5.,counter_propagating=True,rtol=1e-5,atol=1e-9)
    grid,transmission,analytical=simulate_spectrum(.6,[-.1,0,.1],[.7,.9,.6],
        [.13,-.21,-.37],graph,[.05,.04,.03],0.,doppler_enabled=False,transit_enabled=False,
        analytical_enabled=False,doppler_settings=settings,transmission_prefactors=(.1,.2))
    assert np.array_equal(grid,[-.1,0,.1])
    assert np.all(np.isfinite(transmission)) and analytical is None


@pytest.mark.parametrize('species,I,mi',[('Rb85',2.5,2.5),('Cs133',3.5,.5)])
def test_live_arc_partial_rates_agree_with_full_state_dipoles(species,I,mi):
    from arc import Rubidium85,Cesium
    atom=Rubidium85() if species=='Rb85' else Cesium()
    g=calculate_prepared_radiative_graph(atom,levels(species),nuclear_projection=mi)
    assert atom.I==I
    for e in g.edges:
        s,t=g.states[e.source],g.states[e.target]
        ms=s[4] if s[3] is not None else s[5]+mi
        mt=t[4] if t[3] is not None else t[5]+mi
        q=int(mt-ms)
        if s[3] is not None:
            d=atom.getDipoleMatrixElementHFS(*s[:3],s[3],s[4],*t[:3],t[3],t[4],q)
        elif t[3] is not None:
            d=atom.getDipoleMatrixElementHFStoFS(*t[:3],t[3],t[4],*s[:3],s[5],-q)
        else:
            d=atom.getDipoleMatrixElement(*s[:3],s[5],*t[:3],t[5],q)
        reduced=atom.getReducedMatrixElementJ(*s[:3],*t[:3])
        probability=abs(d)**2/(abs(reduced)**2/(2*s[2]+1))
        fs_rate=atom.getTransitionRate(*s[:3],*t[:3],temperature=0.)/1e6
        assert e.branch==pytest.approx(probability,abs=5e-13)
        assert e.rate_per_us==pytest.approx(fs_rate*probability,rel=5e-12)
    assert any((e.source,e.target)==(3,0) and e.rate_per_us>0 for e in g.edges)
