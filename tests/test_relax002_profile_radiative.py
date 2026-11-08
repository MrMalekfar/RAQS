"""Explicit profile radiative selection, stale guards, rollback and real spawn."""
import copy
from dataclasses import replace
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rydberg_aoa.config import atomic_config as ac, parameters as params
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_radiative import (
    RadiativeOptions, resolve_profile_radiative, active_radiative_rates,
)
from rydberg_aoa.arc.prepared_decay import PreparedRadiativeGraph
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.profile_execution import execute_profile
from rydberg_aoa.simulation.profile_spectrum import resolve_profile_medium, run_profile_spectrum
from rydberg_aoa.simulation.runner import SimulationRunner
from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state

ROOT=Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT/'examples/relax002'/f'{mode}_prepared_internal.json').read_text())


@pytest.fixture
def isolated():
    state,pipeline=snapshot_published_atomic_state(),ac._snapshot_pipeline_transaction()
    names=('RF_LO_FREQUENCY','RF_LO_E_FIELD','RF_SIGNAL_FREQUENCY','RF_SIGNAL_E_FIELD','RF_FREQUENCY','IF_FREQ')
    scalars={k:getattr(params,k) for k in names}
    ac.reset_atomic_physics()
    try:yield
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        for k,v in scalars.items():setattr(params,k,v)
        restore_published_atomic_state(state)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_selected_declaration_is_immutable_and_leaves_inputs_unchanged(mode):
    data=document(mode);before=copy.deepcopy(data)
    p=parse_run_profile(data)
    assert data==before and p.radiative==RadiativeOptions('selected_internal_spontaneous_e1',.5)
    assert pickle.loads(pickle.dumps(p))==p
    from rydberg_aoa.config.profile_document import profile_document
    exported=profile_document(p)
    assert exported['radiative']==data['radiative']
    assert parse_run_profile(exported)==p


@pytest.mark.parametrize('bad',[None,True,'0.5',.25,np.nan,np.inf,1+0j,np.array(.5),10**1000])
def test_invalid_projection_is_rejected_before_initialization(bad,monkeypatch):
    data=document();data['radiative']['nuclear_projection']=bad
    monkeypatch.setattr(ac,'initialize_atomic_physics',lambda **kw:pytest.fail('initialization reached'))
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(data)


@pytest.mark.parametrize('change',['unknown_model','missing_mi','extra_key','wrong_channel','old_analytical','wrong_type'])
def test_inconsistent_declaration_is_not_repaired(change):
    data=document()
    if change=='unknown_model':data['radiative']['model']='full_atomic_lifetime'
    elif change=='missing_mi':del data['radiative']['nuclear_projection']
    elif change=='extra_key':data['radiative']['fallback']=True
    elif change=='wrong_channel':data['radiative']['nuclear_projection']=1.5
    elif change=='old_analytical':data['spectrum']['analytical_enabled']=True
    else:data['radiative']=None
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(data)


def test_runtime_api_cannot_bypass_parse_validation(isolated):
    p=parse_run_profile(document())
    bad=replace(p,radiative=RadiativeOptions('selected_internal_spontaneous_e1',1.5))
    with pytest.raises(PhysicsConfigurationError,match='mI'):execute_profile(bad,diagnostics_only=True)
    assert not ac.is_atomic_physics_initialized()


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_real_initialization_report_and_no_stale_rate_rebuilding(mode,isolated,monkeypatch):
    p=parse_run_profile(document(mode));report,result=execute_profile(p,diagnostics_only=True)
    assert result is None and report['radiative_model']['model']=='selected_internal_spontaneous_e1'
    assert report['decay_rates_per_us']['rydberg2->ground']>0
    assert not report['radiative_model']['closed_physical_space']
    assert report['radiative_model']['outside_space_rates_per_us'] is None
    receipt=resolve_profile_radiative(p,ac)
    medium,cfg=resolve_profile_medium(p,ac);cfg=replace(cfg,radiative=receipt)
    request=p.request if mode=='experimental' else p
    runtime=SimulationRunner._current_explicit_runtime(request,mode)
    from rydberg_aoa.arc import prepared_decay
    monkeypatch.setattr(prepared_decay,'calculate_prepared_radiative_graph',lambda *a,**k:pytest.fail('consumer rebuilt rates'))
    assert active_radiative_rates(cfg,runtime,ac,p) is receipt.graph
    restored=pickle.loads(pickle.dumps(cfg))
    assert active_radiative_rates(restored,runtime,ac,p)==receipt.graph
    assert report['applied_fields']==document(mode)['fields']
    assert ac.GAMMA_21==report['initialization_fs_decay_rates_per_us'][0]
    assert report['radiative_model']['nuclear_projection']==.5


def test_missing_selected_graph_is_not_a_legacy_fallback(isolated):
    p=parse_run_profile(document());execute_profile(p,diagnostics_only=True)
    _,cfg=resolve_profile_medium(p,ac)
    runtime=SimulationRunner._current_explicit_runtime(p,'prescribed')
    with pytest.raises(PhysicsConfigurationError,match='execution graph'):
        active_radiative_rates(cfg,runtime,ac,p)


def test_experimental_without_spectrum_cannot_silently_drop_the_selected_graph(isolated):
    data=document('experimental');data['spectrum']=None
    p=parse_run_profile(data);execute_profile(p,diagnostics_only=True)
    _,cfg=resolve_profile_medium(p,ac)
    with pytest.raises(PhysicsConfigurationError,match='execution graph'):
        SimulationRunner().run_laboratory_simulation(p.request,cfg,radiative_options=p.radiative)


def test_resolved_config_export_preserves_declaration_not_runtime_receipt(isolated):
    from rydberg_aoa.config.profile_document import profile_document
    p=parse_run_profile(document());execute_profile(p,diagnostics_only=True)
    receipt=resolve_profile_radiative(p,ac);_,cfg=resolve_profile_medium(p,ac)
    resolved=replace(p,run_config=replace(cfg,radiative=receipt))
    data=profile_document(resolved)
    assert data['radiative']==document()['radiative'] and 'radiative' not in data['execution']
    # A thermal profile has a declaration-only run plan until execution.
    assert parse_run_profile(data).radiative==p.radiative
    with pytest.raises(PhysicsConfigurationError,match='matching declaration'):
        profile_document(replace(resolved,radiative=None))


@pytest.mark.parametrize('change',['atom','runtime','states','declaration','undeclared'])
def test_stale_receipt_is_rejected(change,isolated,monkeypatch):
    p=parse_run_profile(document());execute_profile(p,diagnostics_only=True)
    receipt=resolve_profile_radiative(p,ac);_,cfg=resolve_profile_medium(p,ac)
    cfg=replace(cfg,radiative=receipt)
    runtime=SimulationRunner._current_explicit_runtime(p,'prescribed')
    if change=='atom':monkeypatch.setattr(ac.ATOM,'I',1.5)
    elif change=='runtime':runtime=replace(runtime,rates=(1.,2.,3.))
    elif change=='states':
        changed=copy.deepcopy(ac.ATOMIC_LEVELS);changed['rydberg1']['mj']=.5
        monkeypatch.setattr(ac,'ATOMIC_LEVELS',changed)
    elif change=='declaration':p=replace(p,radiative=RadiativeOptions('legacy_fs_three_edge',None))
    else:p=replace(p,radiative=None)
    with pytest.raises((PhysicsConfigurationError,PhysicsCalculationError)):
        active_radiative_rates(cfg,runtime,ac,p)


def test_failed_graph_build_rolls_back_canonical_initialization(isolated,monkeypatch):
    from rydberg_aoa.arc import prepared_decay
    cause=PhysicsCalculationError('injected prepared-rate failure')
    def fail(*a,**k):raise cause
    before=ac._snapshot_pipeline_transaction()
    lasers=copy.deepcopy(ac.LASER_PARAMS)
    probe_ref,coupling_ref=ac.LASER_PARAMS['probe'],ac.LASER_PARAMS['coupling']
    monkeypatch.setattr(prepared_decay,'calculate_prepared_radiative_graph',fail)
    with pytest.raises(PhysicsCalculationError) as e:
        execute_profile(parse_run_profile(document()),diagnostics_only=True)
    assert e.value is cause and not ac.is_atomic_physics_initialized()
    assert ac.LASER_PARAMS==lasers
    assert ac.LASER_PARAMS['probe'] is probe_ref and ac.LASER_PARAMS['coupling'] is coupling_ref
    for key,cache in [('atomic_levels_cache',ac._ATOMIC_LEVELS_CACHE),('detuning_cache',ac._DETUNING_CACHE),
                      ('dipole_matrix_cache',ac._DIPOLE_MATRIX_CACHE),('optimization_results_cache',ac._OPTIMIZATION_RESULTS_CACHE)]:
        assert cache==before[key]


def test_spectrum_forwards_the_exact_graph_to_both_rf_cases(isolated,monkeypatch):
    from rydberg_aoa.simulation import steady_state
    p=parse_run_profile(document());execute_profile(p,diagnostics_only=True)
    receipt=resolve_profile_radiative(p,ac);medium,cfg=resolve_profile_medium(p,ac)
    cfg=replace(cfg,radiative=receipt);seen=[]
    def capture(omega,grid,omegas,deltas,rates,*a,**kw):
        seen.append(rates)
        assert isinstance(rates,PreparedRadiativeGraph) and not kw['analytical_enabled']
        return np.asarray(grid),np.ones(len(grid)),None
    monkeypatch.setattr(steady_state,'simulate_spectrum',capture)
    result=run_profile_spectrum(p,cfg,medium,ac)
    assert seen==[receipt.graph,receipt.graph]
    assert all(g is receipt.graph for g in seen)
    assert result['radiative_model']==receipt.graph.report()


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_real_worker_tasks_receive_the_same_graph_as_the_spectrum(mode,isolated,monkeypatch):
    from types import SimpleNamespace
    from rydberg_aoa.simulation import runner as runner_module
    p=parse_run_profile(document(mode));execute_profile(p,diagnostics_only=True)
    receipt=resolve_profile_radiative(p,ac);_,cfg=resolve_profile_medium(p,ac)
    cfg=replace(cfg,radiative=receipt);seen=[]
    class Pool:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def map(self,worker,tasks):
            seen.extend(tasks)
            # Real worker/solver; the separate CLI tests exercise real spawn transport.
            return [worker(task) for task in tasks]
    def context(name):
        assert name=='spawn'
        return SimpleNamespace(Pool=lambda processes:Pool())
    monkeypatch.setattr(runner_module.multiprocessing,'get_context',context)
    runner=SimulationRunner()
    if mode=='experimental':
        result=runner.run_laboratory_simulation(p.request,cfg,radiative_options=p.radiative)
    else:result=runner.run_prescribed_simulation(p,cfg)
    assert len(seen)==3 and all(t.decay_rates_per_us is receipt.graph for t in seen)
    assert result['radiative_model']==receipt.graph.report()
    assert np.all(np.isfinite(result['complex_responses']))


def test_profile_parsing_needs_no_atomic_or_worker_import():
    code='''
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in ('arc','rydberg_aoa.config.parameters','rydberg_aoa.config.atomic_config','rydberg_aoa.simulation.time_domain'):
   raise AssertionError('Unexpected import '+fullname)
sys.meta_path.insert(0,Guard())
from rydberg_aoa.config.run_profile import load_run_profile
for mode in ('prescribed','experimental'):
 p=load_run_profile('examples/relax002/'+mode+'_prepared_internal.json')
 assert p.radiative.nuclear_projection==.5
'''
    run=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_real_cli_spawn_spectrum_if_and_saved_arrays(mode,tmp_path):
    # Bound the regression, retaining actual ARC initialization, QuTiP, spawn and IF/AoA.
    data=document(mode)
    data['spectrum']['doppler_enabled']=False
    data['spectrum']['probe_offsets_mhz']=[-.5,0.,.5]
    src=tmp_path/'input.json';src.write_text(json.dumps(data),encoding='utf-8')
    out=tmp_path/'result'
    run=subprocess.run([sys.executable,'-B','main.py','--profile',str(src),'--no-plots','--output-dir',str(out)],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=150)
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads((out/'run_report.json').read_text(encoding='utf-8'))
    assert report['radiative_model']['model']=='selected_internal_spontaneous_e1'
    assert report['decay_rates_per_us']['rydberg2->ground']>0
    assert report['sensing']['paths']==3 and report['sensing']['samples_per_path']==601
    assert np.isfinite(report['sensing']['estimated_aoa_deg'])
    assert report['applied_fields']==data['fields']
    assert report['initialization_fs_decay_rates_per_us']!=list(report['decay_rates_per_us'].values())
    with np.load(out/'simulation_arrays.npz',allow_pickle=False) as arrays:
        assert arrays['time_traces'].shape==(3,601)
        assert arrays['spectrum_rf_on'].shape==(3,)
        assert np.all(np.isfinite(arrays['complex_responses']))
        for name in arrays.files:assert arrays[name].dtype.kind!='O',name


def test_legacy_default_is_reported_without_changing_the_three_rates(isolated):
    data=document();del data['radiative']
    p=parse_run_profile(data);report,_=execute_profile(p,diagnostics_only=True)
    assert p.radiative is None
    assert report['radiative_model']['model']=='legacy_fs_three_edge'
    assert report['decay_rates_per_us']==(ac.GAMMA_21,ac.GAMMA_32,ac.GAMMA_43)
