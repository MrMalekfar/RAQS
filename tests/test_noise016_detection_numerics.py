"""Diagnostic option enforcement, noise coupling and canonical live campaigns."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import poisson
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.study import plan_study
from rydberg_aoa.config.detection_numerics import SolverDiagnostic
from rydberg_aoa.simulation.temporal_diagnostic import diagnostic_grid, run_diagnostic_worker
from rydberg_aoa.simulation.detector_readout import poisson_quantile_counts
from rydberg_aoa.simulation import study_execution as engine
from rydberg_aoa.simulation import detection_validation as validation
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from tests.test_noise014_detection_study import fake_job

ROOT=Path(__file__).resolve().parents[1]


def diagnostic_receipt(solver):
    from dataclasses import asdict
    return dict(solver=None if solver is None else asdict(solver),poisson_quantiles=True,
        acquisition_grid_unchanged=True,solver_method='bdf',nsteps=5000,normalize_output=False,
        option_enforcement='not instrumented' if solver is None else 'spawn-worker mesolve wrapper verifies options and restores on exception')


def document(mode='prescribed'):
    return json.loads((ROOT/f'examples/noise016/{mode}_numerical_validation_pin.json').read_text())


def small_profile(mode='prescribed'):
    d=document(mode);d['study'].update(values=[.001,.005],calibration_trials=9,null_validation_trials=5,trials_per_point=4)
    d['study']['numerical_validation']['cases']=[dict(name='tight_dense',rtol=1e-8,atol=1e-10,output_subdivision=2)]
    return parse_run_profile(d)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_plan_roundtrip_and_same_acquisition_aperture(mode):
    p=small_profile(mode);q=parse_run_profile(profile_document(p))
    assert p==q
    plan=plan_study(p)['numerical_validation']
    assert plan['complete_detection_campaigns']==3 and plan['total_trials']==66
    assert plan['certification_samples_per_case']==[1201]
    grid=diagnostic_grid(p.run_config.tlist_us,2)
    assert np.array_equal(grid[::2],p.run_config.tlist_us)
    assert p.noise.detector.integration_time_us==pytest.approx(.1)


@pytest.mark.parametrize('field,value',[('rtol',True),('atol',0.),('rtol',float('nan')),('output_subdivision',False),('name','')])
def test_invalid_solver_cases_rejected_before_dispatch(field,value):
    d=document();d['study']['numerical_validation']['cases'][0][field]=value
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


def test_poisson_quantiles_independent_cdf_oracle_and_zero_endpoint():
    u=np.array([0.,.01,.1,.5,.9,.999])
    m=np.array([0.,.1,1.,10.,100.,1e4])
    k=poisson_quantile_counts(u,m)
    assert k.dtype==np.int64 and k[0]==0
    assert np.all(poisson.cdf(k[1:],m[1:])>=u[1:])
    assert np.all(poisson.cdf(k[1:]-1,m[1:])<u[1:])
    assert np.array_equal(k,poisson_quantile_counts(u,m))
    assert np.all(poisson_quantile_counts(u,m*2)>=k)


@pytest.mark.parametrize('u,m',[([True],[1.]),([1.],[1.]),([-1.],[1.]),([.5],[float('inf')]),([.5],[-1.]),([.5,.6],[1.])])
def test_invalid_quantile_input_has_no_clipped_or_gaussian_counts(u,m):
    with pytest.raises(PhysicsCalculationError):poisson_quantile_counts(u,m)


def task():
    from types import SimpleNamespace
    return SimpleNamespace(solver_diagnostic=SolverDiagnostic('test',1e-8,1e-10,2),tlist_us=tuple(np.linspace(0.,20.,201)),
                           signed_if_mhz=.1,transient_delay_us=0.)


def test_spawn_local_solver_wrapper_restores_and_selects_exact_samples(monkeypatch):
    from rydberg_aoa.simulation import time_domain as td
    t=task();seen=[]
    def solve(*a,**k):seen.append(k['options'])
    monkeypatch.setattr(td,'mesolve',solve)
    def invoke(grid):
        td.mesolve(options={'method':'bdf','nsteps':5000,'normalize_output':False})
        trace=np.cos(2*np.pi*.1*np.asarray(grid))
        return 99j,np.array([0.]),np.array([0j]),0,trace
    record=run_diagnostic_worker(t,invoke)
    assert td.mesolve is solve and seen[0]['rtol']==1e-8 and seen[0]['atol']==1e-10
    assert len(record[4])==201 and np.array_equal(record[4],np.cos(2*np.pi*.1*np.array(diagnostic_grid(t.tlist_us,2)))[::2])
    assert abs(record[0]-1)<1e-12
    sentinel=RuntimeError('injected')
    def fail(grid):raise sentinel
    with pytest.raises(RuntimeError) as e:run_diagnostic_worker(t,fail)
    assert e.value is sentinel and td.mesolve is solve
    with pytest.raises(PhysicsCalculationError,match='instrumented'):
        run_diagnostic_worker(t,lambda g:(0j,[],[],0,np.zeros(len(g))))


def test_canonical_worker_adapter_passes_tolerances_without_global_leak(monkeypatch):
    from rydberg_aoa.simulation import laboratory_execution as lab, time_domain as td
    from rydberg_aoa.config import parameters
    before=parameters.ATOMIC_TRANSIENT_DELAY_US
    def worker(*args,**kwargs):
        assert parameters.ATOMIC_TRANSIENT_DELAY_US==4.
        td.mesolve(options={'method':'bdf','nsteps':5000,'normalize_output':False})
        times=args[2];trace=np.cos(.2*np.pi*np.asarray(times))
        return 1j,np.array([0.]),np.array([0j]),0,trace
    monkeypatch.setattr(td,'run_single_heterodyne_worker',worker)
    monkeypatch.setattr(td,'mesolve',lambda **kwargs:None)
    t=lab.LaboratoryWorkerTask(0,1j,tuple(np.linspace(0.,20.,201)),.1,(1.,2.,3.),(0.,0.,0.),(1.,1.,1.),(0.,0.,0.),0.,None,4.,solver_diagnostic=SolverDiagnostic('dense',1e-8,1e-10,2))
    result=lab.run_laboratory_worker(t)
    assert len(result[4])==201 and parameters.ATOMIC_TRANSIENT_DELAY_US==before


def test_shared_engine_full_campaigns_cancellation_and_export(monkeypatch,tmp_path):
    p=small_profile();calls=[]
    def run(point,*args):
        calls.append(point);report,result=fake_job(point)
        if args:report['numerical_diagnostic']=diagnostic_receipt(args[0])
        for row in result['noise_trial_results']:
            row['detector_arrays']={'clean_atomic_time_traces':np.zeros((1,len(point.run_config.tlist_us)))}
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',run)
    monkeypatch.setattr(validation,'isolated_diagnostic',run)
    events=[];result=engine.execute_study(p,on_progress=events.append)
    assert sum(event.state=='completed' for event in events)==1
    assert events[-1].completed_points==events[-1].total_points==12
    assert result.status=='completed' and len(calls)==12
    report=result.report()
    assert report['numerical_validation']['production_anchor']['status']=='verified'
    assert report['numerical_validation']['status']=='not_established'
    assert report['numerical_validation']['comparisons'][0]['comparison']['status']=='unassessed_no_supported_selected_field'
    from rydberg_aoa.simulation.study_output import save_study_results
    save_study_results(tmp_path,result)
    assert all((tmp_path/f'case_{i:04d}'/'calibration'/'simulation_arrays.npz').exists() for i in range(3))
    assert json.loads((tmp_path/'study_report.json').read_text())['numerical_validation']['acquisition_grid_and_aperture_fixed']
    with pytest.raises(FileExistsError):save_study_results(tmp_path,result)
    cancelled=engine.execute_study(p,should_cancel=lambda:True)
    assert cancelled.status=='cancelled' and cancelled.report()['numerical_validation']['status']=='incomplete'


def test_bad_anchor_is_rejected_not_repaired(monkeypatch):
    p=small_profile()
    monkeypatch.setattr(engine,'_execute_isolated',fake_job)
    def altered(point,*args):
        report,result=fake_job(point)
        report['numerical_diagnostic']=diagnostic_receipt(args[0])
        for trial in result['noise_trial_results']:trial['detector_arrays']={'clean_atomic_time_traces':np.ones((1,len(point.run_config.tlist_us)))}
        return report,result
    # fake boundary records need explicit clean traces for anchor evidence.
    def anchor(point):
        report,result=fake_job(point)
        for trial in result['noise_trial_results']:trial['detector_arrays']={'clean_atomic_time_traces':np.zeros((1,len(point.run_config.tlist_us)))}
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',anchor)
    monkeypatch.setattr(validation,'isolated_diagnostic',altered)
    with pytest.raises(PhysicsCalculationError,match='comparison failed') as e:engine.execute_study(p)
    assert 'bit-identical' in str(e.value.__cause__)


def test_invalid_diagnostic_controls_clear_stale_runner_results():
    from rydberg_aoa.simulation.runner import SimulationRunner
    runner=SimulationRunner();runner.results=['stale']
    with pytest.raises(PhysicsConfigurationError):runner.run_laboratory_simulation(None,None,poisson_quantiles='false')
    assert runner.results is None


def test_missing_instrumentation_receipt_never_certifies_diagnostic(monkeypatch):
    p=small_profile()
    def run(point,*args):
        report,result=fake_job(point)
        for row in result['noise_trial_results']:
            row['detector_arrays']={'clean_atomic_time_traces':np.zeros((1,len(point.run_config.tlist_us)))}
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',run);monkeypatch.setattr(validation,'isolated_diagnostic',run)
    with pytest.raises(PhysicsCalculationError,match='coupled_baseline'):engine.execute_study(p)


def test_decision_budget_can_fail_despite_passing_if_budget(monkeypatch):
    from tests.test_noise015_sampled_sensitivity import controlled_study
    base,_=controlled_study(monkeypatch,counts=(0,160,200,120))
    case,_=controlled_study(monkeypatch,counts=(0,160,190,120))
    options=replace(small_profile().study.numerical_validation,if_absolute_budget=2.1,if_relative_budget=0.,
                    decision_fraction_budget=.01,probability_difference_budget=.02,require_same_selected_field=False)
    compared=validation.compare_detection_case(base,case,options)
    assert all(row['mixed_if_budget_passed'] for row in compared['jobs'])
    assert compared['jobs'][4]['decision_disagreement_fraction']==pytest.approx(.05)
    assert compared['status']=='failed_declared_comparisons'
    assert any(v is None for row in compared['jobs'] for vector in row['if_relative_errors'] for v in vector)


def test_declared_comparisons_pass_without_claiming_continuum_bound(monkeypatch):
    from tests.test_noise015_sampled_sensitivity import controlled_study
    base,_=controlled_study(monkeypatch)
    options=replace(small_profile().study.numerical_validation,if_absolute_budget=0.,if_relative_budget=0.)
    result=validation.compare_detection_case(base,base,options)
    assert result['status']=='passed_declared_comparisons' and result['selection_assessed']


def test_diagnostic_plot_exposes_case_discrepancies(monkeypatch,tmp_path):
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    p=small_profile()
    def run(point,*args):
        report,result=fake_job(point)
        if args:report['numerical_diagnostic']=diagnostic_receipt(args[0])
        for row in result['noise_trial_results']:
            row['detector_arrays']={'clean_atomic_time_traces':np.zeros((1,len(point.run_config.tlist_us)))}
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',run);monkeypatch.setattr(validation,'isolated_diagnostic',run)
    study=engine.execute_study(p);figures=create_study_figures(study)
    assert {'study_numerical_if_error','study_numerical_decision_error'} <= set(figures)
    for name,fig in figures.items():fig.savefig(tmp_path/(name+'.png'),dpi=120);plt.close(fig)


def test_disabled_numerical_validation_runs_only_original_detection(monkeypatch):
    p=small_profile();calls=[]
    p=replace(p,study=replace(p.study,numerical_validation=replace(p.study.numerical_validation,enabled=False)))
    def run(point):calls.append(point);return fake_job(point)
    monkeypatch.setattr(engine,'_execute_isolated',run)
    monkeypatch.setattr(validation,'isolated_diagnostic',lambda *a:pytest.fail('disabled validation ran diagnostic'))
    result=engine.execute_study(p)
    from rydberg_aoa.simulation.detection_study import DetectionStudyResult
    assert isinstance(result,DetectionStudyResult) and len(calls)==4
    assert 'numerical_validation' not in result.report()


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_diagnostic_production_anchor_and_fixed_detector_grid(tmp_path,mode):
    d=document(mode);d['execution'].update(duration_us=20.,sample_count=201,transient_delay_us=4.)
    d['study'].update(values=[.001,.005],calibration_trials=9,null_validation_trials=5,trials_per_point=4)
    d['study']['numerical_validation']['cases']=[dict(name='tight_dense',rtol=1e-8,atol=1e-10,output_subdivision=2)]
    p=parse_run_profile(d);result=engine.execute_study(p);report=result.report()
    assert result.status=='completed' and report['numerical_validation']['production_anchor']['status']=='verified'
    assert len(result.campaigns)==3
    for name,campaign in result.campaigns:
        for point in (campaign.calibration,campaign.validation,*campaign.points):
            assert len(point.result['tlist'])==201 and point.report['noise_declaration']['detector']['integration_time_us']==.1
            assert point.result['estimated_aoa'] is None
            if name=='tight_dense':assert point.report['numerical_diagnostic']['solver']['rtol']==1e-8
    from rydberg_aoa.simulation.study_output import save_study_results
    save_study_results(tmp_path,result)
    assert (tmp_path/'case_0002'/'point_0001'/'simulation_arrays.npz').exists()
