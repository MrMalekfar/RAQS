"""Independent rank/binomial oracles, campaign isolation and real IF studies."""
from copy import deepcopy
from dataclasses import replace, asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import binom
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.detection_study import detection_jobs, detection_job_profile
from rydberg_aoa.config.study import plan_study, study_capabilities
from rydberg_aoa.simulation import study_execution as engine
from rydberg_aoa.simulation.detection_study import calibrate_threshold, evaluate_detection, binomial_interval
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT/f'examples/noise014/{mode}_detection_pin.json').read_text())


def profile(mode='prescribed'):
    d = document(mode)
    d['study'].update(values=[.001,.005],calibration_trials=9,null_validation_trials=5,trials_per_point=4)
    return parse_run_profile(d)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_planning_roundtrip_and_independent_stage_seeds(mode):
    p=profile(mode); original=deepcopy(p); jobs=detection_jobs(p.study)
    assert len(jobs)==4 and len(set(j['seed'] for j in jobs))==4
    assert plan_study(p)['total_trials']==22 and plan_study(p)['total_atomic_evolutions']==4
    assert plan_study(p)['threshold_rank']==9 and plan_study(p)['finite_threshold_available']
    assert 'if_detection' in study_capabilities(p)['kinds']
    assert parse_run_profile(profile_document(p))==p
    for i,job in enumerate(jobs):
        point=detection_job_profile(p,job)
        assert point.study is None and point.measurement.signal_enabled is (i>=2)
        assert point.noise.seed==job['seed'] and point.noise.trials==job['trials']
        assert point.probe is p.probe and point.coupling is p.coupling and point.lo is p.lo
        assert point.signal.frequency_hz==p.signal.frequency_hz and point.signal.polarization==p.signal.polarization
        assert point.noise.sources==p.noise.sources and point.signed_if_mhz==p.signed_if_mhz
        assert parse_run_profile(profile_document(point))==point
        if mode=='experimental':assert point.request.signal==point.signal and point.request.preparation is p.request.preparation
    assert p==original and detection_jobs(p.study)==jobs


@pytest.mark.parametrize('key,value',[
    ('p_false_alarm',0),('p_false_alarm',1),('p_false_alarm',True),('p_false_alarm','0.1'),
    ('p_detection',float('nan')),('confidence_level',float('inf')),('calibration_trials',False),
    ('null_validation_trials',0),('trials_per_point',2.5),('seed',2**53),('path_index',-1),
    ('values',[0.,.005]),('values',[.005,.001]),('values',[.001,.001]),('values',[True,.005]),
    ('values',[.001,10**1000]),('schema_version',True),('extra','ignored')])
def test_invalid_study_declarations_fail_before_execution(key,value):
    d=document();d['study'][key]=value
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


def test_wrong_receiver_path_or_missing_noise_rejected():
    for change in ('receiver','path','noise','disabled'):
        d=document()
        if change=='receiver':d.pop('measurement')
        elif change=='path':d['study']['path_index']=1
        elif change=='noise':d.pop('noise')
        else:d['noise']['enabled']=False
        with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


def test_rank_correction_ties_and_small_campaign_remain_undefined():
    d=profile().study
    scores=np.array([9.,4.,5.,0.,1.,2.,6.,8.,3.]);before=scores.copy()
    threshold=calibrate_threshold(d,scores)
    assert threshold['value']==9. and threshold['order_statistic_rank_one_based']==9
    assert np.array_equal(scores,before)
    result=evaluate_detection([9.,9.,10.,0.],4,threshold['value'],.8,.05)
    assert result['detections']==[False,False,True,False] and result['count']==1
    small=replace(d,calibration_trials=2)
    unknown=calibrate_threshold(small,[1.,2.])
    assert unknown['value'] is None and unknown['status']=='insufficient_calibration_trials'
    undefined=evaluate_detection([0.,1.],2,None,.8,.05)
    assert undefined['probability'] is undefined['detections'] is undefined['confidence_interval'] is None
    zero=calibrate_threshold(d,np.zeros(9))
    assert zero['value']==0. and evaluate_detection(np.zeros(5),5,0.,.1,.05,null=True)['count']==0


def test_rank_bound_exhaustive_exchangeable_permutations():
    # For n=9 and target=.2, rank 8. Each possible hold-out rank is equally
    # likely; two of ten ranks strictly exceed its calibration threshold.
    d=replace(profile().study,p_false_alarm=.2)
    events=[]
    for held_out in range(10):
        calibration=[v for v in range(10) if v!=held_out]
        events.append(held_out>calibrate_threshold(d,calibration)['value'])
    assert sum(events)==2


@pytest.mark.parametrize('k,n',[(0,20),(20,20),(4,20)])
def test_exact_interval_independent_binomial_inversion(k,n):
    alpha=.1;lo,hi=binomial_interval(k,n,alpha)
    expected_lo=0. if k==0 else brentq(lambda p:binom.sf(k-1,n,p)-alpha/2,0.,1.)
    expected_hi=1. if k==n else brentq(lambda p:binom.cdf(k,n,p)-alpha/2,0.,1.)
    assert [lo,hi]==pytest.approx([expected_lo,expected_hi],abs=1e-11)
    if k==0:assert hi>0
    if k==n:assert lo<1


@pytest.mark.parametrize('scores',[[True]*9,['1']*9,[1j]*9,[float('inf')]*9,[-1.]*9,[1.]*8,[[1.],[2.,3.]]])
def test_invalid_score_arrays_never_become_threshold(scores):
    with pytest.raises(PhysicsCalculationError):calibrate_threshold(profile().study,scores)


@pytest.mark.parametrize('key,value',[('expected',True),('threshold',True),('threshold',-1.),
    ('target','0.5'),('alpha',np.bool_(True)),('alpha',10**1000)])
def test_invalid_probability_evaluation_fails_with_project_context(key,value):
    args=dict(scores=[0.,1.],expected=2,threshold=0.,target=.8,alpha=.05)
    args[key]=value
    with pytest.raises(PhysicsCalculationError):evaluate_detection(**args)


def fake_job(p):
    """Synthetic pipeline-boundary records; not an atomic-physics oracle."""
    count=p.noise.trials; on=p.measurement.signal_enabled
    rng=np.random.default_rng(p.noise.seed)
    responses=rng.normal(0,1,count)+1j*rng.normal(0,1,count)+(100*p.signal.electric_field_vm if on else 0.)
    times=np.asarray(p.run_config.tlist_us);trace=np.zeros(len(times))
    trials=[dict(measurement_mode='if_only',signal_enabled=on,complex_responses=np.array([a]),tlist=times,
        signed_if_mhz=p.signed_if_mhz,estimated_aoa=None,doa_result=None,if_snapshots=None,
        readout_noise={'units':'V','trial_index':i},worker_records=[(a,np.array([0.]),np.array([0j]),0,trace)],
        time_traces=[trace],array_positions=np.array(p.run_config.positions_m),configured_spatial_phases_rad=np.zeros(1),
        xf_0=np.array([0.]),yf_0=np.array([0j])) for i,a in enumerate(responses)]
    result={**trials[0],'run_config':p.run_config,'noise_config':p.noise,'noise_trial_results':trials}
    configured={k:asdict(getattr(p,k)) for k in ('probe','coupling','lo','signal')};applied=deepcopy(configured)
    if not on:applied['signal']['electric_field_vm']=0.
    report=dict(measurement=asdict(p.measurement),configured_fields=configured,applied_fields=applied,
        noise_declaration=asdict(p.noise),configured_rabi_rad_per_us=(1.,2.,3.,4*p.signal.electric_field_vm),
        levels={'channel':'explicit'},rf_orientation={'direction':-1},detunings_rad_per_us=[1.,2.,3.],
        decay_rates_per_us=[1.,2.,3.],resolved_rate_budget_per_us=None,signed_if_mhz=p.signed_if_mhz,
        quantization_axis=[0.,0.,1.],sensing={'execution':asdict(p.run_config)})
    return report,result


def test_shared_dispatch_evidence_frozen_threshold_save_and_figures(monkeypatch,tmp_path):
    p=profile();calls=[]
    def run(point):calls.append(point);return fake_job(point)
    monkeypatch.setattr(engine,'_execute_isolated',run)
    progress=[];result=engine.execute_study(p,on_progress=progress.append)
    report=result.report()
    assert len(calls)==4 and report['status']=='completed'
    assert [x.measurement.signal_enabled for x in calls]==[False,False,True,True]
    oracle=sorted(report['calibration']['summary']['evidence']['scores'])[-1]
    assert report['threshold']['value']==oracle
    alpha=(1-p.study.confidence_level)/3
    for point in [report['null_validation'],*report['points']]:
        prob=point['summary']['probability']
        assert prob['count']==sum(v>oracle for v in prob['scores'])
        assert prob['confidence_interval']==pytest.approx(binomial_interval(prob['count'],prob['trials'],alpha))
    assert progress[-1].completed_points==progress[-1].total_points==4
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    save_study_results(tmp_path,result)
    for name in ('calibration','null_validation','point_0000','point_0001'):
        arrays=np.load(tmp_path/name/'simulation_arrays.npz',allow_pickle=False)
        assert 'trial_complex_responses' in arrays and 'trial_estimated_aoa_deg' not in arrays
    assert json.loads((tmp_path/'study_report.json').read_text())['threshold']==report['threshold']
    with pytest.raises(FileExistsError):save_study_results(tmp_path,result)
    figures=create_study_figures(result)
    assert len(figures)==2
    for name,fig in figures.items():fig.savefig(tmp_path/(name+'.png'));plt.close(fig)


def test_cancellation_and_failure_do_not_advertise_complete_study(monkeypatch):
    p=profile();calls=[]
    def run(point):calls.append(point);return fake_job(point)
    monkeypatch.setattr(engine,'_execute_isolated',run)
    cancelled=engine.execute_study(p,should_cancel=lambda:len(calls)==1)
    assert cancelled.status=='cancelled' and cancelled.validation is None and cancelled.points==()
    assert cancelled.report()['analysis_status']=='incomplete_no_final_detection_claim'
    sentinel=RuntimeError('injected solver failure')
    def fail(point):raise sentinel
    monkeypatch.setattr(engine,'_execute_isolated',fail)
    with pytest.raises(PhysicsCalculationError,match='calibration') as error:engine.execute_study(p)
    assert error.value.__cause__ is sentinel
    with pytest.raises(PhysicsConfigurationError):engine.execute_study(p,should_cancel=lambda:None)


def test_missing_trials_or_changed_fixed_physics_abort_entire_study(monkeypatch):
    p=profile();calls=[]
    def run(point):
        report,result=fake_job(point);calls.append(point)
        if len(calls)==2:report['levels']={'channel':'silently switched'}
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',run)
    with pytest.raises(PhysicsCalculationError,match='null_validation'):engine.execute_study(p)
    assert len(calls)==2


@pytest.mark.parametrize('damage',['missing_trial','wrong_switch','wrong_seed'])
def test_incomplete_or_mismatched_trial_campaign_evidence_rejected(monkeypatch,damage):
    p=profile()
    def run(point):
        report,result=fake_job(point)
        if damage=='missing_trial':result['noise_trial_results'].pop()
        if damage=='wrong_switch':result['signal_enabled']=True
        if damage=='wrong_seed':result['noise_config']=replace(point.noise,seed=1)
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',run)
    with pytest.raises(PhysicsCalculationError,match='calibration'):engine.execute_study(p)


def test_undefined_threshold_is_retained_through_complete_study(monkeypatch,tmp_path):
    p=replace(profile(),study=replace(profile().study,calibration_trials=2))
    monkeypatch.setattr(engine,'_execute_isolated',fake_job)
    result=engine.execute_study(p);report=result.report()
    assert report['status']=='completed' and report['threshold']['value'] is None
    assert report['threshold_frozen'] is False
    assert report['null_validation']['summary']['probability']['status']=='undefined_threshold'
    assert all(row['summary']['probability']['probability'] is None for row in report['points'])
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    save_study_results(tmp_path,result)
    figures=create_study_figures(result)
    assert all(line.get_linestyle()=='--' for line in figures['study_detection_probability'].axes[0].lines)
    for fig in figures.values():plt.close(fig)


def test_nested_noise_counts_seeds_and_signal_noise_contract():
    d=json.loads((ROOT/'examples/noise005/prescribed_rf.json').read_text())
    d['study']=document()['study'];d['measurement']={'mode':'if_only','signal_enabled':True}
    with pytest.raises(PhysicsConfigurationError,match='signal amplitude'):parse_run_profile(d)
    for name,source in d['noise']['sources'].items():source['enabled']=name=='rf_lo_amplitude'
    bare=parse_run_profile(d)
    assert all(detection_job_profile(bare,j).noise.readout is None for j in detection_jobs(bare.study))
    d['noise']['readout']=deepcopy(document()['noise'])
    d['noise']['readout'].update(seed=d['noise']['seed'],trials=d['noise']['trials'])
    p=parse_run_profile(d)
    for job in detection_jobs(p.study):
        point=detection_job_profile(p,job)
        assert point.noise.seed==point.noise.readout.seed==job['seed']
        assert point.noise.trials==point.noise.readout.trials==job['trials']
        assert point.noise.drive==p.noise.drive and point.noise.sources==p.noise.sources


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_detection_study_through_canonical_if_pipeline(tmp_path,mode):
    p=profile(mode)
    result=engine.execute_study(p)
    report=result.report()
    assert result.status=='completed' and len(result.points)==2
    assert report['calibration']['effective_seed']!=report['null_validation']['effective_seed']
    for point in (result.calibration,result.validation,*result.points):
        assert point.result['doa_result'] is point.result['estimated_aoa'] is None
        assert point.summary['evidence']['trace_units']=='V'
        assert len(point.summary['evidence']['scores'])==point.profile.noise.trials
        assert point.report['input_provenance']['classification']=='simulation_assumptions'
    assert result.calibration.report['rabi_rad_per_us'][3]==result.validation.report['rabi_rad_per_us'][3]==0.
    assert all(p.report['rabi_rad_per_us'][3]>0 for p in result.points)
    from rydberg_aoa.simulation.study_output import save_study_results
    save_study_results(tmp_path,result)
    assert json.loads((tmp_path/'study_report.json').read_text())['status']=='completed'
    assert report['mdf_nef_computed'] is False
