"""Selection, auditable offline receipts, refinement planning and live studies."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.detection_study import detection_jobs
from rydberg_aoa.config.sensitivity import refinement_profile
from rydberg_aoa.config.study import plan_study, study_capabilities
from rydberg_aoa.simulation import study_execution as engine
from rydberg_aoa.simulation.sampled_sensitivity import analyze_sampled_sensitivity
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from tests.test_noise014_detection_study import fake_job

ROOT=Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT/f'examples/noise015/{mode}_sampled_sensitivity_pin.json').read_text())


def profile():
    d=document();d['study'].update(values=[.001,.003,.005,.01],calibration_trials=99,
                                  null_validation_trials=200,trials_per_point=200)
    d['study']['sensitivity']['refinement']=None
    return parse_run_profile(d)


def controlled_study(monkeypatch,counts=(0,160,200,120),null_count=0,calibration_trials=99,null_trials=200):
    p=profile();p=replace(p,study=replace(p.study,calibration_trials=calibration_trials,null_validation_trials=null_trials))
    calls=[]
    def run(point):
        i=len(calls);calls.append(point)
        report,result=fake_job(point)
        if i==0:scores=np.linspace(0.,1.,point.noise.trials)
        else:
            k=null_count if i==1 else counts[i-2]
            scores=np.array([4.]*k+[0.]*(point.noise.trials-k))
        for row,score in zip(result['noise_trial_results'],scores):row['complex_responses']=np.array([np.sqrt(score)+0j])
        result['complex_responses']=result['noise_trial_results'][0]['complex_responses']
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',run)
    return engine.execute_study(p),calls


def test_selection_keeps_unresolved_lower_points_and_nonmonotone_evidence(monkeypatch):
    study,calls=controlled_study(monkeypatch)
    receipt=study.report();before=deepcopy(receipt)
    a=receipt['sampled_sensitivity']
    assert a['minimum_supported_field_v_per_m']==.005 and a['selected_point_index']==2
    assert a['grid_minimality_status']=='unresolved_lower_sampled_fields'
    assert [p['status'] for p in a['lower_sampled_points']]==['target_not_met','inconclusive']
    assert a['sampled_estimates_decrease_somewhere'] is True
    assert a['continuous_minimum_computed'] is a['physical_sensitivity_certified'] is False
    assert a['numerical_convergence_status']=='not_verified_for_this_study'
    assert analyze_sampled_sensitivity(receipt)==a and receipt==before
    # JSON persistence, not only the in-memory tuple representation.
    assert analyze_sampled_sensitivity(json.loads(json.dumps(receipt)))==a
    assert len(calls)==6
    alpha=(1-study.definition.confidence_level)/5
    expected_upper=1-(alpha/2)**(1/study.definition.null_validation_trials)
    assert a['false_alarm_validation']['confidence_interval'][1]==pytest.approx(expected_upper)


@pytest.mark.parametrize('counts,status',[
    ((0,100,200,200),'all_lower_sampled_fields_excluded'),
    ((200,0,0,0),'at_lower_grid_boundary')])
def test_sampled_grid_minimality_is_not_continuous_minimum(monkeypatch,counts,status):
    study,_=controlled_study(monkeypatch,counts=counts)
    a=study.report()['sampled_sensitivity']
    assert a['grid_minimality_status']==status and a['continuous_minimum_computed'] is False


@pytest.mark.parametrize('case',['false_alarm','no_detection','small_calibration','small_null'])
def test_no_minimum_is_invented_when_targets_or_counts_insufficient(monkeypatch,case):
    options={}
    if case=='false_alarm':options['null_count']=100
    if case=='no_detection':options['counts']=(0,0,0,0)
    if case=='small_calibration':options['calibration_trials']=2
    if case=='small_null':options['null_trials']=5
    study,_=controlled_study(monkeypatch,**options);a=study.report()['sampled_sensitivity']
    assert a['minimum_supported_field_v_per_m'] is a['selected_point_index'] is None
    if case=='small_null':assert a['statistical_feasibility']['best_case_false_alarm_target_feasible'] is False
    if case=='small_calibration':assert a['status']=='undefined_threshold'


@pytest.mark.parametrize('damage',['threshold','count','bool_count','interval','seed','role','switch','units',
    'missing_point','reorder','field','noise','fit','confidence','cached_status','fixed_channel'])
def test_inconsistent_saved_evidence_cannot_qualify_as_sensitivity(monkeypatch,damage):
    study,_=controlled_study(monkeypatch);r=deepcopy(study.report());point=r['points'][2]
    if damage=='threshold':r['threshold']['value']=0.
    if damage=='count':point['summary']['probability']['count']=1
    if damage=='bool_count':r['null_validation']['summary']['probability']['count']=False
    if damage=='interval':point['summary']['probability']['confidence_interval'][0]=1.
    if damage=='seed':point['effective_seed']=r['calibration']['effective_seed']
    if damage=='role':point['summary']['role']='calibration'
    if damage=='switch':point['run_report']['measurement']['signal_enabled']=False
    if damage=='units':point['summary']['evidence']['trace_units']='W'
    if damage=='missing_point':r['points'].pop()
    if damage=='reorder':r['points'].reverse()
    if damage=='field':point['run_report']['applied_fields']['signal']['electric_field_vm']=.1
    if damage=='noise':
        source=next(s for s in point['run_report']['noise_declaration']['sources'] if s['name']=='electronic_current')
        source['specification']['asd_a_per_sqrt_hz']=1.
    if damage=='fit':point['summary']['evidence']['fit']['complex_peak_amplitude_white_enbw_hz']=0.
    if damage=='confidence':r['confidence']['comparisons']=1
    if damage=='cached_status':point['summary']['joint_target_status']='not_established'
    if damage=='fixed_channel':point['run_report']['levels']={'channel':'wrong'}
    with pytest.raises(PhysicsCalculationError):analyze_sampled_sensitivity(r)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_explicit_refinement_is_independent_pure_and_roundtrips(mode):
    p=parse_run_profile(document(mode));before=deepcopy(p)
    q=refinement_profile(p)
    assert q.study.sensitivity.refinement is None and p==before
    assert q.probe is p.probe and q.coupling is p.coupling and q.lo is p.lo and q.signal is p.signal
    assert q.request is p.request and q.run_config is p.run_config and q.noise is p.noise
    assert q.study.p_false_alarm==p.study.p_false_alarm and q.study.p_detection==p.study.p_detection
    assert not set(j['seed'] for j in detection_jobs(p.study)) & set(j['seed'] for j in detection_jobs(q.study))
    assert parse_run_profile(profile_document(q))==q and parse_run_profile(profile_document(p))==p
    assert plan_study(p)['sensitivity']['separate_refinement']['seed']==15001
    assert study_capabilities(p)['if_detection']['sensitivity']['optional'] is True


@pytest.mark.parametrize('change',['same_seed','out_of_range','hidden_execution','bool_switch','unknown_selection','bad_count','bad_field'])
def test_invalid_refinement_or_selection_rejected_before_dispatch(change):
    d=document();options=d['study']['sensitivity'];r=options['refinement']
    if change=='same_seed':r['seed']=d['study']['seed']
    if change=='out_of_range':r['values'][0]=.0001
    if change=='hidden_execution':r['execution_policy']='run_until_success'
    if change=='bool_switch':options['enabled']=1
    if change=='unknown_selection':options['selection']='interpolated_minimum'
    if change=='bad_count':r['trials_per_point']=True
    if change=='bad_field':r['values'][1]='.002'
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


def test_disabled_analysis_keeps_identical_campaign_and_no_hidden_refinement(monkeypatch):
    p=profile();calls=[]
    def run(point):calls.append(point);return fake_job(point)
    monkeypatch.setattr(engine,'_execute_isolated',run)
    enabled=engine.execute_study(p)
    original=list(calls);calls.clear()
    off=replace(p,study=replace(p.study,sensitivity=replace(p.study.sensitivity,enabled=False)))
    disabled=engine.execute_study(off)
    assert calls==original
    assert disabled.report()['sampled_sensitivity']['status']=='disabled'
    assert enabled.threshold==disabled.threshold
    assert all(np.array_equal(a.result['complex_responses'],b.result['complex_responses']) for a,b in zip(enabled.points,disabled.points))
    cancelled=engine.execute_study(p,should_cancel=lambda:True)
    assert cancelled.report()['sampled_sensitivity']['minimum_supported_field_v_per_m'] is None


def test_sensitivity_report_and_figure_persist_without_altering_raw_arrays(monkeypatch,tmp_path):
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    study,_=controlled_study(monkeypatch)
    save_study_results(tmp_path,study)
    r=json.loads((tmp_path/'study_report.json').read_text())
    assert analyze_sampled_sensitivity(r)==r['sampled_sensitivity']
    arrays=np.load(tmp_path/'point_0002'/'simulation_arrays.npz',allow_pickle=False)
    assert np.array_equal(arrays['trial_complex_responses'],np.array([t['complex_responses'] for t in study.points[2].result['noise_trial_results']]))
    figures=create_study_figures(study)
    plot=figures['study_detection_probability']
    lines=[line for line in plot.axes[0].lines if 'Smallest supported sampled field' in line.get_label()]
    assert len(lines)==1 and np.array_equal(lines[0].get_xdata(),[.005,.005])
    for name,fig in figures.items():fig.savefig(tmp_path/(name+'.png'),dpi=120);plt.close(fig)


def test_exported_refinement_is_only_a_new_profile_not_an_extra_execution(monkeypatch,tmp_path):
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.config.run_profile import load_run_profile
    p=parse_run_profile(document());calls=[]
    def run(point):calls.append(point);return fake_job(point)
    monkeypatch.setattr(engine,'_execute_isolated',run)
    study=engine.execute_study(p)
    assert len(calls)==len(p.study.values)+2
    save_study_results(tmp_path,study)
    assert len(calls)==len(p.study.values)+2
    assert load_run_profile(tmp_path/'refinement_profile.json')==refinement_profile(p)
    assert json.loads((tmp_path/'refinement_plan.json').read_text())['total_trials']==1599
    with pytest.raises(FileExistsError):save_study_results(tmp_path,study)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_two_mode_sensitivity_execution_and_offline_receipt(tmp_path,mode):
    from rydberg_aoa.simulation.study_output import save_study_results
    d=document(mode);d['study'].update(values=[.001,.005],calibration_trials=9,null_validation_trials=5,trials_per_point=4)
    d['study']['sensitivity']['refinement']=None
    p=parse_run_profile(d);study=engine.execute_study(p)
    r=study.report();a=r['sampled_sensitivity']
    assert study.status=='completed' and a['statistical_audit_status']=='recomputed_and_consistent'
    assert a['minimum_supported_field_v_per_m'] is None
    assert a['statistical_feasibility']['best_case_false_alarm_target_feasible'] is False
    assert a['numerical_convergence_status']=='not_verified_for_this_study'
    assert all(point.result['estimated_aoa'] is None for point in study.points)
    save_study_results(tmp_path,study)
    saved=json.loads((tmp_path/'study_report.json').read_text())
    assert analyze_sampled_sensitivity(saved)==a
