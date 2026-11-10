"""Reusable studies: parser/capabilities, immutable planning, isolated jobs and metrics."""
from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from tests.noise_profile_fixtures import legacy_noise_profile

from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.study import study_capabilities,effective_point_profile,plan_study
from rydberg_aoa.exceptions import PhysicsCalculationError,PhysicsConfigurationError
from rydberg_aoa.simulation import study_execution as engine
from rydberg_aoa.simulation.study_metrics import summarize_study_point

ROOT=Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return legacy_noise_profile((ROOT/f'examples/noise008/{mode}_rin_sweep.json'))


@pytest.mark.parametrize('field,bad',[('schema_version',True),('kind',['noise_strength_sweep']),('name',''),
    ('source','photon_shot'),('parameter','enabled'),('values',[0.,0.]),('values',[.2,.1]),
    ('values',[True,.1]),('values',[0.,np.nan]),('values',[0.,10**1000]),('values',[.1]),
    ('trials_per_point',True),('trials_per_point',0),('seed',-1),('seed',2**53),('random_policy','auto'),
    ('metrics',[]),('metrics',['invented']),('metrics',['aoa_rmse_deg','aoa_rmse_deg'])])
def test_invalid_study_is_rejected_before_dispatch(field,bad):
    data=document();data['study'][field]=bad
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(data)


def test_all_effective_points_validated_before_any_physics(monkeypatch):
    data=document();data['study']['values']=[0.,1e-200]
    monkeypatch.setattr(engine,'_execute_isolated',lambda p:pytest.fail('invalid point was dispatched'))
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(data)
    data=document();data['noise']['sources']['laser_probe_intensity']['enabled']=False
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(data)


def test_paired_poisson_is_not_claimed_from_equal_seed():
    data=document()
    data['noise']['readout']=legacy_noise_profile((ROOT/'examples/noise007/prescribed_rin_pin.json'))['noise']['readout']
    with pytest.raises(PhysicsConfigurationError,match='Poisson'):parse_run_profile(data)
    data['study']['random_policy']='independent'
    assert parse_run_profile(data).study.random_policy=='independent'


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_capabilities_and_editable_json_roundtrip(mode):
    p=parse_run_profile(document(mode))
    caps=study_capabilities(p)
    assert {'source':'laser_probe_intensity','parameter':'fractional_rms','unit':'relative_rms',
        'label':'Probe relative intensity RMS',
        'minimum':0.,'minimum_inclusive':True,'configured_enabled':True} in caps['axes']
    assert parse_run_profile(profile_document(p))==p
    for i in (None,0,1):
        point=effective_point_profile(p,i)
        assert point.study is None
        assert parse_run_profile(profile_document(point))==point


def test_independent_seed_and_immutable_points():
    data=document();data['study']['random_policy']='independent'
    p=parse_run_profile(data);before=copy.deepcopy(p)
    points=[effective_point_profile(p,i) for i in range(3)]
    assert p==before
    assert len({v.noise.seed for v in points})==3
    assert all(v.noise.seed<=2**53-1 for v in points)
    assert effective_point_profile(p,1)==points[1]
    for point in points:
        assert point.probe is p.probe and point.request is p.request
        assert point.noise.trials==p.study.trials_per_point


def test_gui_plan_cost_and_johnson_resistance_axis():
    p=parse_run_profile(document())
    preview=plan_study(p)
    assert [r['atomic_evolutions_per_path'] for r in preview['points']]==[1,4,4]
    assert preview['total_atomic_evolutions']==30
    data=document()
    data['noise']=legacy_noise_profile((ROOT/'examples/noise004/prescribed_pin.json'))['noise']
    data['study'].update(source='johnson_current',parameter='shunt_resistance_ohm',values=[1e6,1e8],random_policy='independent')
    parsed=parse_run_profile(data)
    assert effective_point_profile(parsed,0).noise.detector==parsed.noise.detector
    with pytest.raises(PhysicsConfigurationError):
        data['study']['values']=[0.,1e8];parse_run_profile(data)


def result(angles=(0.,0.,0.),responses=None):
    config=SimpleNamespace(transient_delay_us=0.)
    t=np.linspace(0,2,21)
    trace=np.array([np.sin(2*np.pi*t),np.cos(2*np.pi*t)])
    nominal={'complex_responses':np.array([1.+0j,2.+0j]),'estimated_aoa':0.,'true_aoa':0.,
             'tlist':t,'time_traces':trace,'run_config':config}
    trials=[]
    for i,angle in enumerate(angles):
        trial=dict(nominal)
        trial.update(estimated_aoa=angle,complex_responses=np.asarray(responses[i]) if responses is not None else nominal['complex_responses'])
        trials.append(trial)
    return nominal,dict(trials[0],noise_trial_results=trials)


def test_metrics_independent_oracle_wrapped_phase_and_undefined():
    p=parse_run_profile(document());study=replace(p.study,metrics=('aoa_bias_deg','aoa_rmse_deg',
        'if_phase_rmse_deg_per_path','if_amplitude_relative_error_per_path','readout_trace_snr_db_per_path'))
    nominal,trial=result(angles=(-3.,0.,3.),responses=[np.array([1.,2.])*np.exp(1j*np.deg2rad(v)) for v in (-10.,0.,10.)])
    summary=summarize_study_point(nominal,trial,study)
    assert summary['metrics']['aoa_bias_deg']==0.
    assert summary['metrics']['aoa_rmse_deg']==pytest.approx(np.sqrt(6))
    assert summary['metrics']['if_phase_rmse_deg_per_path']==pytest.approx([np.sqrt(200/3)]*2)
    assert summary['metrics']['if_amplitude_relative_error_per_path']==pytest.approx([0.,0.],abs=1e-15)
    assert summary['metrics']['readout_trace_snr_db_per_path']==[None,None]
    trial['noise_trial_results'][1]['complex_responses']=np.array([0.,2.])
    summary=summarize_study_point(nominal,trial,study)
    assert summary['metrics']['if_phase_rmse_deg_per_path'][0] is None
    assert summary['undefined_reasons']['if_phase_rmse_deg_per_path'][0]=='1_of_3_trials_undefined'
    assert summary['metrics']['if_amplitude_relative_error_per_path'][0]==pytest.approx(1/3)


def test_incomplete_trials_cannot_become_successful_summary():
    p=parse_run_profile(document());nominal,trial=result(angles=(0.,))
    with pytest.raises(PhysicsCalculationError,match='complete'):summarize_study_point(nominal,trial,p.study)


def test_reusable_engine_progress_and_point_failure(monkeypatch):
    p=parse_run_profile(document());nominal,trial=result()
    calls=[];events=[]
    def execute(point):
        calls.append(point)
        return {'test':True},nominal if len(calls)==1 else trial
    monkeypatch.setattr(engine,'_execute_isolated',execute)
    out=engine.execute_study(p,on_progress=events.append)
    assert out.status=='completed' and len(out.points)==3 and len(calls)==4
    assert [e.state for e in events]==['reference_started','reference_completed',
        'point_started','point_completed','point_started','point_completed','point_started','point_completed','completed']
    assert all(v.study is None for v in calls)
    assert events[3].point_result is out.points[0]
    failure=PhysicsCalculationError('injected point failure')
    calls.clear()
    def fail(point):
        calls.append(point)
        if len(calls)==3:raise failure
        return {},nominal if len(calls)==1 else trial
    monkeypatch.setattr(engine,'_execute_isolated',fail)
    with pytest.raises(PhysicsCalculationError,match='point 1') as error:engine.execute_study(p)
    assert error.value.__cause__ is failure and len(calls)==3


def test_cancellation_is_explicit_and_never_full_success(monkeypatch):
    p=parse_run_profile(document());nominal,trial=result();calls=[]
    def run(point):
        calls.append(point);return {},nominal if len(calls)==1 else trial
    monkeypatch.setattr(engine,'_execute_isolated',run)
    out=engine.execute_study(p,should_cancel=lambda:len(calls)>=2)
    assert out.status=='cancelled' and len(out.points)==1
    assert out.report()['points_completed']==1 and out.report()['points_requested']==3
    with pytest.raises(PhysicsConfigurationError):engine.execute_study(p,should_cancel=lambda:'yes')


def test_ordinary_execution_cannot_ignore_study():
    from rydberg_aoa.simulation.profile_execution import execute_profile
    with pytest.raises(PhysicsConfigurationError,match='execute_study'):execute_profile(parse_run_profile(document()))


def test_study_figures_use_actual_metrics_and_show_undefined_gaps(monkeypatch):
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    p=parse_run_profile(document());nominal,trial=result()
    monkeypatch.setattr(engine,'_execute_isolated',lambda point:({},nominal if all(not s.specification.enabled for s in point.noise.sources) else trial))
    out=engine.execute_study(p)
    out.points[1].summary['metrics']['if_phase_rmse_deg_per_path'][0]=None
    figures=create_study_figures(out)
    fig=figures['study_if_phase_rmse_deg_per_path']
    assert np.array_equal(fig.axes[0].lines[0].get_xdata(),p.study.values)
    assert np.isnan(fig.axes[0].lines[0].get_ydata()[1])
    assert any('gaps' in t.get_text() for t in fig.texts)
    for f in figures.values():plt.close(f)


def test_actual_spawn_reference_preserves_parent_state(monkeypatch):
    from rydberg_aoa.config import atomic_config as ac
    p=parse_run_profile(document())
    reference=effective_point_profile(p)
    root=ac.LASER_PARAMS;probe=root['probe']
    monkeypatch.setitem(probe,'power_mw',123.456)
    monkeypatch.setattr(ac,'_ATOMIC_PHYSICS_INITIALIZED',True)
    report,result_=engine._execute_isolated(reference)
    assert report['applied_fields']['probe']['power_mw']==reference.probe.power_mw
    assert ac._ATOMIC_PHYSICS_INITIALIZED is True
    assert ac.LASER_PARAMS is root and ac.LASER_PARAMS['probe'] is probe
    assert probe['power_mw']==123.456


def test_actual_readout_study_json_replay_and_saved_curves(tmp_path):
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    from rydberg_aoa.visualization.profile_figures import save_profile_figures
    import matplotlib.pyplot as plt
    data=json.loads((ROOT/'examples/noise008/prescribed_readout_sweep.json').read_text())
    data['study'].update(values=[0.,1e-5],trials_per_point=2)
    p=parse_run_profile(data)
    out=engine.execute_study(p)
    assert out.points[0].summary['metrics']['readout_trace_snr_db_per_path']==[None]*3
    assert all(v is not None for v in out.points[1].summary['metrics']['readout_trace_snr_db_per_path'])
    save_study_results(tmp_path,out)
    assert parse_run_profile(json.loads((tmp_path/'study_profile.json').read_text()))==p
    figures=create_study_figures(out);save_profile_figures(tmp_path,figures)
    assert (tmp_path/'study_aoa_rmse_deg.png').is_file()
    assert parse_run_profile(json.loads((tmp_path/'point_0001'/'profile.json').read_text())).noise.trials==2
    with pytest.raises(FileExistsError):save_study_results(tmp_path,out)
    for f in figures.values():plt.close(f)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_cli_isolated_sweep_and_replayable_profiles(tmp_path,mode):
    data=document(mode)
    data['study'].update(values=[0.,.01],trials_per_point=2)
    # No stationary EIT scan: this test targets temporal sensing and metrics.
    path=tmp_path/'input.json';path.write_text(json.dumps(data))
    out=tmp_path/'output'
    proc=subprocess.run([sys.executable,'-B','main.py','--profile',str(path),'--no-plots','--output-dir',str(out)],
        cwd=ROOT,env=dict(os.environ,MPLBACKEND='Agg'),capture_output=True,text=True,errors='replace',timeout=360)
    assert proc.returncode==0,proc.stdout+proc.stderr
    report=json.loads((out/'study_report.json').read_text(encoding='utf-8'))
    assert report['status']=='completed' and report['points_completed']==2
    assert report['points'][0]['effective_trials']==2
    for i in range(2):
        point=out/f'point_{i:04d}'
        effective=json.loads((point/'profile.json').read_text())
        assert effective['fields']==data['fields'] and 'study' not in effective
        assert parse_run_profile(effective).noise.seed==data['study']['seed']
        with np.load(point/'simulation_arrays.npz',allow_pickle=False) as a:
            assert a['trial_complex_responses'].shape[0]==2
        assert report['points'][i]['run_report']['levels']==report['nominal']['levels']
