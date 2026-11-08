"""Receiver mode, switch, pure moments, no angle inference and real pipeline."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from rydberg_aoa.config.measurement import MeasurementOptions
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.simulation.if_only import summarize_if_trials
from rydberg_aoa.simulation import runner as module
from rydberg_aoa.utils.numerical import extract_if_component
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from tests.test_lab005_laboratory_execution import isolation, request, plan, fake_pool
from tests.test_noise004_detector import config as pin_config
from rydberg_aoa.simulation.detector_readout import DetectorContext

ROOT=Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT/f'examples/noise013/{mode}_null_pin.json').read_text())


@pytest.mark.parametrize('options',[{'mode':'if_only','signal_enabled':0},
    {'mode':'if_only','signal_enabled':'false'},{'mode':'unknown','signal_enabled':False},
    {'mode':'aoa','signal_enabled':False},{'mode':'if_only'},None])
def test_invalid_options_rejected_before_initialization(options):
    d=document();d['measurement']=options
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_single_path_roundtrip_and_unused_method_no_registry_substitute(mode):
    d=document(mode);d['execution']['doa_method']='not_used_in_if_only'
    before=deepcopy(d);p=parse_run_profile(d)
    assert d==before and len(p.run_config.positions_m)==1
    assert p.run_config.measurement_mode=='if_only'
    assert parse_run_profile(profile_document(p))==p
    assert 'measurement_mode' not in profile_document(p)['execution']
    d.pop('measurement')
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


def test_short_window_if_only_needs_no_snapshot_but_keeps_if_rank_contract():
    d=document();d['execution'].update(duration_us=2.,sample_count=21,transient_delay_us=0.)
    p=parse_run_profile(d);assert len(p.run_config.tlist_us)==21
    # DC/Nyquist and <10 retained samples still fail before execution.
    for change in ('dc','cut'):
        changed=deepcopy(d)
        if change=='dc':changed['fields']['signal']['frequency_hz']=changed['fields']['lo']['frequency_hz']
        else:changed['execution']['transient_delay_us']=1.5
        with pytest.raises(PhysicsConfigurationError):parse_run_profile(changed)


def trial(vector,index=0,enabled=False):
    return {'measurement_mode':'if_only','signal_enabled':enabled,'complex_responses':np.array(vector),
        'tlist':np.arange(20)*.1,'signed_if_mhz':.1,'estimated_aoa':None,'doa_result':None,'if_snapshots':None,
        'readout_noise':{'units':'V','trial_index':index}}


def test_if_power_and_joint_iq_covariance_independent_oracle():
    values=np.array([[1+2j,2-1j],[3+4j,4-3j],[5+1j,6-2j]])
    records=[trial(v,i) for i,v in enumerate(values)];before=deepcopy(records)
    out=summarize_if_trials(records,3)
    assert out['mean_squared_if_amplitude_per_path']==pytest.approx(np.mean(abs(values)**2,axis=0))
    assert np.array(out['joint_iq_sample_covariance'])==pytest.approx(np.cov(np.column_stack((values.real,values.imag)),rowvar=False,ddof=1))
    assert out['source_phase_deg_per_path']==[None,None] and out['estimated_aoa_deg'] is None
    assert np.array_equal(records[0]['complex_responses'],before[0]['complex_responses'])
    one=summarize_if_trials([trial([0j])],1)
    assert one['mean_squared_if_amplitude_per_path']==[0.]
    assert one['joint_iq_sample_covariance'] is None and one['covariance_status']=='insufficient_trials'
    assert one['mdf_nef_computed'] is False


@pytest.mark.parametrize('damage',['missing','order','nan','bool','shape','grid','angle','overflow'])
def test_bad_if_evidence_fails_without_dropping_or_clipping(damage):
    rows=[trial([1j],0),trial([2j],1)]
    if damage=='missing':rows.pop()
    if damage=='order':rows[1]['readout_noise']['trial_index']=0
    if damage=='nan':rows[1]['complex_responses']=np.array([np.nan])
    if damage=='bool':rows[1]['complex_responses']=np.array([True])
    if damage=='shape':rows[1]['complex_responses']=np.array([1j,2j])
    if damage=='grid':rows[1]['tlist']=np.arange(20)*.2
    if damage=='angle':rows[1]['estimated_aoa']=0.
    if damage=='overflow':rows[1]['complex_responses']=np.array([1e300])
    with pytest.raises(PhysicsCalculationError):summarize_if_trials(rows,2)


@pytest.mark.parametrize('damage',['switch','timestamp','signed_if','trial_index','units'])
def test_identically_invalid_trial_metadata_is_not_evidence(damage):
    rows=[trial([1j],0),trial([2j],1)]
    for row in rows:
        if damage=='switch':row['signal_enabled']=None
        if damage=='timestamp':row['tlist']=np.array([0.,np.inf])
        if damage=='signed_if':row['signed_if_mhz']=np.inf
        if damage=='units':row['readout_noise']['units']=None
    if damage=='trial_index':rows[0]['readout_noise']['trial_index']=False
    with pytest.raises(PhysicsCalculationError):summarize_if_trials(rows,2)


def null_record(task):
    assert task.omega_signal_complex==0j
    times=np.asarray(task.tlist_us)
    trace=np.full(len(times),-.05)
    if task.rf_drive is not None:
        trace+=.001*np.array([task.rf_drive.at(t)[0]-1 for t in times])
    amp,xf,yf=extract_if_component(trace,times,task.signed_if_mhz,task.transient_delay_us)
    return amp,xf,yf,task.element_index,trace


@pytest.mark.parametrize('readout',['none','pin'])
def test_runner_signal_off_uses_zero_drive_no_doa_or_snapshot(isolation,monkeypatch,readout):
    from rydberg_aoa.simulation import doa_pipeline
    from rydberg_aoa.doa import snapshots
    monkeypatch.setattr(doa_pipeline,'resolve_doa_method',lambda *a,**k:pytest.fail('IF-only resolved DoA'))
    monkeypatch.setattr(module,'estimate_doa_from_results',lambda *a,**k:pytest.fail('IF-only estimated angle'))
    monkeypatch.setattr(snapshots,'extract_if_snapshots',lambda *a,**k:pytest.fail('IF-only made spatial snapshots'))
    tasks=[];fake_pool(monkeypatch,tasks,worker=null_record)
    cfg=plan(measurement_mode='if_only',positions_m=((0.,0.,0.),),doa_method='not_used_in_if_only')
    options={} if readout=='none' else {'noise':pin_config(('photon_shot',)),
        'detector_context':DetectorContext(1e-7,.1,-.5,780.241476e-9)}
    runner=module.SimulationRunner()
    result=runner.run_laboratory_simulation(request(isolation),cfg,measurement=MeasurementOptions('if_only',False),**options)
    assert result['estimated_aoa'] is result['doa_result'] is result['if_snapshots'] is result['element_phases_rad'] is None
    assert result['effective_signal_rabi_rad_per_us']==0.
    assert len(tasks)==1 and tasks[0].omega_signal_complex==0j
    for t in result.get('noise_trial_results',[result]):
        assert t['complex_responses'][0]==extract_if_component(t['time_traces'][0],cfg.tlist_us,.1,4.)[0]
    with pytest.raises(PhysicsConfigurationError,match='IF-only'):
        runner.process_simulation_results()


def test_null_colored_lo_campaign_is_supported_but_signal_noise_rejected(isolation,monkeypatch):
    d=json.loads((ROOT/'examples/noise005/prescribed_rf.json').read_text())['noise']
    d['trials']=2
    for name,s in d['sources'].items():s['enabled']=name=='rf_lo_amplitude'
    cfg=plan(measurement_mode='if_only',positions_m=((0.,0.,0.),));tasks=[]
    fake_pool(monkeypatch,tasks,worker=null_record)
    result=module.SimulationRunner().run_laboratory_simulation(request(isolation),cfg,
        noise=parse_noise_config(d),measurement=MeasurementOptions('if_only',False))
    assert len(tasks)==3 and result['rf_drive_campaign']['drive_noise_included'] is True
    assert result['rf_drive_campaign']['if_statistics']['estimated_aoa_deg'] is None
    assert result['rf_drive_campaign']['if_statistics']['trials_completed']==2
    d['sources']['rf_relative_phase']['enabled']=True
    with pytest.raises(PhysicsConfigurationError,match='disabled signal'):
        module.SimulationRunner().run_laboratory_simulation(request(isolation),cfg,
            noise=parse_noise_config(d),measurement=MeasurementOptions('if_only',False))


def test_if_only_export_has_no_object_angles_phase_or_spatial_covariance(tmp_path):
    from rydberg_aoa.simulation.profile_execution import save_profile_results
    t=trial([1+2j]);times=t['tlist'];trace=np.ones(len(times))
    a,xf,yf=extract_if_component(trace,times,.1)
    t.update(worker_records=[(a,xf,yf,0,trace)],time_traces=[trace],array_positions=np.zeros((1,3)),
        configured_spatial_phases_rad=np.zeros(1),xf_0=xf,yf_0=yf)
    t.pop('readout_noise')
    save_profile_results(tmp_path,t)
    data=np.load(tmp_path/'simulation_arrays.npz',allow_pickle=False)
    assert not set(data.files)&{'angles','power','element_phases_rad','if_snapshots','sample_covariance','trial_estimated_aoa_deg'}
    assert all(data[k].dtype.kind!='O' for k in data.files)


def _run(profile):
    from rydberg_aoa.simulation.profile_execution import execute_profile
    return execute_profile(profile)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_null_pin_profile_report_export_and_figures(tmp_path,mode):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from rydberg_aoa.simulation.profile_execution import save_profile_results
    from rydberg_aoa.visualization.profile_figures import create_profile_figures
    import matplotlib.pyplot as plt
    d=document(mode);d['noise']['trials']=3
    p=parse_run_profile(d)
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as executor:
        report,result=executor.submit(_run,p).result()
    assert report['rabi_rad_per_us'][3]==0. and report['configured_rabi_rad_per_us'][3]>0
    assert report['applied_fields']['signal']['electric_field_vm']==0.
    assert report['configured_fields']['signal']['electric_field_vm']==p.signal.electric_field_vm
    assert report['sensing']['estimated_aoa_deg'] is None and report['sensing']['true_aoa_deg'] is None
    assert report['if_measurement']['trials_completed']==3 and report['noise_included'] is True
    assert report['if_measurement']['source_phase_deg_per_path']==[None]
    assert np.isfinite(report['if_measurement']['mean_squared_if_amplitude_per_path']).all()
    assert report['if_measurement']['fit']['retained_samples']==401
    save_profile_results(tmp_path,result)
    data=np.load(tmp_path/'simulation_arrays.npz',allow_pickle=False)
    assert data['trial_complex_responses'].shape==(3,1) and 'angles' not in data.files
    figures=create_profile_figures(report,result)
    assert set(figures)=={'if_only_time_domain','if_only_trial_amplitude','if_only_iq'}
    for f in figures.values():plt.close(f)
    assert p.signal.electric_field_vm>0 and p.measurement.signal_enabled is False


def test_actual_explicit_aoa_switch_on_is_bit_identical_to_existing_path():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    d=json.loads((ROOT/'examples/noise004/prescribed_pin.json').read_text());d['spectrum']=None
    d['noise']['trials']=1
    for s in d['noise']['sources'].values():s['enabled']=False
    old=parse_run_profile(d);new=replace(old,measurement=MeasurementOptions('aoa',True))
    results=[]
    for p in (old,new):
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as executor:
            results.append(executor.submit(_run,p).result()[1])
    for key in ('time_traces','complex_responses','tlist','angles','power'):
        assert np.array_equal(results[0][key],results[1][key])
    assert results[0]['estimated_aoa']==results[1]['estimated_aoa']


def test_actual_signal_on_receiver_keeps_drive_and_raw_if(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from rydberg_aoa.visualization.profile_figures import create_profile_figures
    from rydberg_aoa.simulation.profile_execution import plot_profile_sensing
    import matplotlib.pyplot as plt
    d=document();d['measurement']['signal_enabled']=True
    d['noise']['trials']=1
    for source in d['noise']['sources'].values():source['enabled']=False
    p=parse_run_profile(d)
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as executor:
        report,result=executor.submit(_run,p).result()
    assert result['effective_signal_rabi_rad_per_us']>0
    assert report['rabi_rad_per_us'][3]==report['configured_rabi_rad_per_us'][3]
    assert report['applied_fields']==report['configured_fields']
    assert result['estimated_aoa'] is result['element_phases_rad'] is None
    actual=extract_if_component(result['time_traces'][0],result['tlist'],
                                result['signed_if_mhz'],result['run_config'].transient_delay_us)[0]
    assert result['complex_responses'][0]==actual and abs(actual)>0
    figures=create_profile_figures(report,result)
    figures['receiver_summary']=plot_profile_sensing(result)
    for name,figure in figures.items():
        figure.savefig(tmp_path/f'{name}.png',dpi=120)
        assert (tmp_path/f'{name}.png').stat().st_size>0
        plt.close(figure)
