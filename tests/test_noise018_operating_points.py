"""Scope-preserving calibration guards and reusable operating-study contracts."""
from copy import deepcopy
from dataclasses import replace, asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.constants import c, epsilon_0, hbar, elementary_charge, Boltzmann
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.study import plan_study, study_capabilities
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.operating_study import operating_profile
from rydberg_aoa.simulation import study_execution as engine
from rydberg_aoa.simulation.operating_study import point_budget
from rydberg_aoa.simulation.measurement_audit import audit_pin_context, check_fixed_readout
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError

ROOT=Path(__file__).resolve().parents[1]


def profile(mode='prescribed'):
    return parse_run_profile(json.loads((ROOT/f'examples/noise018/{mode}_operating_points.json').read_text()))


def fake_point(p):
    t=np.asarray(p.run_config.tlist_us); d=p.noise.detector; mu=1.9e-29
    field=np.sqrt(4*(p.probe.power_mw/1000)/(np.pi*(p.probe.beam_waist_mm/1000)**2*c*epsilon_0))
    omega=mu*field/hbar/1e6
    medium=SimpleNamespace(number_density_m3=1e16,cell_length_m=p.environment.cell_length_m)
    context=SimpleNamespace(incident_probe_power_w=p.probe.power_mw/1000,probe_wavelength_m=p.probe.wavelength_nm*1e-9,
        od_factor=np.pi*medium.cell_length_m/(p.probe.wavelength_nm*1e-9),susceptibility_factor=-2e16*mu**2/(epsilon_0*hbar*omega*1e6))
    receipt=audit_pin_context(p,medium,context,omega,mu)
    amplitude=1e-5*medium.cell_length_m/(1+p.probe.detuning_mhz**2)
    y=(.001+amplitude*np.cos(2*np.pi*p.signed_if_mhz*(t-20)))[None,:]
    report={'detector_calibration':receipt,'levels':{'r1':26},'rf_orientation':{'direction':-1},
            'applied_fields':{k:asdict(getattr(p,k)) for k in ('probe','coupling','lo','signal')},
            'noise_declaration':asdict(p.noise),'measurement':asdict(p.measurement),'mode':p.mode,
            'detunings_rad_per_us':[2*np.pi*p.probe.detuning_mhz,0.,0.],
            'rabi_rad_per_us':[omega,1.,1.,1.],'decay_rates_per_us':[1.,.01,.001],
            'resolved_rate_budget_per_us':{'transit':1.},'quantization_axis':p.quantization_axis}
    result={'clean_time_traces':y,'clean_complex_responses':np.array([amplitude+0j]),'tlist':t,
            'detector_arrays':{'clean_optical_power_w':np.full(y.shape,1e-9), 'clean_detector_transmission':np.full(y.shape,.01)}}
    return report,result


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_json_plan_round_trip_and_rf_first_request_preservation(mode):
    p=profile(mode);before=profile_document(p);plan=plan_study(p)
    assert plan['total_atomic_evolutions']==15
    assert 'operating_point_sweep' in study_capabilities(p)['kinds']
    assert profile_document(parse_run_profile(before))==before
    q=operating_profile(p,0)
    assert q.probe.detuning_mhz==-4 and q.environment.cell_length_m==.001
    assert q.lo==p.lo and q.signal==p.signal and q.noise==p.noise
    if mode=='experimental': assert q.request.probe==q.probe and q.request.bounds==p.request.bounds
    assert profile_document(p)==before


@pytest.mark.parametrize('value',[True,'1',None,float('nan'),float('inf'),-1.,0.,[1.]])
def test_invalid_length_is_loud(value):
    d=profile_document(profile());d['study']['cell_lengths_m']=[value,.01]
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


@pytest.mark.parametrize('damage',['rabi','dipole','chi','od','power','wavelength'])
def test_calibration_mismatch_is_rejected_without_mutation(damage):
    p=profile();mu=1.9e-29
    field=np.sqrt(4*(p.probe.power_mw/1000)/(np.pi*(p.probe.beam_waist_mm/1000)**2*c*epsilon_0));om=mu*field/hbar/1e6
    m=SimpleNamespace(number_density_m3=1e16,cell_length_m=.01)
    ctx=SimpleNamespace(incident_probe_power_w=p.probe.power_mw/1000,probe_wavelength_m=p.probe.wavelength_nm*1e-9,
        od_factor=np.pi*.01/(p.probe.wavelength_nm*1e-9),susceptibility_factor=-2e16*mu**2/(epsilon_0*hbar*om*1e6))
    if damage=='rabi':om*=1.1
    elif damage=='dipole':mu*=1.1
    else:
        key={'chi':'susceptibility_factor','od':'od_factor','power':'incident_probe_power_w','wavelength':'probe_wavelength_m'}[damage]
        setattr(ctx,key,getattr(ctx,key)*1.1)
    before=deepcopy(vars(ctx))
    with pytest.raises(PhysicsCalculationError,match='calibration'):audit_pin_context(p,m,ctx,om,mu)
    assert vars(ctx)==before


def test_fixed_readout_context_change_is_not_solver_convergence():
    first={'readout_noise':{'model':'pin_boxcar_sample_hold','units':'V','detector':{},'forward_context':{'chi':1},'integration_time_s':1e-7}}
    current=deepcopy(first);current['readout_noise']['forward_context']['chi']=2
    with pytest.raises(PhysicsCalculationError,match='forward_context'):check_fixed_readout(first,current)


def test_conditional_variance_uses_independent_sinusoidal_design():
    p=operating_profile(profile(),0);report,result=fake_point(p);b=point_budget(p,report,result)
    t=np.asarray(p.run_config.tlist_us);keep=t>=20;phase=2*np.pi*p.signed_if_mhz*(t[keep]-t[keep][0])
    design=np.column_stack((np.ones(keep.sum()),np.cos(phase),np.sin(phase)))
    projection=np.linalg.pinv(design);weights=projection[1]-1j*projection[2]
    d=p.noise.detector;tau=d.integration_time_us*1e-6;gain=d.transimpedance_v_per_a
    expected={'photon_shot':elementary_charge*d.responsivity_a_per_w*1e-9/tau,
              'dark_shot':elementary_charge*d.dark_current_a/tau,
              'electronic_current':1e-14**2/(2*tau),
              'johnson_current':4*Boltzmann*300/1e9/(2*tau)}
    for name,v in expected.items():assert b['if_noise_power_by_source'][name][0]==pytest.approx(v*np.vdot(weights,weights).real*gain**2,rel=1e-12)
    assert b['total_if_noise_power'][0]==pytest.approx(sum(v[0] for v in b['if_noise_power_by_source'].values()))


def test_disabled_sources_zero_budget_and_undefined_snr():
    p=operating_profile(profile(),0);p=replace(p,noise=replace(p.noise,sources=tuple(replace(s,specification=replace(s.specification,enabled=False)) for s in p.noise.sources)))
    r,o=fake_point(p);b=point_budget(p,r,o)
    assert b['predicted_if_snr_db']==[None] and b['total_if_noise_power']==[0.]


def test_shared_execution_export_progress_and_plot(monkeypatch,tmp_path):
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    monkeypatch.setattr(engine,'_execute_isolated',fake_point)
    events=[];s=engine.execute_study(profile(),on_progress=events.append)
    assert len(s.points)==15 and s.selected_index==8
    assert events[-1].state=='completed' and events[-1].completed_points==15
    assert s.report()['selection']['sensitivity_certified'] is False
    # Save API is shared; isolate only the raw-array writer for this synthetic seam.
    import rydberg_aoa.simulation.study_output as output
    monkeypatch.setattr(output,'save_profile_results',lambda *a:None)
    save_study_results(tmp_path,s)
    assert (tmp_path/'point_0008/profile.json').exists()
    figs=create_study_figures(s)
    assert len(figs)==2
    for f in figs.values():plt.close(f)


def test_cancelled_and_failed_study_never_selects_survivors(monkeypatch):
    monkeypatch.setattr(engine,'_execute_isolated',fake_point)
    result=engine.execute_study(profile(),should_cancel=lambda:True)
    assert result.status=='cancelled' and result.selected_index is None
    def broken(p):raise RuntimeError('injected')
    monkeypatch.setattr(engine,'_execute_isolated',broken)
    with pytest.raises(PhysicsCalculationError) as e:engine.execute_study(profile())
    assert isinstance(e.value.__cause__,RuntimeError)


def test_explicit_calibration_probability_keeps_validation_target():
    from rydberg_aoa.config.detection_study import parse_detection_study
    from rydberg_aoa.simulation.detection_study import calibrate_threshold
    raw=json.loads((ROOT/'examples/noise016/prescribed_numerical_validation_pin.json').read_text())['study']
    old=parse_detection_study(raw)
    assert old.threshold_rank()==90
    raw.update(calibration_false_alarm_probability=.02,calibration_trials=199)
    new=parse_detection_study(raw);assert new.threshold_rank()==196 and new.p_false_alarm==.1
    threshold=calibrate_threshold(new,np.arange(199,dtype=float))
    assert threshold['value']==195 and threshold['calibration_false_alarm_probability']==.02


def test_real_initialization_rejects_corrupt_context_before_worker_and_rolls_back(monkeypatch):
    from rydberg_aoa.config import atomic_config as ac
    from rydberg_aoa.simulation import detector_readout as readout
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation.runner import SimulationRunner
    from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state
    published=snapshot_published_atomic_state();pipeline=ac._snapshot_pipeline_transaction()
    original=readout.resolve_detector_context;called=[]
    def corrupt(*args):
        context=original(*args)
        return replace(context,od_factor=context.od_factor*1.1)
    monkeypatch.setattr(readout,'resolve_detector_context',corrupt)
    monkeypatch.setattr(SimulationRunner,'run_prescribed_simulation',lambda *a,**kw:called.append(True))
    try:
        ac._clear_published_atomic_state()
        entry=ac._snapshot_pipeline_transaction()
        with pytest.raises(PhysicsCalculationError,match='od_factor'):
            execute_profile(operating_profile(profile(),0))
        assert called==[] and not ac.is_atomic_physics_initialized()
        after=ac._snapshot_pipeline_transaction()
        assert after['lambda_p']==entry['lambda_p']
        assert after['atomic_levels_cache']==entry['atomic_levels_cache']
        assert after['dipole_matrix_cache']==entry['dipole_matrix_cache']
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        restore_published_atomic_state(published)


def test_stale_experimental_fields_are_rejected_before_planning():
    p=profile('experimental');p=replace(p,lo=replace(p.lo,electric_field_vm=2.))
    with pytest.raises(PhysicsConfigurationError,match='request'):plan_study(p)


def test_invalid_direct_operating_profile_argument_is_typed():
    with pytest.raises(PhysicsConfigurationError):operating_profile(None,0)


@pytest.mark.parametrize('field',['kind','selection'])
def test_array_valued_direct_design_labels_are_rejected(field):
    p=profile()
    with pytest.raises(PhysicsConfigurationError):replace(p.study,**{field:np.array(['a','b'])})


@pytest.mark.parametrize('bad',[True,'0.02',0.,-.1,1.,float('nan'),1+0j])
def test_invalid_explicit_calibration_probability_is_not_coerced(bad):
    from rydberg_aoa.config.detection_study import parse_detection_study
    raw=json.loads((ROOT/'examples/noise016/prescribed_numerical_validation_pin.json').read_text())['study']
    raw['calibration_false_alarm_probability']=bad
    with pytest.raises(PhysicsConfigurationError):parse_detection_study(raw)
