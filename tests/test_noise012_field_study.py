"""Independent response/derivative oracles and actual two-mode pipeline wiring."""
from copy import deepcopy
from dataclasses import replace, asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from rydberg_aoa.config.field_study import SignalFieldStudy, field_point_profile
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.study import plan_study, study_capabilities
from rydberg_aoa.simulation.field_study import analyze_field_response
from rydberg_aoa.simulation import study_execution as engine
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT/f'examples/noise012/{mode}_field_response.json').read_text())


def definition(values=(1.,2.,5.,9.), region=None, linear_budget=1e-8, derivative_budget=1e-8):
    return SignalFieldStudy(1,'signal_field_sweep','Independent oracle',tuple(values),
        tuple(region or (values[0], values[-1])),'affine',linear_budget,'midpoint_refinement',derivative_budget)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_point_updates_only_signal_field_and_correct_request(mode):
    p = parse_run_profile(document(mode)); before = deepcopy(p)
    points = [field_point_profile(p, i) for i in range(5)]
    assert p == before and [x.signal.electric_field_vm for x in points] == pytest.approx([.001,.002,.003,.004,.005])
    assert plan_study(p)['total_atomic_evolutions'] == 15
    assert 'signal_field_sweep' in study_capabilities(p)['kinds']
    for point in points:
        assert point.study is None and point.noise is p.noise
        assert point.probe is p.probe and point.coupling is p.coupling and point.lo is p.lo
        assert point.signal.frequency_hz == p.signal.frequency_hz and point.signal.polarization == p.signal.polarization
        assert point.signed_if_mhz == p.signed_if_mhz
        if mode == 'experimental':
            assert point.request.signal == point.signal
            assert point.request.preparation is p.request.preparation and point.request.bounds is p.request.bounds
        else:
            assert point.request is p.request
        assert parse_run_profile(profile_document(point)) == point
    assert parse_run_profile(profile_document(p)) == p


@pytest.mark.parametrize('field,value',[
    ('values',[0.,.003,.005]),('values',[True,.003,.005]),('values',['.001',.003,.005]),
    ('values',[.001,.005,.003]),('values',[.001,.001,.005]),('values',[.001,.005]),
    ('values',[.001, float('nan'),.005]),('values',[.001,10**1000,1e308]),
    ('linear_region_v_per_m',[.002,.005]),('linear_region_v_per_m',[.001,.003]),
    ('linearity_max_relative_error',-1),('linearity_max_relative_error',True),
    ('derivative_max_relative_difference','1'),('derivative_max_relative_difference',float('inf')),
    ('linearity_model','through_origin'),('derivative_check','automatic_until_green'),('schema_version',True)])
def test_invalid_declarations_fail_before_physics(field,value):
    d = document(); d['study'][field] = value
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(d)


def test_no_representable_midpoint_rejected():
    a = 1.; b = np.nextafter(a,2.); c = np.nextafter(b,2.)
    with pytest.raises(PhysicsConfigurationError,match='midpoint'):
        definition((a,b,c))


def test_noise_is_not_silently_disabled_and_zero_field_not_invented():
    d = document(); d['noise']['sources']['photon_shot']['enabled'] = True
    with pytest.raises(PhysicsConfigurationError,match='explicitly disabled'):
        parse_run_profile(d)
    d = document(); d['noise']['trials'] = 2
    with pytest.raises(PhysicsConfigurationError,match='trials=1'):
        parse_run_profile(d)
    d = document(); d.pop('noise')
    assert parse_run_profile(d).noise is None  # observable-only deterministic measurement


def test_nonuniform_quadratic_derivatives_and_affine_linearity():
    spec = definition(); x = np.array(spec.refined_values())
    y = np.column_stack((2*x+3,x**2+1))
    before = y.copy()
    output = analyze_field_response(spec,x,y,'V')
    assert np.asarray(output['responsivity_refined_per_path'])[:,0] == pytest.approx(np.full(len(x),2.))
    assert np.asarray(output['responsivity_refined_per_path'])[:,1] == pytest.approx(2*x)
    assert np.asarray(output['responsivity_base_per_path'])[:,1] == pytest.approx(2*x[::2])
    assert output['derivative_comparison']['status_per_path'] == ['passed','passed']
    assert output['linearity']['slope_per_path'][0] == pytest.approx(2.)
    assert output['linearity']['intercept_per_path'][0] == pytest.approx(3.)
    assert output['linearity']['status_per_path'] == ['passed','failed']
    assert output['responsivity_unit'] == 'V/(V/m)' and np.array_equal(y,before)


def test_midpoints_expose_cubic_spacing_error_and_hidden_nonlinearity():
    spec = definition((1.,2.,4.), derivative_budget=.001)
    x = np.array(spec.refined_values())
    out = analyze_field_response(spec,x,(x**3)[:,None],'A')
    assert out['responsivity_base_per_path'][0][0] == pytest.approx(0.,abs=1e-13)
    assert out['responsivity_refined_per_path'][0][0] == pytest.approx(2.5)
    assert out['derivative_comparison']['status_per_path'] == ['failed']
    spec = definition((1.,2.,3.))
    x = np.array(spec.refined_values()); y = 2*x+3; y[1] += 5
    out = analyze_field_response(spec,x,y[:,None],'Im<sigma21>')
    assert out['linearity']['slope_per_path'] == pytest.approx([2.])
    assert out['linearity']['status_per_path'] == ['failed']
    assert out['linearity']['checked_indices'] == [0,1,2,3,4]


def test_fixed_region_never_drops_outside_points_and_zero_fit_is_undefined():
    spec = definition((1.,2.,3.,4.,5.),region=(1.,3.))
    x = np.array(spec.refined_values()); y = 2*x+1
    y[x>3] += 100
    out = analyze_field_response(spec,x,y[:,None],'A')
    assert out['linearity']['status_per_path'] == ['passed']
    assert out['amplitude_per_path'][-1][0] == 111.
    assert len(out['fields_v_per_m']) == 9
    flat = analyze_field_response(spec,x,np.zeros((len(x),1)),'A')
    assert flat['linearity']['status_per_path'] == ['undefined_zero_fitted_amplitude']
    assert flat['derivative_comparison']['status_per_path'] == ['undefined_zero_reference_derivative']
    assert flat['mdf_nef_computed'] is False


@pytest.mark.parametrize('damage',['complex','nan','bool','shape','incomplete','unit','overflow'])
def test_invalid_analysis_does_not_repair_or_clip(damage):
    spec = definition(); x = np.array(spec.refined_values()); y = np.ones((len(x),2)); unit='V'
    if damage == 'complex': y = y.astype(complex)
    if damage == 'bool': y = y.astype(bool)
    if damage == 'nan': y[0,0] = np.nan
    if damage == 'shape': y = y[:,0]
    if damage == 'incomplete': y=y[:-1]; x=x[:-1]
    if damage == 'unit': unit='arbitrary_made_up_gain'
    if damage == 'overflow':
        spec=definition((1e-310,2e-310,3e-310)); x=np.array(spec.refined_values()); y=x[:,None]/1e-310
    with pytest.raises(PhysicsCalculationError):
        analyze_field_response(spec,x,y,unit)


def fake_run(profile):
    # A declared linear transfer oracle for orchestration, not simulated physics.
    t = np.asarray(profile.run_config.tlist_us); e = profile.signal.electric_field_vm
    phi = np.array([0.,-.2,-.4]); a = e*np.array([2.,3.,4.])*np.exp(1j*phi)
    report = {'applied_fields': {k:asdict(getattr(profile,k)) for k in ('probe','coupling','lo','signal')},
        'mode':profile.mode,'noise_included':False,'signed_if_mhz':profile.signed_if_mhz,
        'input_provenance': {'classification':'simulation_assumptions','note':profile.provenance.note,
            'experimental_calibration_verified':False,
            'experimental_mode_meaning':'RF-first atomic resolution, not evidence of hardware calibration'},
        'levels':{'fixture':'fixed'},'rf_orientation':{'direction':-1},'detunings_rad_per_us':(1.,2.,3.),
        'decay_rates_per_us':(1.,2.,3.),'resolved_rate_budget_per_us':{'fixed':True},
        'quantization_axis':profile.quantization_axis,'rabi_rad_per_us':(1.,2.,3.,5*e),
        'sensing':{'estimated_aoa_deg':-30.,'execution':asdict(profile.run_config)}}
    result = {'complex_responses':a,'element_phases_rad':phi,'tlist':t,
              'readout_noise':{'units':'V'}}
    return report,result


def test_execution_progress_cancel_and_failure_preserve_all_evidence(monkeypatch):
    p = parse_run_profile(document()); calls=[]; events=[]
    def run(profile):
        calls.append(profile); return fake_run(profile)
    monkeypatch.setattr(engine,'_execute_isolated',run)
    out = engine.execute_study(p,on_progress=events.append)
    assert len(calls) == 5 and len(out.points) == 5 and events[-1].state == 'completed'
    assert out.analysis['linearity']['status_per_path'] == ['passed']*3
    calls.clear()
    partial = engine.execute_study(p,should_cancel=lambda:len(calls)>=1)
    assert partial.status == 'cancelled' and len(partial.points) == 1 and partial.analysis is None
    assert partial.report()['points_requested'] == 5 and partial.report()['analysis_status'] == 'incomplete_grid'
    failure=PhysicsCalculationError('injected worker failure'); calls.clear()
    def broken(profile):
        calls.append(profile)
        if len(calls)==3: raise failure
        return fake_run(profile)
    monkeypatch.setattr(engine,'_execute_isolated',broken)
    with pytest.raises(PhysicsCalculationError,match='field point 2') as error:
        engine.execute_study(p)
    assert error.value.__cause__ is failure and len(calls)==3


def test_analysis_failure_emits_failed_and_preserves_exception(monkeypatch):
    from rydberg_aoa.simulation import field_study as module
    p=parse_run_profile(document()); events=[]
    monkeypatch.setattr(engine,'_execute_isolated',fake_run)
    error=PhysicsCalculationError('injected derivative analysis failure')
    def broken(*args,**kwargs):raise error
    monkeypatch.setattr(module,'analyze_field_response',broken)
    with pytest.raises(PhysicsCalculationError) as caught:
        engine.execute_study(p,on_progress=events.append)
    assert caught.value is error and events[-1].state=='failed' and events[-1].completed_points==5


@pytest.mark.parametrize('damage',['field','levels','rabi','rabi_nonfinite','units','grid','noise'])
def test_stale_or_changed_runtime_aborts_field_study(monkeypatch,damage):
    p=parse_run_profile(document()); calls=[]
    def run(profile):
        report,result=fake_run(profile); calls.append(profile)
        if len(calls)>1:
            if damage=='field':report['applied_fields']['signal']['electric_field_vm']=1.
            if damage=='levels':report['levels']={'fixture':'changed'}
            if damage=='rabi':report['rabi_rad_per_us']=(1.,2.,3.,999.)
            if damage=='rabi_nonfinite':report['rabi_rad_per_us']=(1.,2.,3.,float('inf'))
            if damage=='units':result['readout_noise']['units']='A'
            if damage=='grid':result['tlist']=result['tlist'][:-1]
            if damage=='noise':report['noise_included']=True
        return report,result
    monkeypatch.setattr(engine,'_execute_isolated',run)
    with pytest.raises(PhysicsCalculationError): engine.execute_study(p)
    assert len(calls)==2


def test_saved_profile_and_plots_have_real_fields_units_and_all_points(tmp_path,monkeypatch):
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    p=parse_run_profile(document()); monkeypatch.setattr(engine,'_execute_isolated',fake_run)
    out=engine.execute_study(p); data=out.report()
    from rydberg_aoa.simulation.study_statistics import analyze_study_statistics
    with pytest.raises(PhysicsConfigurationError,match='noise-strength'):
        analyze_study_statistics(data)
    assert json.loads(json.dumps(data))['analysis']['responsivity_unit']=='V/(V/m)'
    figs=create_study_figures(out)
    assert set(figs)=={'field_if_amplitude','field_responsivity','field_linearity_deviation','field_derivative_spacing'}
    assert np.array_equal(figs['field_if_amplitude'].axes[0].lines[0].get_xdata(),p.study.refined_values())
    assert 'V/(V/m)' in figs['field_responsivity'].axes[0].get_ylabel()
    lines=figs['field_responsivity'].axes[0].lines
    assert all(lines[i].get_color()==lines[i+1].get_color() for i in range(0,6,2))
    for fig in figs.values():plt.close(fig)


def test_offline_configuration_analysis_imports_no_arc_solver_or_rng():
    code='''
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('arc','qutip') or name.endswith(('parameters','atomic_config','noise_random')):
            raise AssertionError('offline field study imported physics: '+name)
sys.meta_path.insert(0,Guard())
from rydberg_aoa.config.run_profile import load_run_profile
from rydberg_aoa.config.study import plan_study
from rydberg_aoa.simulation.field_study import analyze_field_response
import numpy as np
p=load_run_profile('examples/noise012/prescribed_field_response.json')
x=np.array(p.study.refined_values())
assert plan_study(p)['total_atomic_evolutions']==15
assert analyze_field_response(p.study,x,(2*x)[:,None],'A')['mdf_nef_computed'] is False
'''
    child=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert child.returncode==0,child.stderr


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_field_study_uses_canonical_pipeline_and_exports(tmp_path,mode):
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    p=parse_run_profile(document(mode))
    out=engine.execute_study(p)
    assert len(out.points)==5 and out.status=='completed'
    assert out.analysis['response_unit']=='V'
    assert np.isfinite(out.analysis['responsivity_refined_per_path']).all()
    assert out.points[0].report['levels']==out.points[-1].report['levels']
    assert out.points[-1].report['rabi_rad_per_us'][3]/out.points[0].report['rabi_rad_per_us'][3]==pytest.approx(5.)
    assert all(point.report['noise_included'] is False for point in out.points)
    assert p.signal.electric_field_vm==.005
    save_study_results(tmp_path/mode,out)
    stored=json.loads((tmp_path/mode/'study_report.json').read_text())
    assert stored['analysis']==out.analysis and stored['points_completed']==5
    for i,point in enumerate(out.points):
        loaded=json.loads((tmp_path/mode/f'point_{i:04d}/profile.json').read_text())
        assert parse_run_profile(loaded)==point.profile
    figs=create_study_figures(out)
    for fig in figs.values():plt.close(fig)
