"""Nested OU law, certified seam, truthful budgets and profile/API wiring."""
from dataclasses import replace, fields
from pathlib import Path
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.config.study import plan_study,study_capabilities
from rydberg_aoa.config.noise_convergence import convergence_profile
from rydberg_aoa.simulation import noise_convergence as engine
from rydberg_aoa.simulation.rf_drive_noise import RFDriveRealization,build_rf_realization
from rydberg_aoa.simulation.rf_drive_campaign import validate_drive_realizations
from rydberg_aoa.exceptions import PhysicsConfigurationError,PhysicsCalculationError

ROOT=Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT/f'examples/noise010/{mode}_grid_convergence.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('field,bad',[('seed',True),('trials',0),('schema_version',True),('name',''),
    ('output_samples',[601,301]),('output_samples',[True,601]),('output_samples',[2,601]),
    ('noise_steps_us',[.1,.2]),('noise_steps_us',[.2,.07]),('noise_steps_us',[0.,.1]),
    ('noise_steps_us',[True,.1]),('metrics',[]),('metrics',['fake']),
    ('budgets',{'if_phase_deg':True}),('budgets',{'if_phase_deg':-1}),('budgets',{'unselected':1})])
def test_bad_study_pre_dispatch(field,bad,monkeypatch):
    d=document();d['study'][field]=bad
    monkeypatch.setattr(engine,'_isolated_case',lambda *a:pytest.fail('invalid dispatch'))
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


def test_all_grids_and_readout_prevalidated():
    d=document();d['study']['output_samples']=[10,601]
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)
    d=document();d['noise']['readout']=json.loads((ROOT/'examples/noise003/prescribed_campaign.json').read_text())['noise']
    d['noise']['readout'].update(seed=d['noise']['seed'],trials=d['noise']['trials'])
    with pytest.raises(PhysicsConfigurationError,match='readout'):parse_run_profile(d)


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_gui_json_roundtrip_and_cost(mode):
    p=parse_run_profile(document(mode));before=copy.deepcopy(p)
    assert parse_run_profile(profile_document(p))==p
    preview=plan_study(p)
    assert preview['cases']==3 and preview['canonical_evolutions']==27
    assert 'noise_grid_convergence' in study_capabilities(p)['kinds']
    for count in p.study.output_samples:
        case=convergence_profile(p,count,.1)
        assert case.probe is p.probe and case.request is p.request
        assert case.run_config.doa_options==p.run_config.doa_options
        assert parse_run_profile(profile_document(case))==case
    assert p==before


def test_exact_common_knots_and_interpolant_difference():
    p=parse_run_profile(document())
    fine=build_rf_realization(convergence_profile(p,601,.1).noise,trial=0)
    coarse=engine.nested_realization(fine,2)
    for f in fields(RFDriveRealization):
        assert getattr(coarse,f.name)==getattr(fine,f.name)[::2]
    assert coarse.laser_intensity_at(.2)==fine.laser_intensity_at(.2)
    assert coarse.laser_intensity_at(.1)!=fine.laser_intensity_at(.1)
    assert engine.realization_digest(fine)!=engine.realization_digest(coarse)
    validate_drive_realizations(convergence_profile(p,601,.2).noise,(coarse,coarse))
    with pytest.raises(PhysicsConfigurationError):engine.nested_realization(fine,7)


def test_decimated_ou_marginal_covariance_independent_population_oracle():
    p=parse_run_profile(document());cfg=convergence_profile(p,601,.1).noise
    cfg=replace(cfg,trials=2000,drive=replace(cfg.drive,horizon_us=1.))
    sigma=np.sqrt(np.log1p(.02**2));pairs=[]
    for i in range(cfg.trials):
        r=engine.nested_realization(build_rf_realization(cfg,trial=i),2)
        x=np.log(r.probe_intensity)+.5*sigma**2
        pairs.append([x[0],x[2]])
    a=np.asarray(pairs)
    assert np.var(a[:,0])/sigma**2==pytest.approx(1,abs=.08)
    assert np.cov(a.T,ddof=0)[0,1]/sigma**2==pytest.approx(np.exp(-.4/5),abs=.08)


def test_wrong_injected_grid_and_disabled_channel_fail():
    p=parse_run_profile(document());cfg=convergence_profile(p,601,.1).noise
    r=build_rf_realization(cfg,trial=0)
    with pytest.raises(PhysicsConfigurationError):validate_drive_realizations(cfg,(r,))
    with pytest.raises(PhysicsConfigurationError):validate_drive_realizations(cfg,(engine.nested_realization(r,2),)*2)
    forged=replace(r,lo_amplitude=(1.1,)*len(r.times_us))
    with pytest.raises(PhysicsConfigurationError,match='disabled'):validate_drive_realizations(cfg,(forged,forged))


def test_all_seven_noise_channels_coupled_at_common_nodes():
    d=document()
    for filename in ['examples/noise005/prescribed_rf.json','examples/noise006/prescribed_laser_frequency.json']:
        d['noise']['sources'].update(json.loads((ROOT/filename).read_text())['noise']['sources'])
    d['noise']['sources']['laser_coupling_intensity']['enabled']=True
    p=parse_run_profile(d);cfg=convergence_profile(p,601,.1).noise
    fine=build_rf_realization(cfg,trial=0);coarse=engine.nested_realization(fine,2)
    assert all(getattr(coarse,f.name)==getattr(fine,f.name)[::2] for f in fields(RFDriveRealization))
    assert all(np.std(getattr(fine,name))>0 for name in
               ('lo_amplitude','signal_amplitude','relative_phase_rad','probe_frequency_hz',
                'coupling_frequency_hz','probe_intensity','coupling_intensity'))


@pytest.mark.parametrize('supplied',[False,True])
def test_campaign_uses_explicit_inputs_or_canonical_builder_without_shadowing(monkeypatch,supplied):
    from rydberg_aoa.simulation import rf_drive_campaign as campaign
    p=parse_run_profile(document());cfg=convergence_profile(p,601,.1).noise
    paths=tuple(build_rf_realization(cfg,trial=i) for i in range(cfg.trials));calls=[];drives=[]
    def builder(config,*,trial):
        assert not supplied;calls.append(trial);return paths[trial]
    monkeypatch.setattr(campaign,'build_rf_realization',builder)
    times=np.asarray(p.run_config.tlist_us);traces=np.tile(np.sin(times),(3,1));runtime=object()
    class Runner:
        def _run_resolved_sensing(self,*a,**kwargs):
            drives.append(kwargs.get('_drive'))
            return dict(worker_records=[(1+0j,)]*3,time_traces=traces,tlist=times,estimated_aoa=-30.,
                        complex_responses=np.ones(3,dtype=complex))
        def _current_explicit_runtime(self,*a):return runtime
        def _publish_explicit_result(self,*a):pass
    out=campaign.run_rf_drive_campaign(Runner(),p,p.run_config,runtime,'prescribed',cfg,None,
                                      **({'realizations':paths} if supplied else {}))
    assert calls==([] if supplied else [0,1])
    assert tuple(drives[1:])==paths and out['rf_drive_realizations']==list(paths)
    assert out['rf_drive_campaign']['realization_source']==('explicit_validated_diagnostic' if supplied else 'canonical_rng')


def synthetic(vectors,angles):
    return {'rf_drive_trial_results':[{'complex_responses':np.asarray(v,dtype=complex),'estimated_aoa':a}
            for v,a in zip(vectors,angles)]}


def test_independent_metric_oracles_and_strict_budgets():
    d=document();d['study']['budgets']={'if_response_absolute':1.,'if_phase_deg':90.,'aoa_deg':0.}
    study=parse_run_profile(d).study
    ref=synthetic([[1,2],[1,2]],[-30,-30])
    case=synthetic([[1j,3],[1,2]],[-30,-30])
    c=engine.compare_convergence_case(ref,case,study)['metrics']
    assert c['if_response_absolute']['maximum']==pytest.approx(np.sqrt(2))
    assert c['if_response_absolute']['status']=='failed_budget'
    assert c['if_phase_deg']['maximum']==90 and c['if_phase_deg']['status']=='passed_budget'
    assert c['if_amplitude_relative']['maximum']==.5 and c['if_amplitude_relative']['status']=='unassessed'
    assert c['aoa_deg']['maximum']==0
    assert c['aoa_deg']['status']=='passed_budget'  # not sufficient for the other metrics


def test_zero_amplitude_no_survivor_metric_or_fake_success():
    study=parse_run_profile(document()).study
    ref=synthetic([[0,1],[1,1]],[-30,-30]);case=synthetic([[1,0],[1,1]],[-30,-30])
    result=engine.compare_convergence_case(ref,case,study)
    assert result['metrics']['if_phase_deg']['maximum'] is None
    assert result['metrics']['if_response_relative']['status']=='unassessed'
    assert result['metrics']['if_response_absolute']['maximum']==1
    case['rf_drive_trial_results'].pop()
    with pytest.raises(PhysicsCalculationError):engine.compare_convergence_case(ref,case,study)


def mock_run(profile,realizations):
    result=synthetic([[1,2]]*profile.noise.trials,[-30]*profile.noise.trials)
    for trial in result['rf_drive_trial_results']:trial['tlist']=profile.run_config.tlist_us
    result['rf_drive_realizations']=list(realizations)
    report={k:{} for k in ('applied_fields','levels','rf_orientation','detunings_rad_per_us',
                           'rabi_rad_per_us','decay_rates_per_us','resolved_rate_budget_per_us')}
    return report,result


def test_production_dispatch_preserves_paths_and_shared_draws(monkeypatch):
    p=parse_run_profile(document());seen=[]
    def run(case,realizations):
        seen.append(realizations);return mock_run(case,realizations)
    monkeypatch.setattr(engine,'_isolated_case',run)
    events=[]
    from rydberg_aoa.simulation.study_execution import execute_study
    out=execute_study(p,on_progress=events.append)
    assert out.status=='completed' and len(out.points)==3
    assert seen[0] is seen[1]
    assert seen[2][0].probe_intensity==seen[0][0].probe_intensity[::2]
    assert events[-1].state=='completed' and events[1].point_result.index==0
    assert out.report()['points'][0]['comparison']['self_reference'] is True
    assert all(m['status']=='unassessed' for m in out.points[1].comparison['metrics'].values())


def test_failure_context_and_cancellation_no_fake_completion(monkeypatch):
    p=parse_run_profile(document());calls=[]
    def broken(case,r):
        calls.append(1)
        if len(calls)>1:raise ValueError('injected')
        return mock_run(case,r)
    monkeypatch.setattr(engine,'_isolated_case',broken)
    with pytest.raises(PhysicsCalculationError,match='case 1') as caught:engine.execute_convergence_study(p)
    assert isinstance(caught.value.__cause__,ValueError) and len(calls)==2
    calls.clear()
    out=engine.execute_convergence_study(p,should_cancel=lambda:len(calls)==1)
    assert out.status=='cancelled' and len(out.points)==1
    with pytest.raises(PhysicsConfigurationError):engine.execute_convergence_study(p,should_cancel=lambda:1)


@pytest.mark.parametrize('invariant',['levels','resolved_medium','signed_if_mhz','doa_effective_settings','realizations'])
def test_invariants_and_realization_substitution_are_rejected(monkeypatch,invariant):
    p=parse_run_profile(document());calls=[]
    def changed(case,r):
        calls.append(1);report,result=mock_run(case,r)
        if len(calls)>1:
            if invariant=='realizations':result['rf_drive_realizations']=[]
            elif invariant=='doa_effective_settings':report['sensing']={'doa_effective_settings':{'fake':1}}
            else:report[invariant]={'fake':1}
        return report,result
    monkeypatch.setattr(engine,'_isolated_case',changed)
    with pytest.raises(PhysicsCalculationError,match='case 1'):engine.execute_convergence_study(p)


def test_invalid_injection_fails_before_atomic_initialization(monkeypatch):
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.config import atomic_config as ac
    monkeypatch.setattr(ac,'initialize_atomic_physics',lambda *a,**k:pytest.fail('reached atomic init'))
    case=convergence_profile(parse_run_profile(document()),601,.1)
    with pytest.raises(PhysicsConfigurationError):execute_profile(case,drive_realizations=())


def test_cli_profile_dispatch_without_duplicate_physics(tmp_path,monkeypatch,capsys):
    import cli
    import sys
    monkeypatch.setattr(engine,'_isolated_case',mock_run)
    path=tmp_path/'input.json';path.write_text(json.dumps(document()))
    monkeypatch.setattr(sys,'argv',['main.py','--profile',str(path),'--no-plots'])
    report,out=cli.main()
    assert report['kind']=='noise_grid_convergence' and out.status=='completed'
    assert len(out.points)==3
    assert 'noise_grid_convergence' in capsys.readouterr().out


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_canonical_campaign_and_anchor(tmp_path,mode):
    from rydberg_aoa.simulation.study_execution import _execute_isolated
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    d=document(mode);d['study']['trials']=1
    p=parse_run_profile(d)
    out=engine.execute_convergence_study(p)
    assert len(out.points)==3
    reference=out.points[0]
    _,canonical=_execute_isolated(reference.profile)
    assert canonical['rf_drive_realizations']==reference.result['rf_drive_realizations']
    for a,b in zip(canonical['rf_drive_trial_results'],reference.result['rf_drive_trial_results']):
        assert np.array_equal(a['complex_responses'],b['complex_responses'])
    save_study_results(tmp_path,out)
    assert parse_run_profile(json.loads((tmp_path/'study_profile.json').read_text()))==p
    report=json.loads((tmp_path/'study_report.json').read_text())
    assert report['points'][1]['realization_sha256']==report['points'][0]['realization_sha256']
    assert report['points'][2]['realization_sha256']!=report['points'][0]['realization_sha256']
    with np.load(tmp_path/'point_0002'/'simulation_arrays.npz',allow_pickle=False) as arrays:
        assert np.array_equal(arrays['trial_laser_probe_intensity'][0],reference.result['rf_drive_realizations'][0].probe_intensity[::2])
    figures=create_study_figures(out)
    figures['study_grid_if_response_relative'].savefig(tmp_path/'grid.png')
    for figure in figures.values():plt.close(figure)
    with pytest.raises(FileExistsError):save_study_results(tmp_path,out)
