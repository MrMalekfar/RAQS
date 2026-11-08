"""Independent statistical oracles, paired trials and offline/live wiring."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest
from rydberg_aoa.config.study import StudyStatistics, parse_statistics
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.simulation.study_statistics import analyze_study_statistics, load_study_statistics
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def options():
    return StudyStatistics((1,2,4),.95,400,91)


def report(policy='paired'):
    definition = dict(schema_version=1,kind='noise_strength_sweep',name='synthetic oracle',
        source='readout_gaussian',parameter='sigma_observable',values=[0.,.1],
        trials_per_point=4,seed=42,random_policy=policy,
        metrics=['aoa_bias_deg','aoa_rmse_deg','if_phase_rmse_deg_per_path',
                 'if_amplitude_relative_error_per_path','readout_trace_snr_db_per_path'],
        statistics=asdict(options()))
    points = []
    from hashlib import sha256
    for i in range(2):
        rows = [dict(trial_index=k,aoa_error_deg=e+2*i,if_phase_error_deg_per_path=[e,2*e],
            if_amplitude_absolute_relative_error_per_path=[abs(e)/10,0.],
            readout_trace_snr_db_per_path=[None,20.]) for k,e in enumerate([-3.,-1.,1.,3.])]
        seed = 42 if policy=='paired' else int.from_bytes(sha256(f'NOISE-008-v1:42:{i}'.encode()).digest()[:8],'big') & (2**53-1)
        points.append(dict(index=i,value=definition['values'][i],effective_seed=seed,effective_trials=4,summary={'trials':rows}))
    return dict(schema_version=1,study=definition,status='completed',points_requested=2,points_completed=2,points=points)


def test_known_oracles_and_whole_trial_correlation():
    analysis = analyze_study_statistics(report())
    final = analysis['checkpoints'][-1]
    m = final['points'][0]['metrics']
    assert m['aoa_bias_deg']['estimate'] == 0
    assert m['aoa_rmse_deg']['estimate'] == pytest.approx(np.sqrt(5))
    phase = m['if_phase_rmse_deg_per_path']
    assert phase[1]['estimate'] == 2*phase[0]['estimate']
    assert phase[1]['interval'] == pytest.approx(np.array(phase[0]['interval'])*2)
    assert m['if_amplitude_relative_error_per_path'][0]['estimate'] == pytest.approx(.2)
    assert m['readout_trace_snr_db_per_path'][0]['interval'] is None
    assert m['readout_trace_snr_db_per_path'][0]['status']=='undefined_trial_in_prefix'
    assert m['readout_trace_snr_db_per_path'][1]['status']=='degenerate_empirical_distribution'
    assert analysis['checkpoints'][0]['points'][0]['metrics']['aoa_bias_deg']['status']=='insufficient_trials'


def test_paired_differences_preserve_matching_trial_indices():
    paired = analyze_study_statistics(report())['checkpoints'][-1]['point_minus_first'][0]['metrics']['aoa_bias_deg']
    assert paired['estimate']==2 and paired['interval']==pytest.approx([2.,2.],abs=1e-14)
    independent = analyze_study_statistics(report('independent'))['checkpoints'][-1]['point_minus_first'][0]['metrics']['aoa_bias_deg']
    assert independent['interval'][0] < 2 < independent['interval'][1]


def test_no_survivor_statistics_and_prefix_reuse():
    data = report()
    data['points'][0]['summary']['trials'][3]['if_phase_error_deg_per_path'][0] = None
    a = analyze_study_statistics(data)
    assert a['checkpoints'][1]['points'][0]['metrics']['if_phase_rmse_deg_per_path'][0]['interval'] is not None
    assert a['checkpoints'][2]['points'][0]['metrics']['if_phase_rmse_deg_per_path'][0]['estimate'] is None
    assert a['checkpoints'][2]['point_minus_first'][0]['metrics']['if_phase_rmse_deg_per_path'][0]['status']=='undefined_comparison'


def test_offline_reproducible_nonmutating_rng_isolation(tmp_path):
    data = report(); before = deepcopy(data)
    previous = np.random.get_state()
    try:
        np.random.seed(777); state = np.random.get_state()
        result = analyze_study_statistics(data)
        assert all(np.array_equal(x,y) for x,y in zip(state,np.random.get_state()))
    finally:
        np.random.set_state(previous)
    assert data==before and result==analyze_study_statistics(data)
    path = tmp_path/'study_report.json';path.write_text(json.dumps(data))
    assert result==load_study_statistics(path)
    changed = load_study_statistics(path,StudyStatistics((2,),.8,20,4))
    assert len(changed['checkpoints'])==1
    assert changed['checkpoints'][0]['points'][0]['metrics']['aoa_rmse_deg']['estimate']==pytest.approx(np.sqrt(5))


@pytest.mark.parametrize('field,bad',[
    ('trial_checkpoints',[]),('trial_checkpoints',[2,1]),('trial_checkpoints',[True,4]),
    ('trial_checkpoints',[1,1]),('trial_checkpoints',[1,0]),('trial_checkpoints',[1,2.]),
    ('confidence_level',True),('confidence_level',0),('confidence_level',1),
    ('confidence_level',np.nan),('confidence_level',10**1000),('confidence_level','0.95'),
    ('bootstrap_replicates',True),('bootstrap_replicates',1),('seed',-1),('seed',2**53)])
def test_options_fail_loud(field,bad):
    d = asdict(options());d['trial_checkpoints']=list(d['trial_checkpoints']);d[field]=bad
    with pytest.raises(PhysicsConfigurationError):parse_statistics(d)


@pytest.mark.parametrize('damage',['missing_trial','reordered','bad_scalar','bad_path','seed','complete','unknown_metric',
                                  'wrapped_range','negative_amplitude','missing_aoa','metric_path_count'])
def test_incomplete_or_invalid_evidence_rejected(damage):
    d = report(); rows = d['points'][0]['summary']['trials']
    if damage=='missing_trial': rows.pop()
    if damage=='reordered': rows.reverse()
    if damage=='bad_scalar': rows[0]['aoa_error_deg']=True
    if damage=='bad_path': rows[0]['if_phase_error_deg_per_path']=[1.]
    if damage=='seed': d['points'][1]['effective_seed']=99
    if damage=='complete': d['points'].pop();d['points_completed']=1
    if damage=='unknown_metric': d['study']['metrics']=['fake']
    if damage=='wrapped_range': rows[0]['aoa_error_deg']=180.
    if damage=='negative_amplitude': rows[0]['if_amplitude_absolute_relative_error_per_path'][0]=-.1
    if damage=='missing_aoa': rows[0]['aoa_error_deg']=None
    if damage=='metric_path_count':
        for r in rows:r['readout_trace_snr_db_per_path']=[20.]
    with pytest.raises(PhysicsCalculationError):analyze_study_statistics(d)


def test_cancelled_is_honest_and_zero_not_certainty():
    d = report();d.update(status='cancelled',points_completed=0,points=[])
    a = analyze_study_statistics(d)
    assert a['study_status']=='cancelled' and a['checkpoints'][-1]['points']==[]
    d = report()
    for p in d['points']:
        for r in p['summary']['trials']:r['aoa_error_deg']=0.
    m=analyze_study_statistics(d)['checkpoints'][-1]['points'][0]['metrics']['aoa_rmse_deg']
    assert m==dict(estimate=0.,interval=[0.,0.],status='degenerate_empirical_distribution')


def test_profile_roundtrip_maximum_once_and_legacy_compatibility():
    d=json.loads((ROOT/'examples/noise008/prescribed_readout_sweep.json').read_text())
    original=parse_run_profile(d)
    assert original.study.statistics is None
    d['study'].update(trials_per_point=4,statistics=asdict(options()))
    d['study']['statistics']['trial_checkpoints']=[1,2,4]
    p=parse_run_profile(d)
    assert parse_run_profile(profile_document(p))==p
    from rydberg_aoa.config.study import plan_study
    assert all(j['trials']==4 for j in plan_study(p)['points'])
    d['study']['statistics']['trial_checkpoints']=[1,2]
    with pytest.raises(PhysicsConfigurationError):parse_run_profile(d)


def test_example_and_gui_cost_preview():
    from rydberg_aoa.config.study import plan_study,study_capabilities
    p=parse_run_profile(json.loads((ROOT/'examples/noise009/prescribed_readout_statistics.json').read_text(encoding='utf-8')))
    preview=plan_study(p)
    assert preview['bootstrap_index_draws']==3*1000*(5+10+30)
    assert preview['bootstrap_index_matrix_bytes']==8*1000*30
    assert preview['total_atomic_evolutions']==12
    assert study_capabilities(p)['statistics']['offline_reanalysis'] is True


def test_render_offline_checkpoint_estimates_and_bands(tmp_path):
    from rydberg_aoa.visualization.study_figures import create_statistics_figures
    import matplotlib.pyplot as plt
    figs=create_statistics_figures(analyze_study_statistics(report()))
    assert len(figs)==5
    fig=figs['study_trials_aoa_bias_deg']
    assert list(fig.axes[0].lines[0].get_xdata())==[1,2,4]
    assert len(fig.axes[0].collections)==2
    fig.savefig(tmp_path/'checkpoint.png')
    for f in figs.values():plt.close(f)


def test_offline_cli_blocks_physics_imports_and_preserves_files(tmp_path):
    import subprocess
    import sys
    source=tmp_path/'study_report.json';source.write_text(json.dumps(report()))
    target=tmp_path/'analysis.json'
    script = '''import sys, importlib.abc, runpy
class NoPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('arc','qutip') or fullname.endswith(('profile_execution','atomic_config')):
            raise AssertionError('offline analysis imported physics: '+fullname)
sys.meta_path.insert(0,NoPhysics())
sys.argv=['study_statistics',sys.argv[1],'--output',sys.argv[2]]
runpy.run_module('rydberg_aoa.simulation.study_statistics',run_name='__main__')
'''
    before=source.read_bytes()
    process=subprocess.run([sys.executable,'-B','-c',script,str(source),str(target)],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
    assert source.read_bytes()==before
    result=json.loads(target.read_text())
    assert result==json.loads(json.dumps(analyze_study_statistics(report())))
    from rydberg_aoa.simulation.study_statistics import save_statistics_analysis
    with pytest.raises(FileExistsError):save_statistics_analysis(target,result)


def test_checkpoint_and_seed_changes_do_not_change_physical_streams():
    from rydberg_aoa.config.study import effective_point_profile
    d=json.loads((ROOT/'examples/noise008/prescribed_readout_sweep.json').read_text())
    p=parse_run_profile(d)
    d['study']['statistics']=dict(trial_checkpoints=[1,3],confidence_level=.95,bootstrap_replicates=40,seed=999)
    augmented=parse_run_profile(d)
    assert all(effective_point_profile(p,i)==effective_point_profile(augmented,i) for i in [None,0,1,2])


def test_additional_checkpoint_does_not_redraw_existing_bootstrap():
    a=analyze_study_statistics(report())
    b=analyze_study_statistics(report(),StudyStatistics((2,4),.95,400,91))
    assert a['checkpoints'][1:]==b['checkpoints']


def test_legacy_report_requires_explicit_options_and_rejects_more_trials():
    d=report();d['study'].pop('statistics')
    with pytest.raises(PhysicsConfigurationError):analyze_study_statistics(d)
    assert analyze_study_statistics(d,options())['checkpoints'][-1]['trials']==4
    with pytest.raises(PhysicsConfigurationError):
        analyze_study_statistics(d,StudyStatistics((5,),.95,10,91))
    d['random_policy']='independent'
    with pytest.raises(PhysicsCalculationError):analyze_study_statistics(d,options())


def test_extreme_finite_percentiles_do_not_overflow():
    d=report()
    for p in d['points']:
        for k,r in enumerate(p['summary']['trials']):
            r['readout_trace_snr_db_per_path'][1]=(-1 if k%2 else 1)*1e308
    a=analyze_study_statistics(d)
    bounds=a['checkpoints'][-1]['points'][0]['metrics']['readout_trace_snr_db_per_path'][1]['interval']
    assert np.isfinite(bounds).all()


@pytest.mark.parametrize('mode',['prescribed','experimental'])
def test_actual_study_saved_uncertainty_offline_matches_live(tmp_path,mode):
    from rydberg_aoa.simulation.study_execution import execute_study
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    d=json.loads((ROOT/'examples/noise008/prescribed_readout_sweep.json').read_text())
    if mode=='experimental':
        experimental=json.loads((ROOT/'examples/noise008/experimental_rin_sweep.json').read_text())
        experimental.update(noise=d['noise'],study=d['study'])
        d=experimental
    d['study'].update(values=[0.,1e-5],trials_per_point=2,
        statistics=dict(trial_checkpoints=[1,2],confidence_level=.95,bootstrap_replicates=30,seed=91))
    p=parse_run_profile(d)
    out=execute_study(p)
    analysis=out.report()['uncertainty']
    assert [c['trials'] for c in analysis['checkpoints']]==[1,2]
    assert len(out.points)==2
    assert all(len(point.summary['trials'])==2 for point in out.points)
    save_study_results(tmp_path,out)
    assert load_study_statistics(tmp_path/'study_report.json')==json.loads(json.dumps(analysis))
    assert parse_run_profile(json.loads((tmp_path/'study_profile.json').read_text()))==p
    figures=create_study_figures(out)
    assert 'study_trials_aoa_rmse_deg' in figures
    figures['study_trials_if_phase_rmse_deg_per_path'].savefig(tmp_path/'checkpoint.png')
    figures['study_if_phase_rmse_deg_per_path'].savefig(tmp_path/'sweep.png')
    for f in figures.values():plt.close(f)
