"""RIN: independent power/Rabi/absorption oracles and real certified solves."""
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt
from scipy.constants import elementary_charge as Q_E

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.rf_drive_noise import build_rf_realization, RFDriveRealization
from rydberg_aoa.simulation.detector_readout import DetectorContext, detector_forward, apply_pin_readout
from rydberg_aoa.simulation import time_domain as td, runner as module
from tests.test_lab005_laboratory_execution import isolation, request, plan, fake_pool, oracle_record
from tests.test_noise004_detector import config as detector_config

ROOT = Path(__file__).resolve().parents[1]


def declaration():
    return json.loads((ROOT/'examples/noise007/prescribed_rin.json').read_text())['noise']


@pytest.mark.parametrize('field,bad', [('enabled', 1), ('fractional_rms', True),
    ('fractional_rms', np.bool_(True)), ('fractional_rms', '0.1'), ('fractional_rms', 1j),
    ('fractional_rms', np.array(.1)), ('fractional_rms', -1.), ('fractional_rms', np.nan),
    ('fractional_rms', np.inf), ('fractional_rms', 10**1000), ('fractional_rms', 1e-200),
    ('correlation_time_us', 0), ('units', 'dBc/Hz'), ('budget', 'total_including_shot')])
def test_invalid_intensity_contract_even_disabled(field, bad):
    data = declaration(); data['enabled'] = False
    data['sources']['laser_probe_intensity'][field] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


def test_unresolved_nonzero_intensity_fails_instead_of_becoming_zero():
    data = declaration()
    for source in data['sources'].values():
        source['fractional_rms'] = 1e-20
    with pytest.raises(PhysicsCalculationError, match='unresolved'):
        build_rf_realization(parse_noise_config(data), trial=0)


def test_missing_technical_budget_rejected():
    data=declaration()
    del data['sources']['laser_probe_intensity']['budget']
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


def test_mean_rms_and_intensity_covariance_independent_oracle():
    data = declaration(); data['trials'] = 4000
    data['drive'].update(horizon_us=1., noise_step_us=.2)
    data['sources']['laser_probe_intensity']['fractional_rms'] = .3
    cfg = parse_noise_config(data)
    samples = np.array([build_rf_realization(cfg, trial=k).probe_intensity[:2] for k in range(cfg.trials)])
    rms=.3; variance=np.log1p(rms*rms); a=np.exp(-.2/5.)
    assert abs(samples[:,0].mean()-1) < 6*rms/np.sqrt(cfg.trials)
    # Exact lognormal fourth central moment gives variance-estimation uncertainty.
    fourth = np.exp(6*variance)-4*np.exp(3*variance)+6*np.exp(variance)-3
    assert abs(samples[:,0].var(ddof=1)-rms*rms) < 6*np.sqrt((fourth-rms**4)/cfg.trials)
    logs=np.log(samples)+variance/2
    assert abs(np.mean(logs[:,0]*logs[:,1])/variance-a) < 6*np.sqrt((1+a*a)/cfg.trials)
    assert np.all(samples>0)


def test_stream_independence_toggles_interpolation_and_pickle():
    cfg = parse_noise_config(declaration()); r = build_rf_realization(cfg,trial=0)
    assert pickle.loads(pickle.dumps(r)) == r
    assert r.laser_intensity_at(.1) == pytest.approx(tuple((a[0]+a[1])/2 for a in (r.probe_intensity,r.coupling_intensity)))
    for source in cfg.sources:
        changed = replace(cfg,sources=tuple(replace(s,specification=replace(s.specification,enabled=False)) if s.name==source.name else s for s in cfg.sources))
        other = build_rf_realization(changed,trial=0)
        keep = 'coupling_intensity' if source.name=='laser_probe_intensity' else 'probe_intensity'
        assert getattr(other,keep)==getattr(r,keep)
    extended = build_rf_realization(replace(cfg,trials=5,drive=replace(cfg.drive,horizon_us=240)),trial=0)
    assert extended.probe_intensity[:len(r.times_us)] == r.probe_intensity
    data=json.loads((ROOT/'examples/noise005/prescribed_rf.json').read_text())['noise']
    data['sources'].update(json.loads((ROOT/'examples/noise006/prescribed_laser_frequency.json').read_text())['noise']['sources'])
    original=build_rf_realization(parse_noise_config(data),trial=0)
    data['sources'].update(declaration()['sources'])
    mixed=build_rf_realization(parse_noise_config(data),trial=0)
    assert original.at(3.14)==mixed.at(3.14)
    assert original.probe_frequency_hz==mixed.probe_frequency_hz


@pytest.mark.parametrize('kind',['master','sources','zero'])
def test_disabled_and_zero_bit_identical_without_rng(isolation,monkeypatch,kind):
    fake_pool(monkeypatch,[])
    req,run=request(isolation),plan()
    baseline=module.SimulationRunner().run_laboratory_simulation(req,run)
    data=declaration()
    if kind=='master': data['enabled']=False
    else:
        for s in data['sources'].values():
            s['fractional_rms' if kind=='zero' else 'enabled']=0. if kind=='zero' else False
    monkeypatch.setattr(np.random,'SeedSequence',lambda *a,**kw:pytest.fail('disabled RIN RNG'))
    out=module.SimulationRunner().run_laboratory_simulation(req,run,noise=parse_noise_config(data))
    assert np.array_equal(out['time_traces'],baseline['time_traces'])
    assert np.array_equal(out['complex_responses'],baseline['complex_responses'])
    if kind=='zero': assert out['rf_drive_campaign']['atomic_evolutions_per_path']==1


def test_pin_power_and_dynamic_susceptibility_share_same_intensity():
    cfg=detector_config()
    ctx=DetectorContext(1e-7,.7,-.5,780.241476e-9)
    r=np.array([.25,1.,4.,2.])
    y=np.array([[-.1,-.2,-.3,-.4],[-.4,-.3,-.2,-.1]])
    power,current,samples,T=detector_forward(y,cfg.detector,ctx,probe_intensity=r)
    # Canonical model T=exp(-2*od*Im(chi)), chi=K0*y/sqrt(r).
    expected_T=np.exp(-2*ctx.od_factor*ctx.susceptibility_factor*y/np.sqrt(r)[None,:])
    assert np.allclose(T,expected_T,rtol=1e-15,atol=0)
    expected_P=ctx.incident_probe_power_w*r[None,:]*cfg.detector.collection_efficiency*expected_T
    assert np.allclose(power,expected_P,rtol=1e-15,atol=0)
    assert np.allclose(current,cfg.detector.responsivity_a_per_w*expected_P+cfg.detector.dark_current_a)
    assert np.array_equal(samples,current*cfg.detector.transimpedance_v_per_a)
    # Scaling coherence with instantaneous Rabi leaves susceptibility constant.
    _,_,_,scaled_T=detector_forward(np.sqrt(r)[None,:]*np.full((2,4),-.1),cfg.detector,ctx,probe_intensity=r)
    assert np.allclose(scaled_T,scaled_T[:,[0]],rtol=1e-15,atol=0)
    old=detector_forward(y,cfg.detector,ctx)
    one=detector_forward(y,cfg.detector,ctx,probe_intensity=np.ones(4))
    assert all(np.array_equal(a,b) for a,b in zip(old,one))


@pytest.mark.parametrize('bad',[True,[True,True],[True,1.],[np.bool_(False),1.],['1','2'],['1',2.],
    np.ma.array([1.,1.],mask=[True,False]),[1j,2j],[0.,1.],[-1.,1.],[np.nan,1.],[[1.,1.]], [1.]])
def test_invalid_pin_intensity_is_not_repaired(bad):
    cfg=detector_config()
    with pytest.raises(PhysicsCalculationError):
        detector_forward(np.zeros((2,2)),cfg.detector,DetectorContext(1e-7,.1,-1.,780.241476e-9),probe_intensity=bad)


def test_photon_shot_variance_uses_modulated_received_power():
    cfg=detector_config(('photon_shot',),dark_current_a=0.,collection_efficiency=1.)
    ctx=DetectorContext(1e-7,0.,-1.,780.241476e-9)
    r=np.array([.25,1.,4.])
    measured,add,meta,payload=apply_pin_readout(np.zeros((2,3)),cfg,ctx,probe_intensity=r)
    photo=cfg.detector.responsivity_a_per_w*ctx.incident_probe_power_w*r
    tau=cfg.detector.integration_time_us*1e-6
    variance=photo*Q_E/tau
    assert meta['variance_current_a2']['photon_shot']['min_a2']==pytest.approx(variance.min())
    assert meta['variance_current_a2']['photon_shot']['max_a2']==pytest.approx(variance.max())
    assert np.array_equal(payload['incident_probe_power_w'],ctx.incident_probe_power_w*r)
    assert np.array_equal(payload['probe_rabi_multiplier'],np.sqrt(r))
    assert np.array_equal(measured,payload['clean_detector_time_traces']+add)


def test_incident_power_overflow_cannot_be_hidden_by_low_collection():
    cfg=detector_config(collection_efficiency=1e-308)
    ctx=DetectorContext(1e308,0.,-1.,780.241476e-9)
    with pytest.raises(PhysicsCalculationError) as error:
        detector_forward(np.zeros((2,2)),cfg.detector,ctx,probe_intensity=[2.,2.])
    assert isinstance(error.value.__cause__,FloatingPointError)


@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('mixed',[False,True])
def test_real_worker_hermitian_optical_drive_and_no_rng(monkeypatch,direction,mixed):
    from rydberg_aoa.config import parameters as params
    monkeypatch.setattr(params,'ATOMIC_TRANSIENT_DELAY_US',0.)
    data=declaration()
    if mixed:
        for filename in ('examples/noise005/prescribed_rf.json','examples/noise006/prescribed_laser_frequency.json'):
            data['sources'].update(json.loads((ROOT/filename).read_text())['noise']['sources'])
    r=build_rf_realization(parse_noise_config(data),trial=0)
    monkeypatch.setattr(np.random,'SeedSequence',lambda *a,**kw:pytest.fail('RNG in RHS'))
    real=td.mesolve
    def capture(H,*args,**kw):
        evo=qt.QobjEvo(H)
        for t in (.1,3.14159,8.35):
            rp,rc=r.laser_intensity_at(t)
            a_lo,a_sig,phase=r.at(t)
            p,c=2*np.pi*np.array(r.laser_at(t))/1e6
            signal=.05*a_sig*np.exp(1j*(.3+direction*2*np.pi*.1*t+phase))
            expected=np.array([[0,-.1*np.sqrt(rp),0,0],[-.1*np.sqrt(rp),.03+p,-.3*np.sqrt(rc),0],
                [0,-.3*np.sqrt(rc),.01+p+c,-.5*np.conj(.4*a_lo+signal)],
                [0,0,-.5*(.4*a_lo+signal),.01+direction*.01+p+c]],complex)
            assert np.allclose(evo(t).full(),expected,rtol=1e-14,atol=1e-15)
            assert np.array_equal(evo(t).full(),evo(t).full().conj().T)
        assert not kw['options']['normalize_output']
        return real(H,*args,**kw)
    monkeypatch.setattr(td,'mesolve',capture)
    orientation=SimpleNamespace(direction=direction,is_downward=direction<0,
        upper_ladder_index=4 if direction>0 else 3,lower_ladder_index=3 if direction>0 else 4)
    out=td.run_single_heterodyne_worker(0,.05*np.exp(.3j),np.linspace(0,10,101),.1,
        .2,.6,.4,.03,-.02,.01,(.8,.1,.05),(.01,.02,.02),transit_rate_per_us=.2,rf_orientation=orientation,rf_drive=r)
    assert np.isfinite(out[4]).all()


def test_pin_dispatch_uses_worker_realization_at_same_output_times(isolation,monkeypatch):
    data=declaration()
    data['readout']=json.loads((ROOT/'examples/noise004/prescribed_pin.json').read_text())['noise']
    data['readout'].update(seed=data['seed'],trials=data['trials'])
    tasks=[]; fake_pool(monkeypatch,tasks)
    cfg=parse_noise_config(data); run=plan(); ctx=DetectorContext(1e-7,.1,-.5,780.241476e-9)
    out=module.SimulationRunner().run_laboratory_simulation(request(isolation),run,noise=cfg,detector_context=ctx)
    assert len(tasks)==12
    for k,trial in enumerate(out['rf_drive_trial_results']):
        r=out['rf_drive_realizations'][k]
        assert all(t.rf_drive is r for t in tasks[3+3*k:6+3*k])
        factors=np.array([r.laser_intensity_at(t)[0] for t in run.tlist_us])
        assert np.array_equal(trial['detector_arrays']['incident_probe_power_w'],ctx.incident_probe_power_w*factors)
        assert trial['readout_noise']['trial_index']==k
    assert out['rf_drive_campaign']['laser_intensity_noise_included']


def test_coupling_only_does_not_modulate_incident_probe_power(isolation,monkeypatch):
    data=declaration()
    data['sources']['laser_probe_intensity']['enabled']=False
    data['readout']=json.loads((ROOT/'examples/noise004/prescribed_pin.json').read_text())['noise']
    data['readout'].update(seed=data['seed'],trials=data['trials'])
    fake_pool(monkeypatch,[])
    out=module.SimulationRunner().run_laboratory_simulation(request(isolation),plan(),
        noise=parse_noise_config(data),detector_context=DetectorContext(1e-7,.1,-.5,780.241476e-9))
    for trial in out['rf_drive_trial_results']:
        assert not trial['readout_noise']['probe_intensity_modulated']
        assert 'incident_probe_power_w' not in trial['detector_arrays']


def test_zero_rin_keeps_pin_and_is_bit_identical_without_rng(isolation,monkeypatch):
    data=declaration()
    readout=json.loads((ROOT/'examples/noise004/prescribed_pin.json').read_text())['noise']
    readout.update(seed=data['seed'],trials=data['trials'])
    for s in readout['sources'].values(): s['enabled']=False
    data['readout']=readout
    for s in data['sources'].values(): s['fractional_rms']=0.
    fake_pool(monkeypatch,[])
    ctx=DetectorContext(1e-7,.1,-.5,780.241476e-9); req=request(isolation); run=plan()
    monkeypatch.setattr(np.random,'SeedSequence',lambda *a,**kw:pytest.fail('zero RIN/PIN RNG'))
    baseline=module.SimulationRunner().run_laboratory_simulation(req,run,noise=parse_noise_config(readout),detector_context=ctx)
    out=module.SimulationRunner().run_laboratory_simulation(req,run,noise=parse_noise_config(data),detector_context=ctx)
    assert out['readout_noise']['units']==baseline['readout_noise']['units']=='V'
    assert np.array_equal(out['time_traces'],baseline['time_traces'])
    assert np.array_equal(out['complex_responses'],baseline['complex_responses'])
    assert out['rf_drive_campaign']['atomic_evolutions_per_path']==1


def test_real_output_grid_refinement_keeps_same_rin_path(monkeypatch):
    from rydberg_aoa.config import parameters as params
    monkeypatch.setattr(params,'ATOMIC_TRANSIENT_DELAY_US',0.)
    r=build_rf_realization(parse_noise_config(declaration()),trial=0)
    def run(times):
        return td.run_single_heterodyne_worker(0,.05*np.exp(.3j),times,.1,
            .2,.6,.4,.03,-.02,.01,(.8,.1,.05),(.01,.02,.02),transit_rate_per_us=.2,rf_drive=r)
    coarse=run(np.linspace(0,20,201)); fine=run(np.linspace(0,20,401))
    assert np.allclose(coarse[4],fine[4][::2],rtol=1e-4,atol=1e-6)
    assert np.allclose(coarse[0],fine[0],rtol=1e-3,atol=1e-5)


def test_failed_rin_worker_cannot_fall_back_to_nominal(isolation,monkeypatch):
    cause=PhysicsCalculationError('injected RIN ODE failure')
    def fail(task):
        if task.rf_drive is not None: raise cause
        return oracle_record(task)
    fake_pool(monkeypatch,[],worker=fail)
    runner=module.SimulationRunner()
    with pytest.raises(PhysicsCalculationError) as error:
        runner.run_laboratory_simulation(request(isolation),plan(),noise=parse_noise_config(declaration()))
    assert error.value.__cause__ is cause
    assert not runner.results


@pytest.mark.parametrize('mode',['prescribed','experimental','prescribed_pin'])
def test_actual_rin_cli_and_saved_shared_power(tmp_path,mode):
    name='prescribed_rin_pin.json' if mode=='prescribed_pin' else f'{mode}_rin.json'
    data=json.loads((ROOT/'examples/noise007'/name).read_text())
    data['noise']['trials']=1
    if mode=='prescribed_pin': data['noise']['readout']['trials']=1
    data['spectrum']['probe_offsets_mhz']=[-.5,.5]
    path=tmp_path/'profile.json'; path.write_text(json.dumps(data))
    args=[sys.executable,'-B','main.py','--profile',str(path),'--output-dir',str(tmp_path)]
    if mode!='prescribed': args.append('--no-plots')
    p=subprocess.run(args,cwd=ROOT,env=dict(os.environ,MPLBACKEND='Agg'),capture_output=True,text=True,errors='replace',timeout=240)
    assert p.returncode==0,p.stdout+p.stderr
    report=json.loads((tmp_path/'run_report.json').read_text(encoding='utf-8'))
    assert report['laser_intensity_noise_included'] and report['noise_included']
    assert not report['laser_frequency_noise_included'] and not report['rf_drive_included']
    assert report['applied_fields']==data['fields']
    expected=build_rf_realization(parse_noise_config(data['noise']),trial=0)
    with np.load(tmp_path/'simulation_arrays.npz',allow_pickle=False) as a:
        assert np.array_equal(a['trial_laser_probe_intensity'][0],expected.probe_intensity)
        assert np.array_equal(a['trial_laser_probe_rabi_multiplier'][0],np.sqrt(expected.probe_intensity))
        assert not np.array_equal(a['time_traces'],a['nominal_time_traces'])
        if mode=='prescribed_pin':
            factors=np.array([expected.laser_intensity_at(t)[0] for t in a['tlist']])
            assert np.array_equal(a['trial_incident_probe_power_w'][0],data['fields']['probe']['power_mw']/1000*factors)
            assert np.array_equal(a['time_traces'],a['clean_time_traces']+a['additive_readout_noise'])
    if mode=='prescribed': assert (tmp_path/'fig_laser_intensity_noise.png').is_file()
