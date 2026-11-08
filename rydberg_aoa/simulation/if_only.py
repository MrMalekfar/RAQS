"""Receiver-only IF moments; no angle, phase inference, threshold or imputation."""
import numpy as np
from ..exceptions import PhysicsCalculationError


def summarize_if_trials(trials, expected):
    if not isinstance(trials,list) or type(expected) is not int or expected < 1 or len(trials) != expected:
        raise PhysicsCalculationError('NOISE-013: complete explicit IF trial campaign required; no survivor statistics.')
    try:
        first=trials[0]; values=[]
        unit=first.get('readout_noise',{}).get('units','Im<sigma21>')
        from ..config.laboratory import _real, _integer
        signed_if = _real(first['signed_if_mhz'], 'NOISE-013 signed IF [MHz]')
        times = np.asarray(first['tlist'])
        if (type(first['signal_enabled']) is not bool or signed_if == 0 or
                times.ndim != 1 or len(times) < 2 or times.dtype.kind not in 'iuf' or
                not np.isfinite(times).all() or not np.all(np.diff(times) > 0) or
                not isinstance(unit,str) or not unit):
            raise PhysicsCalculationError('NOISE-013: valid switch, units and finite increasing IF grid required.')
        for i, trial in enumerate(trials):
            if (trial.get('measurement_mode') != 'if_only' or trial.get('estimated_aoa') is not None or
                    trial.get('doa_result') is not None or trial.get('if_snapshots') is not None or
                    trial['signal_enabled'] is not first['signal_enabled'] or
                    not np.array_equal(trial['tlist'],first['tlist']) or
                    trial['signed_if_mhz'] != first['signed_if_mhz'] or
                    trial.get('readout_noise',{}).get('units','Im<sigma21>') != unit):
                raise PhysicsCalculationError('NOISE-013: inconsistent mode, switch, units or IF grid between trials.')
            if 'readout_noise' in trial:
                trial_index = _integer(trial['readout_noise']['trial_index'], 'NOISE-013 trial index')
                if trial_index != i:
                    raise PhysicsCalculationError('NOISE-013: readout trial identity/order mismatch.')
            a=np.asarray(trial['complex_responses'])
            if a.ndim != 1 or not len(a) or a.dtype.kind not in 'iufc' or not np.isfinite(a).all():
                raise PhysicsCalculationError('NOISE-013: finite IF vector required.')
            values.append(a)
        a=np.asarray(values)
        if a.ndim != 2:
            raise PhysicsCalculationError('NOISE-013: path count differs between trials.')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            scale=np.max(np.abs(a),axis=0)
            divisor=np.where(scale==0,1.,scale)
            normalized=a/divisor
            mean=np.mean(normalized,axis=0)*divisor
            powers=np.abs(a)**2
            mean_power=np.mean(np.abs(normalized)**2,axis=0)*divisor**2
            iq=np.column_stack((a.real,a.imag))
            covariance=None
            if expected > 1:
                iqscale=np.max(np.abs(iq),axis=0)
                iqdivisor=np.where(iqscale==0,1.,iqscale)
                covariance=np.cov(iq/iqdivisor,rowvar=False,ddof=1)*iqdivisor[:,None]*iqdivisor[None,:]
        if not np.isfinite(mean).all() or not np.isfinite(powers).all() or not np.isfinite(mean_power).all() or (covariance is not None and not np.isfinite(covariance).all()):
            raise PhysicsCalculationError('NOISE-013: IF moments overflow; no clipped statistic.')
        return {'schema_version':1,'trials_requested':expected,'trials_completed':expected,'trace_units':unit,
            'power_units':unit+'^2 (squared IF peak amplitude)',
            'signal_enabled':first['signal_enabled'],'source_phase_deg_per_path':[None]*a.shape[1],
            'phase_status':'not inferred in IF-only mode; I/Q remains fully recorded',
            'estimated_aoa_deg':None,'aoa_status':'not computed: IF-only receiver',
            'mean_if_real_per_path':mean.real.tolist(),'mean_if_imag_per_path':mean.imag.tolist(),
            'mean_squared_if_amplitude_per_path':mean_power.tolist(),
            'joint_iq_sample_covariance':None if covariance is None else covariance.tolist(),
            'covariance_order':'all Re(A) paths, then all Im(A) paths; sample ddof=1',
            'covariance_status':'insufficient_trials' if covariance is None else 'empirical_sample',
            'trials':[{'trial_index':i,'if_real_per_path':v.real.tolist(),'if_imag_per_path':v.imag.tolist(),
                       'if_peak_amplitude_per_path':np.abs(v).tolist(),'squared_if_amplitude_per_path':powers[i].tolist()}
                      for i,v in enumerate(a)],
            'interpretation':'measured IF moments include coherent leakage and numerical residuals; not a certified noise PSD or detection threshold',
            'mdf_nef_computed':False,'hardware_calibration_verified':False}
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-013: IF statistics failed; no replacement moments.') from exc


def summarize_if_result(result):
    trials=result.get('rf_drive_trial_results',result.get('noise_trial_results',[result]))
    expected=result['noise_config'].trials if 'noise_config' in result else 1
    summary=summarize_if_trials(trials,expected)
    from ..utils.numerical import if_fit_noise_weights
    try:
        _,_,fit=if_fit_noise_weights(result['tlist'],result['signed_if_mhz'],result['run_config'].transient_delay_us)
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-013: IF measurement grid/fit invalid.') from exc
    summary['fit']=fit
    summary['complex_response_convention']='raw signed-frequency DC/cos/sin fit, a-i*b; no spatial-polarity conjugation'
    return summary


def save_if_only_arrays(directory,result):
    """Persist numeric data only; no None/object or fictitious spatial arrays."""
    arrays={key:result[key] for key in ('tlist','time_traces','complex_responses','array_positions','xf_0','yf_0','configured_spatial_phases_rad')}
    trials=result.get('rf_drive_trial_results',result.get('noise_trial_results',[result]))
    arrays['trial_indices']=np.arange(len(trials),dtype=int)
    arrays['trial_complex_responses']=np.asarray([t['complex_responses'] for t in trials])
    arrays['trial_time_traces']=np.asarray([t['time_traces'] for t in trials])
    for i,trial in enumerate(trials):
        for path,record in enumerate(trial['worker_records']):
            arrays[f'trial_{i}_fft_frequency_path_{path}']=record[1]
            arrays[f'trial_{i}_fft_complex_path_{path}']=record[2]
    for key in ('clean_time_traces','clean_complex_responses','additive_readout_noise'):
        if key in result:
            arrays[key]=result[key]
            arrays['trial_'+key]=np.asarray([t[key] for t in trials])
    if 'detector_arrays' in result:
        for key in result['detector_arrays']:
            arrays[key]=result['detector_arrays'][key]
            arrays['trial_'+key]=np.asarray([t['detector_arrays'][key] for t in trials])
    for key in ('nominal_time_traces','nominal_complex_responses'):
        if key in result:arrays[key]=result[key]
    if 'rf_drive_realizations' in result and result['rf_drive_realizations'][0] is not None:
        realizations=result['rf_drive_realizations']
        arrays['drive_noise_times_us']=realizations[0].times_us
        for key in ('lo_amplitude','signal_amplitude','relative_phase_rad','probe_frequency_hz','coupling_frequency_hz','probe_intensity','coupling_intensity'):
            arrays['trial_drive_'+key]=np.asarray([getattr(r,key) for r in realizations])
    spectrum=result.get('spectrum')
    if spectrum is not None:
        arrays.update({'spectrum_'+k:v for k,v in spectrum.items() if v is not None and k != 'analytical_contract'})
    if any(np.asarray(v).dtype.kind not in 'iufc' or not np.isfinite(v).all() for v in arrays.values()):
        raise PhysicsCalculationError('NOISE-013: export requires finite numeric arrays; no object placeholders.')
    np.savez_compressed(directory/'simulation_arrays.npz',**arrays)
