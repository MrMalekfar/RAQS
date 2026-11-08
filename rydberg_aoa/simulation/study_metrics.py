"""Study metrics referenced to one certified nominal run; no estimator repair."""
import numpy as np
from ..exceptions import PhysicsCalculationError
from .noise_campaign import azimuth_error, angle_error_statistics, trace_snr_db, _angle
from .if_measurement import measure_readout_if, relative_phase_errors, pooled_if_snr


def _responses(result):
    a = np.asarray(result['complex_responses'])
    if a.ndim != 1 or not len(a) or a.dtype.kind not in 'iufc' or not np.isfinite(a).all():
        raise PhysicsCalculationError('NOISE-008: finite numeric IF response vector required.')
    return a


def _rmse(values):
    a = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(a)))
    return 0. if scale == 0 else float(scale*np.sqrt(np.mean((a/scale)**2)))


def _mean(values):
    a = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(a)))
    return 0. if scale == 0 else float(scale*np.mean(a/scale))


def summarize_study_point(nominal, result, definition):
    """Return aggregate metrics + full trial rows; undefined is None with cause.

    Phase is wrapped error relative to NOMINAL recovered IF, not an absolute
    optical phase calibration. Amplitude metric is mean absolute relative error.
    Any undefined trial makes its aggregate undefined; no survivor statistics.
    """
    trials = result.get('rf_drive_trial_results', result.get('noise_trial_results'))
    if not isinstance(trials, list) or len(trials) != definition.trials_per_point:
        raise PhysicsCalculationError('NOISE-008: complete explicit trial results required; no shortened campaign.')
    reference = _responses(nominal)
    rows = []
    truth = _angle(nominal['true_aoa'], 'study true azimuth')
    for index, trial in enumerate(trials):
        a = _responses(trial)
        if a.shape != reference.shape or not np.array_equal(trial['tlist'], nominal['tlist']):
            raise PhysicsCalculationError('NOISE-008: trial grid/path count differs from nominal reference.')
        phase, amplitude, phase_why, amplitude_why = [], [], [], []
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for value, ref in zip(a, reference):
                missing = 'zero_nominal_if_amplitude' if ref == 0 else 'zero_trial_if_amplitude' if value == 0 else None
                phase.append(None if missing else float(np.rad2deg((np.angle(value)-np.angle(ref)+np.pi)%(2*np.pi)-np.pi)))
                phase_why.append(missing)
                amplitude.append(None if ref == 0 else float(abs(abs(value)/abs(ref)-1.)))
                amplitude_why.append('zero_nominal_if_amplitude' if ref == 0 else None)
        if any(v is not None and not np.isfinite(v) for v in amplitude+phase):
            raise PhysicsCalculationError('NOISE-008: nonfinite phase/amplitude metric; no substitute.')
        snr, why = [None]*len(reference), ['no_additive_readout_layer']*len(reference)
        if 'readout_noise' in trial:
            snr, why = trace_snr_db(trial['clean_time_traces'], trial['additive_readout_noise'],
                                  trial['tlist'], trial['run_config'].transient_delay_us)
        if_evidence = None
        if 'readout_if_snr_db_per_path' in definition.metrics and 'readout_noise' in trial:
            if trial['signed_if_mhz'] != nominal['signed_if_mhz']:
                raise PhysicsCalculationError('NOISE-011: trial IF differs from nominal study declaration.')
            if_evidence = measure_readout_if(trial['clean_time_traces'], trial['additive_readout_noise'],
                trial['tlist'], trial['signed_if_mhz'], trial['run_config'].transient_delay_us)
        relative = None
        if 'relative_if_phase_rmse_deg_per_path' in definition.metrics:
            if not np.array_equal(trial['element_phases_rad'], nominal['element_phases_rad']):
                raise PhysicsCalculationError('NOISE-011: trial spatial phase truth differs from nominal geometry.')
            relative = relative_phase_errors(a, trial['element_phases_rad'], definition.phase_reference_path)
        delta = np.asarray(trial['time_traces'])-np.asarray(nominal['time_traces'])
        deviation, deviation_why = trace_snr_db(nominal['time_traces'], delta, trial['tlist'],
                                               trial['run_config'].transient_delay_us)
        rows.append({'trial_index': index, 'estimated_aoa_deg': _angle(trial['estimated_aoa'], 'study estimate'),
                     'aoa_error_deg': azimuth_error(_angle(trial['estimated_aoa'], 'study estimate'), truth),
                     'if_phase_error_deg_per_path': phase, 'phase_undefined_reason_per_path': phase_why,
                     'if_amplitude_absolute_relative_error_per_path': amplitude,
                     'amplitude_undefined_reason_per_path': amplitude_why,
                     'readout_trace_snr_db_per_path': snr, 'readout_snr_undefined_reason_per_path': why,
                     'waveform_deviation_ratio_db_per_path': deviation,
                     'deviation_undefined_reason_per_path': deviation_why})
        if 'readout_if_snr_db_per_path' in definition.metrics:
            rows[-1]['readout_if_power_log10_per_path'] = ([None]*len(reference) if if_evidence is None else
                                                          if_evidence['if_power_log10_per_path'])
            rows[-1]['readout_if_fit'] = None if if_evidence is None else if_evidence['fit']
        if relative is not None:
            rows[-1].update(relative_if_phase_error_deg_per_path=relative['error_deg_per_path'],
                            relative_phase_undefined_reason_per_path=relative['undefined_reason_per_path'])
    statistics = angle_error_statistics([r['aoa_error_deg'] for r in rows])
    metrics = {'aoa_bias_deg': statistics['bias_deg'], 'aoa_rmse_deg': statistics['rmse_deg']}
    undefined = {}
    pairs = [('if_phase_rmse_deg_per_path','if_phase_error_deg_per_path',_rmse),
             ('if_amplitude_relative_error_per_path','if_amplitude_absolute_relative_error_per_path',_mean),
             ('readout_trace_snr_db_per_path','readout_trace_snr_db_per_path',_mean),
             ('waveform_deviation_ratio_db_per_path','waveform_deviation_ratio_db_per_path',_mean)]
    if 'relative_if_phase_rmse_deg_per_path' in definition.metrics:
        pairs.append(('relative_if_phase_rmse_deg_per_path', 'relative_if_phase_error_deg_per_path', _rmse))
    for metric, key, aggregate in pairs:
        values, reasons = [], []
        for path in range(len(reference)):
            samples = [r[key][path] if r[key] is not None else None for r in rows]
            absent = sum(v is None for v in samples)
            values.append(None if absent else aggregate(samples))
            reasons.append(f'{absent}_of_{len(rows)}_trials_undefined' if absent else None)
        metrics[metric], undefined[metric] = values, reasons
    if 'readout_if_snr_db_per_path' in definition.metrics:
        estimates = [pooled_if_snr([row['readout_if_power_log10_per_path'][path] for row in rows])
                     for path in range(len(reference))]
        metrics['readout_if_snr_db_per_path'] = [value for value, _ in estimates]
        undefined['readout_if_snr_db_per_path'] = [why for _, why in estimates]
    return {'metrics': {m: metrics[m] for m in definition.metrics},
            'undefined_reasons': {m: undefined[m] for m in definition.metrics if m in undefined},
            'trials': rows, 'statistics': statistics,
            'references': {'aoa': 'declared true azimuth; includes nominal recovery bias',
                           'phase': 'wrapped per-path error relative to certified nominal IF response',
                           'amplitude': 'mean absolute relative IF amplitude error versus nominal',
                           'snr': 'mean dB across trials of post-transient clean AC trace/additive readout noise power; not IF SNR',
                           'deviation': 'nominal AC trace power / squared trial-minus-nominal trace; not additive SNR'},
            'if_measurement': {'phase_reference_path': definition.phase_reference_path,
                'reference_zero_is_by_definition': True,
                'relative_phase': 'wrap[(arg A_m-arg A_ref)-(declared phi_m-phi_ref)]; canonical polarity; no per-path calibration',
                'snr': 'ratio of summed clean/additive IF squared peak amplitudes across all trials, not mean dB',
                'conditioning': 'clean readout trace of each driven trial; drive changes not counted as additive readout noise'},
            'sample_std_defined': definition.trials_per_point > 1,
            'uncertainty_status': 'sample statistics only; no confidence interval or convergence claim'}
