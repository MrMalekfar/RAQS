"""NOISE-011: same-stage IF power and relative spatial-phase evidence.

No density evolution, RNG, calibration fit, repairs or estimator changes.
Readout SNR is conditional on the supplied clean trace, including drive noise.
"""
import math
import numpy as np

from ..exceptions import PhysicsCalculationError


def _log_if_power(value):
    """Log10 squared peak amplitude; None denotes exactly zero, not a floor."""
    scale = max(abs(float(value.real)), abs(float(value.imag)))
    if scale == 0:
        return None
    return 2*math.log10(scale) + math.log10((value.real/scale)**2 + (value.imag/scale)**2)


def _validate_power_pair(pair):
    if pair is None:
        return None
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise PhysicsCalculationError('NOISE-011: two IF log-power entries required.')
    for value in pair:
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, float, np.integer, np.floating))):
            raise PhysicsCalculationError('NOISE-011: IF log-power must be real or None for exact zero.')
        if value is not None:
            try:
                finite = math.isfinite(float(value))
            except (ValueError, TypeError, OverflowError) as exc:
                raise PhysicsCalculationError('NOISE-011: IF log-power conversion failed.') from exc
            if not finite:
                raise PhysicsCalculationError('NOISE-011: IF log-power must be finite.')
    return pair


def pooled_if_snr(power_pairs):
    """10log10(sum clean IF powers / sum additive IF powers) over ALL trials.

    Each pair stores log10(|A_clean|^2), log10(|A_noise|^2); a cell None is
    exactly zero power. A whole pair None means no additive-readout evidence.
    Zero-power trials contribute zero; no trial is dropped. A common 1/2 for
    sinusoid mean-square power cancels. Not a mean of per-trial dB ratios.
    """
    if not isinstance(power_pairs, (list, tuple)) or not power_pairs:
        raise PhysicsCalculationError('NOISE-011: nonempty IF power evidence required.')
    parsed = [_validate_power_pair(p) for p in power_pairs]
    if any(p is None for p in parsed):
        return None, 'missing_additive_if_evidence'
    sums = []
    for values in zip(*parsed):
        nonzero = [float(v) for v in values if v is not None]
        if not nonzero:
            sums.append(None)
        else:
            top = max(nonzero)
            sums.append(top + math.log10(math.fsum(10**(v-top) for v in nonzero)))
    reason = ('zero_clean_if_power' if sums[0] is None else
              'zero_additive_if_power' if sums[1] is None else None)
    value = None if reason else 10*(sums[0]-sums[1])
    if value is not None and not math.isfinite(value):
        raise PhysicsCalculationError('NOISE-011: IF SNR overflow; no substitute.')
    return value, reason


def measure_readout_if(clean, additive, times, signed_if_mhz, delay_us):
    """Real path-by-time traces in identical units through the production fit.

    No noisy-minus-clean refit subtraction: supplied additive noise is fitted
    separately, through exactly the same frequency, transient cut and DC term.
    White ENBW is a property of these weights, not a colored-noise prediction.
    """
    # Offline power/bootstrap/phase APIs must not import atomic parameters/ARC.
    from ..utils.numerical import extract_if_component, if_fit_noise_weights
    try:
        clean, additive = np.asarray(clean), np.asarray(additive)
        if (clean.ndim != 2 or additive.shape != clean.shape or not all(clean.shape) or
                any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in (clean, additive))):
            raise PhysicsCalculationError('NOISE-011: finite real, matching path-by-time IF traces required.')
        _, _, fit = if_fit_noise_weights(times, signed_if_mhz, delay_us)
        pairs, values, reasons = [], [], []
        for signal, noise in zip(clean, additive):
            a = extract_if_component(signal, times, signed_if_mhz, delay_us)[0]
            n = extract_if_component(noise, times, signed_if_mhz, delay_us)[0]
            pair = [_log_if_power(a), _log_if_power(n)]
            value, reason = pooled_if_snr([pair])
            pairs.append(pair); values.append(value); reasons.append(reason)
        return {'if_power_log10_per_path': pairs, 'if_snr_db_per_path': values,
                'undefined_reason_per_path': reasons, 'fit': fit,
                'definition': 'same-stage clean IF power / additive-readout IF power; squared peak amplitudes',
                'conditioning': 'clean trace of this trial; drive perturbations are not additive detector noise'}
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-011: IF-stage measurement failed; no trace-SNR fallback.') from exc


def relative_phase_errors(responses, applied_phases_rad, reference_path):
    """Canonical spatial IF responses versus declared phases, relative to ref.

    Caller passes the runner's RF-DIR polarity-corrected responses (one
    conjugation at that boundary). No per-path correction learned from truth.
    Common complex gain and retained-time origin cancel; differential atomic
    transfer, drive noise and readout noise remain observable errors.
    """
    try:
        a, phi = np.asarray(responses), np.asarray(applied_phases_rad)
        if (a.ndim != 1 or len(a) < 2 or phi.shape != a.shape or
                a.dtype.kind not in 'iufc' or phi.dtype.kind not in 'iuf' or
                not np.isfinite(a).all() or not np.isfinite(phi).all()):
            raise PhysicsCalculationError('NOISE-011: finite matching IF responses and declared spatial phases required.')
        if (isinstance(reference_path, (bool, np.bool_)) or not isinstance(reference_path, (int, np.integer)) or
                not 0 <= reference_path < len(a)):
            raise PhysicsCalculationError('NOISE-011: reference path index is outside the actual array.')
        errors, reasons = [], []
        # Reduce individually before differences, avoiding overflow of huge angles.
        measured = np.angle(a)
        declared = np.remainder(phi, 2*np.pi)
        for index, value in enumerate(a):
            why = 'zero_reference_if_amplitude' if a[reference_path] == 0 else 'zero_path_if_amplitude' if value == 0 else None
            error = None if why else float(np.rad2deg(
                (measured[index]-measured[reference_path]-declared[index]+declared[reference_path]+np.pi)%(2*np.pi)-np.pi))
            errors.append(error); reasons.append(why)
        return {'error_deg_per_path': errors, 'undefined_reason_per_path': reasons,
                'reference_path': int(reference_path), 'reference_zero_is_by_definition': True,
                'definition': 'wrap[(arg A_m-arg A_ref)-(phi_m-phi_ref)] in [-180,180) deg',
                'calibration': 'no inferred per-path offsets; declared spatial phases are comparison truth only'}
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-011: relative phase measurement failed.') from exc


def measure_sensing_result(result, reference_path=0):
    """Public postprocessing API for one canonical run or a full campaign.

    Each driven trial uses its own clean detector trace. The representative
    result is never used as a replacement for omitted trial evidence.
    """
    try:
        trials = result.get('rf_drive_trial_results', result.get('noise_trial_results', [result]))
        if not isinstance(trials, list) or not trials:
            raise PhysicsCalculationError('NOISE-011: complete nonempty trial records required.')
        expected = result['noise_config'].trials if 'noise_config' in result else 1
        if len(trials) != expected:
            raise PhysicsCalculationError('NOISE-011: incomplete IF campaign; no survivor measurement.')
        width = len(result['complex_responses'])
        rows, fit = [], None
        for index, trial in enumerate(trials):
            if (len(trial['complex_responses']) != width or
                    not np.array_equal(trial['tlist'], result['tlist']) or
                    trial['signed_if_mhz'] != result['signed_if_mhz'] or
                    not np.array_equal(trial['element_phases_rad'], result['element_phases_rad'])):
                raise PhysicsCalculationError('NOISE-011: trial geometry/IF/grid differs from run.')
            phase = relative_phase_errors(trial['complex_responses'], trial['element_phases_rad'], reference_path)
            readout = None
            if 'readout_noise' in trial:
                if trial['readout_noise']['trial_index'] != index:
                    raise PhysicsCalculationError('NOISE-011: readout trial identity/order mismatch.')
                readout = measure_readout_if(trial['clean_time_traces'], trial['additive_readout_noise'],
                    trial['tlist'], trial['signed_if_mhz'], trial['run_config'].transient_delay_us)
                fit = readout['fit']
            rows.append({'trial_index': index, 'relative_phase': phase, 'readout': readout})
        estimates = [pooled_if_snr([None if r['readout'] is None else
                                  r['readout']['if_power_log10_per_path'][p] for r in rows]) for p in range(width)]
        return {'schema_version': 1, 'trials': rows, 'reference_path': int(reference_path),
                'pooled_readout_if_snr_db_per_path': [value for value, _ in estimates],
                'pooled_snr_undefined_reason_per_path': [why for _, why in estimates],
                'fit': fit, 'readout_if_snr_is_conditional_on_each_clean_trial': True,
                'hardware_calibration_verified': False}
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-011: canonical IF measurement evidence failed.') from exc
