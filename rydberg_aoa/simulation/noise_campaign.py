"""Readout Monte Carlo statistics, conditional on one certified clean evolution.

Angles are azimuths in degrees. SNR is empirical AC trace power / realized
additive-noise power after the same transient cut as the IF fit; not IF SNR,
photocurrent SNR, a PSD, or a detector calibration.
"""

import math

import numpy as np

from ..config.laboratory import _real
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError


def _angle(value, name):
    try:
        return _real(value, 'NOISE-003 ' + name)
    except PhysicsConfigurationError as exc:
        raise PhysicsCalculationError('NOISE-003: invalid ' + name) from exc


def azimuth_error(estimated, reference):
    """Shortest signed azimuth error in [-180, 180), without adjusting outputs."""
    # Reduce each angle first to avoid overflow from subtracting large values.
    return ((estimated % 360 - reference % 360 + 180) % 360) - 180


def _log_power(samples):
    scale = float(np.max(np.abs(samples)))
    if scale == 0:
        return None
    return 2*math.log10(scale) + math.log10(float(np.mean((samples/scale)**2)))


def trace_snr_db(clean, additive, times, delay_us):
    """Per-path empirical SNR, with None and reason when either power is zero."""
    try:
        clean, additive, times = np.asarray(clean), np.asarray(additive), np.asarray(times)
        if (clean.ndim != 2 or additive.shape != clean.shape or times.ndim != 1 or
                clean.shape[1] != len(times) or not all(clean.shape) or
                any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all()
                    for a in (clean, additive, times))):
            raise PhysicsCalculationError('NOISE-003: invalid SNR traces/grid.')
        delay = _angle(delay_us, 'transient delay')
        if delay < 0 or np.any(np.diff(times.astype(float)) <= 0):
            raise PhysicsCalculationError('NOISE-003: invalid SNR time ordering/delay.')
        mask = times >= delay
        if not mask.any():
            raise PhysicsCalculationError('NOISE-003: no retained SNR samples.')
        values, reasons = [], []
        with np.errstate(over='raise', invalid='raise'):
            for signal, noise in zip(clean[:, mask], additive[:, mask]):
                signal_log = _log_power(signal - np.mean(signal))
                noise_log = _log_power(noise)
                reason = ('zero_clean_ac_power' if signal_log is None else
                          'zero_additive_noise_power' if noise_log is None else None)
                value = None if reason is not None else 10*(signal_log-noise_log)
                if value is not None and not math.isfinite(value):
                    raise PhysicsCalculationError('NOISE-003: nonfinite SNR statistic.')
                values.append(value)
                reasons.append(reason)
        return values, reasons
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-003: SNR computation failed; no substitute statistic.') from exc


def angle_error_statistics(errors):
    errors = np.asarray(errors)
    if errors.ndim != 1 or errors.size == 0 or errors.dtype.kind not in 'iuf' or not np.isfinite(errors).all():
        raise PhysicsCalculationError('NOISE-003: finite nonempty angle errors required.')
    return {'bias_deg': float(np.mean(errors)), 'rmse_deg': float(np.sqrt(np.mean(errors**2))),
            'mae_deg': float(np.mean(np.abs(errors))),
            'error_sample_std_deg': float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
            'max_abs_error_deg': float(np.max(np.abs(errors)))}


def summarize_readout_campaign(trials, clean_output, run_config):
    """All trials required; report wrapped errors, not an averaged DoA spectrum."""
    if not trials:
        raise PhysicsCalculationError('NOISE-003: empty campaign.')
    expected = trials[0]['noise_config'].trials
    if len(trials) != expected:
        raise PhysicsCalculationError('NOISE-003: incomplete campaign; no survivor statistics.')
    truth = _angle(run_config.target_azimuth_deg, 'target azimuth')
    clean_angle = _angle(clean_output['estimated_aoa'], 'clean estimated azimuth')
    rows = []
    for index, result in enumerate(trials):
        if result['readout_noise']['trial_index'] != index:
            raise PhysicsCalculationError('NOISE-003: trial identity/order mismatch.')
        angle = _angle(result['estimated_aoa'], 'estimated azimuth')
        snr, reasons = trace_snr_db(result['clean_time_traces'], result['additive_readout_noise'],
                                   result['tlist'], run_config.transient_delay_us)
        rows.append({'trial_index': index, 'estimated_aoa_deg': angle,
                     'error_deg': azimuth_error(angle, truth),
                     'delta_from_clean_deg': azimuth_error(angle, clean_angle),
                     'trace_snr_db_per_path': snr, 'snr_undefined_reason_per_path': reasons})
    errors = np.asarray([row['error_deg'] for row in rows])
    return {'trials_requested': expected, 'trials_completed': len(rows),
            'representative_trial_index': 0, 'atomic_evolutions_per_path': 1,
            'conditional_on_clean_evolution': True,
            'trace_units': trials[0]['readout_noise'].get('units', 'Im<sigma21> per real sample'),
            'true_aoa_deg': truth, 'clean_estimated_aoa_deg': clean_angle,
            'clean_error_deg': azimuth_error(clean_angle, truth),
            **angle_error_statistics(errors),
            'error_convention': 'wrapped azimuth error in [-180,180) deg; std uses ddof=1',
            'snr_definition': 'post-transient clean AC trace power / realized additive-noise power; not IF SNR',
            'failure_policy': 'abort entire campaign; no dropped or substituted trials',
            'trials': rows}
