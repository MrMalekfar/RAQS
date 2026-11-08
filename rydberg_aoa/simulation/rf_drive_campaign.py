"""Certified atomic re-solves per RF/laser realization, then optional readout.

The nominal reference is never a fallback for a failed driven worker. It is
reused only when every drive source is disabled/zero, so H(t) is unchanged.
"""

from dataclasses import replace
import numpy as np

from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from .rf_drive_noise import build_rf_realization
from .noise_campaign import _angle, azimuth_error, angle_error_statistics, trace_snr_db
from ..config.rf_noise import RF_SOURCE_UNITS
from ..config.laser_noise import LASER_SOURCE_UNITS, LASER_INTENSITY_UNITS


def validate_drive_realizations(noise, realizations):
    """Explicit diagnostic inputs, not a fallback or an RNG replacement on failure."""
    if realizations is None:
        return
    from ..config.noise import require_noise_execution_supported
    require_noise_execution_supported(noise)
    from .rf_drive_noise import RFDriveRealization
    if (noise is None or noise.schema_version!=3 or not noise.stochastic_drive or
            not isinstance(realizations,tuple) or len(realizations)!=noise.trials or
            any(not isinstance(r,RFDriveRealization) for r in realizations)):
        raise PhysicsConfigurationError('NOISE-010: complete immutable drive realizations require an active schema-3 campaign.')
    try:
        grid=np.linspace(0.,noise.drive.horizon_us,round(noise.drive.horizon_us/noise.drive.noise_step_us)+1)
    except (MemoryError,ValueError,OverflowError) as exc:
        raise PhysicsConfigurationError('NOISE-010: cannot validate requested realization grid; no shortening.') from exc
    names={'rf_lo_amplitude':'lo_amplitude','rf_signal_amplitude':'signal_amplitude',
           'rf_relative_phase':'relative_phase_rad','laser_probe_frequency':'probe_frequency_hz',
           'laser_coupling_frequency':'coupling_frequency_hz','laser_probe_intensity':'probe_intensity',
           'laser_coupling_intensity':'coupling_intensity'}
    active={s.name for s in noise.sources if s.specification.enabled and s.specification.stationary_std>0}
    for r in realizations:
        # Nested subsets retain original times; allow only floating arithmetic
        # rounding of the mathematically identical uniform grid, not off-grid data.
        if len(r.times_us)!=len(grid) or not np.allclose(r.times_us,grid,rtol=8*np.finfo(float).eps,atol=0):
            raise PhysicsConfigurationError('NOISE-010: supplied realization grid differs from declared noise grid.')
        for source,field in names.items():
            constant=1. if ('amplitude' in field or 'intensity' in field) else 0.
            if source not in active and not np.all(np.asarray(getattr(r,field))==constant):
                raise PhysicsConfigurationError(f'NOISE-010: disabled/zero source {source} has injected noise.')


def run_rf_drive_campaign(runner, request, config, runtime, mode, noise, detector_context, *, realizations=None, receiver=None, solver_diagnostic=None, poisson_quantiles=False):
    validate_drive_realizations(noise,realizations)
    supplied_realizations = realizations
    if_only = receiver is not None and receiver.mode == 'if_only'
    receiver_options = {'measurement':receiver} if if_only else {}
    if solver_diagnostic is not None:receiver_options['solver_diagnostic']=solver_diagnostic
    if poisson_quantiles:receiver_options['poisson_quantiles']=True
    measurement = noise.effective_readout
    if not noise.drive_selected and (measurement is None or not measurement.readout_enabled):
        return runner._run_resolved_sensing(request, config, runtime, mode, **receiver_options)
    nominal_measurement = None if measurement is None else replace(measurement,
        sources=tuple(replace(s, specification=replace(s.specification, enabled=False)) for s in measurement.sources))
    nominal = runner._run_resolved_sensing(request, config, runtime, mode, noise=nominal_measurement,
        detector_context=detector_context, _publish=False, _measurement_trial=0, **receiver_options)
    atomic_reference = nominal.get('atomic_worker_records', nominal['worker_records'])
    if not if_only and np.count_nonzero([record[0] for record in atomic_reference]) < 2:
        raise PhysicsCalculationError('NOISE-005: nominal IF signal is absent; drive noise cannot invent an AoA signal.')
    results, realizations, rows = [], [], []
    truth = None if if_only else _angle(config.target_azimuth_deg, 'target azimuth')
    nominal_angle = None if if_only else _angle(nominal['estimated_aoa'], 'nominal estimated azimuth')
    for index in range(noise.trials):
        try:
            realization = build_rf_realization(noise, trial=index) if supplied_realizations is None else supplied_realizations[index]
            result = runner._run_resolved_sensing(request, config, runtime, mode, noise=measurement,
                detector_context=detector_context, _drive=realization, _publish=False, _measurement_trial=index,
                _records_override=atomic_reference if realization is None else None,
                **({'_temporal_reports_override':nominal['temporal_doppler_reports']}
                   if realization is None and config.temporal_doppler is not None else {}), **receiver_options)
            delta = np.asarray(result['time_traces'])-np.asarray(nominal['time_traces'])
            ratio, reasons = trace_snr_db(nominal['time_traces'], delta, result['tlist'], config.transient_delay_us)
            angle = None if if_only else _angle(result['estimated_aoa'], 'RF trial estimated azimuth')
            row = {'trial_index': index, 'estimated_aoa_deg': angle,
                   'error_deg': None if if_only else azimuth_error(angle, truth),
                   'delta_from_nominal_deg': None if if_only else azimuth_error(angle, nominal_angle),
                   'waveform_deviation_ratio_db_per_path': ratio,
                   'deviation_undefined_reason_per_path': reasons}
            if not if_only and 'readout_noise' in result:
                snr, why = trace_snr_db(result['clean_time_traces'], result['additive_readout_noise'],
                                        result['tlist'], config.transient_delay_us)
                row.update(readout_trace_snr_db_per_path=snr, readout_snr_undefined_reason_per_path=why)
            rows.append(row)
            results.append(result)
            realizations.append(realization)
        except Exception as exc:
            raise PhysicsCalculationError(f'NOISE-005: trial {index} failed; entire RF campaign aborted without retry/substitution.') from exc
    summary = {'trials_requested': noise.trials, 'trials_completed': len(results),
               'representative_trial_index': 0, 'true_aoa_deg': truth,
               'nominal_estimated_aoa_deg': nominal_angle, 'nominal_error_deg': None if if_only else azimuth_error(nominal_angle, truth),
               'trace_units': results[0].get('readout_noise', {}).get('units', 'Im<sigma21> per real sample'),
               'atomic_evolutions_per_path': 1+noise.trials if noise.stochastic_drive else 1,
               'rf_drive_included': noise.enabled and any(s.name in RF_SOURCE_UNITS and
                   s.specification.enabled and s.specification.stationary_std > 0 for s in noise.sources),
               'laser_frequency_noise_included': noise.enabled and any(s.name in LASER_SOURCE_UNITS and
                   s.specification.enabled and s.specification.stationary_std > 0 for s in noise.sources),
               'drive_noise_included': noise.stochastic_drive,
               'laser_intensity_noise_included': noise.enabled and any(s.name in LASER_INTENSITY_UNITS and
                   s.specification.enabled and s.specification.stationary_std > 0 for s in noise.sources),
               'laser_intensity_model': 'mean-one lognormal intensity at nodes; linear intensity interpolation, Rabi=sqrt(intensity)',
               'laser_intensity_input': 'stationary fractional intensity RMS at nodes, not PSD or Rabi RMS',
               'laser_frequency_model': 'stationary OU offsets [Hz], linear interpolation, phase is their integral',
               'laser_frequency_budget': 'independent residual, not a second realization of the declared Markov linewidth',
               'laser_frame': 'instantaneous probe/coupling phase; probe-referenced coherence; shared optical shifts on R1/R2',
               'preparation': noise.drive.preparation, 'spatial_scope': noise.drive.spatial_scope,
               'interpolation': noise.drive.interpolation, 'noise_step_us': noise.drive.noise_step_us,
               'noise_horizon_us': noise.drive.horizon_us,
               'amplitude_model': 'stationary mean-one lognormal node factors, linearly interpolated',
               'phase_model': 'stationary OU phase of complex signal Rabi envelope; not laser/RF linewidth or diffusion',
               'waveform_deviation_definition': 'nominal clean AC power / squared measured-minus-nominal waveform; not additive detector SNR',
               'failure_policy': 'abort entire campaign, no survivor statistics',
               'realization_source': 'canonical_rng' if supplied_realizations is None else 'explicit_validated_diagnostic',
               **({} if if_only else angle_error_statistics([row['error_deg'] for row in rows])), 'trials': rows}
    if if_only:
        from .if_only import summarize_if_trials
        summary['if_statistics'] = summarize_if_trials(results,noise.trials)
    output = dict(results[0])
    output.update(rf_drive_campaign=summary, rf_drive_trial_results=results, rf_drive_realizations=realizations,
                  nominal_time_traces=np.asarray(nominal['time_traces']),
                  nominal_complex_responses=nominal['complex_responses'], noise_config=noise)
    if runner._current_explicit_runtime(request, mode) != runtime:
        raise PhysicsConfigurationError('NOISE-005: runtime changed during RF campaign; no stale publication.')
    runner._publish_explicit_result(output, request, runtime, mode)
    return output
