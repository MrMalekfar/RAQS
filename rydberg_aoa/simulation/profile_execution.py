"""LAB-006 explicit profile execution and reports, without optical auto-tuning."""

from dataclasses import asdict, is_dataclass
import copy
import math

from ..exceptions import PhysicsConfigurationError


def execute_profile(profile, *, diagnostics_only=False, drive_realizations=None, solver_diagnostic=None, poisson_quantiles=False):
    """Install explicit inputs, then use the canonical resolution seam.

    A failed run restores input dictionaries, aliases, caches and RF scalar
    views and leaves physics unpublished. An existing published run is refused
    before writes; this is an entry point, not an implicit reset/force API.
    """
    from ..config.run_profile import RunProfile
    if not isinstance(profile, RunProfile) or type(diagnostics_only) is not bool:
        raise PhysicsConfigurationError('LAB-006: a parsed RunProfile and boolean diagnostics_only are required.')
    from ..config.temporal_doppler import validate_temporal_profile
    validate_temporal_profile(profile)
    from ..config.run_numerics import validate_profile_numerics, ode_receipt
    validate_profile_numerics(profile)
    from ..config.profile_radiative import (validate_profile_radiative, resolve_profile_radiative,
                                          radiative_report, named_internal_rates)
    validate_profile_radiative(profile)
    from ..config.magnetic_core import validate_magnetic_core
    validate_magnetic_core(profile)
    from ..config.lo_phase_reference import validate_lo_phase_reference,require_uniform_lo_execution,lo_phase_receipt
    validate_lo_phase_reference(profile)
    if not diagnostics_only:require_uniform_lo_execution(profile)
    if profile.magnetic_core is not None and not diagnostics_only:
        raise PhysicsConfigurationError('BBR-003: magnetic core supports explicit diagnostics only until '
                                        'outside-FS kinetics and the multilevel receiver are validated; no four-state substitute.')
    from ..config.detection_numerics import SolverDiagnostic
    if ((solver_diagnostic is not None and not isinstance(solver_diagnostic,SolverDiagnostic)) or
            type(poisson_quantiles) is not bool or (diagnostics_only and (solver_diagnostic is not None or poisson_quantiles))):
        raise PhysicsConfigurationError('NOISE-016: valid temporal diagnostic controls required before initialization.')
    from ..config.provenance import provenance_report
    input_provenance = provenance_report(profile.provenance)
    from ..config.measurement import resolved_measurement, validate_measurement_profile
    receiver=resolved_measurement(profile.measurement)
    if profile.measurement is not None:
        validate_measurement_profile(profile)
    if profile.study is not None:
        raise PhysicsConfigurationError('NOISE-008: a study profile requires execute_study; do not silently run one ordinary point.')
    if drive_realizations is not None:
        from .rf_drive_campaign import validate_drive_realizations
        validate_drive_realizations(profile.noise,drive_realizations)
        if diagnostics_only:
            raise PhysicsConfigurationError('NOISE-010: diagnostic-only initialization cannot consume drive realizations.')
    from ..config.noise import require_noise_execution_supported
    require_noise_execution_supported(profile.noise)
    from ..config.rf_noise import validate_rf_noise_execution
    validate_rf_noise_execution(profile.noise, profile.run_config)
    from ..config.detector import validate_detector_sampling
    validate_detector_sampling(profile.noise, profile.run_config)
    if profile.noise is not None and profile.noise.physical_readout_enabled and profile.environment is None:
        raise PhysicsConfigurationError('NOISE-004: explicit environment required for optical detector forward model.')
    if profile.noise is not None and profile.noise.detector is not None:
        from .detector_readout import pin_efficiency
        pin_efficiency(profile.noise.detector.responsivity_a_per_w, profile.probe.wavelength_nm*1e-9)
    if profile.run_config is None and not diagnostics_only:
        raise PhysicsConfigurationError('LAB-006: prescribed profile supports --diagnostics-only; '
                                        'use schema_version 2 with explicit execution for sensing.')
    if not diagnostics_only and any(value == 0 for value in
            (profile.probe.power_mw, profile.coupling.power_mw,
             profile.lo.electric_field_vm, profile.signal.electric_field_vm)):
        raise PhysicsConfigurationError('LAB-007: driven AoA requires nonzero applied fields; no angle substitute.')
    from ..config import atomic_config as ac, parameters as params
    from ..config.rf_configuration import configure_rf_fields
    if ac.is_atomic_physics_initialized():
        raise PhysicsConfigurationError('LAB-006: physics is already published; explicitly reset before applying a profile.')
    pipeline = ac._snapshot_pipeline_transaction()
    scalar_names = ('RF_LO_FREQUENCY', 'RF_LO_E_FIELD', 'RF_SIGNAL_FREQUENCY',
                    'RF_SIGNAL_E_FIELD', 'RF_FREQUENCY', 'IF_FREQ')
    scalars = {name: getattr(params, name) for name in scalar_names}
    def restore_reference_pipeline():
        ac._rollback_pipeline_transaction(pipeline)
        for name, value in scalars.items():
            setattr(params, name, value)
        ac._clear_published_atomic_state()
    try:
        from ..config.atom_species import (effective_atom_spec, validate_profile_atom,
                                           atom_report)
        ac.configure_atom(effective_atom_spec(profile))
        validate_profile_atom(profile, ac.ATOM)
        configure_rf_fields(lo=profile.lo, signal=profile.signal)
        for name in ('probe', 'coupling'):
            ac.LASER_PARAMS[name].clear()
            ac.LASER_PARAMS[name].update(asdict(getattr(profile, name)))
        # Every declaration is validated before inputs are applied. ARC/solver
        # errors retain their original exception identity and cause chain.
        request = profile.request.resolution_request() if profile.mode == 'prescribed' else profile.request
        ac.initialize_atomic_physics(resolution_request=request)
        from .profile_spectrum import resolve_profile_medium, run_profile_spectrum
        medium, config = resolve_profile_medium(profile, ac)
        radiative = resolve_profile_radiative(profile, ac)
        if radiative is not None and config is not None:
            from dataclasses import replace
            config = replace(config, radiative=radiative)
        if profile.temporal_doppler is not None and profile.temporal_doppler.enabled:
            from dataclasses import replace
            from ..config.temporal_doppler import ResolvedTemporalDoppler
            config = replace(config, temporal_doppler=ResolvedTemporalDoppler(
                profile.temporal_doppler, medium.temperature_k, medium.atom_mass_kg,
                profile.probe.wavelength_nm*1e-9, profile.coupling.wavelength_nm*1e-9))
        effective_detunings = ac.get_cached_detunings()
        if profile.mode == 'prescribed' and profile.environment is not None:
            from ..config.prescribed_runtime import capture_prescribed_runtime
            effective_detunings = capture_prescribed_runtime(ac, profile).detunings
        if diagnostics_only:
            result = None
        else:
            from .runner import SimulationRunner
            spectrum = run_profile_spectrum(profile, config, medium, ac)
            runner = SimulationRunner()
            from .detector_readout import resolve_detector_context
            detector_context = resolve_detector_context(profile, ac, medium)
            detector_calibration = None
            if detector_context is not None:
                from .measurement_audit import audit_pin_context
                detector_calibration = audit_pin_context(profile, medium, detector_context,
                    ac.OMEGA_P, ac.get_cached_dipole_matrix()['probe']['dipole_si'])
            detector_options = {'detector_context': detector_context} if detector_context is not None else {}
            if profile.mode == 'experimental' and profile.radiative is not None:
                detector_options['radiative_options'] = profile.radiative
            if drive_realizations is not None:
                detector_options['drive_realizations'] = drive_realizations
            if solver_diagnostic is not None:detector_options['solver_diagnostic']=solver_diagnostic
            if poisson_quantiles:detector_options['poisson_quantiles']=True
            if profile.measurement is not None and profile.mode == 'experimental':
                detector_options['measurement'] = receiver
            result = (runner.run_laboratory_simulation(profile.request, config, **(
                          {'noise': profile.noise} if profile.noise is not None else {}), **detector_options) if profile.mode == 'experimental'
                      else runner.run_prescribed_simulation(profile, config, **detector_options))
            result['spectrum'] = spectrum
        report = {
            'atom': atom_report(ac.ATOM, ac.ATOM_SPEC, profile.atom is not None),
            'schema_version': 2 if profile.environment is not None else 1, 'mode': profile.mode, 'diagnostics_only': diagnostics_only,
            'prepared_single_channel_assumption': True,
            'input_provenance': input_provenance,
            'lo_phase_reference':lo_phase_receipt(profile,diagnostics_only),
            'quantization_axis': profile.quantization_axis,
            'applied_fields': {name: asdict(getattr(profile, name)) for name in ('probe', 'coupling', 'lo', 'signal')},
            'levels': copy.deepcopy(ac.ATOMIC_LEVELS),
            'rf_orientation': asdict(ac.get_rf_orientation()),
            'detunings_rad_per_us': effective_detunings,
            'initialization_detunings_rad_per_us': ac.get_cached_detunings(),
            'detunings_mhz': tuple(v / (2*math.pi) for v in effective_detunings),
            'rabi_rad_per_us': (ac.OMEGA_P, ac.OMEGA_C, ac.OMEGA_RF_BASE, ac.OMEGA_SIGNAL_RF_BASE),
            'decay_rates_per_us': (ac.GAMMA_21, ac.GAMMA_32, ac.GAMMA_43),
            'decay_rate_basis': 'existing ARC n,l,j rates; not resolved magnetic branching',
            'radiative_model': radiative_report(radiative),
            'signed_if_mhz': (profile.signal.frequency_hz - profile.lo.frequency_hz) / 1e6,
            'zeeman_included': False, 'noise_included': False,
            'doppler_ensemble_included': bool(result is not None and profile.spectrum is not None and profile.spectrum.doppler_enabled),
            'temporal_doppler_ensemble_included': bool(result is not None and config.temporal_doppler is not None),
            'temporal_doppler': asdict(config.temporal_doppler) if config is not None and config.temporal_doppler is not None else None,
        }
        if radiative is not None:
            report['initialization_fs_decay_rates_per_us'] = report['decay_rates_per_us']
            report['decay_rates_per_us'] = named_internal_rates(radiative)
            report['decay_rate_basis'] = 'prepared-state partial rates; selected internal spontaneous E1 graph, 0 K'
        if profile.magnetic_core is not None:
            from ..physics.magnetic_core import build_magnetic_core
            core = build_magnetic_core(ac.ATOM, ac.ATOMIC_LEVELS, report['applied_fields'],
                report['rabi_rad_per_us'][:3], profile.radiative.nuclear_projection, profile.magnetic_core)
            report['magnetic_core'] = copy.deepcopy(core.report)
            report['initialization_prepared_radiative_reference'] = report['radiative_model']
            report['radiative_model'] = {'model': profile.magnetic_core.model,
                'scope': 'internal working manifolds only; see magnetic_core.internal_e1_lines'}
            report['reference_four_state_detunings_rad_per_us'] = report['detunings_rad_per_us']
            report['detunings_rad_per_us'] = core.report['detunings_reference_rad_per_us']
            report['detunings_mhz'] = tuple(x/(2*math.pi) for x in report['detunings_rad_per_us'])
            report['reference_prepared_decay_rates_per_us'] = report['decay_rates_per_us']
            report['decay_rates_per_us'] = None
            report['decay_rate_basis'] = 'not a three-gamma model; see magnetic_core.internal_e1_lines'
            report['initialization_is_reference_only'] = True
            report['prepared_single_channel_assumption'] = False
            report['preparation_contract'] = 'reference ground HFS state initially occupied; all core magnetic channels retained'
        if profile.measurement is not None:
            report['measurement']=asdict(receiver)
            report['configured_fields']=copy.deepcopy(report['applied_fields'])
            report['configured_rabi_rad_per_us']=report['rabi_rad_per_us']
            report['effective_signal_field_v_per_m']=profile.signal.electric_field_vm if receiver.signal_enabled else 0.
            if not receiver.signal_enabled:
                report['applied_fields']['signal']['electric_field_vm']=0.
                report['rabi_rad_per_us']=(*report['rabi_rad_per_us'][:3],0.)
            report['signal_switch_semantics']='configured positive source setpoint/channel calibration retained; signal OFF removes its Hamiltonian drive'
        if profile.magnetic_core is not None and profile.magnetic_core.temporal_diagnostic is not None:
            from .runner import SimulationRunner
            from ..physics.susceptibility.absorption_const import calculate_absorption_constant
            prefactors=None if medium is None else calculate_absorption_constant(
                mu_21_c_m=core.probe_reference_dipole_c_m,omega_p_rad_per_s=ac.OMEGA_P*1e6,
                n0_m3=medium.number_density_m3,cell_length_m=medium.cell_length_m,
                lambda_p_m=profile.probe.wavelength_nm*1e-9)
            temporal=SimulationRunner.run_magnetic_core_diagnostic(core,profile.magnetic_core.temporal_diagnostic,
                None if profile.numerics is None else profile.numerics.temporal_ode,report['rabi_rad_per_us'][3],report['signed_if_mhz'],
                absorption_prefactors=prefactors,transit_rate_per_us=config.transit_rate_per_us)
            report['internal_core_temporal']=temporal
            report['magnetic_core']['signal_drive_included']=temporal['signal_drive_included']
            report['magnetic_core']['prepared_transit_executed']=bool(config.transit_rate_per_us)
            report['magnetic_core']['IF_extraction_executed']=temporal['IF_extraction_executed']
            if config.transit_rate_per_us:
                report['magnetic_core']['nonradiative_channels_included']=True
                report['magnetic_core']['nonradiative_scope']='prepared-state transit reset only; no external-FS return'
            report['preparation_contract']=('incoming atoms reset to declared prepared ground; ' if config.transit_rate_per_us else '')+'initial state: '+temporal['initial_state']
        report['temporal_ode'] = ode_receipt(config.temporal_ode if config is not None else None,solver_diagnostic)
        report['temporal_ode']['executed'] = result is not None or 'internal_core_temporal' in report
        if 'internal_core_temporal' in report and profile.magnetic_core.temporal_diagnostic.backend=='charge_block_bdf':
            report['temporal_ode']['scope']='internal magnetic-core SciPy BDF; exact charge reduction and sparse Jacobian'
            report['temporal_ode']['backend']='scipy.integrate.BDF'
        if 'internal_core_temporal' in report and profile.magnetic_core.temporal_diagnostic.backend=='charge_block_periodic':
            report['temporal_ode'].update(executed=False,effective=None,source='not_applicable_periodic',
                scope='ODE controls not executed; explicit periodic Fourier solve selected')
            report['periodic_solver']=temporal['backend_report']
            report['preparation_contract']='prepared incoming atoms; LO steady is initial reference for late-window error estimate, returned densities are periodic steady states'
        if solver_diagnostic is not None or poisson_quantiles:
            report['numerical_diagnostic']={'solver':None if solver_diagnostic is None else asdict(solver_diagnostic),
                'poisson_quantiles':poisson_quantiles,'acquisition_grid_unchanged':True,
                'solver_method':'bdf','nsteps':report['temporal_ode']['effective']['nsteps'],'normalize_output':False,
                'option_enforcement':'not instrumented' if solver_diagnostic is None else
                    'spawn-worker mesolve wrapper verifies options and restores on exception'}
        if profile.noise is not None:
            if not diagnostics_only and detector_calibration is not None:
                report['detector_calibration'] = detector_calibration
            report['noise_declaration'] = asdict(profile.noise)
            report['noise_status'] = ('not_executed_diagnostics' if diagnostics_only and (profile.noise.readout_enabled or profile.noise.drive_selected)
                                     else 'configured_disabled')
            # Metadata only: the disabled execution path instantiates no RNG.
            from .noise_random import NoiseRandomPlan
            report['noise_random_plan'] = NoiseRandomPlan(profile.noise).metadata()
            if result is not None and 'readout_noise' in result:
                report['readout_noise'] = result['readout_noise']
                report['noise_included'] = result['readout_noise']['noise_applied']
                report['noise_status'] = ('applied' if report['noise_included'] else
                    'detector_noiseless' if profile.noise.physical_readout_enabled else 'enabled_zero_sigma')
                report['spectrum_measurement_noise_included'] = False
                if 'noise_campaign' in result:
                    report['noise_campaign'] = result['noise_campaign']
                report['sensing_trace_units'] = result['readout_noise']['units']
            if result is not None and 'rf_drive_campaign' in result:
                report['rf_drive_campaign'] = result['rf_drive_campaign']
                report['rf_drive_included'] = result['rf_drive_campaign']['rf_drive_included']
                report['laser_frequency_noise_included'] = result['rf_drive_campaign']['laser_frequency_noise_included']
                report['laser_intensity_noise_included'] = result['rf_drive_campaign']['laser_intensity_noise_included']
                report['noise_included'] = report['noise_included'] or result['rf_drive_campaign']['drive_noise_included']
                report['noise_status'] = 'applied' if report['noise_included'] else 'enabled_zero_noise'
                report['spectrum_drive_noise_included'] = False
                report['sensing_trace_units'] = result['rf_drive_campaign']['trace_units']
        if profile.mode == 'experimental':
            runtime = ac.get_laboratory_runtime()
            report['optical_energy_convention'] = profile.request.optical_energy_convention.value
            report['rf_mismatch_hz'] = runtime.selection.winner.manifold.mismatch_hz
            report['proposed_wavelengths_nm_not_applied'] = {
                'probe': runtime.selection.optical_selection.probe.proposed_wavelength_nm,
                'coupling': runtime.selection.winner.manifold.coupling.proposed_wavelength_nm,
            }
        else:
            report['optical_energy_convention'] = 'fs_centroid'
        if medium is not None:
            report['environment'] = asdict(profile.environment)
            report['resolved_medium'] = asdict(medium)
        if config is not None:
            report['resolved_rate_budget_per_us'] = {'pure_dephasing': config.dephasing_rates_per_us,
                                                    'transit_reset': config.transit_rate_per_us}
        if profile.spectrum is not None:
            report['spectrum_declaration'] = asdict(profile.spectrum)
            report['spectrum_computed'] = not diagnostics_only
        if result is not None:
            if config.temporal_doppler is not None:
                report['temporal_doppler_reports'] = result['temporal_doppler_reports']
            if receiver.mode == 'if_only':
                from .if_only import summarize_if_result
                report['if_measurement']=summarize_if_result(result)
            else:
                from .if_measurement import measure_sensing_result
                report['if_measurement'] = measure_sensing_result(result)
            report['sensing'] = {'true_aoa_deg': result['true_aoa'], 'estimated_aoa_deg': result['estimated_aoa'],
                                 'sensor_detunings_rad_per_us': result['sensor_detunings_rad_per_us'],
                                 'paths': len(result['worker_records']),
                                 'samples_per_path': len(result['tlist']),
                                 'execution': {key: value for key, value in asdict(config).items() if key not in ('tlist_us','measurement_mode','radiative')},
                                 'duration_us': config.tlist_us[-1],
                                 'sample_step_us': config.tlist_us[1]-config.tlist_us[0]}
            if receiver.mode == 'aoa':
                report['sensing']['doa_effective_settings'] = result['doa_result'].diagnostics['effective_settings']
                report['sensing']['doa_settings_source'] = result['doa_result'].diagnostics['settings_source']
            else:
                report['sensing']['doa_status']='not executed: IF-only receiver'
        if profile.magnetic_core is not None:
            report['initialization_optical_energy_convention'] = report['optical_energy_convention']
            report['optical_energy_convention'] = core.report['energy_convention']
            restore_reference_pipeline()
            report['compatibility_runtime_published'] = False
        return report, result
    except Exception:
        restore_reference_pipeline()
        raise


def json_default(value):
    """Serialize only actual report values; no placeholder for unknown types."""
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f'LAB-006: unsupported report value {type(value).__name__}')


def plot_profile_sensing(result):
    """Plot measured trace, FFT, spatial phase and actual DoA output only."""
    import matplotlib.pyplot as plt
    import numpy as np
    if result.get('measurement_mode') == 'if_only':
        from .if_only import summarize_if_result
        summary = summarize_if_result(result)
        figure, panels = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for index, trace in enumerate(result['time_traces']):
            panels[0].plot(result['tlist'], trace, label=f'Path {index + 1}')
        panels[0].set(xlabel='Time [us]', ylabel=summary['trace_units'], title='IF-only receiver')
        panels[0].legend()
        for trial in summary['trials']:
            panels[1].plot(trial['if_real_per_path'], trial['if_imag_per_path'], 'o')
        panels[1].set(xlabel=f"Re(A_IF) [{summary['trace_units']}]",
                      ylabel=f"Im(A_IF) [{summary['trace_units']}]", title='Recorded I/Q')
        return figure
    figure, panels = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels[0, 0].plot(result['tlist'], result['time_traces'][0])
    panels[0, 0].set(xlabel='Time [us]', ylabel='Worker observable, path 0')
    frequency_order = np.argsort(result['xf_0'])
    panels[0, 1].plot(np.asarray(result['xf_0'])[frequency_order],
                      np.abs(np.asarray(result['yf_0'])[frequency_order]))
    panels[0, 1].set_yscale('log')
    if 'signed_if_mhz' in result:
        frequency = abs(result['signed_if_mhz'])
        panels[0, 1].set_xlim(-3*frequency, 3*frequency)
        panels[0, 1].axvline(frequency, color='gray', linestyle=':', label='Declared IF magnitude')
        panels[0, 1].legend(fontsize=9)
    panels[0, 1].set(xlabel='Frequency [MHz]', ylabel='FFT magnitude, path 0')
    panels[1, 0].plot(np.arange(len(result['complex_responses'])),
                      np.unwrap(np.angle(result['complex_responses'])), 'o-')
    panels[1, 0].set(xlabel='Path index', ylabel='Normalized IF phase [rad]')
    if result['power'] is None:
        panels[1, 1].text(.5, .5, f"Estimated AoA: {result['estimated_aoa']:.6g} deg", ha='center')
        panels[1, 1].set_axis_off()
    else:
        panels[1, 1].plot(result['angles'], result['power'])
        panels[1, 1].set(xlabel='Azimuth [deg]', ylabel='DoA spectrum')
    return figure


def save_profile_results(directory, result):
    """Persist computed arrays, including complex IF/FFT, without invented curves."""
    import numpy as np
    if result.get('measurement_mode') == 'if_only':
        from .if_only import save_if_only_arrays
        return save_if_only_arrays(directory,result)
    arrays = {name: result[name] for name in ('tlist', 'time_traces', 'complex_responses',
              'array_positions', 'element_phases_rad', 'xf_0', 'yf_0')}
    for index, record in enumerate(result['worker_records']):
        arrays[f'fft_frequency_path_{index}'] = record[1]
        arrays[f'fft_complex_path_{index}'] = record[2]
    snapshots = result['if_snapshots']
    if 'readout_noise' in result:
        for name in ('clean_time_traces', 'clean_complex_responses', 'additive_readout_noise'):
            arrays[name] = result[name]
        for index, record in enumerate(result['clean_worker_records']):
            arrays[f'clean_fft_frequency_path_{index}'] = record[1]
            arrays[f'clean_fft_complex_path_{index}'] = record[2]
        trials = result['noise_trial_results'] if 'noise_trial_results' in result else result['rf_drive_trial_results']
        arrays['noise_trial_indices'] = np.arange(len(trials), dtype=int)
        for name in ('time_traces', 'complex_responses', 'additive_readout_noise'):
            arrays['trial_' + name] = np.asarray([trial[name] for trial in trials])
        arrays['trial_estimated_aoa_deg'] = np.asarray([trial['estimated_aoa'] for trial in trials])
        campaign = result.get('noise_campaign', result.get('rf_drive_campaign'))
        arrays['trial_error_deg'] = np.asarray([row['error_deg'] for row in campaign['trials']])
        arrays['trial_if_snapshots'] = np.asarray([trial['if_snapshots'].X for trial in trials])
        if 'detector_arrays' in result:
            arrays.update(result['detector_arrays'])
            for key in result['detector_arrays']:
                if (key.startswith('measured_') or key.endswith('_electron_counts') or key.endswith('_current_noise_a') or
                        key in ('clean_optical_power_w', 'noiseless_current_a', 'clean_detector_transmission',
                                'incident_probe_power_w', 'probe_rabi_multiplier')):
                    arrays['trial_'+key] = np.asarray([trial['detector_arrays'][key] for trial in trials])
        for trial_index, trial in enumerate(trials):
            for path, record in enumerate(trial['worker_records']):
                arrays[f'trial_{trial_index}_fft_frequency_path_{path}'] = record[1]
                arrays[f'trial_{trial_index}_fft_complex_path_{path}'] = record[2]
        if all(trial['power'] is not None for trial in trials):
            arrays['trial_power'] = np.asarray([trial['power'] for trial in trials])
    if 'rf_drive_campaign' in result:
        trials = result['rf_drive_trial_results']
        arrays.update(nominal_time_traces=result['nominal_time_traces'],
                      nominal_complex_responses=result['nominal_complex_responses'],
                      trial_time_traces=np.asarray([t['time_traces'] for t in trials]),
                      trial_complex_responses=np.asarray([t['complex_responses'] for t in trials]),
                      trial_if_snapshots=np.asarray([t['if_snapshots'].X for t in trials]),
                      trial_estimated_aoa_deg=np.asarray([t['estimated_aoa'] for t in trials]),
                      trial_error_deg=np.asarray([r['error_deg'] for r in result['rf_drive_campaign']['trials']]))
        for trial_index, trial in enumerate(trials):
            for path, record in enumerate(trial['worker_records']):
                arrays[f'trial_{trial_index}_fft_frequency_path_{path}'] = record[1]
                arrays[f'trial_{trial_index}_fft_complex_path_{path}'] = record[2]
        if all(t['power'] is not None for t in trials):
            arrays['trial_power'] = np.asarray([t['power'] for t in trials])
        realizations = result['rf_drive_realizations']
        if realizations[0] is not None:
            arrays['drive_noise_times_us'] = realizations[0].times_us
            for key in ('lo_amplitude', 'signal_amplitude', 'relative_phase_rad'):
                arrays['trial_rf_'+key] = np.asarray([getattr(r, key) for r in realizations])
            for key in ('probe_frequency_hz', 'coupling_frequency_hz'):
                arrays['trial_laser_'+key] = np.asarray([getattr(r, key) for r in realizations])
            arrays['trial_laser_integrated_phase_rad'] = np.asarray([r.laser_phase_nodes() for r in realizations])
            for key in ('probe_intensity', 'coupling_intensity'):
                arrays['trial_laser_'+key] = np.asarray([getattr(r, key) for r in realizations])
                arrays['trial_laser_'+key.replace('intensity', 'rabi_multiplier')] = np.sqrt(arrays['trial_laser_'+key])
    from ..doa.snapshots import sample_covariance
    arrays.update(if_snapshots=snapshots.X, snapshot_centers_us=snapshots.time_centers_us,
                  sample_covariance=sample_covariance(snapshots.X))
    if result['power'] is not None:
        arrays.update(angles=result['angles'], power=result['power'])
    spectrum = result.get('spectrum')
    if spectrum is not None:
        arrays.update({'spectrum_' + key: value for key, value in spectrum.items()
                       if value is not None and key not in ('analytical_contract', 'radiative_model', 'decay_rates_per_us')})
    np.savez_compressed(directory / 'simulation_arrays.npz', **arrays)


def plot_profile_spectrum(result):
    """Draw actual computed transmission and the separately labeled approximation."""
    import matplotlib.pyplot as plt
    spectrum = result.get('spectrum')
    if spectrum is None:
        return None
    figure, panel = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = spectrum['probe_offsets_mhz']
    for name, label in (('rf_off', 'RF off'), ('rf_on', 'RF LO on')):
        panel.plot(x, spectrum[name], 'o-', label=label + ', numerical')
        reference = spectrum[name + '_analytical']
        if reference is not None:
            panel.plot(x, reference, 'o--', label=label + ', zero-velocity analytical approximation')
    panel.set(xlabel='Probe offset from applied center [MHz]', ylabel='Cell transmission')
    panel.legend()
    return figure


def plot_profile_rf_levels(report):
    """RF subdiagram in physical frequency units; preserve declared ladder identities."""
    import matplotlib.pyplot as plt
    figure, panel = plt.subplots(figsize=(7, 4), constrained_layout=True)
    signed_ghz = report['rf_orientation']['signed_frequency_hz'] / 1e9
    for x, role, energy in ((0, 'rydberg1', 0.), (1, 'rydberg2', signed_ghz)):
        level = report['levels'][role]
        panel.hlines(energy, x - .25, x + .25)
        panel.annotate(f"{role}: n={level['n']}, l={level['l']}, j={level['j']}, mJ={level['mj']}",
                       (x, energy), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
    panel.annotate('', (1, signed_ghz), (0, 0), arrowprops={'arrowstyle': '<->'})
    panel.set(xlim=(-.6, 1.6), xticks=[], ylabel='(E - E_R1)/h [GHz]',
              title='Declared Rydberg pair: actual signed RF energy ordering')
    panel.margins(y=.25)
    return figure
