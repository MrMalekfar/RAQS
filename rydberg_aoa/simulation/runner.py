"""
Simulation runner module for orchestrating Rydberg AoA simulations.
Handles the high-level simulation workflow and data processing.
"""
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from typing import Tuple, List, Optional

from rydberg_aoa.physics.susceptibility.absorption_const import calculate_absorption_constant

# Import our modules
from ..config import parameters as params, atomic_config as acfg
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError
from .steady_state import simulate_spectrum
from .time_domain import run_single_heterodyne_worker, atomic_if_polarity
from .doa_pipeline import estimate_doa_from_results, merge_doa_into_results
from ..signal_processing.array_geometry import get_array_phases
from ..tools import analysis
from ..physics.lindblad.calculate_dephasing_rates import calculate_dephasing_rates
from ..physics.lindblad.calculate_transit_rate import resolve_effective_transit_rate
from ..physics.susceptibility.absorption_const import calculate_absorption_constant


def _validate_worker_records(records, tlist_len=None, expected_count=None,
                             require_contiguous_indices=False):
    """Fail-loud validation of a batch of heterodyne worker records (F05-WORKER-001).

    A failed element must abort the run, never be published as None-filled
    data.  The batch is materialized to a list here (so a generator cannot be
    consumed during validation and leave processing with an exhausted input);
    callers MUST consume the returned list.  Each record must be the 5-element
    success tuple ``(complex_amp, xf, yf, element_index, signal_trace)`` with:

    * amplitude: a real OR complex numeric SCALAR (int/float/complex or their
      NumPy counterparts), excluding bool/np.bool_, strings/bytes, EVERY
      ndarray (including 0-D), containers and conversion-only objects; finite
      real and imaginary parts (zero valid).  Conversion success is never
      accepted as proof of a valid type.
    * xf: nonempty, finite, 1-D REAL numeric array;
    * yf: nonempty, finite, 1-D REAL OR COMPLEX numeric array (its imaginary
      part is preserved);
    * matching xf/yf lengths;
    * signal_trace: nonempty, finite, 1-D REAL numeric array (of the
      dispatched time-grid length when ``tlist_len`` is given);
    * element_index: a nonnegative Python/NumPy integer (not bool), unique.

    ``expected_count`` enforces the batch size; ``require_contiguous_indices``
    enforces the index set equals range(count).  Never repairs/pads/drops or
    replaces records, never flattens a malformed amplitude, never discards an
    imaginary component; raises ``PhysicsCalculationError`` on any violation.
    """
    if records is None:
        raise PhysicsCalculationError(
            "F05-WORKER-001: the time-domain worker batch is None; no results "
            "may be published or consumed.")
    try:
        records = list(records)
    except TypeError as exc:
        raise PhysicsCalculationError(
            f"F05-WORKER-001: the time-domain worker batch is not iterable "
            f"(type={type(records).__name__}); no results may be published or "
            f"consumed.") from exc
    if expected_count is not None and len(records) != expected_count:
        raise PhysicsCalculationError(
            f"F05-WORKER-001: expected exactly {expected_count} worker "
            f"result(s), got {len(records)}; a partial or over-full batch is "
            f"never published.")
    if len(records) == 0:
        raise PhysicsCalculationError(
            "F05-WORKER-001: the time-domain worker batch is empty.")

    def _array(value, pos, name, allow_complex):
        try:
            arr = np.asarray(value)
        except Exception as exc:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} {name} is "
                f"not a constructible array ({exc}).") from exc
        if arr.ndim != 1 or arr.size == 0:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} {name} must "
                f"be a nonempty 1-D array (got shape {arr.shape}).")
        if arr.dtype == np.bool_ or arr.dtype == object \
                or arr.dtype.kind in ("U", "S", "V") \
                or not np.issubdtype(arr.dtype, np.number):
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} {name} must "
                f"be numeric (got dtype {arr.dtype}).")
        if np.issubdtype(arr.dtype, np.complexfloating) and not allow_complex:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} {name} must "
                f"be a real array (got complex dtype {arr.dtype}).")
        try:
            finite = bool(np.all(np.isfinite(arr)))
        except TypeError as exc:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} {name} is not "
                f"finite-checkable (dtype {arr.dtype}).") from exc
        if not finite:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} {name} "
                f"contains non-finite values.")
        return arr

    seen = set()
    for pos, rec in enumerate(records):
        if not isinstance(rec, tuple) or len(rec) != 5:
            length = len(rec) if hasattr(rec, "__len__") else "n/a"
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} is not the "
                f"5-element success tuple (got {type(rec).__name__} of length "
                f"{length}); error tuples must never be produced or consumed.")
        amp, xf, yf, idx, trace = rec
        if amp is None or xf is None or yf is None or idx is None \
                or trace is None:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} contains a "
                f"None field; a failed element must abort the run, not be "
                f"published as None-filled data.")
        if isinstance(amp, (bool, np.bool_)) or not isinstance(
                amp, (int, float, complex, np.integer, np.floating,
                      np.complexfloating)):
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} amplitude "
                f"must be a real or complex numeric scalar, got {amp!r} "
                f"(type={type(amp).__name__}).")
        try:
            amp_val = complex(amp)
        except (OverflowError, TypeError, ValueError) as exc:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} amplitude "
                f"{amp!r} cannot be converted to a complex scalar ({exc}).") \
                from exc
        if not (np.isfinite(amp_val.real) and np.isfinite(amp_val.imag)):
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} has a "
                f"non-finite amplitude {amp!r}.")
        if isinstance(idx, (bool, np.bool_)) or not isinstance(
                idx, (int, np.integer)):
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} has a "
                f"non-integer element index {idx!r}.")
        idx_int = int(idx)
        if idx_int < 0:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} has a "
                f"negative element index {idx_int}.")
        if idx_int in seen:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: duplicate element index {idx_int} in the "
                f"worker batch.")
        seen.add(idx_int)
        xf_arr = _array(xf, pos, "xf", allow_complex=False)
        yf_arr = _array(yf, pos, "yf", allow_complex=True)
        if xf_arr.shape[0] != yf_arr.shape[0]:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} has mismatched"
                f" xf/yf lengths ({xf_arr.shape[0]} vs {yf_arr.shape[0]}).")
        tr = _array(trace, pos, "signal trace", allow_complex=False)
        if tlist_len is not None and tr.shape[0] != tlist_len:
            raise PhysicsCalculationError(
                f"F05-WORKER-001: worker record at position {pos} trace length "
                f"{tr.shape[0]} does not match the dispatched time grid "
                f"{tlist_len}.")
    if require_contiguous_indices and seen != set(range(len(records))):
        raise PhysicsCalculationError(
            f"F05-WORKER-001: worker element indices {sorted(seen)} do not match "
            f"the dispatched set {set(range(len(records)))}.")
    return records


class SimulationRunner:
    """
    Orchestrates the complete Rydberg AoA simulation workflow.
    """

    @staticmethod
    def run_magnetic_core_diagnostic(model,diagnostic,ode,omega_signal_rad_per_us,signed_if_mhz,*,absorption_prefactors=None,transit_rate_per_us=0.):
        """BBR-014: explicit multilevel worker; never enters a four-state solver."""
        from .magnetic_core_temporal import run_magnetic_core_temporal
        return run_magnetic_core_temporal(model,diagnostic,ode,omega_signal_rad_per_us,signed_if_mhz,
                                         absorption_prefactors=absorption_prefactors,transit_rate_per_us=transit_rate_per_us)

    def __init__(self):
        """Initialize the simulation runner."""
        self.array_positions = None
        self.element_phases_rad = None
        self.results = None
        self._laboratory_result_runtime = None
        self._prescribed_result_context = None
        self._result_measurement_mode = None

    def setup_array_geometry(self) -> np.ndarray:
        """
        Set up the antenna array geometry.

        Returns:
            Array positions as (N_elements, 2) array
        """
        self.array_positions = np.zeros((params.N_ELEMENTS, 2))
        self.array_positions[:, 0] = np.arange(params.N_ELEMENTS) * params.SPACING
        return self.array_positions

    def get_element_phases(self) -> np.ndarray:
        """
        Calculate the true element phases for the target AoA.

        Returns:
            Element phases in radians
        """
        if self.array_positions is None:
            self.setup_array_geometry()

        self.element_phases_rad = get_array_phases(
            self.array_positions,
            params.RF_FREQUENCY,
            params.TARGET_AOA_DEG
        )
        return self.element_phases_rad

    def run_steady_state_spectrum(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the steady-state spectrum validation.

        ARCH-001: requires explicit atomic initialization — resolved Rabi
        frequencies, decay rates and cached detunings are only valid after
        ``rydberg_aoa.config.atomic_config.initialize_atomic_physics()``.

        Returns:
            Tuple of (dp_scan, trans_no_rf, trans_no_rf_analytic, trans_with_rf, trans_with_rf_analytic)
        """
        # ARCH-001: running the spectrum requires resolved atomic physics.
        acfg.require_atomic_physics_initialized()
        # 1. PREPARE DATA (Extract numbers from Config)
        
        # Scan Range
        dp_scan_mhz = np.linspace(-params.SCAN_RANGE_MHZ, params.SCAN_RANGE_MHZ, params.N_SCAN)
        
        # Omegas (Probe, Coupling, RF_Base)
        p_omegas = (acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE)
        
        # Detunings (Center values)
        p_deltas = acfg.get_cached_detunings()
        
        # Decay Rates
        p_gammas = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
        # RF-DIR-001B: resolved energy orientation of the |3> <-> |4> RF edge
        # (published by initialize_atomic_physics; None → legacy ladder-index
        # jump inside create_sensor_model).
        rf_orientation = acfg.get_rf_orientation()
        
        # Dephasing Rates (Using your physics helper)
        p_dephasing = calculate_dephasing_rates() # Returns [g2, g3, g4]
        # DPH-002B: transit is a loss/repopulation channel, threaded explicitly
        # to every production model build (never part of p_dephasing).
        # PHY-SW-001: gated by params.TRANSIT_ENABLED (True → legacy rate).
        transit_rate = resolve_effective_transit_rate()
        print(f"DPH-002B: effective Markovian transit rate Gamma_t = {transit_rate:.4f} 1/us")
        absorption_const = calculate_absorption_constant()
        p_omegas_no_rf = list(p_omegas)
        p_omegas_no_rf[2] = 0.0  # omega_rf = 0
        dp_scan, trans_no_rf, trans_no_rf_analytic = simulate_spectrum(
            0.0, 
            dp_scan_mhz, 
            tuple(p_omegas_no_rf), p_deltas, p_gammas, p_dephasing, absorption_const,
            transit_rate_per_us=transit_rate,
            rf_orientation=rf_orientation,
        )
        dp_scan, trans_with_rf, trans_with_rf_analytic = simulate_spectrum(
            acfg.OMEGA_RF_BASE, 
            dp_scan_mhz, 
            p_omegas, p_deltas, p_gammas, p_dephasing, absorption_const,
            transit_rate_per_us=transit_rate,
            rf_orientation=rf_orientation,
        )
        return dp_scan, trans_no_rf, trans_no_rf_analytic, trans_with_rf, trans_with_rf_analytic

    def _sensor_probe_detunings(self, p_deltas):
        """OP-001 frozen sensor operating point — apply the sensor probe detuning.

        Concept separation (phase-scan vs sensing):
          A. Spectrum-scan detunings stay at the level-optimization center
             (run_steady_state_spectrum / simulate_spectrum — EIT/AT plots).
          B. Sensor operating detuning SENSOR_PROBE_DETUNING_MHZ [MHz] is
             applied ONLY to the time-domain / heterodyne sensing path.

        Conversion (C-001/OP-001 unit contract):
            Delta_sensor [rad/us] = 2*pi * SENSOR_PROBE_DETUNING_MHZ
        Exactly ONE 2*pi factor; NO extra 2*pi. Additive to the
        level-optimization residual, matching the HET-001 detuning-sweep
        convention (sensor peak at Delta = -0.5 MHz).

        Args:
            p_deltas: (delta_p, delta_c, delta_rf) from get_cached_detunings

        Returns:
            tuple: (delta_p_sensor, delta_c, delta_rf) in rad/us
        """
        # ARCH-001: running the sensor requires resolved atomic physics.
        acfg.require_atomic_physics_initialized()
        delta_p, delta_c, delta_rf = p_deltas
        return (
            delta_p + 2.0 * np.pi * params.SENSOR_PROBE_DETUNING_MHZ,
            delta_c,
            delta_rf,
        )

    def run_time_domain_simulation(self) -> Tuple[List, np.ndarray]:
        """
        Run the parallel time-domain heterodyne simulation.

        Returns:
            Tuple of (simulation results, time array)
        """
        # F05-WORKER-001: invalidate any previous results at the start of a
        # new run so a failure here cannot leave stale/partial data consumable.
        self.results = None
        self._laboratory_result_runtime = None
        self._prescribed_result_context = None
        self._result_measurement_mode = None
        if self.element_phases_rad is None:
            self.get_element_phases()
                # 1. CALCULATE DEPHASING RATES ONCE (Main Process)
        # This uses your physics logic!
        dephasing_rates_list = calculate_dephasing_rates() 
        # DPH-002B: effective transit reset rate — threaded explicitly to workers
        # PHY-SW-001: gated by params.TRANSIT_ENABLED (True → legacy rate).
        transit_rate = resolve_effective_transit_rate()
        print(f"DPH-002B: effective Markovian transit rate Gamma_t = {transit_rate:.4f} 1/us")
        # Result example: [0.35, 0.42, 0.42]
        tlist = np.linspace(0, params.T_MAX, params.T_STEPS)
        # 1. EXTRACT ALL PHYSICS PARAMETERS HERE (Main Process Only)
        # This triggers the ARC calculation ONCE.
        p_omegas = (acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE)
        
        # Get detunings (ensure these are calculated). OP-001: the SENSOR
        # operating probe detuning (SENSOR_PROBE_DETUNING_MHZ = -0.5 MHz) is
        # applied to the TIME-DOMAIN / heterodyne sensing path ONLY — the
        # steady-state spectrum scan keeps the level-optimization center.
        p_deltas = self._sensor_probe_detunings(acfg.get_cached_detunings()) 
        
        # Get decay rates
        p_gammas = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
        # RF-DIR-001B: resolved energy orientation of the |3> <-> |4> RF edge
        # (threaded to every worker; picklable frozen dataclass).
        rf_orientation = acfg.get_rf_orientation()
        omega_S_complex_array = acfg.OMEGA_SIGNAL_RF_BASE * np.exp(1j * self.element_phases_rad)

        # Debug: Print complex Rabi frequencies
        print("DEBUG: Complex Rabi frequencies |Omega|:")
        for i, omega in enumerate(omega_S_complex_array):
            amplitude = np.abs(omega)
            phase_rad = np.angle(omega)
            phase_deg = np.rad2deg(phase_rad)
            print(f"  Element {i}: |Omega| = {amplitude:.4f}, phase = {phase_rad:.4f} rad ({phase_deg:.1f} deg)")
        # 2. Prepare Arguments for Workers
        # We pass pure numbers, so workers don't need to import ARC/Config
        args_list = []
        for i in range(params.N_ELEMENTS):
            args = (
                i, 
                omega_S_complex_array[i], 
                tlist, 
                params.IF_FREQ,
                # Expand tuples for clarity
                p_omegas[0], p_omegas[1], p_omegas[2],
                p_deltas[0], p_deltas[1], p_deltas[2],
                p_gammas,
                dephasing_rates_list,
                # noise_model=None (positional slot), DPH-002B transit rate,
                # then RF-DIR-001B energy orientation of the RF-edge jump
                None,
                transit_rate,
                rf_orientation
            )
            args_list.append(args)

        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel simulation.")

        # Use multiprocessing.Pool for better serialization control
        import multiprocessing as mp

        print(f"Using multiprocessing.Pool with {num_cores} processes")

        with mp.Pool(processes=num_cores) as pool:
            raw_results = pool.starmap(run_single_heterodyne_worker, args_list)

        # F05-WORKER-001: validate the COMPLETE batch before publishing.  On
        # any failure self.results stays None (unavailable); no previous or
        # partial batch is ever published.  Sort by element index only AFTER
        # validation.  Zero amplitude is valid worker data.
        validated = _validate_worker_records(
            raw_results, tlist_len=len(tlist),
            expected_count=params.N_ELEMENTS,
            require_contiguous_indices=True,
        )
        validated = sorted(validated, key=lambda r: int(r[3]))
        self.results = validated
        return self.results, tlist

    def run_laboratory_simulation(self, request, run_config, *, force=False, noise=None, detector_context=None, drive_realizations=None, measurement=None, solver_diagnostic=None, poisson_quantiles=False, radiative_options=None) -> dict:
        """LAB-005 explicit prepared-channel sensing, spawn workers and DoA.

        No spectrum scan, optical tuning or Doppler ensemble is implied.
        Applied fields must match request (LAB-004). Pure dephasing, transit,
        record, geometry and sensing offset are explicit in run_config.
        Results are published only after the entire batch and DoA succeed.
        """
        from ..config.detection_numerics import validate_diagnostic_controls
        self.results = None
        self._laboratory_result_runtime = None
        self._prescribed_result_context = None
        self._result_measurement_mode = None
        validate_diagnostic_controls(solver_diagnostic,poisson_quantiles)
        from ..config.noise import require_noise_execution_supported
        require_noise_execution_supported(noise)
        from ..config.rf_noise import validate_rf_noise_execution
        validate_rf_noise_execution(noise, run_config)
        if drive_realizations is not None:
            from .rf_drive_campaign import validate_drive_realizations
            validate_drive_realizations(noise,drive_realizations)
        from .detector_readout import require_detector_context
        require_detector_context(noise, run_config, detector_context)
        from .laboratory_execution import LaboratoryRunConfig, LaboratoryWorkerTask, run_laboratory_worker
        from ..config.laboratory import ExperimentalLabRequest
        from ..config import doa_config
        from ..doa.snapshots import extract_if_snapshots
        from .doa_pipeline import resolve_doa_method

        if not isinstance(request, ExperimentalLabRequest) or not isinstance(run_config, LaboratoryRunConfig):
            raise PhysicsConfigurationError('LAB-005: explicit laboratory request and LaboratoryRunConfig required.')
        if radiative_options is not None:
            from ..config.profile_radiative import require_radiative_selection
            require_radiative_selection(radiative_options, run_config.radiative)
        run_config.validate_sampling(request.signed_if_mhz)
        from ..config.measurement import resolved_measurement
        receiver = resolved_measurement(measurement)
        if receiver.mode != run_config.measurement_mode:
            raise PhysicsConfigurationError('NOISE-013: receiver and execution modes disagree.')
        if receiver.mode == 'aoa':
            selected = resolve_doa_method(run_config.doa_method)
        if receiver.mode == 'aoa' and run_config.doa_options is None and doa_config.DOA_ELEVATION_DEG != 0:
            raise PhysicsConfigurationError('LAB-005: this azimuth-only sensing profile requires DoA elevation=0 degrees.')
        if any(v == 0 for v in (request.probe.power_mw, request.coupling.power_mw,
                               request.lo.electric_field_vm, request.signal.electric_field_vm)):
            raise PhysicsConfigurationError('LAB-005: driven beat-note AoA needs nonzero probe, coupling, LO and signal; no invented angle.')
        acfg.initialize_atomic_physics(resolution_request=request, force=force)
        runtime = acfg.get_laboratory_runtime()
        options = {} if drive_realizations is None else {'_drive_realizations':drive_realizations}
        if measurement is not None:
            options['measurement'] = receiver
        if solver_diagnostic is not None:options['solver_diagnostic']=solver_diagnostic
        if poisson_quantiles:options['poisson_quantiles']=True
        return self._run_resolved_sensing(request, run_config, runtime, 'experimental', noise=noise, detector_context=detector_context, **options)

    def run_prescribed_simulation(self, profile, run_config, *, force=False, detector_context=None, drive_realizations=None, solver_diagnostic=None, poisson_quantiles=False):
        """LAB-007: use the unchanged prescribed seam and shared sensing pipeline."""
        from ..config.magnetic_core import require_four_state_profile
        require_four_state_profile(profile)
        from ..config.detection_numerics import validate_diagnostic_controls
        self.results = None
        self._laboratory_result_runtime = None
        self._prescribed_result_context = None
        self._result_measurement_mode = None
        validate_diagnostic_controls(solver_diagnostic,poisson_quantiles)
        from ..config.run_profile import RunProfile
        from ..config.laboratory_runtime import require_matching_applied_fields
        if not isinstance(profile, RunProfile) or profile.mode != 'prescribed':
            raise PhysicsConfigurationError('LAB-007: an explicit prescribed RunProfile is required.')
        if (profile.temporal_doppler is not None and profile.temporal_doppler.enabled
                and run_config.temporal_doppler is None):
            raise PhysicsConfigurationError('DOP-TD-001: enabled profile requires resolved temporal ensemble; use execute_profile.')
        from ..config.noise import require_noise_execution_supported
        require_noise_execution_supported(profile.noise)
        from ..config.rf_noise import validate_rf_noise_execution
        validate_rf_noise_execution(profile.noise, run_config)
        if drive_realizations is not None:
            from .rf_drive_campaign import validate_drive_realizations
            validate_drive_realizations(profile.noise,drive_realizations)
        from .detector_readout import require_detector_context
        require_detector_context(profile.noise, run_config, detector_context)
        self._validate_explicit_sensing(profile, run_config)
        require_matching_applied_fields(acfg, profile)
        acfg.initialize_atomic_physics(resolution_request=profile.request.resolution_request(), force=force)
        runtime = self._current_explicit_runtime(profile, 'prescribed')
        options = {} if drive_realizations is None else {'_drive_realizations':drive_realizations}
        if solver_diagnostic is not None:options['solver_diagnostic']=solver_diagnostic
        if poisson_quantiles:options['poisson_quantiles']=True
        return self._run_resolved_sensing(profile, run_config, runtime, 'prescribed', noise=profile.noise, detector_context=detector_context, **options)

    @staticmethod
    def _current_explicit_runtime(request, mode):
        if mode == 'experimental':
            return acfg.get_laboratory_runtime()
        from ..config.prescribed_runtime import capture_prescribed_runtime
        return capture_prescribed_runtime(acfg, request)

    @staticmethod
    def _validate_explicit_sensing(request, run_config):
        from .laboratory_execution import LaboratoryRunConfig
        from ..config import doa_config
        from .doa_pipeline import resolve_doa_method
        if not isinstance(run_config, LaboratoryRunConfig):
            raise PhysicsConfigurationError('LAB-007: resolved LaboratoryRunConfig required.')
        run_config.validate_sampling(request.signed_if_mhz)
        from ..config.measurement import resolved_measurement
        receiver=resolved_measurement(getattr(request,'measurement',None))
        if receiver.mode != run_config.measurement_mode:
            raise PhysicsConfigurationError('NOISE-013: receiver and execution modes disagree.')
        if receiver.mode == 'aoa':
            resolve_doa_method(run_config.doa_method)
        if receiver.mode == 'aoa' and run_config.doa_options is None and doa_config.DOA_ELEVATION_DEG != 0:
            raise PhysicsConfigurationError('LAB-007: explicit sensing currently requires elevation=0.')
        if any(v == 0 for v in (request.probe.power_mw, request.coupling.power_mw,
                               request.lo.electric_field_vm, request.signal.electric_field_vm)):
            raise PhysicsConfigurationError('LAB-007: beat-note sensing requires nonzero applied fields.')

    def _run_resolved_sensing(self, request, run_config, runtime, mode, *, noise=None, detector_context=None,
                              _drive=None, _publish=True, _measurement_trial=None, _records_override=None, _drive_realizations=None, _temporal_reports_override=None, measurement=None, solver_diagnostic=None, poisson_quantiles=False):
        from ..config.profile_radiative import active_radiative_rates, radiative_report
        rates = active_radiative_rates(run_config, runtime, acfg, request)
        from ..config.detection_numerics import SolverDiagnostic
        if (solver_diagnostic is not None and not isinstance(solver_diagnostic,SolverDiagnostic)) or type(poisson_quantiles) is not bool:
            raise PhysicsConfigurationError('NOISE-016: typed diagnostic and boolean Poisson coupling required.')
        from ..config.measurement import resolved_measurement, validate_signal_noise
        receiver=resolved_measurement(measurement if measurement is not None else getattr(request,'measurement',None))
        validate_signal_noise(receiver,noise)
        if_only=receiver.mode == 'if_only'
        if receiver.mode != run_config.measurement_mode:
            raise PhysicsConfigurationError('NOISE-013: receiver mode does not match resolved execution.')
        if noise is not None and noise.schema_version == 3:
            from .rf_drive_campaign import run_rf_drive_campaign
            options = {} if _drive_realizations is None else {'realizations':_drive_realizations}
            if if_only:
                options['receiver'] = receiver
            if solver_diagnostic is not None:options['solver_diagnostic']=solver_diagnostic
            if poisson_quantiles:options['poisson_quantiles']=True
            return run_rf_drive_campaign(self, request, run_config, runtime, mode, noise, detector_context, **options)
        if _drive_realizations is not None:
            raise PhysicsConfigurationError('NOISE-010: supplied realizations cannot be ignored by a non-drive campaign.')
        from .laboratory_execution import LaboratoryWorkerTask, run_laboratory_worker
        from ..doa.snapshots import extract_if_snapshots
        from .doa_pipeline import resolve_doa_method
        selected = None if if_only else resolve_doa_method(run_config.doa_method)
        orientation = runtime.selection.rf_orientation if mode == 'experimental' else runtime.rf_orientation
        positions = np.asarray(run_config.positions_m)
        phases = get_array_phases(positions, request.signal.frequency_hz, run_config.target_azimuth_deg)
        if not np.all(np.isfinite(phases)):
            raise PhysicsConfigurationError('LAB-005: spatial phase overflowed; no phase substitution.')
        rabi = runtime.selection.rabi if mode == 'experimental' else runtime.rabi
        detunings = (runtime.detunings[0] + 2*np.pi*run_config.sensor_probe_offset_mhz,
                     *runtime.detunings[1:])
        if not np.all(np.isfinite(detunings)):
            raise PhysicsConfigurationError('LAB-005: sensing detunings overflowed; no clipping.')
        effective_signal_rabi = rabi.signal if receiver.signal_enabled else 0.
        from ..config import atomic_config as ac
        from ..config.atom_species import resolved_atom
        atom_receipt = resolved_atom(ac.ATOM, ac.ATOM_SPEC)
        tasks = [LaboratoryWorkerTask(i, complex(effective_signal_rabi*np.exp(1j*phase)), run_config.tlist_us,
                                     request.signed_if_mhz, (rabi.probe, rabi.coupling, rabi.lo),
                                     detunings, rates, run_config.dephasing_rates_per_us,
                                     run_config.transit_rate_per_us, orientation,
                                     run_config.transient_delay_us,
                                     atom=atom_receipt,
                                     measurement_mode=receiver.mode,
                                     temporal_ode=run_config.temporal_ode,
                                     **({'solver_diagnostic':solver_diagnostic} if solver_diagnostic is not None else {}),
                                     **({'rf_drive': _drive} if _drive is not None else {}),
                                     **({'temporal_doppler': run_config.temporal_doppler} if run_config.temporal_doppler is not None else {})) for i, phase in enumerate(phases)]
        # Explicit spawn also on platforms where fork is the ambient default.
        # No ARC instance, global physics state or Qobj crosses this boundary.
        try:
            if _records_override is not None:
                # Explicit reuse only for a zero-drive realization within the
                # same immutable runtime, never a failed worker fallback.
                raw = _records_override
            else:
                with multiprocessing.get_context('spawn').Pool(processes=run_config.processes) as pool:
                    raw = pool.map(run_laboratory_worker, tasks)
        except PhysicsError:
            raise
        except Exception as exc:
            raise PhysicsCalculationError('LAB-005: spawn worker dispatch failed; no partial batch or serial retry.') from exc
        temporal_reports = None
        if run_config.temporal_doppler is not None:
            from .temporal_doppler import DopplerWorkerResult
            if _records_override is not None:
                if _temporal_reports_override is None:
                    raise PhysicsCalculationError('DOP-TD-001: reused records require ensemble receipts.')
                temporal_reports = _temporal_reports_override
            else:
                if len(raw) != len(positions) or any(not isinstance(r, DopplerWorkerResult) for r in raw):
                    raise PhysicsCalculationError('DOP-TD-001: complete ensemble worker receipts required.')
                temporal_reports = [r.report for r in raw]
                raw = [r.record for r in raw]
            from dataclasses import asdict
            if (len(temporal_reports) != len(positions) or
                    any(not isinstance(r, dict) or r.get('converged') is not True or
                        r.get('plan') != asdict(run_config.temporal_doppler) or
                        r.get('element_index') != i for i, r in enumerate(temporal_reports))):
                raise PhysicsCalculationError('DOP-TD-001: stale/incomplete temporal ensemble receipts; no reuse.')
        records = sorted(_validate_worker_records(raw, tlist_len=len(run_config.tlist_us),
                                                  expected_count=len(positions), require_contiguous_indices=True),
                         key=lambda record: int(record[3]))
        if self._current_explicit_runtime(request, mode) != runtime:
            raise PhysicsConfigurationError('LAB-005: runtime changed during worker execution; refusing stale results.')
        active_radiative_rates(run_config, runtime, acfg, request)
        clean_records = records
        atomic_records = records
        physical = noise is not None and noise.physical_readout_enabled
        atomic_traces = np.asarray([record[4] for record in records]) if physical else None
        detector_kwargs = {}
        if physical and _drive is not None and any(v != 1. for v in _drive.probe_intensity):
            detector_kwargs['probe_intensity'] = np.asarray([
                _drive.laser_intensity_at(t)[0] for t in run_config.tlist_us])

        if physical:
            if not if_only and np.count_nonzero([record[0] for record in atomic_records]) < 2:
                raise PhysicsCalculationError('NOISE-004: fewer than two nonzero atomic IF responses; detector noise cannot invent a signal.')
            from .detector_readout import detector_forward
            from ..utils.numerical import extract_if_component
            _, _, clean_detector, _ = detector_forward(atomic_traces, noise.detector, detector_context, **detector_kwargs)
            retained = clean_detector[:, np.asarray(run_config.tlist_us) >= run_config.transient_delay_us]
            if not if_only and np.count_nonzero(np.ptp(retained, axis=1)) < 2:
                raise PhysicsCalculationError('NOISE-004: clean detector has no resolved temporal modulation; no noise-invented angle.')
            converted = []
            for record, trace in zip(records, clean_detector):
                try:
                    amp, xf, yf = extract_if_component(trace, run_config.tlist_us,
                        request.signed_if_mhz, run_config.transient_delay_us)
                except Exception as exc:
                    raise PhysicsCalculationError('NOISE-004: clean detector IF extraction failed; no atomic-observable fallback.') from exc
                converted.append((amp, xf, yf, record[3], trace))
            clean_records = records = _validate_worker_records(converted, tlist_len=len(run_config.tlist_us),
                expected_count=len(positions), require_contiguous_indices=True)

        def process_records(records, noise=None, trial=0):
            readout = additive = clean_traces = None
            detector_arrays = None
            if noise is not None and noise.readout_enabled:
                from .readout_noise import apply_gaussian_readout
                from ..utils.numerical import extract_if_component
                clean_traces = np.asarray([record[4] for record in records])
                if physical:
                    from .detector_readout import apply_pin_readout
                    measured, additive, readout, detector_arrays = apply_pin_readout(
                        atomic_traces, noise, detector_context, trial=trial, **detector_kwargs,
                        **({'poisson_quantiles':True} if poisson_quantiles else {}))
                else:
                    measured, additive, readout = apply_gaussian_readout(clean_traces, noise, trial=trial)
                if readout['noise_applied']:
                    rebuilt = []
                    for record, trace in zip(records, measured):
                        try:
                            amp, xf, yf = extract_if_component(trace, run_config.tlist_us,
                                request.signed_if_mhz, run_config.transient_delay_us)
                        except PhysicsError:
                            raise
                        except Exception as exc:
                            raise PhysicsCalculationError(
                                f'NOISE-002: measured IF extraction failed for element {record[3]}; no clean fallback.') from exc
                        rebuilt.append((amp, xf, yf, record[3], trace))
                    records = _validate_worker_records(rebuilt, tlist_len=len(run_config.tlist_us),
                        expected_count=len(positions), require_contiguous_indices=True)
            responses = np.asarray([r[0] for r in records], dtype=complex)
            if not if_only and atomic_if_polarity(orientation) < 0:
                responses = np.conj(responses)
            if not if_only and np.count_nonzero(responses) < 2:
                raise PhysicsCalculationError('LAB-005: fewer than two nonzero IF responses; azimuth is unidentifiable.')
            times = np.asarray(run_config.tlist_us)
            traces = [r[4] for r in records]
            if not if_only:
                snapshots = extract_if_snapshots(times, traces, request.signed_if_mhz,
                                                start_us=run_config.transient_delay_us,
                                                cycles_per_snapshot=run_config.cycles_per_snapshot)
                doa, snapshots = estimate_doa_from_results(positions, responses, times, traces,
                                                           method=selected, if_snapshots=snapshots,
                                                           rf_orientation=orientation,
                                                           doa_options=run_config.doa_options)
            output = dict(resolution_mode=mode, run_config=run_config, array_positions=positions,
                          element_phases_rad=phases, worker_records=records, complex_responses=responses,
                          time_traces=traces, tlist=times, true_aoa=None if if_only else run_config.target_azimuth_deg,
                          signal_carrier_hz=request.signal.frequency_hz, signed_if_mhz=request.signed_if_mhz,
                          sensor_detunings_rad_per_us=detunings, xf_0=records[0][1], yf_0=records[0][2])
            if temporal_reports is not None:
                output['temporal_doppler_reports'] = temporal_reports
            if readout is not None:
                clean_responses = np.asarray([r[0] for r in clean_records], dtype=complex)
                if not if_only and atomic_if_polarity(orientation) < 0:
                    clean_responses = np.conj(clean_responses)
                output.update(clean_worker_records=clean_records, clean_time_traces=clean_traces,
                              clean_complex_responses=clean_responses, additive_readout_noise=additive,
                              readout_noise=readout, noise_config=noise)
                if detector_arrays is not None:
                    output.update(detector_arrays=detector_arrays, atomic_worker_records=atomic_records)
            if if_only:
                output.update(measurement_mode='if_only', signal_enabled=receiver.signal_enabled,
                    effective_signal_rabi_rad_per_us=effective_signal_rabi, estimated_aoa=None,
                    angles=None,power=None,if_snapshots=None,doa_result=None)
                output.update(configured_spatial_phases_rad=phases,element_phases_rad=None)
            else:
                merge_doa_into_results(output, doa, snapshots)
            return output, records

        if _measurement_trial is not None:
            output, records = process_records(clean_records, noise, _measurement_trial)
        elif noise is not None and noise.readout_enabled:
            # Clean evolution was certified once. Only the readout is stochastic.
            if not if_only and np.count_nonzero([record[0] for record in clean_records]) < 2:
                raise PhysicsCalculationError('NOISE-002: fewer than two nonzero clean IF responses; noise cannot invent an identifiable signal.')
            clean_output, _ = process_records(clean_records)
            trials = []
            for trial in range(noise.trials):
                try:
                    measured_output, _ = process_records(clean_records, noise, trial)
                    trials.append(measured_output)
                except Exception as exc:
                    if noise.trials == 1:
                        raise
                    raise PhysicsCalculationError(
                        f'NOISE-003: trial {trial} failed; entire campaign aborted without dropping trials.') from exc
            from .noise_campaign import summarize_readout_campaign
            if if_only:
                from .if_only import summarize_if_trials
                summary = summarize_if_trials(trials,noise.trials)
            else:
                summary = summarize_readout_campaign(trials, clean_output, run_config)
            output = dict(trials[0])
            output.update(noise_campaign=summary, noise_trial_results=trials)
            records = output['worker_records']
        else:
            output, records = process_records(records)
        if self._current_explicit_runtime(request, mode) != runtime:
            raise PhysicsConfigurationError('LAB-005: runtime changed during DoA processing; no stale publication.')
        active_radiative_rates(run_config, runtime, acfg, request)
        if run_config.radiative is not None:
            output['radiative_model'] = radiative_report(run_config.radiative)
        output['laboratory_runtime' if mode == 'experimental' else 'prescribed_runtime'] = runtime
        if _publish:
            self._publish_explicit_result(output, request, runtime, mode)
        return output

    def _publish_explicit_result(self, output, request, runtime, mode):
        self._result_measurement_mode = output.get('measurement_mode','aoa')
        self.array_positions, self.element_phases_rad = output['array_positions'], output['element_phases_rad']
        self.results = output['worker_records']
        self._laboratory_result_runtime = runtime if mode == 'experimental' else None
        self._prescribed_result_context = (request, runtime) if mode == 'prescribed' else None

    def process_simulation_results(self, rf_orientation=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Process the raw simulation results.

        RF-DIR-001D: this is the physics -> phase/AoA boundary for the
        per-element complex IF coefficients.  When the RF-edge energy
        orientation conjugates the atomic chain (direction = -1, see
        ``time_domain.atomic_if_polarity``), the coefficients are flipped
        here ONCE so ``complex_responses`` / ``extracted_phases`` always
        carry the PHYSICAL (+Delta_phi) polarity.  The raw atomic value
        stays inspectable in the ``self.results`` worker tuples; the
        snapshot branch is normalized at its own production point in
        ``doa_pipeline.estimate_doa_from_results`` with the same rule.
        Default ``rf_orientation=None`` keeps the historical behaviour.

        Returns:
            Tuple of (complex_responses, xf_0, yf_0, time_traces)
        """
        if self.results is None:
            raise ValueError("No simulation results available. Run simulation first.")
        if getattr(self,'_result_measurement_mode',None) == 'if_only':
            raise PhysicsConfigurationError('NOISE-013: IF-only records cannot be consumed as spatial AoA responses.')

        if self._prescribed_result_context is not None:
            profile, stored = self._prescribed_result_context
            if self._current_explicit_runtime(profile, 'prescribed') != stored:
                raise PhysicsConfigurationError('LAB-007: prescribed runtime changed after sensing; no stale results.')
            if rf_orientation is None:
                rf_orientation = stored.rf_orientation
            elif rf_orientation != stored.rf_orientation:
                raise PhysicsConfigurationError('LAB-007: conflicting prescribed RF orientation.')
        if self._laboratory_result_runtime is not None:
            runtime = acfg.get_laboratory_runtime()
            if runtime != self._laboratory_result_runtime:
                raise PhysicsConfigurationError('LAB-005: these worker records belong to an older laboratory runtime.')
            if rf_orientation is not None and rf_orientation != runtime.selection.rf_orientation:
                raise PhysicsConfigurationError('LAB-005: explicit RF orientation conflicts with these laboratory records.')
            rf_orientation = runtime.selection.rf_orientation

        # F05-WORKER-001: reject malformed manually supplied records before any
        # phase processing.  This standalone method intentionally accepts
        # smaller valid batches, so it does NOT impose config.N_ELEMENTS; it
        # validates structure only (no tlist-length or contiguous-index check,
        # which belong to the dispatched run).  The validated, materialized
        # batch is consumed (assigned back) so a generator input cannot be
        # exhausted by validation and leave the reads below empty.
        self.results = _validate_worker_records(self.results)

        complex_responses = np.array([r[0] for r in self.results])
        xf_0 = self.results[0][1]
        yf_0 = self.results[0][2]
        time_traces = [r[4] for r in self.results]

        if atomic_if_polarity(rf_orientation) < 0.0 and np.iscomplexobj(complex_responses):
            complex_responses = np.conj(complex_responses)

        return complex_responses, xf_0, yf_0, time_traces

    def extract_phases(self, complex_responses: np.ndarray) -> np.ndarray:
        """
        Extract phases from complex responses.

        Args:
            complex_responses: Complex amplitude responses

        Returns:
            Unwrapped phase array
        """
        return np.unwrap(np.angle(complex_responses))

    def estimate_angle_of_arrival(self,
                                complex_responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform AoA estimation using beamforming.

        Args:
            complex_responses: Complex responses from all elements

        Returns:
            Tuple of (angles, power_spectrum, estimated_aoa)
        """
        if self.array_positions is None:
            self.setup_array_geometry()

        scan_angle_list = np.linspace(-90, 90, 181)
        angles, power = analysis.estimate_aoa_beamforming(
            complex_responses,
            self.array_positions,
            params.RF_FREQUENCY,
            scan_angle_list
        )

        estimated_aoa = angles[np.argmax(power)]
        return angles, power, estimated_aoa

    def run_complete_simulation(self) -> dict:
        """
        Run the complete simulation pipeline.

        Returns:
            Dictionary containing all simulation results
        """
        print("Setting up array geometry...")
        array_pos = self.setup_array_geometry()
        phases = self.get_element_phases()

        print(f"Array geometry defined for {params.N_ELEMENTS} elements with {params.SPACING*100:.1f} cm spacing.")
        print(f"Target AoA: {params.TARGET_AOA_DEG} deg")
        print(f"Vapor cell length: {params.CELL_LENGTH*100:.1f} cm")

        # Debug: Print array positions and phases
        print("DEBUG: Array positions (x, y):")
        for i, pos in enumerate(array_pos):
            print(f"  Element {i}: ({pos[0]*100:.1f} cm, {pos[1]*100:.1f} cm)")

        print("DEBUG: Element phases (radians):")
        for i, phase in enumerate(phases):
            print(f"  Element {i}: {phase:.4f} rad ({np.rad2deg(phase):.1f} deg)")

        print("\nRunning steady-state spectrum simulation...")
        dp_scan, trans_no_rf, trans_no_rf_analytic, trans_with_rf, trans_with_rf_analytic = self.run_steady_state_spectrum()

        print("\nRunning time-domain heterodyne simulation...")
        results, tlist = self.run_time_domain_simulation()
        self.results = results  # Update stored results

        print("\nProcessing results...")
        # RF-DIR-001D: the published RF-edge orientation drives the single
        # physics -> phase polarity conversion (vector here, snapshots in
        # the DOA dispatch below - one rule: atomic_if_polarity).
        rf_orientation = acfg.get_rf_orientation()
        complex_responses, xf_0, yf_0, time_traces = self.process_simulation_results(
            rf_orientation=rf_orientation
        )
        extracted_phases = self.extract_phases(complex_responses)

        results = {
            'array_positions': array_pos,
            'element_phases': phases,
            'dp_scan': dp_scan,
            'trans_no_rf': trans_no_rf,
            'trans_no_rf_analytic': trans_no_rf_analytic,
            'trans_with_rf': trans_with_rf,
            'trans_with_rf_analytic': trans_with_rf_analytic,
            'complex_responses': complex_responses,
            'xf_0': xf_0,
            'yf_0': yf_0,
            'time_traces': time_traces,
            'tlist': tlist,
            'extracted_phases': extracted_phases,
            'true_aoa': params.TARGET_AOA_DEG,
            'cell_length': params.CELL_LENGTH,
            'element_spacing': params.SPACING
        }

        # DOA-INTEGRATE-001: single production DOA dispatch.  The estimator
        # is selected by config.doa_config.DOA_METHOD (default
        # "phase_beamforming", preserving the validated legacy behavior
        # exactly), the geometry is built from the ACTUAL path positions,
        # and the complex IF snapshots are extracted EXACTLY ONCE from the
        # existing time-domain output so every estimator — now or in later
        # same-run comparisons — consumes the SAME X matrix.
        print("\nRunning DOA estimation (production dispatcher)...")
        doa_result, if_snapshots = estimate_doa_from_results(
            array_positions=array_pos,
            complex_responses=complex_responses,
            tlist_us=tlist,
            time_traces=time_traces,
            rf_orientation=rf_orientation,
        )
        # Structured DOA contract + legacy keys (estimated_aoa, angles,
        # power).  estimated_aoa is always the selected estimator's
        # azimuth; angles/power carry the spectrum only for spectrum
        # methods (None for search-free estimators — no fabricated data).
        merge_doa_into_results(results, doa_result, if_snapshots)
        estimated_aoa = results['estimated_aoa']

        print(f"\nTrue AoA: {params.TARGET_AOA_DEG:.2f} deg")
        print(f"Estimated AoA ({doa_result.algorithm}): {estimated_aoa:.2f} deg")

        return results
