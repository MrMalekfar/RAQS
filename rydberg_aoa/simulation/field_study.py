"""NOISE-012 response analysis and canonical isolated field-study execution."""
from dataclasses import dataclass, asdict
import numpy as np
from ..config.field_study import SignalFieldStudy, validate_field_study_profile, field_point_profile
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError


def analyze_field_response(definition, fields_v_per_m, amplitudes, response_unit):
    """Analyze the complete base+midpoint grid, retaining every measured point.

    Fits affine response on declared BASE points in the declared region;
    checks deviation at ALL sampled points inside it. Derivatives use actual
    nonuniform coordinates, quadratic three-point formulas, one-sided ends.
    One midpoint refinement is a spacing diagnostic, not a rigorous error bound.
    """
    try:
        x, y = np.asarray(fields_v_per_m), np.asarray(amplitudes)
        if (not isinstance(definition, SignalFieldStudy) or x.ndim != 1 or y.ndim != 2 or
                y.shape[0] != len(x) or not y.shape[1] or
                any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in (x,y)) or
                np.any(y < 0) or not np.array_equal(x, definition.refined_values())):
            raise PhysicsCalculationError('NOISE-012: complete finite real field-by-path amplitude grid required.')
        if response_unit not in ('Im<sigma21>', 'A', 'V'):
            raise PhysicsCalculationError('NOISE-012: explicit observable/A/V response units required.')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            xs = float(x[-1]); z = x/xs
            scale = np.max(y, axis=0)
            ys = np.where(scale == 0, 1., scale)
            normalized = y/ys
            coarse = np.gradient(normalized[::2], z[::2], axis=0, edge_order=2)
            fine = np.gradient(normalized, z, axis=0, edge_order=2)
            coarse_si, fine_si = coarse*(ys/xs), fine*(ys/xs)
            delta = np.abs(fine[::2]-coarse)
            delta_si = delta*(ys/xs)
            relative, derivative_status = [], []
            for path in range(y.shape[1]):
                denominator = float(np.max(np.abs(fine[::2, path])))
                value = None if denominator == 0 else float(np.max(delta[:, path])/denominator)
                relative.append(value)
                derivative_status.append('undefined_zero_reference_derivative' if value is None else
                    'passed' if value <= definition.derivative_max_relative_difference else 'failed')
            lo, hi = definition.linear_region_v_per_m
            region = (x >= lo) & (x <= hi)
            base_region = region[::2]
            center = lo+(hi-lo)*.5
            halfspan = (hi-lo)*.5
            u = (x-center)/halfspan
            design = np.column_stack((np.ones(np.count_nonzero(base_region)), u[::2][base_region]))
            coefficients, _, rank, _ = np.linalg.lstsq(design, normalized[::2][base_region], rcond=None)
            if rank != 2:
                raise PhysicsCalculationError('NOISE-012: affine fit is rank deficient; no through-origin fallback.')
            prediction_normalized = coefficients[0]+u[:, None]*coefficients[1]
            predictions = prediction_normalized*ys
            slope = coefficients[1]*(ys/halfspan)
            intercept = coefficients[0]*ys-slope*center
            deviations, maxima, linear_status = [], [], []
            for path in range(y.shape[1]):
                values = [None if fitted == 0 else float(abs(actual-fitted)/abs(fitted))
                          for actual, fitted in zip(normalized[:, path], prediction_normalized[:, path])]
                selected = [v for v, inside in zip(values, region) if inside]
                maximum = None if any(v is None for v in selected) else max(selected)
                deviations.append(values); maxima.append(maximum)
                linear_status.append('undefined_zero_fitted_amplitude' if maximum is None else
                    'passed' if maximum <= definition.linearity_max_relative_error else 'failed')
        numerical = (coarse_si, fine_si, delta_si, predictions, slope, intercept)
        if (any(not np.isfinite(v).all() for v in numerical) or
                any(v is not None and not np.isfinite(v) for v in relative+maxima) or
                any(v is not None and not np.isfinite(v) for row in deviations for v in row)):
            raise PhysicsCalculationError('NOISE-012: nonfinite response diagnostic; no clipping or point removal.')
        return {'response_unit': response_unit, 'responsivity_unit': response_unit+'/(V/m)',
                'fields_v_per_m': x.tolist(), 'amplitude_per_path': y.tolist(),
                'base_indices': list(range(0, len(x), 2)),
                'responsivity_base_per_path': coarse_si.tolist(), 'responsivity_refined_per_path': fine_si.tolist(),
                'derivative_method': 'three-point nonuniform gradient, edge_order=2; one-sided boundary estimates',
                'derivative_comparison': {'absolute_difference_at_base_per_path': delta_si.tolist(),
                    'max_absolute_difference_per_path': np.max(delta_si, axis=0).tolist(),
                    'max_difference_over_max_refined_derivative_per_path': relative,
                    'relative_budget': definition.derivative_max_relative_difference,
                    'status_per_path': derivative_status,
                    'interpretation': 'single sampled midpoint refinement; not solver convergence or certified derivative error'},
                'linearity': {'region_v_per_m': list(definition.linear_region_v_per_m),
                    'fit_base_indices': [i for i in range(0, len(x), 2) if region[i]],
                    'checked_indices': np.flatnonzero(region).tolist(), 'model': 'amplitude = slope*E + intercept',
                    'slope_per_path': slope.tolist(), 'intercept_per_path': intercept.tolist(),
                    'predicted_amplitude_per_path': predictions.tolist(),
                    'relative_deviation_per_path': np.asarray(deviations, dtype=object).T.tolist(),
                    'max_relative_deviation_in_region_per_path': maxima,
                    'relative_budget': definition.linearity_max_relative_error, 'status_per_path': linear_status,
                    'policy': 'region and budgets declared before execution; no extension, trimming or forced-zero intercept'},
                'hardware_calibration_verified': False, 'mdf_nef_computed': False}
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-012: field-response analysis failed; no substitute derivative.') from exc


def _point_evidence(profile, report, result):
    """Read canonical IF amplitudes and check that exactly this field was run."""
    from .if_measurement import relative_phase_errors
    from ..config.provenance import provenance_report
    try:
        if (report['applied_fields'] != {k: asdict(getattr(profile,k)) for k in ('probe','coupling','lo','signal')} or
                report['mode'] != profile.mode or report['noise_included'] is not False or
                report['signed_if_mhz'] != profile.signed_if_mhz or
                report['input_provenance'] != provenance_report(profile.provenance)):
            raise PhysicsCalculationError('NOISE-012: run fields/noise/mode do not match point declarations.')
        a = np.asarray(result['complex_responses'])
        rabi = np.asarray(report['rabi_rad_per_us'])
        if rabi.shape != (4,) or rabi.dtype.kind not in 'iuf' or not np.isfinite(rabi).all() or np.any(rabi <= 0):
            raise PhysicsCalculationError('NOISE-012: initialized driven Rabi values must be positive finite real scalars.')
        if (a.ndim != 1 or len(a) != len(profile.run_config.positions_m) or a.dtype.kind not in 'iufc' or
                not np.isfinite(a).all() or not np.array_equal(result['tlist'], profile.run_config.tlist_us)):
            raise PhysicsCalculationError('NOISE-012: finite canonical IF responses on the declared grid required.')
        if 'noise_trial_results' in result or 'rf_drive_trial_results' in result:
            rows = result.get('rf_drive_trial_results', result.get('noise_trial_results'))
            if len(rows) != 1:
                raise PhysicsCalculationError('NOISE-012: expected one deterministic response, not a stochastic average.')
        pin = profile.noise is not None and profile.noise.physical_readout_enabled
        unit = profile.noise.detector.output_units if pin else 'Im<sigma21>'
        if pin and result['readout_noise']['units'] != unit:
            raise PhysicsCalculationError('NOISE-012: PIN trace units do not match declared detector.')
        amplitudes = np.abs(a)
        if not np.isfinite(amplitudes).all():
            raise PhysicsCalculationError('NOISE-012: IF magnitude overflow.')
        relative = relative_phase_errors(a, result['element_phases_rad'], 0)
        return {'response_unit': unit, 'if_amplitude_per_path': amplitudes.tolist(),
                'if_real_per_path': a.real.tolist(), 'if_imag_per_path': a.imag.tolist(),
                'relative_phase_reference_0': relative,
                'signal_field_v_per_m': profile.signal.electric_field_vm}
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-012: canonical point evidence is missing or invalid.') from exc


def _check_fixed_physics(first, report, first_field, field):
    # Report values are canonical initialized values, not recalculated physics.
    keys = ('levels', 'rf_orientation', 'detunings_rad_per_us', 'decay_rates_per_us',
            'resolved_rate_budget_per_us', 'signed_if_mhz', 'quantization_axis', 'sensing')
    for key in keys:
        if key == 'sensing':
            a, b = dict(first[key]), dict(report[key])
            for varied in ('estimated_aoa_deg',):
                a.pop(varied); b.pop(varied)
            if a != b:
                raise PhysicsCalculationError('NOISE-012: sensing geometry/settings/grid changed between fields.')
        elif first[key] != report[key]:
            raise PhysicsCalculationError(f'NOISE-012: fixed physics {key} changed between fields; no channel switching.')
    a, b = first['rabi_rad_per_us'], report['rabi_rad_per_us']
    if a[:3] != b[:3] or not np.isclose(b[3], a[3]*(field/first_field), rtol=1e-12, atol=0):
        raise PhysicsCalculationError('NOISE-012: signal Rabi does not scale with declared field or other drives changed.')


@dataclass(frozen=True)
class FieldStudyResult:
    status: str
    definition: SignalFieldStudy
    points: tuple
    base_profile: object
    analysis: object
    # Shared output adapter sees no extra nominal job, because every point is a
    # deterministic measurement at its declared field, including the first.
    nominal_profile: object = None
    nominal_report: object = None
    nominal_result: object = None

    def report(self):
        return {'schema_version': 1, 'study': asdict(self.definition), 'status': self.status,
                'points_requested': len(self.definition.refined_values()), 'points_completed': len(self.points),
                'axis_unit': 'V/m', 'analysis': self.analysis,
                'analysis_status': 'complete_sampled_diagnostics' if self.analysis is not None else 'incomplete_grid',
                'noise_policy': 'all stochastic sources explicitly off; deterministic PIN retained if enabled',
                'isolation': 'fresh canonical spawn process per field; no shared atomic runtime',
                'points': [{'index': p.index, 'value': p.value, 'role': 'base' if p.index%2 == 0 else 'midpoint',
                            'summary': p.summary, 'run_report': p.report} for p in self.points]}


def execute_field_study(profile, *, on_progress=None, should_cancel=None):
    from . import study_execution as engine
    validate_field_study_profile(profile)
    for callback in (on_progress, should_cancel):
        if callback is not None and not callable(callback):
            raise PhysicsConfigurationError('NOISE-012: progress/cancellation callbacks must be callable.')
    grid = profile.study.refined_values(); points = []
    def emit(state, index=None):
        if on_progress is not None:
            on_progress(engine.StudyProgress(state, len(points), len(grid), index,
                None if index is None else grid[index], points[-1] if state == 'point_completed' else None))
    for i, field in enumerate(grid):
        if should_cancel is not None:
            answer = should_cancel()
            if type(answer) is not bool:
                raise PhysicsConfigurationError('NOISE-012: cancellation predicate must return boolean.')
            if answer:
                emit('cancelled')
                return FieldStudyResult('cancelled', profile.study, tuple(points), profile, None)
        point = field_point_profile(profile, i)
        emit('point_started', i)
        try:
            report, result = engine._execute_isolated(point)
            summary = _point_evidence(point, report, result)
            if points:
                _check_fixed_physics(points[0].report, report, points[0].value, field)
                if summary['response_unit'] != points[0].summary['response_unit']:
                    raise PhysicsCalculationError('NOISE-012: response units changed between fields.')
        except Exception as exc:
            emit('failed', i)
            raise PhysicsCalculationError(f'NOISE-012: field point {i}, E={field!r} V/m failed; no dropped point or reused response.') from exc
        points.append(engine.StudyPointResult(i, field, point, report, result, summary))
        emit('point_completed', i)
    try:
        analysis = analyze_field_response(profile.study, grid,
            [p.summary['if_amplitude_per_path'] for p in points], points[0].summary['response_unit'])
    except Exception:
        emit('failed')
        raise
    emit('completed')
    return FieldStudyResult('completed', profile.study, tuple(points), profile, analysis)
