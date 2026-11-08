"""NOISE-014 independent null calibration/validation and IF detection sweep."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import beta
from ..config.detection_study import DetectionStudy, detection_jobs, detection_job_profile, validate_detection_profile
from ..config.laboratory import _real, _integer
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError


def _scores(values, expected):
    try:
        expected = _integer(expected, 'NOISE-014 expected trials', minimum=1)
        a = np.asarray(values)
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-014: invalid trial count or score-array structure.') from exc
    if (a.ndim != 1 or len(a) != expected or a.dtype.kind not in 'iuf' or
            not np.isfinite(a).all() or np.any(a < 0)):
        raise PhysicsCalculationError('NOISE-014: complete finite nonnegative real trial scores required; no clipping or survivor analysis.')
    return a


def _real_calculation(value, name, **bounds):
    try:
        return _real(value, 'NOISE-014 '+name, **bounds)
    except PhysicsConfigurationError as exc:
        raise PhysicsCalculationError(f'NOISE-014: invalid {name}.') from exc


def calibrate_threshold(definition, scores):
    if not isinstance(definition, DetectionStudy):
        raise PhysicsCalculationError('NOISE-014: parsed DetectionStudy required.')
    scores = _scores(scores, definition.calibration_trials)
    rank = definition.threshold_rank()
    available = rank <= len(scores)
    result = {'status': 'finite_order_statistic' if available else 'insufficient_calibration_trials',
            'value': float(np.sort(scores)[rank-1]) if available else None,
            'order_statistic_rank_one_based': rank, 'trials': len(scores),
            'target_false_alarm_probability': definition.p_false_alarm,
            'comparison': 'strict score > threshold; ties are nondetections',
            'method': 'ceil((n+1)*(1-P_FA)) order statistic; no interpolation',
            'interpretation': 'marginal rank bound under exchangeable null trials; not a conditional guarantee for this realized threshold',
            'calibration_scores': scores.tolist()}
    if definition.calibration_false_alarm_probability is not None:
        result['calibration_false_alarm_probability'] = definition.calibration_false_alarm_probability
        result['method'] = 'ceil((n+1)*(1-P_FA_calibration)) order statistic; no interpolation'
    return result


def binomial_interval(successes, trials, alpha):
    """Two-sided Clopper-Pearson interval; alpha is this interval's error rate."""
    try:
        n = _integer(trials, 'NOISE-014 binomial trials', minimum=1)
        k = _integer(successes, 'NOISE-014 binomial count')
        alpha = _real(alpha, 'NOISE-014 interval alpha', positive=True)
        if k > n or alpha >= 1:
            raise PhysicsCalculationError('NOISE-014: invalid binomial interval arguments.')
        lo = 0. if k == 0 else float(beta.ppf(alpha/2, k, n-k+1))
        hi = 1. if k == n else float(beta.isf(alpha/2, k+1, n-k))
        if not np.isfinite([lo,hi]).all() or not 0 <= lo <= hi <= 1:
            raise PhysicsCalculationError('NOISE-014: invalid exact-binomial interval; no substitute interval.')
        return [lo, hi]
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-014: binomial interval calculation failed.') from exc


def evaluate_detection(scores, expected, threshold, target, alpha, *, null=False):
    scores = _scores(scores, expected)
    expected = len(scores)
    target = _real_calculation(target, 'target probability', positive=True)
    alpha = _real_calculation(alpha, 'interval alpha', positive=True)
    if target >= 1 or alpha >= 1 or type(null) is not bool:
        raise PhysicsCalculationError('NOISE-014: invalid probability/comparison design.')
    if threshold is None:
        return {'trials': expected, 'scores': scores.tolist(), 'detections': None,
                'count': None, 'probability': None, 'confidence_interval': None,
                'status': 'undefined_threshold'}
    threshold = _real_calculation(threshold, 'threshold', nonnegative=True)
    mask = scores > threshold
    k = int(np.count_nonzero(mask)); lo, hi = binomial_interval(k, expected, alpha)
    supported = hi <= target if null else lo >= target
    contradicted = lo > target if null else hi < target
    return {'trials': expected, 'scores': scores.tolist(), 'detections': mask.tolist(),
            'count': k, 'probability': k/expected, 'confidence_interval': [lo,hi],
            'target': target, 'per_interval_confidence': 1-alpha,
            'status': 'target_supported' if supported else 'target_not_met' if contradicted else 'inconclusive'}


def _evidence(profile, report, result, definition):
    from .if_only import summarize_if_result
    summary = summarize_if_result(result)
    if (result['measurement_mode'] != 'if_only' or result['signal_enabled'] is not profile.measurement.signal_enabled or
            report['measurement'] != asdict(profile.measurement) or
            report['configured_fields']['signal'] != asdict(profile.signal) or
            report['noise_declaration'] != asdict(profile.noise) or
            result.get('noise_config') != profile.noise or
            summary['trials_completed'] != profile.noise.trials):
        raise PhysicsCalculationError('NOISE-014: actual receiver/switch/field/noise/trial evidence differs from the planned job.')
    actual_field = profile.signal.electric_field_vm if profile.measurement.signal_enabled else 0.
    if report['applied_fields']['signal']['electric_field_vm'] != actual_field:
        raise PhysicsCalculationError('NOISE-014: effective signal field differs from the declared switch.')
    values = [t['squared_if_amplitude_per_path'][definition.path_index] for t in summary['trials']]
    return {'scores': _scores(values, profile.noise.trials).tolist(), 'trace_units': summary['trace_units'],
            'statistic_units': summary['power_units'], 'path_index': definition.path_index,
            'fit': summary['fit'], 'response_convention': summary['complex_response_convention']}


def _fixed_conditions(reference, report, first_evidence, evidence):
    from .measurement_audit import check_fixed_readout
    check_fixed_readout(reference, report)
    for key in ('levels','rf_orientation','detunings_rad_per_us','decay_rates_per_us',
                'resolved_rate_budget_per_us','signed_if_mhz','quantization_axis'):
        if reference[key] != report[key]:
            raise PhysicsCalculationError(f'NOISE-014: fixed physics {key} changed; no automatic retuning/channel switching.')
    for key in ('trace_units','statistic_units','path_index','fit','response_convention'):
        if first_evidence[key] != evidence[key]:
            raise PhysicsCalculationError(f'NOISE-014: measurement {key} changed between null and signal jobs.')
    a, b = reference['configured_rabi_rad_per_us'], report['configured_rabi_rad_per_us']
    ratio = report['configured_fields']['signal']['electric_field_vm']/reference['configured_fields']['signal']['electric_field_vm']
    if a[:3] != b[:3] or not np.isclose(b[3], a[3]*ratio, rtol=1e-12, atol=0):
        raise PhysicsCalculationError('NOISE-014: calibrated drive conditions changed beyond the declared signal-field axis.')
    for key in ('probe','coupling','lo'):
        if reference['applied_fields'][key] != report['applied_fields'][key]:
            raise PhysicsCalculationError('NOISE-014: fixed applied optical/LO fields changed.')
    if reference['sensing']['execution'] != report['sensing']['execution']:
        raise PhysicsCalculationError('NOISE-014: geometry, transient or sampling conditions changed.')


@dataclass(frozen=True)
class DetectionStudyResult:
    status: str
    definition: DetectionStudy
    base_profile: object
    calibration: object
    validation: object
    points: tuple
    threshold: object
    # Kept for the common CLI representative-figure adapter.
    nominal_result: object = None

    def report(self):
        d = self.definition
        def record(p):
            if p is None:
                return None
            return {'index': p.index, 'value': p.value, 'effective_seed': p.profile.noise.seed,
                    'effective_trials': p.profile.noise.trials, 'summary': p.summary, 'run_report': p.report}
        report = {'schema_version': 1, 'study': asdict(d), 'status': self.status,
                'execution_scope':'single_detection_campaign',
                'points_requested': len(d.values), 'points_completed': len(self.points), 'axis_unit': 'V/m',
                'statistic': 'squared raw IF peak magnitude on the predeclared path; no phase or angle inference',
                'threshold': self.threshold, 'calibration': record(self.calibration),
                'null_validation': record(self.validation), 'points': [record(p) for p in self.points],
                'confidence': {'family_level': d.confidence_level, 'method': 'Clopper-Pearson with Bonferroni',
                               'comparisons': len(d.values)+1, 'scope': 'P_FA and all predeclared P_D points, conditional on the realized calibration threshold',
                               'assumption': 'independent IID trials within each campaign; time samples are not trial replicates'},
                'seed_derivation': 'SHA256 NOISE-014-v1:seed:role first 64 bits masked to 53; collisions rejected',
                'independence': 'distinct calibration, null-validation and field-point seed domains; no paired innovations',
                'analysis_status': 'complete_declared_grid' if self.status == 'completed' else 'incomplete_no_final_detection_claim',
                'threshold_frozen': self.threshold is not None and self.threshold['value'] is not None,
                'hardware_calibration_verified': False, 'mdf_nef_computed': False,
                'interpretation': 'sampled probabilities only; no monotonicity enforcement, interpolation, minimum-field selection or NEF'}
        if d.sensitivity is not None:
            if d.sensitivity.enabled:
                from .sampled_sensitivity import analyze_sampled_sensitivity
                report['sampled_sensitivity'] = analyze_sampled_sensitivity(report)
            else:
                report['sampled_sensitivity'] = {'status':'disabled','minimum_supported_field_v_per_m':None}
        return report


def execute_detection_study(profile, *, on_progress=None, should_cancel=None, solver_diagnostic=None, poisson_quantiles=False):
    from . import study_execution as engine
    validate_detection_profile(profile)
    for callback in (on_progress, should_cancel):
        if callback is not None and not callable(callback):
            raise PhysicsConfigurationError('NOISE-014: callable progress/cancellation hooks required.')
    d = profile.study; jobs = detection_jobs(d); completed = []; threshold = None
    alpha = (1-d.confidence_level)/(len(d.values)+1)
    def outcome(status):
        return DetectionStudyResult(status,d,profile, completed[0] if completed else None,
                                    completed[1] if len(completed)>1 else None, tuple(completed[2:]), threshold)
    def emit(state, i=None):
        if on_progress is not None:
            on_progress(engine.StudyProgress(state,len(completed),len(jobs),i,
                jobs[i]['field_v_per_m'] if i is not None else None,
                completed[-1] if state == 'point_completed' else None))
    for i, job in enumerate(jobs):
        if should_cancel is not None:
            answer = should_cancel()
            if type(answer) is not bool:
                raise PhysicsConfigurationError('NOISE-014: cancellation predicate must return boolean.')
            if answer:
                emit('cancelled',i)
                return outcome('cancelled')
        point = detection_job_profile(profile,job)
        emit('point_started',i)
        try:
            if solver_diagnostic is None and not poisson_quantiles:
                report, result = engine._execute_isolated(point)
            else:
                from .detection_validation import isolated_diagnostic
                report,result=isolated_diagnostic(point,solver_diagnostic,poisson_quantiles)
            evidence = _evidence(point,report,result,d)
            if completed:
                _fixed_conditions(completed[0].report,report,completed[0].summary['evidence'],evidence)
            if i == 0:
                threshold = calibrate_threshold(d,evidence['scores'])
                summary = {'role': job['role'], 'evidence': evidence, 'threshold': threshold}
            else:
                probability = evaluate_detection(evidence['scores'],job['trials'],threshold['value'],
                    d.p_false_alarm if i == 1 else d.p_detection,alpha,null=i==1)
                summary = {'role': job['role'], 'evidence': evidence, 'probability': probability}
                if i > 1:
                    null_status = completed[1].summary['probability']['status']
                    summary['joint_target_status'] = ('supported_at_sampled_field' if null_status == 'target_supported' and
                        probability['status'] == 'target_supported' else 'not_established')
        except Exception as exc:
            emit('failed',i)
            raise PhysicsCalculationError(f'NOISE-014: {job["role"]} failed; study aborted without retry, replacement or dropped trials.') from exc
        completed.append(engine.StudyPointResult(i-2,job['field_v_per_m'],point,report,result,summary))
        emit('point_completed',i)
    emit('completed')
    return outcome('completed')
