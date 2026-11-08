"""NOISE-014 explicit IF detection design and pure GUI planning."""
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from hashlib import sha256
import math
from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class DetectionStudy:
    schema_version: int
    kind: str
    name: str
    values: tuple
    path_index: int
    p_false_alarm: float
    p_detection: float
    confidence_level: float
    calibration_trials: int
    null_validation_trials: int
    trials_per_point: int
    seed: int
    sensitivity: object = None
    numerical_validation: object = None
    calibration_false_alarm_probability: object = None

    def __post_init__(self):
        if self.calibration_false_alarm_probability is not None:
            value = _real(self.calibration_false_alarm_probability, 'NOISE-018 calibration probability', positive=True)
            if value >= 1:
                raise PhysicsConfigurationError('NOISE-018: calibration probability must be below one.')
            object.__setattr__(self,'calibration_false_alarm_probability',value)
        from .sensitivity import SensitivityOptions
        if self.sensitivity is not None and not isinstance(self.sensitivity,SensitivityOptions):
            raise PhysicsConfigurationError('NOISE-015: parsed SensitivityOptions required.')
        from .detection_numerics import DetectionNumerics
        if self.numerical_validation is not None and not isinstance(self.numerical_validation,DetectionNumerics):
            raise PhysicsConfigurationError('NOISE-016: parsed DetectionNumerics required.')
        if (type(self.schema_version) is not int or self.schema_version != 1 or
                not isinstance(self.kind,str) or self.kind != 'if_detection'):
            raise PhysicsConfigurationError('NOISE-014: detection schema 1 and kind if_detection required.')
        if not isinstance(self.name, str) or not self.name.strip():
            raise PhysicsConfigurationError('NOISE-014: nonempty study name required.')
        if not isinstance(self.values, tuple) or len(self.values) < 2:
            raise PhysicsConfigurationError('NOISE-014: >=2 explicit positive field points required.')
        values = tuple(_real(v, 'NOISE-014 field [V/m]', positive=True) for v in self.values)
        if any(b <= a for a, b in zip(values, values[1:])):
            raise PhysicsConfigurationError('NOISE-014: field values must strictly increase; no sorting or refinement.')
        object.__setattr__(self, 'values', values)
        for key in ('p_false_alarm', 'p_detection', 'confidence_level'):
            value = _real(getattr(self, key), f'NOISE-014 {key}', positive=True)
            if value >= 1:
                raise PhysicsConfigurationError(f'NOISE-014: {key} must lie strictly between zero and one.')
            object.__setattr__(self, key, value)
        for key in ('path_index', 'seed', 'calibration_trials', 'null_validation_trials', 'trials_per_point'):
            value = _integer(getattr(self, key), f'NOISE-014 {key}', minimum=0 if key in ('path_index','seed') else 1)
            if value > 2**53-1:
                raise PhysicsConfigurationError('NOISE-014: counts/seed/index must fit exact JSON integers.')
            object.__setattr__(self, key, value)

    def threshold_rank(self):
        # Preserve the declared decimal probability at an integer boundary.
        probability = self.p_false_alarm if self.calibration_false_alarm_probability is None else self.calibration_false_alarm_probability
        return math.ceil((self.calibration_trials+1)*(1-Fraction(str(probability))))


def parse_detection_study(document):
    keys = {f.name for f in fields(DetectionStudy)}
    if (not isinstance(document,dict) or not keys-{'sensitivity','numerical_validation','calibration_false_alarm_probability'} <= set(document) <= keys or
            not isinstance(document['values'],list)):
        raise PhysicsConfigurationError(f'NOISE-014: detection study requires exactly {sorted(keys)} and a values array.')
    data = {**document,'values':tuple(document['values'])}
    if data.get('sensitivity') is not None:
        from .sensitivity import parse_sensitivity
        data['sensitivity'] = parse_sensitivity(data['sensitivity'])
    if data.get('numerical_validation') is not None:
        from .detection_numerics import parse_detection_numerics
        data['numerical_validation']=parse_detection_numerics(data['numerical_validation'])
    return DetectionStudy(**data)


def detection_jobs(definition):
    """Independent seed domains; no paired null/alternative innovations."""
    jobs = [('calibration', None, definition.calibration_trials),
            ('null_validation', None, definition.null_validation_trials)]
    jobs += [(f'point_{i:04d}', value, definition.trials_per_point) for i, value in enumerate(definition.values)]
    seeds = [int.from_bytes(sha256(f'NOISE-014-v1:{definition.seed}:{role}'.encode('ascii')).digest()[:8],
                            'big') & (2**53-1) for role, _, _ in jobs]
    if len(set(seeds)) != len(seeds):
        raise PhysicsConfigurationError('NOISE-014: independent seed collision; no replacement seed.')
    return tuple({'role': role, 'field_v_per_m': value, 'trials': count, 'seed': seed}
                 for (role, value, count), seed in zip(jobs, seeds))


def detection_job_profile(profile, job):
    from .measurement import MeasurementOptions
    def noise_for_job(noise):
        return replace(noise, seed=job['seed'], trials=job['trials'],
                       readout=noise_for_job(noise.readout) if noise.readout is not None else None)
    on = job['field_v_per_m'] is not None
    signal = replace(profile.signal, electric_field_vm=job['field_v_per_m']) if on else profile.signal
    request = replace(profile.request, signal=signal) if profile.mode == 'experimental' else profile.request
    return replace(profile, study=None, signal=signal, request=request,
                   measurement=MeasurementOptions('if_only', on), noise=noise_for_job(profile.noise))


def validate_detection_profile(profile):
    from .run_profile import RunProfile
    from .measurement import validate_measurement_profile
    if (not isinstance(profile, RunProfile) or not isinstance(profile.study, DetectionStudy) or
            profile.run_config is None or profile.measurement is None or profile.measurement.mode != 'if_only' or
            profile.noise is None or not profile.noise.enabled or not profile.noise.readout_enabled and not profile.noise.drive_selected):
        raise PhysicsConfigurationError('NOISE-014: explicit IF-only temporal profile and enabled trial noise execution required.')
    if profile.study.path_index >= len(profile.run_config.positions_m):
        raise PhysicsConfigurationError('NOISE-014: selected path outside declared receiver; no best-path selection.')
    if set(profile.noise.active_sources) & {'rf_signal_amplitude','rf_relative_phase'}:
        raise PhysicsConfigurationError('NOISE-014: signal amplitude/relative-phase noise must be explicitly off for this shared null/alternative design.')
    from .rf_noise import validate_rf_noise_execution
    from .detector import validate_detector_sampling
    for job in detection_jobs(profile.study):
        point = detection_job_profile(profile, job)
        validate_measurement_profile(point)
        point.run_config.validate_sampling(point.signed_if_mhz)
        validate_rf_noise_execution(point.noise, point.run_config)
        validate_detector_sampling(point.noise, point.run_config)
        if point.noise.physical_readout_enabled and point.environment is None:
            raise PhysicsConfigurationError('NOISE-014: PIN requires an explicit optical environment.')
        if point.mode == 'experimental' and any(getattr(point.request, key) != getattr(point, key)
                                               for key in ('probe','coupling','lo','signal')):
            raise PhysicsConfigurationError('NOISE-014: applied fields differ from the experimental request.')
    if profile.study.sensitivity is not None and profile.study.sensitivity.refinement is not None:
        from .sensitivity import refinement_profile
        refinement_profile(profile)
    if profile.study.numerical_validation is not None and profile.study.numerical_validation.enabled:
        if profile.study.sensitivity is None or not profile.study.sensitivity.enabled:
            raise PhysicsConfigurationError('NOISE-016: enable sampled sensitivity for numerical validation.')
        from .detection_numerics import validate_grid_subdivision
        for case in profile.study.numerical_validation.cases:
            validate_grid_subdivision(profile.run_config.tlist_us,case.output_subdivision)


def plan_detection_study(profile):
    validate_detection_profile(profile)
    jobs = detection_jobs(profile.study)
    paths = len(profile.run_config.positions_m)
    alpha = (1-profile.study.confidence_level)/(len(profile.study.values)+1)
    result = {'schema_version': 1, 'jobs': list(jobs), 'paths': paths,
            'samples_per_trace': len(profile.run_config.tlist_us),
            'total_trials': sum(j['trials'] for j in jobs),
            'total_atomic_evolutions': paths*sum(1+j['trials'] if profile.noise.stochastic_drive else 1 for j in jobs),
            'threshold_rank': profile.study.threshold_rank(),
            'finite_threshold_available': profile.study.threshold_rank() <= profile.study.calibration_trials,
            'confidence_scope': 'simultaneous P_FA and all P_D points, Bonferroni exact binomial intervals',
            'per_interval_alpha': alpha,
            'zero_false_alarm_upper_bound': -math.expm1(math.log(alpha/2)/profile.study.null_validation_trials),
            'all_detections_lower_bound': math.exp(math.log(alpha/2)/profile.study.trials_per_point),
            'cost_note': 'full canonical campaigns; no automatic sample reduction or stopping'}
    result['best_case_false_alarm_target_feasible'] = result['zero_false_alarm_upper_bound'] <= profile.study.p_false_alarm
    result['best_case_detection_target_feasible'] = result['all_detections_lower_bound'] >= profile.study.p_detection
    if profile.study.sensitivity is not None:
        from .sensitivity import sensitivity_plan
        result['sensitivity'] = sensitivity_plan(profile)
    if profile.study.numerical_validation is not None:
        from .detection_numerics import numerical_plan
        result['numerical_validation']=numerical_plan(profile)
    return result
