"""NOISE-015 sampled-grid selection and explicitly separate refinement jobs."""
from dataclasses import dataclass, fields, replace
from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class RefinementGrid:
    execution_policy: str
    values: tuple
    seed: int
    calibration_trials: int
    null_validation_trials: int
    trials_per_point: int

    def __post_init__(self):
        if not isinstance(self.execution_policy,str) or self.execution_policy != 'separate_explicit_run':
            raise PhysicsConfigurationError('NOISE-015: refinement requires separate_explicit_run; no adaptive hidden continuation.')
        if not isinstance(self.values,tuple) or len(self.values) < 2:
            raise PhysicsConfigurationError('NOISE-015: >=2 explicit refinement field points required.')
        values = tuple(_real(v,'NOISE-015 refinement field [V/m]',positive=True) for v in self.values)
        if any(b <= a for a,b in zip(values,values[1:])):
            raise PhysicsConfigurationError('NOISE-015: refinement fields must strictly increase; no sorting/interpolation.')
        object.__setattr__(self,'values',values)
        for key in ('seed','calibration_trials','null_validation_trials','trials_per_point'):
            value = _integer(getattr(self,key),'NOISE-015 '+key,minimum=0 if key=='seed' else 1)
            if value > 2**53-1:
                raise PhysicsConfigurationError('NOISE-015: refinement integers must fit exact JSON integers.')
            object.__setattr__(self,key,value)


@dataclass(frozen=True)
class SensitivityOptions:
    enabled: bool
    selection: str
    refinement: object = None

    def __post_init__(self):
        if (type(self.enabled) is not bool or not isinstance(self.selection,str) or
                self.selection != 'minimum_supported_sampled_field'):
            raise PhysicsConfigurationError('NOISE-015: boolean enabled and minimum_supported_sampled_field selection required.')
        if self.refinement is not None and (not isinstance(self.refinement,RefinementGrid) or not self.enabled):
            raise PhysicsConfigurationError('NOISE-015: typed refinement requires enabled sampled sensitivity analysis.')


def parse_sensitivity(document):
    keys = {f.name for f in fields(SensitivityOptions)}
    if not isinstance(document,dict) or not keys-{'refinement'} <= set(document) <= keys:
        raise PhysicsConfigurationError('NOISE-015: sensitivity requires enabled/selection and optional refinement; unknown fields rejected.')
    data = dict(document)
    if data.get('refinement') is not None:
        refinement = data['refinement']; required = {f.name for f in fields(RefinementGrid)}
        if not isinstance(refinement,dict) or set(refinement) != required or not isinstance(refinement['values'],list):
            raise PhysicsConfigurationError('NOISE-015: refinement requires a full explicit design and JSON field array.')
        data['refinement'] = RefinementGrid(**{**refinement,'values':tuple(refinement['values'])})
    return SensitivityOptions(**data)


def refinement_profile(profile):
    """Pure new-profile construction; no simulation, files or merged inference."""
    from .detection_study import DetectionStudy, detection_jobs, validate_detection_profile
    if not isinstance(getattr(profile,'study',None),DetectionStudy):
        raise PhysicsConfigurationError('NOISE-015: detection profile required for refinement.')
    options = profile.study.sensitivity
    if options is None or not options.enabled or options.refinement is None:
        raise PhysicsConfigurationError('NOISE-015: explicitly enabled refinement design required; no automatic point proposal.')
    r = options.refinement
    if r.values[0] < profile.study.values[0] or r.values[-1] > profile.study.values[-1]:
        raise PhysicsConfigurationError('NOISE-015: declared refinement must lie inside the original sampled field range; no clipping.')
    study = replace(profile.study,values=r.values,seed=r.seed,calibration_trials=r.calibration_trials,
                    null_validation_trials=r.null_validation_trials,trials_per_point=r.trials_per_point,
                    sensitivity=replace(options,refinement=None))
    if set(j['seed'] for j in detection_jobs(study)) & set(j['seed'] for j in detection_jobs(profile.study)):
        raise PhysicsConfigurationError('NOISE-015: refinement reuses an original seed domain; declare an independent seed.')
    refined = replace(profile,study=study)
    validate_detection_profile(refined)
    return refined


def sensitivity_plan(profile):
    """GUI preview describes statistical feasibility, never guarantees success."""
    from .detection_study import plan_detection_study
    d = profile.study
    options = d.sensitivity
    plan = {'enabled': options is not None and options.enabled,
            'selection': 'minimum_supported_sampled_field',
            'refinement_execution': 'not executed automatically',
            'numerical_convergence': 'not certified by binomial intervals or density-matrix gates',
            'continuous_minimum_computed': False, 'nef_computed': False}
    if options is not None and options.refinement is not None:
        refined = refinement_profile(profile)
        p = plan_detection_study(refined)
        plan['separate_refinement'] = {'values':list(refined.study.values),'seed':refined.study.seed,
            'total_trials':p['total_trials'],'total_atomic_evolutions':p['total_atomic_evolutions'],
            'finite_threshold_available':p['finite_threshold_available'],
            'confidence_scope':'new independent study only; no pooled or repeated-study confidence claim'}
    return plan
