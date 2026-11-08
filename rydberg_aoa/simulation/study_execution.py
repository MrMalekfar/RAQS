"""Reusable study API; execution/progress are independent of CLI and plotting."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import multiprocessing
from ..config.study import validate_study_profile, effective_point_profile, axis_spec
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from .study_metrics import summarize_study_point


@dataclass(frozen=True)
class StudyProgress:
    state: str
    completed_points: int
    total_points: int
    point_index: object = None
    axis_value: object = None
    point_result: object = None


@dataclass(frozen=True)
class StudyPointResult:
    index: int
    value: float
    profile: object
    report: dict
    result: dict
    summary: dict


@dataclass(frozen=True)
class StudyResult:
    status: str
    definition: object
    nominal_profile: object
    nominal_report: object
    nominal_result: object
    points: tuple
    base_profile: object

    def report(self):
        report = {'schema_version': 1, 'study': asdict(self.definition), 'status': self.status,
                'points_requested': len(self.definition.values), 'points_completed': len(self.points),
                'axis_unit': axis_spec(self.definition.source,self.definition.parameter)[0],
                'random_policy': self.definition.random_policy,
                'seed_derivation': 'paired Gaussian/OU innovations: same seed; independent SHA256 NOISE-008-v1:seed:index first 64 bits masked to 53 bits, collisions rejected',
                'isolation': 'fresh non-daemon spawn process per nominal/point; no shared atomic runtime',
                'cancellation': 'between point jobs; active solver completes before cancellation is observed',
                'nominal': self.nominal_report,
                'points': [{'index':p.index, 'value':p.value, 'effective_seed':p.profile.noise.seed,
                            'effective_trials':p.profile.noise.trials, 'summary':p.summary,
                            'run_report':p.report} for p in self.points]}
        if self.definition.statistics is not None:
            from .study_statistics import analyze_study_statistics
            report['uncertainty'] = analyze_study_statistics(report)
        return report


def _execute_point(profile):
    from .profile_execution import execute_profile
    return execute_profile(profile)


def _execute_isolated(profile):
    # ProcessPoolExecutor workers are non-daemon: the canonical sensing worker
    # can itself use its existing spawn Pool. One executor per job guarantees
    # a fresh interpreter, including after a successful previous initialization.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
        return executor.submit(_execute_point, profile).result()


def execute_study(profile, *, on_progress=None, should_cancel=None):
    """Run validated user study, return structured data; no files/GUI side effects.

    Call from a GUI background thread. Callbacks run in that calling thread;
    the frontend owns UI-thread scheduling. Cancellation is between jobs.
    Errors abort with point context/cause; no successful partial aggregate.
    """
    from ..config.magnetic_core import require_four_state_profile
    require_four_state_profile(profile)
    from ..config.operating_study import OperatingStudy
    if isinstance(getattr(profile,'study',None),OperatingStudy):
        from .operating_study import execute_operating_study
        return execute_operating_study(profile,on_progress=on_progress,should_cancel=should_cancel)
    from ..config.detection_study import DetectionStudy
    if isinstance(getattr(profile,'study',None),DetectionStudy):
        options=profile.study.numerical_validation
        if options is not None and options.enabled:
            from .detection_validation import execute_detection_validation
            return execute_detection_validation(profile,on_progress=on_progress,should_cancel=should_cancel)
        from .detection_study import execute_detection_study
        return execute_detection_study(profile,on_progress=on_progress,should_cancel=should_cancel)
    from ..config.field_study import SignalFieldStudy
    if isinstance(getattr(profile,'study',None), SignalFieldStudy):
        from .field_study import execute_field_study
        return execute_field_study(profile,on_progress=on_progress,should_cancel=should_cancel)
    from ..config.noise_convergence import NoiseConvergenceStudy
    if isinstance(getattr(profile,'study',None),NoiseConvergenceStudy):
        from .noise_convergence import execute_convergence_study
        return execute_convergence_study(profile,on_progress=on_progress,should_cancel=should_cancel)
    validate_study_profile(profile)
    for callback in (on_progress, should_cancel):
        if callback is not None and not callable(callback):
            raise PhysicsConfigurationError('NOISE-008: progress/cancellation callbacks must be callable.')
    def cancelled():
        if should_cancel is None:
            return False
        answer = should_cancel()
        if type(answer) is not bool:
            raise PhysicsConfigurationError('NOISE-008: cancellation predicate must return boolean.')
        return answer
    def emit(state, points, index=None):
        if on_progress is not None:
            on_progress(StudyProgress(state, len(points), len(profile.study.values), index,
                                      profile.study.values[index] if index is not None else None,
                                      points[-1] if state == 'point_completed' else None))
    points = []
    nominal_profile = effective_point_profile(profile)
    if cancelled():
        emit('cancelled', points)
        return StudyResult('cancelled', profile.study, nominal_profile, None, None, (), profile)
    emit('reference_started', points)
    try:
        nominal_report, nominal = _execute_isolated(nominal_profile)
    except Exception as exc:
        emit('failed', points)
        raise PhysicsCalculationError('NOISE-008: nominal study reference failed; no point execution or fallback.') from exc
    emit('reference_completed', points)
    for index, value in enumerate(profile.study.values):
        if cancelled():
            emit('cancelled', points)
            return StudyResult('cancelled', profile.study, nominal_profile, nominal_report, nominal, tuple(points), profile)
        point_profile = effective_point_profile(profile, index)
        emit('point_started', points, index)
        try:
            report, result = _execute_isolated(point_profile)
            summary = summarize_study_point(nominal, result, profile.study)
        except Exception as exc:
            emit('failed', points, index)
            raise PhysicsCalculationError(f'NOISE-008: point {index}, {profile.study.parameter}={value!r} failed; study aborted, no dropped trials.') from exc
        points.append(StudyPointResult(index, value, point_profile, report, result, summary))
        emit('point_completed', points, index)
    emit('completed', points)
    return StudyResult('completed', profile.study, nominal_profile, nominal_report, nominal, tuple(points), profile)
