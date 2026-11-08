"""Coupled output/noise-grid comparisons through the canonical certified solver."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, fields
from hashlib import sha256
import multiprocessing
import numpy as np
from ..config.noise_convergence import validate_convergence_profile, convergence_profile
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from .rf_drive_noise import RFDriveRealization, build_rf_realization
from .rf_drive_campaign import validate_drive_realizations
from .noise_campaign import azimuth_error, _angle


def realization_digest(realization):
    if not isinstance(realization,RFDriveRealization):
        raise PhysicsConfigurationError('NOISE-010: digest requires a validated realization.')
    digest=sha256()
    for field in fields(RFDriveRealization):
        digest.update(field.name.encode('ascii'))
        digest.update(np.asarray(getattr(realization,field.name),dtype='<f8').tobytes())
    return digest.hexdigest()


def nested_realization(fine, stride):
    """Retain exact knots/factors; no interpolation, redraw or renormalization.

    Subsampling exact stationary OU nodes preserves the coarser OU marginal
    law. Linear interpolants between knots differ: that is the tested error.
    """
    if not isinstance(fine,RFDriveRealization) or type(stride) is not int or stride<1 or (len(fine.times_us)-1)%stride:
        raise PhysicsConfigurationError('NOISE-010: stride must divide the finest interval count.')
    return RFDriveRealization(**{f.name:getattr(fine,f.name)[::stride] for f in fields(RFDriveRealization)})


def _run_case(profile, realizations):
    from .profile_execution import execute_profile
    return execute_profile(profile,drive_realizations=realizations)


def _isolated_case(profile, realizations):
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as executor:
        return executor.submit(_run_case,profile,realizations).result()


def _trials(result, count):
    trials=result.get('rf_drive_trial_results')
    if not isinstance(trials,list) or len(trials)!=count:
        raise PhysicsCalculationError('NOISE-010: missing complete canonical trial results.')
    return trials


def _response(result):
    a=np.asarray(result['complex_responses'])
    if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iufc' or not np.isfinite(a).all():
        raise PhysicsCalculationError('NOISE-010: finite canonical IF vector required.')
    return a


def compare_convergence_case(reference, result, study):
    """Worst discrepancy over ALL trials/paths; undefined never drops a sample."""
    rows=[]
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for index,(ref,case) in enumerate(zip(_trials(reference,study.trials),_trials(result,study.trials))):
                a,b=_response(ref),_response(case)
                if a.shape!=b.shape:
                    raise PhysicsCalculationError('NOISE-010: differing IF path counts.')
                amplitude=np.abs(a)
                absolute=np.abs(b-a)
                relative=[None if r==0 else float(d/r) for d,r in zip(absolute,amplitude)]
                phase=[None if r==0 or v==0 else float(abs(np.rad2deg((np.angle(v)-np.angle(r)+np.pi)%(2*np.pi)-np.pi)))
                       for v,r in zip(b,a)]
                amp=[None if r==0 else float(abs(abs(v)/r-1)) for v,r in zip(b,amplitude)]
                angle=abs(azimuth_error(_angle(case['estimated_aoa'],'convergence case AoA'),
                                        _angle(ref['estimated_aoa'],'convergence reference AoA')))
                rows.append({'trial_index':index,'if_response_absolute':absolute.tolist(),
                             'if_response_relative':relative,'if_phase_deg':phase,
                             'if_amplitude_relative':amp,'aoa_deg':angle})
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-010: comparison failed; no substituted metric.') from exc
    summary={};budgets=dict(study.budgets)
    for metric in study.metrics:
        samples=[v for row in rows for v in (row[metric] if isinstance(row[metric],list) else [row[metric]])]
        if any(v is None for v in samples):
            summary[metric]={'maximum':None,'budget':budgets.get(metric),'status':'unassessed',
                             'reason':'undefined_reference_or_trial_IF_amplitude'}
            continue
        if not np.isfinite(samples).all():
            raise PhysicsCalculationError('NOISE-010: nonfinite discrepancy.')
        maximum=max(samples)
        budget=budgets.get(metric)
        status='unassessed' if budget is None else 'passed_budget' if maximum<=budget else 'failed_budget'
        summary[metric]={'maximum':maximum,'budget':budget,'status':status,
                         'reason':'no_explicit_budget' if budget is None else None}
    return {'metrics':summary,'trials':rows,
            'meaning':'agreement with the declared finest case; not an error bound against the continuum',
            'aoa_warning':'fixed/zero AoA discrepancy alone does not certify temporal/noise-grid convergence'}


@dataclass(frozen=True)
class ConvergencePoint:
    index: int
    axis: str
    value: float
    profile: object
    report: dict
    result: dict
    comparison: object


@dataclass(frozen=True)
class ConvergenceResult:
    status: str
    base_profile: object
    points: tuple

    @property
    def definition(self):
        return self.base_profile.study

    def report(self):
        definition=asdict(self.definition);definition['budgets']=dict(self.definition.budgets)
        requested=len(self.definition.output_samples)+len(self.definition.noise_steps_us)-1
        return {'schema_version':1,'kind':self.definition.kind,'study':definition,'status':self.status,
                'points_requested':requested,'points_completed':len(self.points),'reference_index':0 if self.points else None,
                'coupling':'finest canonical OU node realization per trial; exact nested subsets; common paths',
                'rng_protocol':'NOISE-005/006/007 canonical source streams on finest noise grid only',
                'solver_policy':'canonical BDF/certification/settings unchanged; no tolerance overrides',
                'sampling_policy':'fixed duration/transient/IF/DoA/physics; finest-noise output scan then finest-output noise scan',
                'budget_policy':'explicit maxima over all trials and paths; no survivor statistics or automatic stopping',
                'points':[{'index':p.index,'axis':p.axis,'value':p.value,
                    'output_samples':len(p.profile.run_config.tlist_us),'noise_step_us':p.profile.noise.drive.noise_step_us,
                    'realization_sha256':[realization_digest(r) for r in p.result['rf_drive_realizations']],
                    'comparison':p.comparison,'run_report':p.report} for p in self.points],
                'limitations':['finest case is a comparison reference, not a certified continuum solution',
                               'no readout sampling/bandwidth study; stochastic readout must be explicitly disabled',
                               'same seed with a different grid is not trajectory identity; coarse cases use declared coupling',
                               'this finite trial campaign does not establish population statistical adequacy']}


def execute_convergence_study(profile, *, on_progress=None, should_cancel=None):
    validate_convergence_profile(profile)
    for callback in (on_progress,should_cancel):
        if callback is not None and not callable(callback):
            raise PhysicsConfigurationError('NOISE-010: callable progress/cancellation required.')
    from .study_execution import StudyProgress
    study=profile.study
    fine_samples,fine_step=study.output_samples[-1],study.noise_steps_us[-1]
    cases=[('reference',fine_samples,fine_step)]
    cases += [('output_samples',n,fine_step) for n in study.output_samples[:-1]]
    cases += [('noise_step_us',fine_samples,step) for step in study.noise_steps_us[:-1]]
    points=[];finest=None
    def emit(state,index=None):
        if on_progress:
            on_progress(StudyProgress(state,len(points),len(cases),index,
                       None if index is None else cases[index][1 if cases[index][0]=='output_samples' else 2],
                       points[-1] if state=='point_completed' else None))
    for index,(axis,samples,step) in enumerate(cases):
        if should_cancel is not None:
            answer=should_cancel()
            if type(answer) is not bool:
                raise PhysicsConfigurationError('NOISE-010: cancellation must return boolean.')
            if answer:
                emit('cancelled');return ConvergenceResult('cancelled',profile,tuple(points))
        emit('point_started',index)
        try:
            case=convergence_profile(profile,samples,step)
            if finest is None:
                finest=tuple(build_rf_realization(case.noise,trial=i) for i in range(study.trials))
            stride=round(step/fine_step)
            realizations=finest if stride==1 else tuple(nested_realization(r,stride) for r in finest)
            validate_drive_realizations(case.noise,realizations)
            report,result=_isolated_case(case,realizations)
            _trials(result,study.trials)
            if tuple(result.get('rf_drive_realizations',())) != realizations:
                raise PhysicsCalculationError('NOISE-010: canonical run did not retain supplied realizations.')
            if any(not np.array_equal(t['tlist'],case.run_config.tlist_us) for t in result['rf_drive_trial_results']):
                raise PhysicsCalculationError('NOISE-010: canonical trial grid differs from case.')
            if points:
                for key in ('applied_fields','levels','rf_orientation','detunings_rad_per_us',
                            'rabi_rad_per_us','decay_rates_per_us','resolved_rate_budget_per_us'):
                    if report[key]!=points[0].report[key]:
                        raise PhysicsCalculationError(f'NOISE-010: physical invariant {key} changed between cases.')
                for key in ('environment','resolved_medium','optical_energy_convention','signed_if_mhz','sensing_trace_units'):
                    if report.get(key)!=points[0].report.get(key):
                        raise PhysicsCalculationError(f'NOISE-010: medium/unit invariant {key} changed between cases.')
                for key in ('doa_effective_settings','doa_settings_source'):
                    if report.get('sensing',{}).get(key)!=points[0].report.get('sensing',{}).get(key):
                        raise PhysicsCalculationError(f'NOISE-010: DoA invariant {key} changed between cases.')
            comparison=compare_convergence_case(result if index==0 else points[0].result,result,study)
            comparison['self_reference'] = index==0
            value=samples if axis=='output_samples' else step
            points.append(ConvergencePoint(index,axis,value,case,report,result,comparison))
        except Exception as exc:
            emit('failed',index)
            raise PhysicsCalculationError(f'NOISE-010: case {index} ({axis}, samples={samples}, step={step}us) failed; no retry/dropped trials.') from exc
        emit('point_completed',index)
    emit('completed')
    return ConvergenceResult('completed',profile,tuple(points))
