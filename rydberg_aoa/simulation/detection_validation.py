"""NOISE-016 paired numerical diagnostics on complete IF detection campaigns."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import multiprocessing
from importlib.metadata import version
import numpy as np
from ..config.detection_study import validate_detection_profile
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from .detection_study import execute_detection_study


def _run_diagnostic(profile,solver,quantiles):
    from .profile_execution import execute_profile
    return execute_profile(profile,solver_diagnostic=solver,poisson_quantiles=quantiles)


def isolated_diagnostic(profile,solver,quantiles):
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as executor:
        return executor.submit(_run_diagnostic,profile,solver,quantiles).result()


def _jobs(study):
    if study.status!='completed' or study.calibration is None or study.validation is None:
        raise PhysicsCalculationError('NOISE-016: full detection campaign required; no survivor comparison.')
    return (study.calibration,study.validation,*study.points)


def _trials(point):
    result=point.result
    rows=result.get('rf_drive_trial_results',result.get('noise_trial_results'))
    if not isinstance(rows,list) or len(rows)!=point.profile.noise.trials:
        raise PhysicsCalculationError('NOISE-016: complete canonical trials required.')
    return rows


def _vector(trial):
    if np.ma.isMaskedArray(trial['complex_responses']):raise PhysicsCalculationError('NOISE-016: masked IF evidence rejected.')
    a=np.asarray(trial['complex_responses'])
    if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iufc' or not np.isfinite(a).all():
        raise PhysicsCalculationError('NOISE-016: finite complex IF vector required.')
    return a


def _clean(trial,noise):
    try:
        if noise.physical_readout_enabled:a=np.asarray(trial['detector_arrays']['clean_atomic_time_traces'])
        elif noise.readout_enabled:a=np.asarray(trial['clean_time_traces'])
        else:a=np.asarray(trial['time_traces'])  # Explicit drive-only acquisition, no readout model.
    except (KeyError,TypeError) as exc:
        raise PhysicsCalculationError('NOISE-016: missing canonical clean atomic trace evidence.') from exc
    if a.ndim!=2 or not all(a.shape) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise PhysicsCalculationError('NOISE-016: finite real clean traces required.')
    return a


def _check_drives(a,b):
    from .noise_convergence import realization_digest
    left=a.result.get('rf_drive_realizations');right=b.result.get('rf_drive_realizations')
    if (left is None)!=(right is None):raise PhysicsCalculationError('NOISE-016: drive realization evidence differs.')
    if left is not None:
        if len(left)!=len(right):raise PhysicsCalculationError('NOISE-016: drive realization count differs.')
        for x,y in zip(left,right):
            if (x is None)!=(y is None) or (x is not None and realization_digest(x)!=realization_digest(y)):
                raise PhysicsCalculationError('NOISE-016: coupled OU/laser realization changed.')


def production_anchor_check(anchor,baseline):
    rows=[]
    for a,b in zip(_jobs(anchor),_jobs(baseline)):
        _live_evidence(a,anchor.definition);_live_evidence(b,baseline.definition)
        _check_drives(a,b)
        if a.profile!=b.profile:raise PhysicsCalculationError('NOISE-016: anchor/coupled baseline profile differs.')
        for x,y in zip(_trials(a),_trials(b)):
            if not np.array_equal(_clean(x,a.profile.noise),_clean(y,b.profile.noise)):
                raise PhysicsCalculationError('NOISE-016: diagnostic baseline is not bit-identical to production atomic evolution.')
        rows.append({'role':a.summary['role'],'atomic_traces_bit_identical':True})
    return {'status':'verified','jobs':rows,
            'scope':'canonical atomic evolution only; diagnostic Poisson quantiles intentionally differ from production Poisson RNG'}


def _live_evidence(point,definition):
    from .detection_study import _evidence
    from .sampled_sensitivity import _match
    _match(_evidence(point.profile,point.report,point.result,definition),point.summary['evidence'],'live raw IF evidence')


def verify_diagnostic_evidence(study,solver,quantiles):
    from .sampled_sensitivity import _match
    for point in _jobs(study):
        actual=point.report.get('numerical_diagnostic')
        base = getattr(getattr(getattr(point,'profile',None),'run_config',None),'temporal_ode',None)
        effective_nsteps = (5000 if base is None else base.nsteps) if solver is None else solver.nsteps
        if solver is None and not quantiles:
            if actual is not None:raise PhysicsCalculationError('NOISE-016: production anchor was instrumented.')
        else:
            # Old receipts already recorded the effective canonical limit at
            # the outer level. Accept that explicit witness for the newly
            # duplicated nested field; never infer a noncanonical limit.
            if (solver is not None and solver.nsteps == 5000 and isinstance(actual,dict)
                    and actual.get('nsteps') == 5000 and isinstance(actual.get('solver'),dict)
                    and 'nsteps' not in actual['solver']):
                actual={**actual,'solver':{**actual['solver'],'nsteps':5000}}
            expected={'solver':None if solver is None else asdict(solver),'poisson_quantiles':quantiles,
                'acquisition_grid_unchanged':True,'solver_method':'bdf','nsteps':effective_nsteps,'normalize_output':False,
                'option_enforcement':'not instrumented' if solver is None else
                    'spawn-worker mesolve wrapper verifies options and restores on exception'}
            _match(expected,actual,'actual numerical diagnostic')


def compare_detection_case(baseline,case,options):
    from .detection_study import _fixed_conditions
    rows=[];passed=True;decisions_assessed=True
    try:
        for a,b in zip(_jobs(baseline),_jobs(case)):
            _live_evidence(a,baseline.definition);_live_evidence(b,case.definition)
            if a.profile!=b.profile:raise PhysicsCalculationError('NOISE-016: numerical case changed physical profile.')
            _check_drives(a,b)
            _fixed_conditions(a.report,b.report,a.summary['evidence'],b.summary['evidence'])
            deltas=[];ratios=[];criterion=[]
            for x,y in zip(_trials(a),_trials(b)):
                if not np.array_equal(x['tlist'],y['tlist']):raise PhysicsCalculationError('NOISE-016: detector acquisition grid changed.')
                u,v=_vector(x),_vector(y)
                if u.shape!=v.shape:raise PhysicsCalculationError('NOISE-016: IF path shape changed.')
                with np.errstate(over='raise',invalid='raise'):
                    delta=np.abs(v-u);budget=options.if_absolute_budget+options.if_relative_budget*np.abs(u)
                if not np.isfinite(delta).all() or not np.isfinite(budget).all():
                    raise PhysicsCalculationError('NOISE-016: nonfinite error comparison; no clipping.')
                deltas.append(delta.tolist());criterion.append(bool(np.all(delta<=budget)))
                ratios.append([None if abs(p)==0 else float(d/abs(p)) for p,d in zip(u,delta)])
                if any(value is not None and not np.isfinite(value) for value in ratios[-1]):
                    raise PhysicsCalculationError('NOISE-016: relative diagnostic overflow; no clipped ratio.')
                # Constant-variance Gaussian sources must retain identical additive draws.
                if a.profile.noise.schema_version==1 and not np.array_equal(x['additive_readout_noise'],y['additive_readout_noise']):
                    raise PhysicsCalculationError('NOISE-016: Gaussian innovations changed between numerical cases.')
                if 'detector_arrays' in x:
                    for key in ('electronic_current_current_noise_a','johnson_current_current_noise_a'):
                        if key in x['detector_arrays'] and not np.array_equal(x['detector_arrays'][key],y['detector_arrays'][key]):
                            raise PhysicsCalculationError('NOISE-016: electronic/Johnson Gaussian draws changed.')
            role=a.summary['role'];decision=None;frozen=None;probability_difference=None
            if role!='calibration':
                p,q=a.summary['probability'],b.summary['probability']
                if p['detections'] is None or q['detections'] is None:decisions_assessed=False
                else:
                    decision=float(np.mean(np.asarray(p['detections'])!=np.asarray(q['detections'])))
                    threshold=baseline.threshold['value']
                    bscores=np.asarray(b.summary['evidence']['scores'])
                    frozen=float(np.mean(np.asarray(p['detections'])!=(bscores>threshold)))
                    probability_difference=abs(p['probability']-q['probability'])
                    passed &= max(decision,frozen)<=options.decision_fraction_budget and probability_difference<=options.probability_difference_budget
            ok=all(criterion);passed &= ok
            rows.append({'role':role,'if_absolute_errors':deltas,'if_relative_errors':ratios,
                         'maximum_if_absolute_error':float(np.max(deltas)),'mixed_if_budget_passed':ok,
                         'decision_disagreement_fraction':decision,'frozen_baseline_threshold_disagreement_fraction':frozen,
                         'probability_difference':probability_difference})
        left=baseline.report()['sampled_sensitivity']['minimum_supported_field_v_per_m']
        right=case.report()['sampled_sensitivity']['minimum_supported_field_v_per_m']
        same=left==right
        selection_assessed=left is not None and right is not None
        if options.require_same_selected_field:passed &= same and selection_assessed
        status=('unassessed_undefined_threshold' if not decisions_assessed else
                'unassessed_no_supported_selected_field' if options.require_same_selected_field and not selection_assessed else
                'passed_declared_comparisons' if passed else 'failed_declared_comparisons')
        return {'status':status,'jobs':rows,'selected_field_same':same,'selection_assessed':selection_assessed,
                'baseline_selected_field_v_per_m':left,'case_selected_field_v_per_m':right,
                'threshold_absolute_difference':None if baseline.threshold['value'] is None or case.threshold['value'] is None else
                    abs(baseline.threshold['value']-case.threshold['value']),
                'relative_zero_reference':'null, never fabricated zero/inf',
                'scope':'paired finite-setting agreement; not a continuum error bound or additional independent trials'}
    except PhysicsCalculationError:raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-016: numerical case comparison failed; no replacement metric.') from exc


@dataclass(frozen=True)
class DetectionValidationProgress:
    state: str
    completed_points: int
    total_points: int
    point_index: object
    axis_value: object
    point_result: object
    campaign_name: str


@dataclass(frozen=True)
class DetectionValidationResult:
    status: str
    base_profile: object
    campaigns: tuple
    anchor_evidence: object
    comparisons: tuple

    @property
    def definition(self):return self.base_profile.study
    @property
    def points(self):return self.campaigns[1][1].points if len(self.campaigns)>1 else ()

    def report(self):
        baseline=self.campaigns[1][1] if len(self.campaigns)>1 else None
        r=baseline.report() if baseline is not None else {'schema_version':1,'study':asdict(self.definition),
            'status':self.status,'points':[],'calibration':None,'null_validation':None,'threshold':None}
        r['status']=self.status
        r['execution_scope']='detection_numerical_validation'
        assessed=len(self.comparisons)==len(self.definition.numerical_validation.cases)
        verdict='incomplete' if self.status!='completed' or not assessed else (
            'passed_declared_comparisons' if all(c['comparison']['status']=='passed_declared_comparisons' for c in self.comparisons) else
            'not_established')
        r['numerical_validation']={'status':verdict,'production_anchor':self.anchor_evidence,
            'comparisons':list(self.comparisons),'environment':{p:version(p) for p in ('qutip','numpy','scipy')},
            'campaigns':[{'name':name,'report':study.report()} for name,study in self.campaigns],
            'noise_protocol':'OU nodes and Gaussian draws common; Poisson inverse-CDF uniforms common in diagnostic campaigns',
            'acquisition_grid_and_aperture_fixed':True,'observation_and_transient_fixed':True,
            'continuum_error_bound':False,'physical_sensitivity_certified':False,
            'independence':'numerical cases are paired, not extra trial replicates; confidence applies within each detection study only'}
        if 'sampled_sensitivity' in r:
            r['sampled_sensitivity']['numerical_comparison_status']=verdict
            r['sampled_sensitivity']['numerical_convergence_status']=('passed_declared_finite_comparisons' if
                verdict=='passed_declared_comparisons' else 'not_established_by_declared_comparisons')
        return r


def execute_detection_validation(profile,*,on_progress=None,should_cancel=None):
    validate_detection_profile(profile);options=profile.study.numerical_validation
    if options is None or not options.enabled:raise PhysicsConfigurationError('NOISE-016: enabled numerical design required.')
    for callback in (on_progress,should_cancel):
        if callback is not None and not callable(callback):raise PhysicsConfigurationError('NOISE-016: callable hooks required.')
    campaigns=[];comparisons=[];anchor=None
    cases=[('production_anchor',None,False),('coupled_baseline',None,True)]+[(c.name,c,True) for c in options.cases]
    jobs_per_case=len(profile.study.values)+2;total=len(cases)*jobs_per_case
    for case_index,(name,solver,quantiles) in enumerate(cases):
        def relay(event):
            if on_progress is not None:
                on_progress(DetectionValidationProgress('campaign_completed' if event.state=='completed' else event.state,
                    case_index*jobs_per_case+event.completed_points,total,
                    None if event.point_index is None else case_index*jobs_per_case+event.point_index,
                    event.axis_value,event.point_result,name))
        try:
            study=execute_detection_study(profile,on_progress=relay,should_cancel=should_cancel,
                                          solver_diagnostic=solver,poisson_quantiles=quantiles)
        except Exception as exc:
            raise PhysicsCalculationError(f'NOISE-016: campaign {name} failed; no alternate solver, retry or survivor statistics.') from exc
        campaigns.append((name,study))
        if study.status!='completed':return DetectionValidationResult('cancelled',profile,tuple(campaigns),anchor,tuple(comparisons))
        try:
            verify_diagnostic_evidence(study,solver,quantiles)
            if len(campaigns)==2:anchor=production_anchor_check(campaigns[0][1],study)
            if len(campaigns)>2:
                comparison=compare_detection_case(campaigns[1][1],study,options)
                comparisons.append({'name':name,'solver':asdict(solver),'comparison':comparison})
        except Exception as exc:
            if on_progress is not None:
                on_progress(DetectionValidationProgress('failed',(case_index+1)*jobs_per_case,total,None,None,None,name))
            raise PhysicsCalculationError(f'NOISE-016: evidence/comparison failed in {name}; no substituted validation.') from exc
    if on_progress is not None:
        on_progress(DetectionValidationProgress('completed',total,total,None,None,None,cases[-1][0]))
    return DetectionValidationResult('completed',profile,tuple(campaigns),anchor,tuple(comparisons))
