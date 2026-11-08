"""NOISE-015 audit and offline sampled-field selection; no physical-model copy."""
from collections.abc import Mapping
from dataclasses import asdict, replace, fields
import numpy as np
from ..config.detection_study import parse_detection_study, detection_jobs
from ..config.noise import parse_noise_config, NoiseConfig
from ..config.laboratory import _real, _integer
from ..exceptions import PhysicsCalculationError


def _match(expected, recorded, context):
    """Compare recorded claims to recomputation; bool is never a numeric witness."""
    if isinstance(expected,Mapping):
        if not isinstance(recorded,Mapping) or set(recorded) != set(expected):
            raise PhysicsCalculationError(f'NOISE-015: incomplete or extra recorded evidence at {context}.')
        for key,value in expected.items():
            _match(value,recorded[key],context+'.'+key)
    elif isinstance(expected,(tuple,list)):
        if not isinstance(recorded,(tuple,list)) or len(recorded) != len(expected):
            raise PhysicsCalculationError(f'NOISE-015: evidence length/type differs at {context}.')
        for i,(a,b) in enumerate(zip(expected,recorded)):
            _match(a,b,f'{context}[{i}]')
    elif type(expected) is bool:
        if type(recorded) is not bool or recorded is not expected:
            raise PhysicsCalculationError(f'NOISE-015: boolean evidence differs at {context}.')
    elif isinstance(expected,(int,float,np.integer,np.floating)):
        value = _real(recorded,'NOISE-015 '+context)
        if isinstance(expected,(int,np.integer)):
            _integer(recorded,'NOISE-015 '+context,minimum=-2)
        if value != expected:
            raise PhysicsCalculationError(f'NOISE-015: numerical evidence differs at {context}.')
    elif type(recorded) is not type(expected) or recorded != expected:
        raise PhysicsCalculationError(f'NOISE-015: recorded evidence differs at {context}.')


def _job_evidence(record, job, index, definition):
    if not isinstance(record,Mapping):
        raise PhysicsCalculationError('NOISE-015: full job record required.')
    _match(index,record['index'],'job.index')
    _match(job['field_v_per_m'],record['value'],'job.field')
    _match(job['seed'],record['effective_seed'],'job.seed')
    _match(job['trials'],record['effective_trials'],'job.trials')
    summary = record['summary']; run = record['run_report']; evidence = summary['evidence']
    _match(job['role'],summary['role'],'job.role')
    _match({'mode':'if_only','signal_enabled':index >= 0},run['measurement'],'job.measurement')
    configured = _real(run['configured_fields']['signal']['electric_field_vm'],'NOISE-015 configured field',positive=True)
    applied = _real(run['applied_fields']['signal']['electric_field_vm'],'NOISE-015 applied field',nonnegative=True)
    if applied != (job['field_v_per_m'] if index >= 0 else 0.) or (index >= 0 and configured != applied):
        raise PhysicsCalculationError('NOISE-015: effective/configured signal field evidence differs from the job.')
    noise = _recorded_noise(run['noise_declaration'])
    if noise.seed != job['seed'] or noise.trials != job['trials']:
        raise PhysicsCalculationError('NOISE-015: recorded noise seed/count differs from the independent design.')
    _match(definition.path_index,evidence['path_index'],'selected path')
    if evidence['trace_units'] not in ('V','A','Im<sigma21>','Im<sigma21> per real sample') or evidence['statistic_units'] != evidence['trace_units']+'^2 (squared IF peak amplitude)':
        raise PhysicsCalculationError('NOISE-015: actual squared IF statistic units required.')
    from ..utils.numerical import if_fit_noise_weights
    sensing = run['sensing']; execution = sensing['execution']
    if 'tlist_us' in execution:
        times = np.asarray(execution['tlist_us'])
        if times.dtype.kind not in 'iuf' or times.ndim != 1:
            raise PhysicsCalculationError('NOISE-015: real recorded time grid required.')
    else:
        count = _integer(sensing['samples_per_path'],'NOISE-015 output samples',minimum=10)
        duration = _real(sensing['duration_us'],'NOISE-015 duration [us]',positive=True)
        times = np.linspace(0.,duration,count)
        _match(float(times[1]-times[0]),sensing['sample_step_us'],'sampling step')
    _,_,fit = if_fit_noise_weights(times,run['signed_if_mhz'],execution['transient_delay_us'])
    _match(fit,evidence['fit'],'recomputed IF fit')
    return summary,run,evidence,noise


def _recorded_noise(raw):
    """Validate the actual asdict report format through the canonical parser."""
    if not isinstance(raw,Mapping) or set(raw) != {f.name for f in fields(NoiseConfig)}:
        raise PhysicsCalculationError('NOISE-015: complete recorded noise declaration required.')
    sequence = raw['sources']
    if not isinstance(sequence,(tuple,list)) or not sequence:
        raise PhysicsCalculationError('NOISE-015: recorded noise-source entries required.')
    sources = {}
    for source in sequence:
        if (not isinstance(source,Mapping) or set(source) != {'name','specification'} or
                not isinstance(source['name'],str) or source['name'] in sources):
            raise PhysicsCalculationError('NOISE-015: invalid or duplicate recorded noise source.')
        sources[source['name']] = source['specification']
    data = {key:raw[key] for key in ('schema_version','enabled','seed','trials')}
    data['sources'] = sources
    if raw['schema_version'] == 2:
        data['detector'] = raw['detector']
    elif raw['schema_version'] == 3:
        from ..config.profile_document import noise_document
        data['drive'] = raw['drive']
        data['readout'] = noise_document(_recorded_noise(raw['readout'])) if raw['readout'] is not None else None
    noise = parse_noise_config(data)
    _match(asdict(noise),raw,'recorded noise declaration')
    return noise


def _noise_at(noise, job):
    return replace(noise,seed=job['seed'],trials=job['trials'],
                   readout=_noise_at(noise.readout,job) if noise.readout is not None else None)


def analyze_sampled_sensitivity(report):
    """Recompute from stored scores, then select only a simultaneously supported field.

    This audits consistency of recorded evidence, not authenticity of artifacts,
    numerical convergence, hardware calibration, or a continuous minimum.
    """
    from .detection_study import calibrate_threshold, evaluate_detection, _fixed_conditions
    try:
        if not isinstance(report,Mapping) or report.get('schema_version') != 1 or type(report.get('schema_version')) is not int:
            raise PhysicsCalculationError('NOISE-015: versioned detection report required.')
        # Native asdict reports contain tuples; JSON files contain lists.
        # Normalize representation only, without filling/sorting any evidence.
        def lists(value):
            if isinstance(value,Mapping):return {k:lists(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [lists(v) for v in value]
            return value
        d = parse_detection_study(lists(report['study']))
        result = {'schema_version':1,'status':'incomplete_study','minimum_supported_field_v_per_m':None,
            'selected_point_index':None,'unit':'V/m','criterion':'P_FA upper <= target and P_D lower >= target; simultaneous intervals',
            'grid_minimality_status':'not_established','lower_sampled_points':[],
            'continuous_minimum_computed':False,'mdf_nef_computed':False,
            'numerical_convergence_status':'not_verified_for_this_study',
            'physical_sensitivity_certified':False,'hardware_calibration_verified':False,
            'evidence_scope':'consistency audit and recomputation from recorded scores; not artifact authenticity',
            'refinement':'separate explicit study with independent seeds; no pooling or repeated-study confidence claim'}
        if report['status'] == 'cancelled':
            return result
        if report['status'] != 'completed':
            raise PhysicsCalculationError('NOISE-015: unsupported study execution status.')
        _match(len(d.values),report['points_requested'],'points requested')
        _match(len(d.values),report['points_completed'],'points completed')
        if not isinstance(report['points'],list) or len(report['points']) != len(d.values):
            raise PhysicsCalculationError('NOISE-015: complete predeclared grid required; no surviving-point selection.')
        confidence = report['confidence']
        _match(d.confidence_level,confidence['family_level'],'confidence level')
        _match(len(d.values)+1,confidence['comparisons'],'confidence comparisons')
        _match('Clopper-Pearson with Bonferroni',confidence['method'],'confidence method')
        jobs = detection_jobs(d)
        calibration,reference,first_evidence,base_noise = _job_evidence(report['calibration'],jobs[0],-2,d)
        threshold = calibrate_threshold(d,first_evidence['scores'])
        _match(threshold,report['threshold'],'threshold')
        _match(threshold,calibration['threshold'],'calibration threshold')
        _match(threshold['value'] is not None,report['threshold_frozen'],'threshold frozen')
        alpha = (1-d.confidence_level)/(len(d.values)+1)
        probabilities = []
        records = [report['null_validation'],*report['points']]
        for j,(record,job) in enumerate(zip(records,jobs[1:])):
            index = j-1
            summary,run,evidence,noise = _job_evidence(record,job,index,d)
            _match(asdict(_noise_at(base_noise,job)),asdict(noise),'fixed noise conditions')
            for key in ('levels','rf_orientation','detunings_rad_per_us','decay_rates_per_us',
                        'resolved_rate_budget_per_us','signed_if_mhz','quantization_axis'):
                _match(reference[key],run[key],'fixed physics '+key)
            _match(reference['sensing']['execution'],run['sensing']['execution'],'fixed execution')
            _fixed_conditions(reference,run,first_evidence,evidence)
            evaluated = evaluate_detection(evidence['scores'],job['trials'],threshold['value'],
                d.p_false_alarm if j == 0 else d.p_detection,alpha,null=j==0)
            _match(evaluated,summary['probability'],'recorded probability')
            probabilities.append(evaluated)
            if j > 0:
                joint = ('supported_at_sampled_field' if probabilities[0]['status']=='target_supported' and
                         evaluated['status']=='target_supported' else 'not_established')
                _match(joint,summary['joint_target_status'],'joint decision')
        pfa = probabilities[0]; pd = probabilities[1:]
        result['statistical_audit_status'] = 'recomputed_and_consistent'
        result['false_alarm_validation'] = pfa
        result['sampled_points'] = [{'index':i,'field_v_per_m':v,'probability':p} for i,(v,p) in enumerate(zip(d.values,pd))]
        result['statistical_feasibility'] = {'finite_threshold_available':threshold['value'] is not None,
            'best_case_false_alarm_target_feasible': -np.expm1(np.log(alpha/2)/d.null_validation_trials) <= d.p_false_alarm,
            'best_case_detection_target_feasible': np.exp(np.log(alpha/2)/d.trials_per_point) >= d.p_detection}
        # Make GUI/JSON booleans native, never numpy scalars.
        result['statistical_feasibility'] = {k:bool(v) for k,v in result['statistical_feasibility'].items()}
        if threshold['value'] is None:
            result['status'] = 'undefined_threshold'
        elif pfa['status'] != 'target_supported':
            result['status'] = 'false_alarm_target_not_established'
        else:
            supported = [i for i,p in enumerate(pd) if p['status']=='target_supported']
            if not supported:
                result['status'] = 'no_supported_sampled_field'
            else:
                selected = supported[0]
                result.update(status='supported_sampled_field',selected_point_index=selected,
                              minimum_supported_field_v_per_m=d.values[selected])
                result['lower_sampled_points'] = [{'index':i,'field_v_per_m':d.values[i],'status':pd[i]['status']}
                                                  for i in range(selected)]
                result['grid_minimality_status'] = ('at_lower_grid_boundary' if selected==0 else
                    'all_lower_sampled_fields_excluded' if all(p['status']=='target_not_met' for p in pd[:selected]) else
                    'unresolved_lower_sampled_fields')
        estimates = [p['probability'] for p in pd]
        result['sampled_estimates_decrease_somewhere'] = None if any(p is None for p in estimates) else any(b<a for a,b in zip(estimates,estimates[1:]))
        return result
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-015: incomplete or invalid sensitivity evidence; no substituted minimum field.') from exc
