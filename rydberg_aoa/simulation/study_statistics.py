"""Offline whole-trial percentile bootstrap; never executes physics or uses its RNG.

Trials must be independent realizations within a point. Paired points share
resampled trial indices, preserving cross-path and cross-point dependence.
Intervals describe empirical Monte Carlo sampling uncertainty, not numerical
convergence, calibration uncertainty or guaranteed population coverage.
"""
from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from ..config.study import StudyDefinition, StudyStatistics, parse_statistics, parse_study, axis_spec
from ..config.laboratory import _real
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from .study_metrics import _mean, _rmse
from .if_measurement import pooled_if_snr, _validate_power_pair

_FIELDS = {
    'aoa_bias_deg': ('aoa_error_deg', _mean),
    'aoa_rmse_deg': ('aoa_error_deg', _rmse),
    'if_phase_rmse_deg_per_path': ('if_phase_error_deg_per_path', _rmse),
    'if_amplitude_relative_error_per_path': ('if_amplitude_absolute_relative_error_per_path', _mean),
    'readout_trace_snr_db_per_path': ('readout_trace_snr_db_per_path', _mean),
    'waveform_deviation_ratio_db_per_path': ('waveform_deviation_ratio_db_per_path', _mean),
    'relative_if_phase_rmse_deg_per_path': ('relative_if_phase_error_deg_per_path', _rmse),
    'readout_if_snr_db_per_path': ('readout_if_power_log10_per_path', None),
}


def _rng(seed, checkpoint, point):
    # Separate namespace; physical SeedSequence/source IDs are never consumed.
    key = sha256(f'NOISE-009-bootstrap-v1:{seed}:{checkpoint}:{point}'.encode('ascii')).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(key,'big')))


def _columns(rows, metric):
    field, _ = _FIELDS[metric]
    try:
        values = [r[field] for r in rows]
        vector = metric.endswith('_per_path')
        if vector:
            if any(not isinstance(v,(tuple,list)) or not v for v in values):
                raise ValueError('per-path arrays required')
            width = len(values[0])
            if any(len(v) != width for v in values):
                raise ValueError('inconsistent path count')
        else:
            values = [[v] for v in values]
        columns = []
        for col in zip(*values):
            if metric == 'readout_if_snr_db_per_path':
                columns.append([_validate_power_pair(v) for v in col])
                continue
            parsed = [None if v is None else _real(v,'NOISE-009 trial metric') for v in col]
            if metric in ('aoa_bias_deg','aoa_rmse_deg','if_phase_rmse_deg_per_path', 'relative_if_phase_rmse_deg_per_path'):
                if any(v is not None and not -180 <= v < 180 for v in parsed):
                    raise ValueError('wrapped degree errors must be in [-180,180)')
            if metric=='if_amplitude_relative_error_per_path' and any(v is not None and v<0 for v in parsed):
                raise ValueError('absolute relative amplitude error must be nonnegative')
            if not vector and any(v is None for v in parsed):
                raise ValueError('AoA error requires complete finite evidence')
            columns.append(parsed)
        return columns
    except (KeyError,TypeError,ValueError,PhysicsConfigurationError) as exc:
        raise PhysicsCalculationError(f'NOISE-009: invalid trial evidence for {metric}.') from exc


def _estimate(column, aggregate, indices):
    if aggregate is None:
        estimate, why = pooled_if_snr(column)
        if why is not None:
            return None, None, why
        draws = [pooled_if_snr([column[i] for i in ix]) for ix in indices]
        if any(why is not None for _, why in draws):
            return estimate, None, 'undefined_bootstrap_power'
        return estimate, np.asarray([value for value, _ in draws]), None
    if any(v is None for v in column):
        return None, None, 'undefined_trial_in_prefix'
    values = np.asarray(column,dtype=float)
    estimate = aggregate(values)
    samples = np.asarray([aggregate(values[ix]) for ix in indices])
    if not np.isfinite(samples).all() or not np.isfinite(estimate):
        raise PhysicsCalculationError('NOISE-009: nonfinite bootstrap estimate; no substitute.')
    return estimate, samples, None


def _interval(estimate, samples, why, count, config):
    if why is not None:
        return {'estimate':estimate,'interval':None,'status':why}
    if count < 2:
        return {'estimate':estimate,'interval':None,'status':'insufficient_trials'}
    alpha = (1-config.confidence_level)/2
    scale = float(np.max(np.abs(samples)))
    bounds = ([0.,0.] if scale==0 else
              (scale*np.quantile(samples/scale,[alpha,1-alpha],method='linear')).tolist())
    if not np.isfinite(bounds).all():
        raise PhysicsCalculationError('NOISE-009: nonfinite percentile bounds.')
    return {'estimate':estimate,'interval':bounds,
            'status':'degenerate_empirical_distribution' if bounds[0]==bounds[1] else 'estimated'}


def analyze_study_statistics(report, statistics=None):
    """Analyze a live or JSON-reloaded report; optional options override is pure.

    All checkpoints use prefixes of the already completed campaign. No omitted
    trials, imputation, fresh physical draws, automatic stopping or solver calls.
    Point-minus-first-point intervals preserve the declared pairing policy.
    Cancelled reports retain their status; absent points are never fabricated.
    """
    if not isinstance(report,Mapping) or type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise PhysicsCalculationError('NOISE-009: study report schema 1 required.')
    try:
        # JSON normalizes tuple declarations from in-memory asdict reports.
        definition = parse_study(json.loads(json.dumps(report['study'],allow_nan=False)))
    except (KeyError,TypeError,ValueError,PhysicsConfigurationError) as exc:
        raise PhysicsCalculationError('NOISE-009: invalid stored study declaration.') from exc
    if not isinstance(definition, StudyDefinition):
        raise PhysicsConfigurationError('NOISE-009: bootstrap requires a noise-strength study with trials; field/spacing studies do not supply trial uncertainty.')
    if 'random_policy' in report and report['random_policy'] != definition.random_policy:
        raise PhysicsCalculationError('NOISE-009: contradictory recorded random policies.')
    config = definition.statistics if statistics is None else statistics
    if isinstance(config,dict):
        config = parse_statistics(config)
    if not isinstance(config,StudyStatistics):
        raise PhysicsConfigurationError('NOISE-009: explicit statistics configuration required.')
    if config.trial_checkpoints[-1] > definition.trials_per_point:
        raise PhysicsConfigurationError('NOISE-009: checkpoints exceed stored trials; rerun explicitly to obtain more data.')
    points = report.get('points')
    status = report.get('status')
    if (status not in ('completed','cancelled') or not isinstance(points,list) or
            type(report.get('points_completed')) is not int or type(report.get('points_requested')) is not int or
            len(points) != report.get('points_completed') or
            report.get('points_requested') != len(definition.values) or
            (status=='completed' and len(points)!=len(definition.values))):
        raise PhysicsCalculationError('NOISE-009: invalid completion evidence.')
    all_rows = []
    path_count = None
    for index,point in enumerate(points):
        try:
            rows = point['summary']['trials']
            expected_seed = definition.seed if definition.random_policy=='paired' else int.from_bytes(
                sha256(f'NOISE-008-v1:{definition.seed}:{index}'.encode('ascii')).digest()[:8],'big') & (2**53-1)
            if (type(point['index']) is not int or point['index'] != index or
                    _real(point['value'],'NOISE-009 stored axis value') != definition.values[index] or
                    type(point.get('effective_seed')) is not int or point['effective_seed'] != expected_seed or
                    type(point['effective_trials']) is not int or point['effective_trials'] != definition.trials_per_point or
                    not isinstance(rows,list) or len(rows)!=definition.trials_per_point or
                    any(not isinstance(r,Mapping) or type(r.get('trial_index')) is not int or
                        r['trial_index']!=i for i,r in enumerate(rows))):
                raise ValueError('incomplete or reordered trial/point evidence')
            for metric in definition.metrics:
                columns = _columns(rows,metric)
                if metric == 'relative_if_phase_rmse_deg_per_path':
                    reference = definition.phase_reference_path
                    if reference >= len(columns) or any(v is not None and v != 0 for v in columns[reference]):
                        raise ValueError('relative reference must exist and be zero by definition or undefined')
                if metric.endswith('_per_path'):
                    if path_count is not None and len(columns)!=path_count:
                        raise ValueError('path count differs across metrics/points')
                    path_count = len(columns)
            all_rows.append(rows)
        except (KeyError,TypeError,ValueError,PhysicsConfigurationError) as exc:
            raise PhysicsCalculationError(f'NOISE-009: invalid point {index} evidence.') from exc
    checkpoints = []
    for count in config.trial_checkpoints:
        samples_by_point, results = [], []
        for index,rows in enumerate(all_rows):
            generator = _rng(config.seed,count,0 if definition.random_policy=='paired' else index)
            try:
                indices = generator.integers(0,count,size=(config.bootstrap_replicates,count))
            except (MemoryError,ValueError,OverflowError) as exc:
                raise PhysicsCalculationError(f'NOISE-009: cannot allocate requested bootstrap {config.bootstrap_replicates}x{count}; no workload reduction.') from exc
            metrics, samples = {}, {}
            for metric in definition.metrics:
                aggregate = _FIELDS[metric][1]
                estimates = [_estimate(col,aggregate,indices) for col in _columns(rows[:count],metric)]
                output = [_interval(e,s,w,count,config) for e,s,w in estimates]
                metrics[metric] = output if metric.endswith('_per_path') else output[0]
                samples[metric] = estimates
            results.append({'index':index,'value':definition.values[index],'metrics':metrics})
            samples_by_point.append(samples)
        differences = []
        for index in range(1,len(results)):
            metrics = {}
            for metric in definition.metrics:
                first, current = samples_by_point[0][metric],samples_by_point[index][metric]
                if len(first)!=len(current):
                    raise PhysicsCalculationError('NOISE-009: cross-point path counts differ.')
                outputs = []
                for (a,sa,wa),(b,sb,wb) in zip(first,current):
                    why = 'undefined_comparison' if wa or wb else None
                    with np.errstate(over='raise',invalid='raise'):
                        try:
                            delta = None if why else b-a
                            draws = None if why else sb-sa
                        except FloatingPointError as exc:
                            raise PhysicsCalculationError('NOISE-009: comparison overflow.') from exc
                    if delta is not None and (not np.isfinite(delta) or not np.isfinite(draws).all()):
                        raise PhysicsCalculationError('NOISE-009: nonfinite point difference.')
                    outputs.append(_interval(delta,draws,why,count,config))
                metrics[metric] = outputs if metric.endswith('_per_path') else outputs[0]
            differences.append({'index':index,'reference_index':0,'metrics':metrics})
        checkpoints.append({'trials':count,'points':results,'point_minus_first':differences})
    from dataclasses import asdict
    configuration = asdict(config)
    configuration['trial_checkpoints'] = list(config.trial_checkpoints)
    return {'schema_version':1,'method':'whole_trial_percentile_bootstrap',
            'configuration':configuration,'study_status':status,'random_policy':definition.random_policy,
            'axis':{'source':definition.source,'parameter':definition.parameter,
                    'unit':axis_spec(definition.source,definition.parameter)[0]},
            'rng':'PCG64; SHA256 NOISE-009-bootstrap-v1:seed:checkpoint:point; independent of physics',
            'numpy_version':np.__version__,'quantile_method':'linear','checkpoints':checkpoints,
            'limitations':['empirical Monte Carlo uncertainty; coverage not guaranteed',
                          'degenerate intervals do not prove zero population error; AoA may be grid-quantized',
                          'prefix checkpoints are correlated; no automatic convergence/adequacy claim',
                          'not solver, time-grid or noise-grid convergence; trials assumed independent within each point']}


def load_study_statistics(path, statistics=None):
    """Reanalyze stored JSON without importing ARC or rerunning physics."""
    try:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        raise PhysicsCalculationError('NOISE-009: cannot read study report.') from exc
    return analyze_study_statistics(report,statistics)


def save_statistics_analysis(path, analysis):
    """Export a separate analysis artifact, refusing to overwrite any file."""
    encoded = json.dumps(analysis,ensure_ascii=False,allow_nan=False,indent=2)
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(encoded+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NOISE-009: offline whole-trial bootstrap, no atomic rerun.')
    parser.add_argument('report',help='stored study_report.json')
    parser.add_argument('--statistics',help='JSON file with explicit statistical options override')
    parser.add_argument('--output',required=True,help='new uncertainty analysis JSON path; refuses overwrite')
    args = parser.parse_args()
    override = None
    if args.statistics:
        try:
            override = json.loads(Path(args.statistics).read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:
            raise PhysicsConfigurationError('NOISE-009: cannot read statistics options.') from exc
    save_statistics_analysis(args.output,load_study_statistics(args.report,override))
