"""UI-independent, versioned study declarations and discoverable capabilities."""
from dataclasses import dataclass, replace, fields
from hashlib import sha256
import numpy as np

from .laboratory import _integer, _real
from ..exceptions import PhysicsConfigurationError

# Explicit whitelist, never eval or arbitrary attribute/path mutation.
# Values are (field, unit, minimum); minimum zero is inclusive, >0 required for resistance.
PARAMETERS = {
    'readout_gaussian': ('sigma_observable', 'Im<sigma21>', False),
    'rf_lo_amplitude': ('stationary_std', 'log_amplitude', False),
    'rf_signal_amplitude': ('stationary_std', 'log_amplitude', False),
    'rf_relative_phase': ('stationary_std', 'rad', False),
    'laser_probe_frequency': ('stationary_std', 'Hz', False),
    'laser_coupling_frequency': ('stationary_std', 'Hz', False),
    'laser_probe_intensity': ('fractional_rms', 'relative_rms', False),
    'laser_coupling_intensity': ('fractional_rms', 'relative_rms', False),
    'electronic_current': ('asd_a_per_sqrt_hz', 'A/sqrt(Hz)', False),
    'johnson_current': ('temperature_k', 'K', False),
}
METRICS = ('aoa_bias_deg', 'aoa_rmse_deg', 'if_phase_rmse_deg_per_path',
           'if_amplitude_relative_error_per_path', 'readout_trace_snr_db_per_path',
           'waveform_deviation_ratio_db_per_path', 'readout_if_snr_db_per_path',
           'relative_if_phase_rmse_deg_per_path')
METRIC_LABELS = {
    'aoa_bias_deg':'Azimuth bias [deg]', 'aoa_rmse_deg':'Azimuth RMSE [deg]',
    'if_phase_rmse_deg_per_path':'Nominal-referenced IF phase RMSE [deg]',
    'if_amplitude_relative_error_per_path':'Mean absolute relative IF amplitude error',
    'readout_trace_snr_db_per_path':'Mean additive-readout trace SNR [dB]',
    'waveform_deviation_ratio_db_per_path':'Mean waveform deviation ratio [dB]',
    'readout_if_snr_db_per_path':'Pooled additive-readout IF SNR [dB]',
    'relative_if_phase_rmse_deg_per_path':'Relative spatial IF phase RMSE [deg; reference is zero by definition]',
}
AXIS_LABELS = {
    'readout_gaussian':'Gaussian readout standard deviation',
    'rf_lo_amplitude':'RF LO log-amplitude standard deviation',
    'rf_signal_amplitude':'RF signal log-amplitude standard deviation',
    'rf_relative_phase':'RF relative envelope phase standard deviation',
    'laser_probe_frequency':'Probe frequency-noise standard deviation',
    'laser_coupling_frequency':'Coupling frequency-noise standard deviation',
    'laser_probe_intensity':'Probe relative intensity RMS',
    'laser_coupling_intensity':'Coupling relative intensity RMS',
    'electronic_current':'Electronic current ASD', 'johnson_current':'Johnson noise temperature',
}


def axis_spec(source, parameter):
    if source == 'johnson_current' and parameter == 'shunt_resistance_ohm':
        return 'ohm', True
    field, unit, positive = PARAMETERS[source]
    if parameter != field:
        raise PhysicsConfigurationError(f'NOISE-008: unsupported parameter {parameter!r} for {source}.')
    return unit, positive


def axis_label(source, parameter):
    return 'Johnson shunt resistance' if parameter == 'shunt_resistance_ohm' else AXIS_LABELS[source]


@dataclass(frozen=True)
class StudyStatistics:
    """Percentile whole-trial bootstrap; checkpoints reuse one physical campaign."""
    trial_checkpoints: tuple
    confidence_level: float
    bootstrap_replicates: int
    seed: int

    def __post_init__(self):
        if not isinstance(self.trial_checkpoints, tuple) or not self.trial_checkpoints:
            raise PhysicsConfigurationError('NOISE-009: nonempty immutable trial checkpoints required.')
        checks = tuple(_integer(v, 'NOISE-009 checkpoint', minimum=1) for v in self.trial_checkpoints)
        if any(b <= a for a,b in zip(checks,checks[1:])):
            raise PhysicsConfigurationError('NOISE-009: checkpoints must be strictly increasing.')
        level = _real(self.confidence_level, 'NOISE-009 confidence level')
        if not 0 < level < 1:
            raise PhysicsConfigurationError('NOISE-009: confidence level must lie strictly between 0 and 1.')
        reps = _integer(self.bootstrap_replicates, 'NOISE-009 bootstrap replicates', minimum=2)
        seed = _integer(self.seed, 'NOISE-009 statistical seed')
        if max(*checks,reps,seed) > 2**53-1:
            raise PhysicsConfigurationError('NOISE-009: integers must fit the exact JSON integer range.')
        object.__setattr__(self,'trial_checkpoints',checks)
        object.__setattr__(self,'confidence_level',level)
        object.__setattr__(self,'bootstrap_replicates',reps)
        object.__setattr__(self,'seed',seed)


def parse_statistics(document):
    if not isinstance(document,dict) or set(document) != {f.name for f in fields(StudyStatistics)}:
        raise PhysicsConfigurationError('NOISE-009: explicit checkpoints, confidence_level, bootstrap_replicates and seed required.')
    data = dict(document)
    if not isinstance(data['trial_checkpoints'],list):
        raise PhysicsConfigurationError('NOISE-009: trial_checkpoints must be a JSON array.')
    data['trial_checkpoints'] = tuple(data['trial_checkpoints'])
    return StudyStatistics(**data)


@dataclass(frozen=True)
class StudyDefinition:
    schema_version: int
    kind: str
    name: str
    source: str
    parameter: str
    values: tuple
    trials_per_point: int
    seed: int
    random_policy: str
    metrics: tuple
    statistics: object = None
    phase_reference_path: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'phase_reference_path',
                           _integer(self.phase_reference_path, 'NOISE-011 phase reference path'))
        if self.statistics is not None:
            if not isinstance(self.statistics,StudyStatistics):
                raise PhysicsConfigurationError('NOISE-009: statistics must be StudyStatistics or None.')
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise PhysicsConfigurationError('NOISE-008: study schema_version must be integer 1.')
        if not isinstance(self.kind, str) or self.kind != 'noise_strength_sweep':
            raise PhysicsConfigurationError('NOISE-008: unsupported study kind.')
        if not isinstance(self.name, str) or not self.name.strip():
            raise PhysicsConfigurationError('NOISE-008: a nonempty user study name is required.')
        if not isinstance(self.source, str) or self.source not in PARAMETERS:
            raise PhysicsConfigurationError('NOISE-008: source has no supported numeric noise-strength axis; no invented shot-noise sigma.')
        if not isinstance(self.parameter, str):
            raise PhysicsConfigurationError('NOISE-008: parameter must be a string.')
        _, positive = axis_spec(self.source,self.parameter)
        if not isinstance(self.values, tuple) or len(self.values) < 2:
            raise PhysicsConfigurationError('NOISE-008: immutable values with at least two points required.')
        values = tuple(_real(v, 'NOISE-008 axis value', positive=positive, nonnegative=not positive) for v in self.values)
        if any(b <= a for a, b in zip(values, values[1:])):
            raise PhysicsConfigurationError('NOISE-008: axis values must be strictly increasing; nonuniform spacing is allowed.')
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'seed', _integer(self.seed, 'NOISE-008 seed'))
        object.__setattr__(self, 'trials_per_point', _integer(self.trials_per_point, 'NOISE-008 trials', minimum=1))
        if self.statistics is not None and self.statistics.trial_checkpoints[-1] != self.trials_per_point:
            raise PhysicsConfigurationError('NOISE-009: final checkpoint must equal trials_per_point; execute maximum once.')
        if self.seed > 2**53-1 or self.trials_per_point > 2**53-1:
            raise PhysicsConfigurationError('NOISE-008: seed/trials must fit the exact JSON/JavaScript integer range (0..2^53-1).')
        if not isinstance(self.random_policy, str) or self.random_policy not in ('paired', 'independent'):
            raise PhysicsConfigurationError('NOISE-008: random_policy must be paired or independent.')
        if (not isinstance(self.metrics, tuple) or not self.metrics or
                any(not isinstance(m, str) or m not in METRICS for m in self.metrics) or
                len(set(self.metrics)) != len(self.metrics)):
            raise PhysicsConfigurationError('NOISE-008: nonempty unique supported metrics required.')


def parse_study(document):
    if not isinstance(document,dict) or not isinstance(document.get('kind'),str):
        raise PhysicsConfigurationError('NOISE-008: study kind must be an explicit string.')
    if isinstance(document,dict) and document.get('kind') == 'operating_point_sweep':
        from .operating_study import parse_operating_study
        return parse_operating_study(document)
    if isinstance(document,dict) and document.get('kind') == 'if_detection':
        from .detection_study import parse_detection_study
        return parse_detection_study(document)
    if isinstance(document,dict) and isinstance(document.get('kind'),str) and document['kind'] == 'signal_field_sweep':
        from .field_study import parse_field_study
        return parse_field_study(document)
    if isinstance(document,dict) and isinstance(document.get('kind'),str) and document['kind']=='noise_grid_convergence':
        from .noise_convergence import parse_convergence_study
        return parse_convergence_study(document)
    keys = {f.name for f in fields(StudyDefinition)}
    optional = {'statistics', 'phase_reference_path'}
    if (not isinstance(document, dict) or not keys-optional <= set(document) or not set(document) <= keys):
        raise PhysicsConfigurationError(f'NOISE-008: study requires {sorted(keys-optional)}; statistics and phase_reference_path are optional, extra fields are rejected.')
    data = dict(document)
    if data.get('statistics') is not None:
        data['statistics'] = parse_statistics(data['statistics'])
    for key in ('values', 'metrics'):
        if not isinstance(data[key], list):
            raise PhysicsConfigurationError(f'NOISE-008: {key} must be a JSON array.')
        data[key] = tuple(data[key])
    return StudyDefinition(**data)


def _owners(noise):
    if noise is None:
        return ()
    return (noise,) + ((noise.readout,) if noise.schema_version == 3 and noise.readout is not None else ())


def study_capabilities(profile):
    """Plain descriptors for a future GUI; no ARC, writes or simulation."""
    from .run_profile import RunProfile
    if not isinstance(profile,RunProfile):
        raise PhysicsConfigurationError('NOISE-008: capabilities require a RunProfile.')
    axes = []
    for owner in _owners(profile.noise):
        for source in owner.sources:
            if source.name in PARAMETERS:
                parameter, unit, positive = PARAMETERS[source.name]
                axes.append({'source': source.name, 'parameter': parameter, 'unit': unit,
                             'label':axis_label(source.name,parameter),
                             'minimum': 0., 'minimum_inclusive': not positive,
                             'configured_enabled': source.specification.enabled})
                if source.name == 'johnson_current':
                    axes.append({'source':source.name,'parameter':'shunt_resistance_ohm','unit':'ohm',
                                 'label':'Johnson shunt resistance',
                                 'minimum':0.,'minimum_inclusive':False,
                                 'configured_enabled':source.specification.enabled})
    from .noise_convergence import CONVERGENCE_METRICS
    from .temporal_doppler import temporal_doppler_capabilities
    from .atom_species import species_capabilities
    from .run_numerics import numerics_capabilities
    return {'schema_version': 1, 'numerics': numerics_capabilities(), 'atom': species_capabilities(), 'temporal_doppler': temporal_doppler_capabilities(), 'kinds': ('noise_strength_sweep','noise_grid_convergence','signal_field_sweep','if_detection','operating_point_sweep'), 'axes': axes,
            'operating_point_sweep': {'axes':({'field':'probe_detunings_mhz','unit':'MHz','label':'Applied probe detuning offset'},
                                             {'field':'cell_lengths_m','unit':'m','minimum_exclusive':0.,'label':'Cell length'}),
                                     'minimum_points_per_axis':2,'measurement_mode':'if_only','noise_schema':2,
                                     'selection':'maximum_predicted_if_snr', 'hardware_calibration_verified':False},
            'if_detection': {'statistic':'squared raw IF peak magnitude', 'path_selection':'predeclared path_index',
                             'calibration_false_alarm_probability':{'optional':True,'range_exclusive':(0.,1.),
                                                                  'legacy_none_meaning':'use p_false_alarm for calibration too'},
                             'axis':'signal.electric_field_vm', 'unit':'V/m', 'minimum_exclusive':0.,
                             'reference_stages':('calibration','null_validation'),
                             'independent_seed_domains':True, 'automatic_stopping':False,
                             'threshold_method':'finite corrected order statistic',
                             'interval_method':'Clopper-Pearson simultaneous Bonferroni',
                             'numerical_validation': {'optional':True,'enabled_switch':True,
                                 'controls':('rtol','atol','output_subdivision'),
                                 'acquisition_aperture_and_observation_fixed':True,
                                 'poisson_coupling':'diagnostic common inverse-CDF uniforms',
                                 'production_anchor':'unchanged RNG and solver; atomic traces compared bitwise'},
                             'sensitivity': {'optional':True, 'enabled_switch':True,
                                             'selection':'minimum_supported_sampled_field',
                                             'refinement_policy':'separate_explicit_run',
                                             'continuous_minimum_computed':False},
                             'mdf_nef_computed':False},
            'signal_field_sweep': {'axis': 'signal.electric_field_vm', 'unit': 'V/m', 'minimum_exclusive': 0.,
                                  'minimum_base_points': 3, 'linearity_model': 'affine',
                                  'derivative_check': 'midpoint_refinement', 'stochastic_sources_supported': False},
            'noise_grid_convergence':{'metrics':CONVERGENCE_METRICS,'output_samples_minimum':10,
                                     'coupling':'exact nested subsets of finest OU knots',
                                     'stochastic_readout_supported':False,'automatic_stopping':False},
            'metrics': METRICS, 'random_policies': ('paired', 'independent'),
            'metric_labels':dict(METRIC_LABELS), 'axis_labels':dict(AXIS_LABELS),
            'phase_reference': {'field': 'phase_reference_path', 'default': 0,
                                'convention': 'zero-based actual path index; reference error zero by definition'},
            'statistics': {'method':'whole_trial_percentile_bootstrap',
                           'configuration_fields':tuple(f.name for f in fields(StudyStatistics)),
                           'offline_reanalysis':True,'automatic_stopping':False},
            'cancellation': 'between atomic point jobs; no interruption of an active solver'}


def _replace_noise(noise, study, value, seed, *, nominal=False):
    sources = []
    for source in noise.sources:
        spec = source.specification
        if source.name == study.source and value is not None:
            spec = replace(spec, **{study.parameter: value})
        if nominal:
            spec = replace(spec, enabled=False)
        sources.append(replace(source, specification=spec))
    readout = _replace_noise(noise.readout, study, value, seed, nominal=nominal) if noise.readout is not None else None
    return replace(noise, sources=tuple(sources), seed=seed,
                   trials=1 if nominal else study.trials_per_point, readout=readout)


def effective_point_profile(profile, index=None):
    """Clone immutable configuration; None means all stochastic sources off."""
    from .run_profile import RunProfile
    if not isinstance(profile,RunProfile) or profile.noise is None:
        raise PhysicsConfigurationError('NOISE-008: point planning requires a RunProfile with noise.')
    study = profile.study
    if not isinstance(study, StudyDefinition):
        raise PhysicsConfigurationError('NOISE-008: explicit StudyDefinition required.')
    if index is None:
        seed, value = study.seed, None
    else:
        index = _integer(index, 'NOISE-008 point index')
        if index >= len(study.values):
            raise PhysicsConfigurationError('NOISE-008: point index outside study.')
        value = study.values[index]
        seed = study.seed if study.random_policy == 'paired' else int.from_bytes(
            sha256(f'NOISE-008-v1:{study.seed}:{index}'.encode('ascii')).digest()[:8], 'big') & (2**53-1)
    return replace(profile, study=None, noise=_replace_noise(profile.noise, study, value, seed, nominal=index is None))


def validate_study_profile(profile):
    """Validate EVERY effective point before dispatch or atomic initialization."""
    from .operating_study import OperatingStudy, validate_operating_profile
    if isinstance(getattr(profile,'study',None), OperatingStudy):
        return validate_operating_profile(profile)
    from .detection_study import DetectionStudy, validate_detection_profile
    if isinstance(getattr(profile,'study',None),DetectionStudy):
        return validate_detection_profile(profile)
    from .run_profile import RunProfile
    from .field_study import SignalFieldStudy, validate_field_study_profile
    if isinstance(getattr(profile,'study',None), SignalFieldStudy):
        return validate_field_study_profile(profile)
    from .noise_convergence import NoiseConvergenceStudy, validate_convergence_profile
    if isinstance(getattr(profile,'study',None),NoiseConvergenceStudy):
        return validate_convergence_profile(profile)
    if not isinstance(profile,RunProfile):
        raise PhysicsConfigurationError('NOISE-008: study requires a RunProfile.')
    study, noise = profile.study, profile.noise
    if not isinstance(study, StudyDefinition) or noise is None or not noise.enabled or profile.run_config is None:
        raise PhysicsConfigurationError('NOISE-008: temporal profile, enabled noise and explicit study required.')
    if study.phase_reference_path >= len(profile.run_config.positions_m):
        raise PhysicsConfigurationError('NOISE-011: phase reference path outside the actual array; no index replacement.')
    found = [(o,s) for o in _owners(noise) for s in o.sources if s.name == study.source]
    if len(found) != 1 or not found[0][0].enabled or not found[0][1].specification.enabled:
        raise PhysicsConfigurationError('NOISE-008: selected source must be uniquely declared and enabled.')
    if found[0][0].schema_version == 2 and not found[0][0].physical_readout_enabled:
        raise PhysicsConfigurationError('NOISE-008: selected detector source requires enabled PIN.')
    if study.random_policy == 'paired' and set(noise.active_sources) & {'photon_shot','dark_shot'}:
        raise PhysicsConfigurationError('NOISE-008: paired innovations unsupported with active Poisson sources; choose independent or disable counting sources.')
    from .rf_noise import validate_rf_noise_execution
    from .detector import validate_detector_sampling
    seeds = []
    for index in [None, *range(len(study.values))]:
        point = effective_point_profile(profile, index)
        validate_rf_noise_execution(point.noise, point.run_config)
        validate_detector_sampling(point.noise, point.run_config)
        if index is not None: seeds.append(point.noise.seed)
    if study.random_policy == 'independent' and len(set(seeds)) != len(seeds):
        raise PhysicsConfigurationError('NOISE-008: derived independent point seed collision; choose another explicit study seed.')


def plan_study(profile):
    """Pure GUI cost/validation preview; never reduce the requested workload."""
    from .operating_study import OperatingStudy, plan_operating_study
    if isinstance(getattr(profile,'study',None), OperatingStudy):
        return plan_operating_study(profile)
    from .detection_study import DetectionStudy, plan_detection_study
    if isinstance(getattr(profile,'study',None),DetectionStudy):
        return plan_detection_study(profile)
    from .field_study import SignalFieldStudy, plan_field_study
    if isinstance(getattr(profile,'study',None), SignalFieldStudy):
        return plan_field_study(profile)
    from .noise_convergence import NoiseConvergenceStudy,plan_convergence_study
    if isinstance(getattr(profile,'study',None),NoiseConvergenceStudy):
        return plan_convergence_study(profile)
    validate_study_profile(profile)
    jobs=[]
    for i,value in enumerate(profile.study.values):
        p=effective_point_profile(profile,i)
        evolutions=1+profile.study.trials_per_point if p.noise.stochastic_drive else 1
        jobs.append({'index':i,'value':value,'seed':p.noise.seed,
                     'trials':p.noise.trials,'atomic_evolutions_per_path':evolutions})
    paths=len(profile.run_config.positions_m)
    statistics = profile.study.statistics
    return {'schema_version':1,'points':jobs,'nominal_atomic_evolutions_per_path':1,
            'bootstrap_index_draws':None if statistics is None else
                len(jobs)*statistics.bootstrap_replicates*sum(statistics.trial_checkpoints),
            'bootstrap_index_matrix_bytes':None if statistics is None else
                8*statistics.bootstrap_replicates*statistics.trial_checkpoints[-1],
            'total_atomic_evolutions':paths*(1+sum(p['atomic_evolutions_per_path'] for p in jobs)),
            'paths':paths,'samples_per_trace':len(profile.run_config.tlist_us),
            'cost_note':'planned canonical evolutions; no duration/memory guarantee and no automatic shortening'}
