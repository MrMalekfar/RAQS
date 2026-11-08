"""Explicit two-axis diagnostic study; no solver tolerance or model changes."""
from dataclasses import dataclass, replace, fields
import numpy as np
from .laboratory import _integer, _real
from ..exceptions import PhysicsConfigurationError

CONVERGENCE_METRICS = ('if_response_absolute','if_response_relative','if_phase_deg',
                       'if_amplitude_relative','aoa_deg')


@dataclass(frozen=True)
class NoiseConvergenceStudy:
    schema_version: int
    kind: str
    name: str
    seed: int
    trials: int
    output_samples: tuple
    noise_steps_us: tuple
    metrics: tuple
    budgets: tuple

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1 or not isinstance(self.kind,str) or self.kind!='noise_grid_convergence':
            raise PhysicsConfigurationError('NOISE-010: study schema 1 and noise_grid_convergence kind required.')
        if not isinstance(self.name,str) or not self.name.strip():
            raise PhysicsConfigurationError('NOISE-010: nonempty name required.')
        seed=_integer(self.seed,'NOISE-010 seed'); trials=_integer(self.trials,'NOISE-010 trials',minimum=1)
        if max(seed,trials)>2**53-1:
            raise PhysicsConfigurationError('NOISE-010: seed/trials must fit JSON exact integer range.')
        if not isinstance(self.output_samples,tuple) or len(self.output_samples)<2:
            raise PhysicsConfigurationError('NOISE-010: at least two immutable output sample counts required.')
        counts=tuple(_integer(v,'NOISE-010 samples',minimum=10) for v in self.output_samples)
        if any(v>2**53-1 for v in counts) or any(b<=a for a,b in zip(counts,counts[1:])):
            raise PhysicsConfigurationError('NOISE-010: output counts must increase and fit exact integers.')
        if not isinstance(self.noise_steps_us,tuple) or len(self.noise_steps_us)<2:
            raise PhysicsConfigurationError('NOISE-010: at least two immutable noise steps required.')
        steps=tuple(_real(v,'NOISE-010 noise step [us]',positive=True) for v in self.noise_steps_us)
        if any(b>=a for a,b in zip(steps,steps[1:])):
            raise PhysicsConfigurationError('NOISE-010: noise steps must strictly decrease.')
        if (not isinstance(self.metrics,tuple) or not self.metrics or
                any(not isinstance(m,str) or m not in CONVERGENCE_METRICS for m in self.metrics) or
                len(set(self.metrics))!=len(self.metrics)):
            raise PhysicsConfigurationError('NOISE-010: unique supported metrics required.')
        if not isinstance(self.budgets,tuple):
            raise PhysicsConfigurationError('NOISE-010: immutable explicit metric budgets required; empty means unassessed.')
        parsed=[]
        for item in self.budgets:
            if not isinstance(item,tuple) or len(item)!=2 or not isinstance(item[0],str) or item[0] not in self.metrics:
                raise PhysicsConfigurationError('NOISE-010: budget key must be a selected metric.')
            parsed.append((item[0],_real(item[1],'NOISE-010 absolute comparison budget',nonnegative=True)))
        if len({k for k,v in parsed})!=len(parsed):
            raise PhysicsConfigurationError('NOISE-010: duplicate budget keys rejected.')
        for name,value in [('seed',seed),('trials',trials),('output_samples',counts),
                           ('noise_steps_us',steps),('budgets',tuple(parsed))]:
            object.__setattr__(self,name,value)


def parse_convergence_study(document):
    if not isinstance(document,dict) or set(document)!={f.name for f in fields(NoiseConvergenceStudy)}:
        raise PhysicsConfigurationError('NOISE-010: explicit study fields required; no defaults.')
    data=dict(document)
    for name in ('output_samples','noise_steps_us','metrics'):
        if not isinstance(data[name],list):
            raise PhysicsConfigurationError(f'NOISE-010: {name} must be a JSON array.')
        data[name]=tuple(data[name])
    if not isinstance(data['budgets'],dict):
        raise PhysicsConfigurationError('NOISE-010: budgets must be a JSON object.')
    data['budgets']=tuple(data['budgets'].items())
    return NoiseConvergenceStudy(**data)


def convergence_profile(profile, samples, step):
    """Clone one declared case, preserving duration, fields, source flags and DoA."""
    from .run_profile import RunProfile
    from .noise import NoiseConfig
    if (not isinstance(profile,RunProfile) or not isinstance(profile.study,NoiseConvergenceStudy) or
            profile.run_config is None or not isinstance(profile.noise,NoiseConfig) or profile.noise.schema_version!=3):
        raise PhysicsConfigurationError('NOISE-010: temporal convergence profile required.')
    study=profile.study
    samples=_integer(samples,'NOISE-010 case sample count',minimum=10)
    step=_real(step,'NOISE-010 case noise step',positive=True)
    if samples not in study.output_samples or step not in study.noise_steps_us:
        raise PhysicsConfigurationError('NOISE-010: case must use declared study grids.')
    try:
        config=replace(profile.run_config,tlist_us=tuple(float(v) for v in
                       np.linspace(0,profile.run_config.tlist_us[-1],samples)))
    except (MemoryError,ValueError,OverflowError) as exc:
        raise PhysicsConfigurationError(f'NOISE-010: cannot construct requested output grid {samples}; no shortening.') from exc
    readout=profile.noise.readout
    if readout is not None: readout=replace(readout,seed=study.seed,trials=study.trials)
    noise=replace(profile.noise,seed=study.seed,trials=study.trials,
                  drive=replace(profile.noise.drive,noise_step_us=step),readout=readout)
    return replace(profile,study=None,noise=noise,run_config=config)


def validate_convergence_profile(profile):
    from .run_profile import RunProfile
    from .noise import NoiseConfig
    if not isinstance(profile,RunProfile) or not isinstance(profile.study,NoiseConvergenceStudy):
        raise PhysicsConfigurationError('NOISE-010: explicit convergence RunProfile required.')
    noise=profile.noise
    if profile.run_config is None or not isinstance(noise,NoiseConfig) or noise.schema_version!=3 or not noise.stochastic_drive:
        raise PhysicsConfigurationError('NOISE-010: temporal schema-3 profile with active nonzero drive noise required.')
    if noise.effective_readout is not None and noise.effective_readout.readout_enabled:
        raise PhysicsConfigurationError('NOISE-010: stochastic readout changes sampling/bandwidth; disable its sources explicitly for drive-grid convergence.')
    fine=profile.study.noise_steps_us[-1]
    for step in profile.study.noise_steps_us:
        ratio=step/fine
        if not np.isfinite(ratio) or not np.isclose(ratio,round(ratio),rtol=1e-12,atol=0):
            raise PhysicsConfigurationError('NOISE-010: nested noise grids require integer coarse/fine step ratios.')
    from .rf_noise import validate_rf_noise_execution
    from .detector import validate_detector_sampling
    for samples in profile.study.output_samples:
        for step in profile.study.noise_steps_us:
            case=convergence_profile(profile,samples,step)
            case.run_config.validate_sampling(case.signed_if_mhz)
            validate_rf_noise_execution(case.noise,case.run_config)
            validate_detector_sampling(case.noise,case.run_config)


def plan_convergence_study(profile):
    validate_convergence_profile(profile)
    study=profile.study
    cases=len(study.output_samples)+len(study.noise_steps_us)-1
    return {'schema_version':1,'kind':study.kind,'cases':cases,'trials_per_case':study.trials,
            'canonical_evolutions':cases*(1+study.trials)*len(profile.run_config.positions_m),
            'noise_coupling':'one finest stationary OU realization per trial; coarse nodes are exact subsets',
            'output_coupling':'identical finest realization for every output-grid case',
            'budgets':dict(study.budgets),'automatic_stopping':False}
