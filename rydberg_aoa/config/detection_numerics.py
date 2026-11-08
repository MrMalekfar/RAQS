"""NOISE-016 bounded diagnostic controls; never implicit production defaults."""
from dataclasses import dataclass, fields
from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class SolverDiagnostic:
    """Explicit diagnostic accuracy/work limits; production defaults stay fixed."""
    name: str
    rtol: float
    atol: float
    output_subdivision: int
    nsteps: int = 5000

    def __post_init__(self):
        if not isinstance(self.name,str) or not self.name.strip():
            raise PhysicsConfigurationError('NOISE-016: diagnostic case name required.')
        for key in ('rtol','atol'):
            value=_real(getattr(self,key),'NOISE-016 '+key,positive=True)
            if value >= 1:
                raise PhysicsConfigurationError('NOISE-016: tolerances must be strictly between zero and one.')
            object.__setattr__(self,key,value)
        subdivision=_integer(self.output_subdivision,'NOISE-016 subdivision',minimum=1)
        if subdivision>2**53-1:raise PhysicsConfigurationError('NOISE-016: subdivision must fit exact JSON integers.')
        object.__setattr__(self,'output_subdivision',subdivision)
        nsteps=_integer(self.nsteps,'NUM-ODE-001 nsteps',minimum=1)
        if nsteps > 2**31-1:
            raise PhysicsConfigurationError('NUM-ODE-001: nsteps exceeds the signed 32-bit integrator limit.')
        object.__setattr__(self,'nsteps',nsteps)


@dataclass(frozen=True)
class DetectionNumerics:
    enabled: bool
    cases: tuple
    if_absolute_budget: float
    if_relative_budget: float
    decision_fraction_budget: float
    probability_difference_budget: float
    require_same_selected_field: bool

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.require_same_selected_field) is not bool:
            raise PhysicsConfigurationError('NOISE-016: numerical flags must be boolean.')
        if (not isinstance(self.cases,tuple) or not self.cases or
                any(not isinstance(c,SolverDiagnostic) for c in self.cases) or
                len({c.name for c in self.cases}) != len(self.cases) or
                any(c.name in ('production_anchor','coupled_baseline') for c in self.cases)):
            raise PhysicsConfigurationError('NOISE-016: unique explicit diagnostic cases required.')
        for key in ('if_absolute_budget','if_relative_budget','decision_fraction_budget','probability_difference_budget'):
            value=_real(getattr(self,key),'NOISE-016 '+key,nonnegative=True)
            if key.endswith('fraction_budget') or key=='probability_difference_budget':
                if value > 1:raise PhysicsConfigurationError('NOISE-016: fraction/probability budgets must be <=1.')
            object.__setattr__(self,key,value)


def parse_detection_numerics(document):
    if not isinstance(document,dict) or set(document) != {f.name for f in fields(DetectionNumerics)}:
        raise PhysicsConfigurationError('NOISE-016: complete numerical-validation design required; no implicit budgets.')
    if not isinstance(document['cases'],list):
        raise PhysicsConfigurationError('NOISE-016: cases must be a JSON array.')
    cases=[]
    for raw in document['cases']:
        allowed={f.name for f in fields(SolverDiagnostic)}
        if (not isinstance(raw,dict) or not (allowed-{'nsteps'} <= set(raw))
                or not set(raw) <= allowed):
            raise PhysicsConfigurationError('NOISE-016: complete solver case required.')
        cases.append(SolverDiagnostic(**raw))
    return DetectionNumerics(**{**document,'cases':tuple(cases)})


def numerical_plan(profile):
    from .detection_study import detection_jobs
    d=profile.study; options=d.numerical_validation
    if options is None or not options.enabled:return {'enabled':False}
    paths=len(profile.run_config.positions_m); jobs=detection_jobs(d)
    runs=2+len(options.cases)
    evolutions=paths*sum(1+j['trials'] if profile.noise.stochastic_drive else 1 for j in jobs)
    return {'enabled':True,'complete_detection_campaigns':runs,'total_trials':runs*sum(j['trials'] for j in jobs),
            'canonical_evolutions':runs*evolutions,'production_anchor':True,
            'acquisition_samples':len(profile.run_config.tlist_us),
            'certification_samples_per_case':[(len(profile.run_config.tlist_us)-1)*c.output_subdivision+1 for c in options.cases],
            'acquisition_and_aperture':'fixed; subdivision changes solver/certification grid only',
            'readout_coupling':'common Gaussian draws; common Poisson quantiles in diagnostic campaigns',
            'observation_window_and_transient':'fixed; their variation is a different measurement study',
            'automatic_stopping':False,'continuum_error_bound':False}


def validate_diagnostic_controls(solver,quantiles):
    if (solver is not None and not isinstance(solver,SolverDiagnostic)) or type(quantiles) is not bool:
        raise PhysicsConfigurationError('NOISE-016: typed diagnostic and boolean readout coupling required.')


def validate_grid_subdivision(times,subdivision):
    """O(acquisition samples) preview; never allocate the refined grid here."""
    import numpy as np
    subdivision=_integer(subdivision,'NOISE-016 subdivision',minimum=1)
    a=np.asarray(times)
    if a.ndim!=1 or len(a)<2 or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() or np.any(np.diff(a)<=0):
        raise PhysicsConfigurationError('NOISE-016: finite increasing acquisition grid required.')
    if subdivision>1:
        gap=np.diff(a)/subdivision
        spacing=np.maximum(np.abs(np.spacing(a[:-1])),np.abs(np.spacing(a[1:])))
        if np.any(gap<spacing):
            raise PhysicsConfigurationError('NOISE-016: requested nested grid exceeds floating-point resolution; no rounded substitute.')
