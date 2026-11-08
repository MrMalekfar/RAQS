"""NUM-ODE-002 immutable temporal ODE controls, separate from quadrature."""
from dataclasses import dataclass, asdict
from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class TemporalODE:
    method: str
    rtol: float
    atol: float
    nsteps: int

    def __post_init__(self):
        if not isinstance(self.method,str) or self.method != 'bdf':
            raise PhysicsConfigurationError('NUM-ODE-002: only the established bdf method is supported.')
        for key in ('rtol','atol'):
            value = _real(getattr(self,key), 'NUM-ODE-002 '+key, positive=True)
            if value >= 1:
                raise PhysicsConfigurationError('NUM-ODE-002: ODE tolerances must be below one.')
            object.__setattr__(self,key,value)
        value = _integer(self.nsteps, 'NUM-ODE-002 nsteps', minimum=1)
        if value > 2**31-1:
            raise PhysicsConfigurationError('NUM-ODE-002: nsteps exceeds the integrator integer range.')
        object.__setattr__(self,'nsteps',value)

    def options(self):
        return {k:getattr(self,k) for k in ('method','rtol','atol','nsteps')} | {'normalize_output':False}


@dataclass(frozen=True)
class BDFOrderControls(TemporalODE):
    """Explicit maximum BDF order; never selected automatically after failure."""
    max_order: int

    def __post_init__(self):
        super().__post_init__()
        order = _integer(self.max_order,'DOP-CONV-001 maximum BDF order',minimum=1)
        if order > 5:
            raise PhysicsConfigurationError('DOP-CONV-001: BDF maximum order must be within 1..5.')
        object.__setattr__(self,'max_order',order)

    def options(self):
        return {**super().options(),'order':self.max_order}


def parse_temporal_ode(raw):
    from dataclasses import fields
    if not isinstance(raw,dict):
        raise PhysicsConfigurationError('NUM-ODE-002: temporal_ode must be a mapping.')
    cls = BDFOrderControls if 'max_order' in raw else TemporalODE
    if set(raw) != {f.name for f in fields(cls)}:
        raise PhysicsConfigurationError('NUM-ODE-002: complete typed temporal ODE fields required.')
    return cls(**raw)


@dataclass(frozen=True)
class RunNumerics:
    temporal_ode: TemporalODE

    def __post_init__(self):
        if not isinstance(self.temporal_ode,TemporalODE):
            raise PhysicsConfigurationError('NUM-ODE-002: typed temporal ODE controls required.')


def validate_profile_numerics(profile):
    declared = profile.numerics
    effective = getattr(profile.run_config, 'temporal_ode', None)
    if declared is not None and (not isinstance(declared,RunNumerics) or profile.run_config is None):
        raise PhysicsConfigurationError('NUM-ODE-002: numerics requires a temporal execution plan.')
    if effective != (None if declared is None else declared.temporal_ode):
        raise PhysicsConfigurationError('NUM-ODE-002: profile and execution ODE controls differ.')


def ode_receipt(controls, diagnostic=None):
    """Read effective defaults for metadata only; never inject them into a run."""
    if controls is None:
        from qutip.solver.integrator.scipy_integrator import IntegratorScipyBDF
        defaults = IntegratorScipyBDF.integrator_options
        options = dict(method='bdf',nsteps=5000,normalize_output=False,
                       rtol=defaults['rtol'],atol=defaults['atol'])
        source = 'legacy_qutip_defaults'
    else:
        options, source = controls.options(), 'profile'
    if diagnostic is not None:
        options.update(rtol=diagnostic.rtol,atol=diagnostic.atol,nsteps=diagnostic.nsteps)
        source = 'explicit_diagnostic_override'
    return {'configured': None if controls is None else asdict(controls),
            'effective': options,'source': source,'automatic_retry':False,
            'scope':'temporal mesolve only; not steady-state or velocity-quadrature tolerances'}


def numerics_capabilities():
    """Pure descriptors for CLI/GUI; no solver or ARC import."""
    return {'path':'numerics.temporal_ode','methods':['bdf'],
            'required':['method','rtol','atol','nsteps'],
            'tolerances':{'minimum_exclusive':0.,'maximum_exclusive':1.,'unit':'dimensionless density matrix'},
            'nsteps':{'minimum':1,'maximum':2**31-1,'unit':'internal steps between output points'},
            'max_order':{'optional':True,'minimum':1,'maximum':5,'scope':'BDF maximum order; omission preserves backend default'},
            'normalize_output':False,'omission':'preserve legacy QuTiP tolerances and nsteps=5000',
            'automatic_retry':False,'velocity_quadrature_controls_separate':True}
