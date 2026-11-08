"""DOP-TD-001: explicit optical-only 1-D Maxwell ensemble controls."""
from dataclasses import dataclass, fields
from .laboratory import _integer, _real
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class TemporalDoppler:
    enabled: bool
    method: str
    initial_classes: int
    max_classes: int
    n_sigma: float
    rtol: float
    atol: float
    counter_propagating: bool

    def __post_init__(self):
        for name in ('enabled', 'counter_propagating'):
            if type(getattr(self, name)) is not bool:
                raise PhysicsConfigurationError(f'DOP-TD-001: {name} must be boolean.')
        if (not isinstance(self.method,str) or
                (self.method not in ('gauss_hermite', 'uniform') and
                 not (self.method == 'sinh' and isinstance(self,SinhTemporalDoppler)))):
            raise PhysicsConfigurationError('DOP-TD-001: declare a supported typed velocity method.')
        for name in ('initial_classes', 'max_classes'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=3))
        if self.initial_classes % 2 != 1 or self.max_classes < 4*self.initial_classes-3:
            raise PhysicsConfigurationError('DOP-TD-001: odd initial_classes and budget for two refinements required.')
        for name in ('n_sigma', 'rtol', 'atol'):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        if self.rtol >= 1:
            raise PhysicsConfigurationError('DOP-TD-001: rtol must be strictly between zero and one.')


@dataclass(frozen=True)
class SinhTemporalDoppler(TemporalDoppler):
    """Declared m/s scale for a transformed, truncated Maxwell quadrature."""
    velocity_scale_ms: float

    def __post_init__(self):
        super().__post_init__()
        if self.method != 'sinh':
            raise PhysicsConfigurationError('DOP-CONV-001: sinh controls require method=sinh.')
        object.__setattr__(self,'velocity_scale_ms',
            _real(self.velocity_scale_ms,'DOP-CONV-001 velocity scale [m/s]',positive=True))


@dataclass(frozen=True)
class VelocityErrorBudget:
    """Coherence/IF peak-amplitude bounds; phase in rad, never degrees."""
    trace_absolute: float
    trace_relative: float
    if_absolute: float
    if_relative: float
    if_phase_rad: float
    phase_amplitude_floor: float

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _real(getattr(self, f.name),
                'DOP-PROFILE-001 '+f.name, positive=True))
        if self.trace_relative >= 1 or self.if_relative >= 1 or self.if_phase_rad >= 1:
            raise PhysicsConfigurationError('DOP-PROFILE-001: relative/phase budgets must be below one.')


@dataclass(frozen=True)
class AdaptiveTemporalDoppler:
    """Positive local Simpson integration plus independent leaf quadrature.

    No fixed class count. Limits include all unique native velocity solves
    per sensor; elapsed time cannot preempt a native solver call.
    """
    enabled: bool
    method: str
    counter_propagating: bool
    n_sigma: float
    velocity_scale_ms: float
    initial_y_intervals: int
    hint_offsets_ms: tuple
    max_panels: int
    mass_relative_tolerance: float
    independent_legendre_order: int
    max_new_solves: int
    max_wall_seconds: float
    budgets: VelocityErrorBudget

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.counter_propagating) is not bool:
            raise PhysicsConfigurationError('DOP-PROFILE-001: boolean switches required.')
        if self.method != 'adaptive_simpson':
            raise PhysicsConfigurationError('DOP-PROFILE-001: method must be adaptive_simpson.')
        for name in ('n_sigma', 'velocity_scale_ms', 'mass_relative_tolerance', 'max_wall_seconds'):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        if self.mass_relative_tolerance >= 1:
            raise PhysicsConfigurationError('DOP-PROFILE-001: mass tolerance must be below one.')
        for name, minimum in (('initial_y_intervals', 2), ('max_panels', 4),
                              ('independent_legendre_order', 2), ('max_new_solves', 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        if self.initial_y_intervals % 2 or self.max_panels < self.initial_y_intervals+2:
            raise PhysicsConfigurationError('DOP-PROFILE-001: even initial intervals and two refinements required.')
        if not isinstance(self.hint_offsets_ms, (tuple, list)) or not self.hint_offsets_ms:
            raise PhysicsConfigurationError('DOP-PROFILE-001: nonempty hint offsets [m/s] required.')
        offsets = tuple(_real(v, 'hint offset [m/s]') for v in self.hint_offsets_ms)
        if 0. not in offsets or tuple(sorted(set(offsets))) != offsets:
            raise PhysicsConfigurationError('DOP-PROFILE-001: sorted unique offsets including zero required.')
        object.__setattr__(self, 'hint_offsets_ms', offsets)
        if isinstance(self.budgets, dict) and set(self.budgets) == {f.name for f in fields(VelocityErrorBudget)}:
            object.__setattr__(self, 'budgets', VelocityErrorBudget(**self.budgets))
        if not isinstance(self.budgets, VelocityErrorBudget):
            raise PhysicsConfigurationError('DOP-PROFILE-001: complete typed error budgets required.')


def parse_temporal_doppler(raw):
    if not isinstance(raw,dict):
        raise PhysicsConfigurationError('DOP-TD-001: temporal_doppler must be a mapping.')
    method=raw.get('method')
    classes = {'adaptive_simpson': AdaptiveTemporalDoppler, 'sinh': SinhTemporalDoppler}
    cls = classes.get(method, TemporalDoppler) if isinstance(method, str) else TemporalDoppler
    if set(raw) != {f.name for f in fields(cls)}:
        raise PhysicsConfigurationError('DOP-CONV-001: complete method-specific velocity controls required.')
    return cls(**raw)


@dataclass(frozen=True)
class ResolvedTemporalDoppler:
    controls: TemporalDoppler | AdaptiveTemporalDoppler
    temperature_k: float
    atom_mass_kg: float
    probe_wavelength_m: float
    coupling_wavelength_m: float

    def __post_init__(self):
        if not isinstance(self.controls, (TemporalDoppler, AdaptiveTemporalDoppler)) or not self.controls.enabled:
            raise PhysicsConfigurationError('DOP-TD-001: enabled typed controls required.')
        for f in fields(self):
            if f.name != 'controls':
                object.__setattr__(self, f.name, _real(getattr(self, f.name), f.name, positive=True))


def temporal_doppler_capabilities():
    """JSON-compatible descriptors for CLI/GUI clients, without physics imports."""
    return {'methods': ['gauss_hermite', 'uniform', 'sinh', 'adaptive_simpson'], 'velocity_dimension': 1,
            'adaptive_simpson': {'required': [f.name for f in fields(AdaptiveTemporalDoppler)],
                'budgets_required': [f.name for f in fields(VelocityErrorBudget)],
                'units': {'n_sigma': 'velocity standard deviations', 'velocity_scale_ms': 'm/s',
                          'hint_offsets_ms': 'm/s', 'max_wall_seconds': 's',
                          'trace_absolute': 'coherence', 'if_absolute': 'coherence peak IF',
                          'phase_amplitude_floor': 'coherence peak IF', 'if_phase_rad': 'rad'},
                'refinement': 'split leaf with largest raw full/AC/IF/mass defect; two global passes',
                'independent_check': 'Gauss-Legendre on every final leaf; mandatory',
                'work_limits_scope': 'per path, includes independent check; checked before/after native solves',
                'cache': 'exact velocities, in-memory within one path only',
                'qualification': 'conditional finite Gaussian domain at declared ODE/output grid'},
            'sinh':{'velocity_scale_ms':{'required':True,'minimum_exclusive':0.,'unit':'m/s'},
                    'map':'v=scale*sinh(y); same declared +/-n_sigma sigma domain',
                    'weights':'Gaussian density times Jacobian and trapezoidal endpoint factors'},
            'switch': 'enabled', 'class_schedule': 'n -> 2*n-1; two consecutive accepted refinements',
            'class_schedule_scope': 'gauss_hermite, uniform and sinh only',
            'initial_classes_minimum': 3, 'initial_classes_odd': True,
            'temperature_source': 'environment.temperature_k', 'mass_source': 'resolved ARC atom',
            'rtol_units': 'dimensionless', 'atol_units': 'coherence',
            'rf_doppler_included': False, 'propagation_included': False,
            'failure_policy': 'abort on failed class or unconverged ensemble'}


def validate_temporal_profile(profile):
    controls = profile.temporal_doppler
    if controls is not None:
        if not isinstance(controls, (TemporalDoppler, AdaptiveTemporalDoppler)):
            raise PhysicsConfigurationError('DOP-TD-001: typed temporal Doppler controls required.')
        if controls.enabled and (profile.environment is None or profile.run_config is None):
            raise PhysicsConfigurationError('DOP-TD-001: explicit environment and temporal execution required.')
