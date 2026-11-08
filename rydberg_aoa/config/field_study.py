"""NOISE-012 deterministic signal-field study; no ARC or input mutation."""
from dataclasses import dataclass, fields, replace
from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class SignalFieldStudy:
    schema_version: int
    kind: str
    name: str
    values: tuple
    linear_region_v_per_m: tuple
    linearity_model: str
    linearity_max_relative_error: float
    derivative_check: str
    derivative_max_relative_difference: float

    def __post_init__(self):
        if (type(self.schema_version) is not int or self.schema_version != 1 or
                not isinstance(self.kind, str) or self.kind != 'signal_field_sweep'):
            raise PhysicsConfigurationError('NOISE-012: field study schema 1 and kind signal_field_sweep required.')
        if not isinstance(self.name, str) or not self.name.strip():
            raise PhysicsConfigurationError('NOISE-012: nonempty study name required.')
        if not isinstance(self.values, tuple) or len(self.values) < 3:
            raise PhysicsConfigurationError('NOISE-012: >=3 immutable positive signal fields [V/m] required.')
        values = tuple(_real(v, 'NOISE-012 field [V/m]', positive=True) for v in self.values)
        if any(b <= a for a, b in zip(values, values[1:])):
            raise PhysicsConfigurationError('NOISE-012: fields must be strictly increasing; nonuniform spacing allowed.')
        if not isinstance(self.linear_region_v_per_m, tuple) or len(self.linear_region_v_per_m) != 2:
            raise PhysicsConfigurationError('NOISE-012: explicit two-endpoint linear region required.')
        region = tuple(_real(v, 'NOISE-012 linear-region field', positive=True) for v in self.linear_region_v_per_m)
        if region[0] >= region[1] or any(v not in values for v in region) or sum(region[0] <= v <= region[1] for v in values) < 3:
            raise PhysicsConfigurationError('NOISE-012: linear region endpoints must be base-grid values enclosing >=3 points.')
        if (not isinstance(self.linearity_model, str) or self.linearity_model != 'affine' or
                not isinstance(self.derivative_check, str) or self.derivative_check != 'midpoint_refinement'):
            raise PhysicsConfigurationError('NOISE-012: explicit affine fit and midpoint_refinement required; no substitute method.')
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'linear_region_v_per_m', region)
        for name in ('linearity_max_relative_error', 'derivative_max_relative_difference'):
            object.__setattr__(self, name, _real(getattr(self, name), 'NOISE-012 '+name, nonnegative=True))
        # Floating neighbours without a representable midpoint cannot support this diagnostic.
        self.refined_values()

    def refined_values(self):
        output = []
        for a, b in zip(self.values, self.values[1:]):
            midpoint = a+(b-a)*.5
            if not a < midpoint < b:
                raise PhysicsConfigurationError('NOISE-012: field interval has no representable interior midpoint.')
            output.extend((a, midpoint))
        return tuple(output)+(self.values[-1],)


def parse_field_study(document):
    keys = {f.name for f in fields(SignalFieldStudy)}
    if not isinstance(document, dict) or set(document) != keys:
        raise PhysicsConfigurationError(f'NOISE-012: field study requires exactly {sorted(keys)}.')
    data = dict(document)
    for key in ('values', 'linear_region_v_per_m'):
        if not isinstance(data[key], list):
            raise PhysicsConfigurationError(f'NOISE-012: {key} must be a JSON array.')
        data[key] = tuple(data[key])
    return SignalFieldStudy(**data)


def field_point_profile(profile, index):
    """Update both applied signal and the experimental immutable request."""
    from .run_profile import RunProfile
    if not isinstance(profile,RunProfile) or not isinstance(profile.study, SignalFieldStudy):
        raise PhysicsConfigurationError('NOISE-012: explicit SignalFieldStudy required.')
    index = _integer(index, 'NOISE-012 point index')
    grid = profile.study.refined_values()
    if index >= len(grid):
        raise PhysicsConfigurationError('NOISE-012: point outside declared/refined field grid.')
    signal = replace(profile.signal, electric_field_vm=grid[index])
    request = replace(profile.request, signal=signal) if profile.mode == 'experimental' else profile.request
    return replace(profile, signal=signal, request=request, study=None)


def validate_field_study_profile(profile):
    from .run_profile import RunProfile
    if not isinstance(profile, RunProfile) or not isinstance(profile.study, SignalFieldStudy) or profile.run_config is None:
        raise PhysicsConfigurationError('NOISE-012: explicit temporal profile and SignalFieldStudy required.')
    if profile.mode == 'experimental' and any(getattr(profile.request, k) != getattr(profile, k) for k in ('probe','coupling','lo','signal')):
        raise PhysicsConfigurationError('NOISE-012: profile applied fields disagree with experimental request.')
    if any(v == 0 for v in (profile.probe.power_mw, profile.coupling.power_mw, profile.lo.electric_field_vm)):
        raise PhysicsConfigurationError('NOISE-012: nonzero optical and LO fields required for sensing.')
    noise = profile.noise
    if noise is not None:
        if noise.trials != 1:
            raise PhysicsConfigurationError('NOISE-012: deterministic study requires noise.trials=1; no silent reduction.')
        if noise.active_sources:
            raise PhysicsConfigurationError('NOISE-012: stochastic sources must be explicitly disabled; PIN may remain enabled.')
    from .rf_noise import validate_rf_noise_execution
    from .detector import validate_detector_sampling
    validate_rf_noise_execution(noise, profile.run_config)
    validate_detector_sampling(noise, profile.run_config)
    if noise is not None and noise.physical_readout_enabled and profile.environment is None:
        raise PhysicsConfigurationError('NOISE-012: PIN requires explicit optical environment.')
    # Validate derived fields before any dispatch, preserving frequencies/q/bounds.
    for i in range(len(profile.study.refined_values())):
        point = field_point_profile(profile, i)
        point.run_config.validate_sampling(point.signed_if_mhz)


def plan_field_study(profile):
    validate_field_study_profile(profile)
    grid = profile.study.refined_values()
    paths = len(profile.run_config.positions_m)
    return {'schema_version': 1, 'kind': 'signal_field_sweep', 'axis_unit': 'V/m',
            'points': [{'index': i, 'value': value, 'role': 'base' if i%2 == 0 else 'midpoint',
                        'atomic_evolutions_per_path': 1} for i, value in enumerate(grid)],
            'paths': paths, 'samples_per_trace': len(profile.run_config.tlist_us),
            'total_atomic_evolutions': paths*len(grid), 'nominal_atomic_evolutions_per_path': 0,
            'cost_note': 'one deterministic canonical run per field; midpoints are actual runs, no automatic refinement'}
