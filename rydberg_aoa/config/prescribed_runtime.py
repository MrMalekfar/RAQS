"""Immutable consumer snapshot for the existing prescribed initialization seam."""

from dataclasses import dataclass, asdict, is_dataclass
from .laboratory import _real
from .laboratory_runtime import require_matching_applied_fields
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError


def _freeze(value):
    if is_dataclass(value):
        return _freeze(asdict(value))
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SensorRabi:
    probe: float
    coupling: float
    lo: float
    signal: float


@dataclass(frozen=True)
class PrescribedSensorRuntime:
    identity: tuple
    rabi: SensorRabi
    rates: tuple
    detunings: tuple
    rf_orientation: object


def capture_prescribed_runtime(ac, profile):
    """Read all published inputs/caches without cache repair or an ARC scan."""
    ac.require_atomic_physics_initialized()
    from .atom_species import require_profile_atom
    require_profile_atom(profile, ac)
    if not ac._INITIALIZED_REQUEST.is_prescribed:
        raise PhysicsConfigurationError('LAB-007: prescribed sensing requires a prescribed published runtime.')
    require_matching_applied_fields(ac, profile)
    declared = profile.request.resolution_request()
    if ac._INITIALIZED_REQUEST != declared:
        raise PhysicsConfigurationError('LAB-007: prescribed declaration differs from the published ladder.')
    key = ac._resolution_cache_key(ac._INITIALIZED_REQUEST)
    caches = (ac._ATOMIC_LEVELS_CACHE, ac._DIPOLE_MATRIX_CACHE, ac._DETUNING_CACHE, ac._OPTIMIZATION_RESULTS_CACHE)
    if any(key not in cache for cache in caches):
        raise PhysicsCalculationError('LAB-007: missing prescribed runtime cache; no repair or replacement.')
    rabi = SensorRabi(*(_real(getattr(ac, name), name, nonnegative=True) for name in
                       ('OMEGA_P', 'OMEGA_C', 'OMEGA_RF_BASE', 'OMEGA_SIGNAL_RF_BASE')))
    rates = tuple(_real(getattr(ac, name), name, nonnegative=True) for name in ('GAMMA_21', 'GAMMA_32', 'GAMMA_43'))
    detunings = tuple(_real(value, 'prescribed detuning') for value in ac.get_cached_detunings())
    published = tuple(_real(getattr(ac, name), name) for name in ('DELTA_P', 'DELTA_C', 'DELTA_RF'))
    if detunings != published:
        raise PhysicsCalculationError('LAB-007: prescribed detuning cache differs from published values.')
    if ac.ATOMIC_LEVELS != ac._ATOMIC_LEVELS_CACHE[key]:
        raise PhysicsCalculationError('LAB-007: prescribed level cache differs from published levels.')
    identity = (id(ac.ATOM), ac._INITIALIZED_RESOLUTION_IDENTITY,
                _freeze(ac.ATOMIC_LEVELS), tuple(_freeze(cache[key]) for cache in caches),
                _freeze(ac.LASER_PARAMS), _freeze(ac.RF_PARAMS), _freeze(ac.SIGNAL_RF_PARAMS),
                ac.params.OMEGA_LO_BASE, ac.params.OMEGA_S_BASE)
    # Legacy prescribed detuning calculations use wavelength centers only.
    # Version-2 profile offsets are explicit consumer inputs, added once here;
    # the canonical prescribed cache and publication remain unchanged.
    if profile.environment is not None:
        import math
        detunings = (detunings[0] + 2*math.pi*profile.probe.detuning_mhz,
                     detunings[1] + 2*math.pi*profile.coupling.detuning_mhz, detunings[2])
        if not all(math.isfinite(v) for v in detunings):
            raise PhysicsCalculationError('LAB-007: prescribed applied offsets overflowed.')
    return PrescribedSensorRuntime(identity, rabi, rates, detunings, ac.get_rf_orientation())
