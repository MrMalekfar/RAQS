"""Explicit thermal and spectrum declarations; no atomic imports during parsing."""

from dataclasses import dataclass
import numpy as np
from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


def _flag(value, name):
    if type(value) is not bool:
        raise PhysicsConfigurationError(f'LAB-007: {name} must be boolean.')
    return value


@dataclass(frozen=True)
class ProfileEnvironment:
    temperature_k: float
    cell_length_m: float
    rate_source: str
    linewidth_rates_per_us: tuple
    linewidth_enabled: bool
    collision_enabled: bool
    transit_enabled: bool

    def __post_init__(self):
        for name in ('temperature_k', 'cell_length_m'):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        if self.rate_source not in ('explicit', 'thermal_budget'):
            raise PhysicsConfigurationError('LAB-007: rate_source must be explicit or thermal_budget.')
        if not isinstance(self.linewidth_rates_per_us, tuple) or len(self.linewidth_rates_per_us) != 3:
            raise PhysicsConfigurationError('LAB-007: three linewidth contributions [1/us] required.')
        object.__setattr__(self, 'linewidth_rates_per_us', tuple(
            _real(v, 'linewidth contribution [1/us]', nonnegative=True) for v in self.linewidth_rates_per_us))
        for name in ('linewidth_enabled', 'collision_enabled', 'transit_enabled'):
            _flag(getattr(self, name), name)
        if self.rate_source == 'explicit' and (any(self.linewidth_rates_per_us) or
                self.linewidth_enabled or self.collision_enabled or self.transit_enabled):
            raise PhysicsConfigurationError('LAB-007: explicit total rates cannot be combined with thermal component declarations.')


@dataclass(frozen=True)
class UniformProbeScan:
    """Uniform probe offsets [MHz], including both declared endpoints.

    This creates solver input points, not interpolation of a plotted curve.
    Unrepresentable grids fail without reducing the requested point count.
    """
    start_offset_mhz: float
    stop_offset_mhz: float
    point_count: int

    def offsets_mhz(self):
        start = _real(self.start_offset_mhz, 'probe_scan.start_offset_mhz')
        stop = _real(self.stop_offset_mhz, 'probe_scan.stop_offset_mhz')
        count = _integer(self.point_count, 'probe_scan.point_count', minimum=2)
        if start >= stop:
            raise PhysicsConfigurationError('LAB-007: probe_scan requires start_offset_mhz < stop_offset_mhz.')
        if count > np.iinfo(np.intp).max:
            raise PhysicsConfigurationError('LAB-007: probe_scan.point_count exceeds platform array-index capacity.')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                grid = np.linspace(start, stop, count, endpoint=True)
            if not np.isfinite(grid).all() or not (grid[1:] > grid[:-1]).all():
                raise PhysicsConfigurationError('LAB-007: probe_scan grid must be finite and strictly increasing at floating-point precision.')
            return tuple(grid)
        except (FloatingPointError, OverflowError, ValueError, TypeError, MemoryError) as exc:
            raise PhysicsConfigurationError('LAB-007: cannot construct probe_scan grid; no substitute point count or range.') from exc


@dataclass(frozen=True)
class NonuniformProbeScan:
    """Resample an explicit offset template at equally spaced template indices.

    Anchor locations [MHz] define sampling density, not spectrum values.
    With point_count == len(anchors), the original points are returned exactly.
    Other counts preserve endpoints; interior anchors need not be sampled.
    Every resulting offset is an input to the actual spectrum solver.
    """
    anchor_offsets_mhz: tuple
    point_count: int

    def offsets_mhz(self):
        if not isinstance(self.anchor_offsets_mhz, tuple) or len(self.anchor_offsets_mhz) < 2:
            raise PhysicsConfigurationError('LAB-007: probe_scan requires at least two anchor_offsets_mhz.')
        anchors = tuple(_real(v, 'probe_scan.anchor offset [MHz]') for v in self.anchor_offsets_mhz)
        if any(b <= a for a, b in zip(anchors, anchors[1:])):
            raise PhysicsConfigurationError('LAB-007: probe_scan anchors must be strictly increasing.')
        count = _integer(self.point_count, 'probe_scan.point_count', minimum=2)
        if count > np.iinfo(np.intp).max:
            raise PhysicsConfigurationError('LAB-007: probe_scan.point_count exceeds platform array-index capacity.')
        if count == len(anchors):
            return anchors
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                # Explicit arithmetic, avoiding np.interp's silent overflow repair.
                index = np.linspace(0., len(anchors) - 1, count, endpoint=True)
                left = np.minimum(np.floor(index).astype(np.intp), len(anchors) - 2)
                fraction = index - left
                values = np.asarray(anchors)
                grid = values[left] + fraction * (values[left + 1] - values[left])
                grid[0], grid[-1] = anchors[0], anchors[-1]
            if not np.isfinite(grid).all() or not (grid[1:] > grid[:-1]).all():
                raise PhysicsConfigurationError('LAB-007: probe_scan grid must be finite and strictly increasing at floating-point precision.')
            return tuple(grid)
        except (FloatingPointError, OverflowError, ValueError, TypeError, MemoryError) as exc:
            raise PhysicsConfigurationError('LAB-007: cannot construct nonuniform probe_scan; no substitute count or template.') from exc


@dataclass(frozen=True)
class ProfileSpectrum:
    probe_offsets_mhz: tuple
    doppler_enabled: bool
    method: str
    n_classes: int
    n_sigma: float
    rtol: float
    atol: float
    counter_propagating: bool
    analytical_enabled: bool
    steady_state_backend: str | None = None

    def __post_init__(self):
        if self.steady_state_backend is not None and (type(self.steady_state_backend) is not str
                                                     or self.steady_state_backend not in ('direct', 'qutip')):
            raise PhysicsConfigurationError('LAB-007: spectrum steady_state_backend must be direct or qutip.')
        if not isinstance(self.probe_offsets_mhz, tuple) or not self.probe_offsets_mhz:
            raise PhysicsConfigurationError('LAB-007: declare nonempty spectrum offsets [MHz].')
        object.__setattr__(self, 'probe_offsets_mhz', tuple(_real(v, 'spectrum offset [MHz]') for v in self.probe_offsets_mhz))
        if self.method not in ('adaptive', 'gauss_hermite', 'uniform'):
            raise PhysicsConfigurationError('LAB-007: unsupported Doppler method.')
        object.__setattr__(self, 'n_classes', _integer(self.n_classes, 'velocity classes', minimum=1))
        if self.method == 'uniform' and self.n_classes < 2:
            raise PhysicsConfigurationError('LAB-007: uniform Maxwell sampling needs >=2 classes to span both velocity signs.')
        for name in ('n_sigma', 'rtol', 'atol'):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        for name in ('doppler_enabled', 'counter_propagating', 'analytical_enabled'):
            _flag(getattr(self, name), name)
