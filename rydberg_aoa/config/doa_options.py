"""Explicit method-specific DoA options for the single-RF-source profile model."""

from dataclasses import dataclass
import numpy as np

from .laboratory import _integer, _real
from ..exceptions import PhysicsConfigurationError

SEARCH_FREE = ('esprit', 'root_music')
SUBSPACE = ('music', 'esprit', 'root_music')
LOADING = ('mvdr', 'capon')


@dataclass(frozen=True)
class ProfileDOAOptions:
    scan_azimuth_deg: tuple = None
    n_sources: int = None
    loading_rel: float = None

    def validate(self, method):
        if method not in ('phase_beamforming', 'bartlett', 'music', 'mvdr', 'capon', 'esprit', 'root_music'):
            raise PhysicsConfigurationError(f'DOA-PROFILE-001: unknown method {method!r}; no substitute.')
        if method in SEARCH_FREE:
            if self.scan_azimuth_deg is not None:
                raise PhysicsConfigurationError('DOA-PROFILE-001: search-free methods reject scan grids.')
        else:
            if not isinstance(self.scan_azimuth_deg, tuple) or len(self.scan_azimuth_deg) < 2:
                raise PhysicsConfigurationError('DOA-PROFILE-001: declare >=2 immutable scan angles [deg].')
            grid = tuple(_real(v, 'DoA scan angle [deg]') for v in self.scan_azimuth_deg)
            if not all(-90 <= v <= 90 for v in grid) or any(b <= a for a, b in zip(grid, grid[1:])):
                raise PhysicsConfigurationError('DOA-PROFILE-001: scan angles must increase inside [-90,90] deg.')
        if method in SUBSPACE:
            if _integer(self.n_sources, 'DoA n_sources', minimum=1) != 1:
                raise PhysicsConfigurationError('DOA-PROFILE-001: this profile simulates one RF source; n_sources must be 1.')
        elif self.n_sources is not None:
            raise PhysicsConfigurationError('DOA-PROFILE-001: n_sources is unused by the selected method.')
        if method in LOADING:
            _real(self.loading_rel, 'DoA relative diagonal loading', nonnegative=True)
        elif self.loading_rel is not None:
            raise PhysicsConfigurationError('DOA-PROFILE-001: loading_rel is unused by the selected method.')

    def estimator_kwargs(self, method):
        self.validate(method)
        # Explicit profiles model a wave with zero elevation. Never borrow a
        # different elevation or algorithm parameter from mutable globals.
        result = {'elevation_deg': 0.}
        if method not in SEARCH_FREE:
            result['scan_azimuth_deg'] = np.asarray(self.scan_azimuth_deg, dtype=float)
        if method in SUBSPACE:
            result['n_sources'] = self.n_sources
        if method in LOADING:
            result['loading_rel'] = self.loading_rel
        return result


def parse_doa_options(value, method):
    """Strict JSON contract; all applicable settings are required when supplied."""
    keys = set()
    if method not in SEARCH_FREE:
        keys.add('azimuth_scan')
    if method in SUBSPACE:
        keys.add('n_sources')
    if method in LOADING:
        keys.add('loading_rel')
    if not isinstance(value, dict) or set(value) != keys:
        raise PhysicsConfigurationError(f'DOA-PROFILE-001: {method} doa_options requires exactly {sorted(keys)}.')
    grid = None
    if 'azimuth_scan' in value:
        scan = value['azimuth_scan']
        if not isinstance(scan, dict) or set(scan) != {'min_deg', 'max_deg', 'point_count'}:
            raise PhysicsConfigurationError('DOA-PROFILE-001: azimuth_scan requires min_deg,max_deg,point_count.')
        lo, hi = (_real(scan[key], f'DoA {key} [deg]') for key in ('min_deg', 'max_deg'))
        count = _integer(scan['point_count'], 'DoA grid point_count', minimum=2)
        if not -90 <= lo < hi <= 90 or count > np.iinfo(np.intp).max:
            raise PhysicsConfigurationError('DOA-PROFILE-001: invalid visible azimuth interval or point count.')
        try:
            grid = tuple(np.linspace(lo, hi, count))
        except (MemoryError, ValueError, OverflowError) as exc:
            raise PhysicsConfigurationError('DOA-PROFILE-001: cannot construct declared grid; no shorter substitute.') from exc
    result = ProfileDOAOptions(grid, value.get('n_sources'), value.get('loading_rel'))
    result.validate(method)
    return result
