"""Residual laser frequency and classical excess intensity noise declarations."""

from dataclasses import dataclass
import math
from .laboratory import _real
from ..exceptions import PhysicsConfigurationError

LASER_SOURCE_UNITS = {'laser_probe_frequency': 'Hz', 'laser_coupling_frequency': 'Hz'}
LASER_INTENSITY_UNITS = {'laser_probe_intensity': 'relative_rms', 'laser_coupling_intensity': 'relative_rms'}


@dataclass(frozen=True)
class LaserFrequencyNoise:
    enabled: bool
    stationary_std: float
    correlation_time_us: float
    units: str
    budget: str

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise PhysicsConfigurationError('NOISE-006: laser enabled must be boolean.')
        object.__setattr__(self, 'stationary_std', _real(
            self.stationary_std, 'NOISE-006 frequency std [Hz]', nonnegative=True))
        object.__setattr__(self, 'correlation_time_us', _real(
            self.correlation_time_us, 'NOISE-006 correlation time [us]', positive=True))
        if not isinstance(self.units, str) or self.units != 'Hz':
            raise PhysicsConfigurationError('NOISE-006: laser frequency noise units must be Hz (not linewidth/FWHM).')
        if not isinstance(self.budget, str) or self.budget != 'independent_residual':
            raise PhysicsConfigurationError('NOISE-006: laser budget must explicitly be independent_residual; do not count the same linewidth twice.')


@dataclass(frozen=True)
class LaserIntensityNoise:
    """Classical excess RIN at nodes, excluding photon-counting shot noise."""
    enabled: bool
    fractional_rms: float
    correlation_time_us: float
    units: str
    budget: str

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise PhysicsConfigurationError('NOISE-007: intensity enabled must be boolean.')
        object.__setattr__(self, 'fractional_rms', _real(
            self.fractional_rms, 'NOISE-007 fractional intensity RMS', nonnegative=True))
        object.__setattr__(self, 'correlation_time_us', _real(
            self.correlation_time_us, 'NOISE-007 correlation time [us]', positive=True))
        if not isinstance(self.units, str) or self.units != 'relative_rms':
            raise PhysicsConfigurationError('NOISE-007: units must be relative_rms, not dBc/Hz or Rabi RMS.')
        if not isinstance(self.budget, str) or self.budget != 'technical_excess':
            raise PhysicsConfigurationError('NOISE-007: budget must be technical_excess; photon shot noise belongs to the detector counting model.')
        if self.fractional_rms > 0 and self.stationary_std == 0:
            raise PhysicsConfigurationError('NOISE-007: positive RMS is below representable log-intensity variance; no zero substitute.')

    @property
    def stationary_std(self):
        """Std of underlying Gaussian log-intensity: sqrt(log(1+RMS^2))."""
        r = self.fractional_rms
        variance = math.log1p(r*r) if r <= 1 else 2*math.log(r)+math.log1p((1/r)**2)
        return math.sqrt(variance)
