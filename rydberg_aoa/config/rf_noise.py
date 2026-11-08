"""Residual colored RF envelope noise; separate from atomic level selection."""

from dataclasses import dataclass
import math
import numpy as np

from .laboratory import _real
from ..exceptions import PhysicsConfigurationError

RF_SOURCE_UNITS = {'rf_lo_amplitude': 'log_amplitude', 'rf_signal_amplitude': 'log_amplitude',
                   'rf_relative_phase': 'rad'}


@dataclass(frozen=True)
class RFOrnsteinUhlenbeck:
    enabled: bool
    stationary_std: float
    correlation_time_us: float
    units: str

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise PhysicsConfigurationError('NOISE-005: RF enabled must be boolean.')
        object.__setattr__(self, 'stationary_std', _real(self.stationary_std, 'NOISE-005 stationary std', nonnegative=True))
        object.__setattr__(self, 'correlation_time_us', _real(self.correlation_time_us, 'NOISE-005 correlation time', positive=True))
        if not isinstance(self.units, str) or self.units not in ('rad', 'log_amplitude'):
            raise PhysicsConfigurationError('NOISE-005: RF units rad/log_amplitude required.')


@dataclass(frozen=True)
class RFDriveSettings:
    noise_step_us: float
    horizon_us: float
    interpolation: str
    spatial_scope: str
    preparation: str
    process_initialization: str
    phase_budget: str

    def __post_init__(self):
        for name in ('noise_step_us', 'horizon_us'):
            object.__setattr__(self, name, _real(getattr(self, name), 'NOISE-005 '+name, positive=True))
        expected = {'interpolation': 'linear', 'spatial_scope': 'common_all_paths',
                    'preparation': 'nominal_lo_steady_state_then_switch', 'process_initialization': 'stationary',
                    'phase_budget': 'independent_residual'}
        for name, value in expected.items():
            if not isinstance(getattr(self, name), str) or getattr(self, name) != value:
                raise PhysicsConfigurationError(f'NOISE-005: {name} must explicitly be {value!r}.')
        count = self.horizon_us/self.noise_step_us
        if not math.isfinite(count) or count < 1 or count > 2**53 or not np.isclose(count, round(count), rtol=1e-12, atol=0):
            raise PhysicsConfigurationError('NOISE-005: horizon must contain an integer number of noise steps.')


def validate_rf_noise_execution(noise, run_config):
    if noise is None or noise.schema_version != 3 or not noise.stochastic_drive:
        return
    if run_config is None:
        raise PhysicsConfigurationError('NOISE-005: explicit temporal execution required.')
    if noise.drive.horizon_us <= run_config.tlist_us[-1]:
        raise PhysicsConfigurationError('NOISE-005: explicit noise horizon must exceed output end for BDF internal evaluations; no extrapolation.')
    for source in noise.sources:
        if not source.specification.enabled or source.specification.stationary_std == 0:
            continue
        if noise.drive.noise_step_us > source.specification.correlation_time_us/10:
            raise PhysicsConfigurationError('NOISE-005: noise step must be <= tau/10; refine the declared realization grid.')
