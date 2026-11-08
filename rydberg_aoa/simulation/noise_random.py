"""NOISE-001 reproducible stream factory; no physical noise injection.

Calling generator twice with the same key restarts the same stream. Consumers
must construct each logical stream once and retain it. IDs are independent of
PID, dispatch order, and process count; source IDs must never be renumbered.
"""

from dataclasses import dataclass

import numpy as np

from ..config.laboratory import _integer
from ..config.noise import NoiseConfig
from ..exceptions import PhysicsConfigurationError

STREAM_VERSION = 1
_SOURCE_IDS = {'readout_gaussian': 1, 'photon_shot': 2, 'dark_shot': 3,
               'electronic_current': 4, 'johnson_current': 5,
               'rf_lo_amplitude': 6, 'rf_signal_amplitude': 7, 'rf_relative_phase': 8,
               'laser_probe_frequency': 9, 'laser_coupling_frequency': 10,
               'laser_probe_intensity': 11, 'laser_coupling_intensity': 12}


@dataclass(frozen=True)
class NoiseRandomPlan:
    config: NoiseConfig

    def __post_init__(self):
        if not isinstance(self.config, NoiseConfig):
            raise PhysicsConfigurationError('NOISE-001: NoiseRandomPlan requires NoiseConfig.')

    def generator(self, source, *, trial, path=None):
        """Return PCG64 stream or None for a disabled source; never draw here.

        path=None identifies a common stream. A nonnegative integer identifies
        a local path stream. This namespace does not itself impose correlation.
        """
        if not isinstance(source, str) or source not in _SOURCE_IDS:
            raise PhysicsConfigurationError('NOISE-001: unknown RNG source.')
        trial = _integer(trial, 'NOISE-001 trial')
        if trial >= self.config.trials:
            raise PhysicsConfigurationError('NOISE-001: trial outside declared campaign.')
        if path is not None:
            path = _integer(path, 'NOISE-001 path')
        if source not in self.config.active_sources:
            return None
        # Fixed key components; no shared mutable spawn counter or Python hash.
        key = (STREAM_VERSION, _SOURCE_IDS[source], trial,
               0 if path is None else 1, 0 if path is None else path)
        sequence = np.random.SeedSequence(self.config.seed, spawn_key=key)
        return np.random.Generator(np.random.PCG64(sequence))

    def metadata(self):
        return {'stream_version': STREAM_VERSION, 'bit_generator': 'PCG64',
                'seed': self.config.seed, 'trials': self.config.trials,
                'active_sources': self.config.active_sources,
                'key_layout': ('stream_version', 'source_id', 'trial', 'scope', 'path'),
                'source_ids': dict(_SOURCE_IDS), 'numpy_version': np.__version__}
