"""Explicit GUI-ready Cartesian operating-point design; no atomic queries."""
from dataclasses import dataclass, fields, replace
from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class OperatingStudy:
    schema_version: int
    kind: str
    name: str
    probe_detunings_mhz: tuple
    cell_lengths_m: tuple
    path_index: int
    selection: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or not isinstance(self.kind,str) or self.kind != 'operating_point_sweep':
            raise PhysicsConfigurationError('NOISE-018: operating_point_sweep schema 1 required.')
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.selection,str) or self.selection != 'maximum_predicted_if_snr':
            raise PhysicsConfigurationError('NOISE-018: name and explicit maximum_predicted_if_snr selection required.')
        for name in ('probe_detunings_mhz', 'cell_lengths_m'):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) < 2:
                raise PhysicsConfigurationError('NOISE-018: each operating axis needs >=2 immutable points.')
            values = tuple(_real(v, 'NOISE-018 '+name, positive=name=='cell_lengths_m') for v in values)
            if any(b <= a for a, b in zip(values, values[1:])):
                raise PhysicsConfigurationError('NOISE-018: operating axes must strictly increase; no sorting.')
            object.__setattr__(self, name, values)
        object.__setattr__(self, 'path_index', _integer(self.path_index, 'NOISE-018 path index'))


def parse_operating_study(raw):
    if not isinstance(raw, dict) or set(raw) != {f.name for f in fields(OperatingStudy)}:
        raise PhysicsConfigurationError('NOISE-018: complete explicit operating design required.')
    data = dict(raw)
    for key in ('probe_detunings_mhz','cell_lengths_m'):
        if not isinstance(data[key], list):
            raise PhysicsConfigurationError('NOISE-018: axes must be JSON arrays.')
        data[key] = tuple(data[key])
    return OperatingStudy(**data)


def operating_points(profile):
    d = profile.study
    return tuple((delta, length) for delta in d.probe_detunings_mhz for length in d.cell_lengths_m)


def operating_profile(profile, index):
    from .run_profile import RunProfile
    if not isinstance(profile,RunProfile) or not isinstance(profile.study,OperatingStudy) or profile.environment is None:
        raise PhysicsConfigurationError('NOISE-018: parsed operating profile and environment required.')
    points = operating_points(profile)
    index = _integer(index, 'NOISE-018 operating index')
    if index >= len(points): raise PhysicsConfigurationError('NOISE-018: operating point outside design.')
    delta, length = points[index]
    probe = replace(profile.probe, detuning_mhz=delta)
    request = replace(profile.request, probe=probe) if profile.mode == 'experimental' else profile.request
    return replace(profile, study=None, probe=probe, request=request,
                   environment=replace(profile.environment, cell_length_m=length))


def validate_operating_profile(profile):
    from .run_profile import RunProfile
    if (not isinstance(profile, RunProfile) or not isinstance(profile.study, OperatingStudy) or
            profile.environment is None or profile.run_config is None or profile.measurement is None or
            profile.measurement.mode != 'if_only' or not profile.measurement.signal_enabled or
            profile.noise is None or profile.noise.schema_version != 2 or not profile.noise.physical_readout_enabled or
            profile.noise.trials != 1 or profile.spectrum is not None):
        raise PhysicsConfigurationError('NOISE-018: temporal IF-only signal ON, schema-2 PIN, one pilot trial, environment, spectrum=null required.')
    if profile.study.path_index >= len(profile.run_config.positions_m):
        raise PhysicsConfigurationError('NOISE-018: predeclared path outside receiver.')
    if profile.mode == 'experimental' and any(getattr(profile.request,k) != getattr(profile,k) for k in ('probe','coupling','lo','signal')):
        raise PhysicsConfigurationError('NOISE-018: experimental request and applied fields disagree; no silent reconciliation.')
    from .detector import validate_detector_sampling
    for i in range(len(operating_points(profile))):
        p = operating_profile(profile, i)
        p.run_config.validate_sampling(p.signed_if_mhz)
        validate_detector_sampling(p.noise, p.run_config)
        if p.mode == 'experimental' and p.request.probe != p.probe:
            raise PhysicsConfigurationError('NOISE-018: request/profile applied probe mismatch.')


def plan_operating_study(profile):
    validate_operating_profile(profile)
    points = operating_points(profile)
    return {'kind':'operating_point_sweep', 'points':[{'index':i,'probe_detuning_mhz':d,'cell_length_m':l} for i,(d,l) in enumerate(points)],
            'total_atomic_evolutions':len(points)*len(profile.run_config.positions_m),
            'total_trials':len(points), 'selection':'predicted conditional detector IF SNR, not observed lucky pilot noise',
            'validation':'a separately declared held-out detection study; never automatic retuning',
            'sampling_and_solver':'unchanged', 'hardware_calibration_verified':False}
