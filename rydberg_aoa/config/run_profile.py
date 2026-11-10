"""LAB-006 JSON declarations. Parsing performs no ARC query or config write."""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np

from .laboratory import (AppliedLaser, AppliedRF, PreparedHyperfineState,
                         PreparedChannel, LabScanBounds, ExperimentalLabRequest,
                         OpticalEnergyConvention, _real, _integer)
from ..exceptions import PhysicsConfigurationError
from ..simulation.laboratory_execution import LaboratoryRunConfig, ThermalRunPlan
from .profile_effects import ProfileEnvironment, ProfileSpectrum, UniformProbeScan, NonuniformProbeScan


def _object(value, keys, name):
    keys = set(keys)
    if not isinstance(value, dict) or set(value) != keys:
        raise PhysicsConfigurationError(f'LAB-006: {name} requires exactly {sorted(keys)}; got {value!r}.')
    return dict(value)


def _construct(kind, value, name):
    return kind(**_object(value, (f.name for f in fields(kind)), name))


def _tuple(value, name):
    if not isinstance(value, list):
        raise PhysicsConfigurationError(f'LAB-006: {name} requires a JSON array.')
    return tuple(value)


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise PhysicsConfigurationError(f'LAB-006: duplicate JSON key {key!r}; no last-value substitution.')
        output[key] = value
    return output


def _reject_constant(token):
    raise PhysicsConfigurationError(f'LAB-006: nonfinite JSON constant {token}.')


@dataclass(frozen=True)
class RunProfile:
    mode: str
    probe: AppliedLaser
    coupling: AppliedLaser
    lo: AppliedRF
    signal: AppliedRF
    request: object
    run_config: object
    quantization_axis: tuple
    environment: object = None
    spectrum: object = None
    noise: object = None
    study: object = None
    provenance: object = None
    measurement: object = None
    temporal_doppler: object = None
    atom: object = None
    numerics: object = None
    radiative: object = None
    magnetic_core: object = None
    lo_phase_reference: object = None
    schema_version: int = 1

    @property
    def signed_if_mhz(self):
        return (self.signal.frequency_hz - self.lo.frequency_hz) / 1e6


@dataclass(frozen=True)
class DeclaredLadder:
    """Validated immutable declarations; defer ARC-package imports to execution."""
    entries: tuple

    def level_mappings(self):
        return {role: dict(values) for role, values in self.entries}

    def resolution_request(self):
        from .atomic_resolution import AtomicResolutionRequest, PrescribedLadderSpec
        return AtomicResolutionRequest(mode='prescribed', prescribed=PrescribedLadderSpec(**self.level_mappings()))


def parse_run_profile(document):
    """Strict versioned schemas; all physical inputs are explicitly declared.

    Version 1 retains prescribed diagnostics and experimental sensing.
    Version 2 adds explicit thermal/spectrum inputs and prescribed sensing.
    Time samples are generated only from the user's duration/count declaration.
    """
    if not isinstance(document, dict):
        raise PhysicsConfigurationError('LAB-006: profile must be a JSON object.')
    mode = document.get('mode')
    version = document.get('schema_version')
    if type(version) is not int or version not in (1, 2):
        raise PhysicsConfigurationError('LAB-006: schema_version must be integer 1 or 2.')
    common = {'schema_version', 'mode', 'fields', 'optical_energy_convention'}
    if mode == 'experimental':
        required = common | {'preparation', 'bounds', 'execution'}
    elif mode == 'prescribed':
        required = common | {'ladder', 'quantization_axis'}
    else:
        raise PhysicsConfigurationError('LAB-006: explicit mode experimental or prescribed is required.')
    if version == 2:
        required |= {'environment', 'spectrum', 'execution'}
    if 'atom' in document:
        required.add('atom')
    if 'numerics' in document:
        required.add('numerics')
    if 'radiative' in document:
        required.add('radiative')
    if 'magnetic_core' in document:
        required.add('magnetic_core')
    if 'lo_phase_reference' in document:
        required.add('lo_phase_reference')
    if 'temporal_doppler' in document:
        required.add('temporal_doppler')
    if 'noise' in document:
        required.add('noise')
    if 'study' in document:
        required.add('study')
    if 'provenance' in document:
        required.add('provenance')
    if 'measurement' in document:
        required.add('measurement')
    data = _object(document, required, 'profile')
    if mode == 'experimental' and data['execution'] is None:
        raise PhysicsConfigurationError('LAB-006: experimental profiles require an execution object.')
    if version == 2 and data['spectrum'] is not None:
        if data['environment'] is None or data['execution'] is None:
            raise PhysicsConfigurationError('LAB-007: a spectrum requires explicit environment and execution.')
    numerics = None
    if 'numerics' in data:
        from .run_numerics import RunNumerics, parse_temporal_ode
        raw = _object(data['numerics'], ('temporal_ode',), 'numerics')
        numerics = RunNumerics(parse_temporal_ode(raw['temporal_ode']))
    noise = None
    provenance = None
    measurement = None
    if 'measurement' in data:
        from .measurement import MeasurementOptions
        measurement = _construct(MeasurementOptions,data['measurement'],'measurement')
    if 'provenance' in data:
        from .provenance import InputProvenance
        provenance = _construct(InputProvenance,data['provenance'],'provenance')
    if 'noise' in data:
        from .noise import parse_noise_config
        noise = parse_noise_config(data['noise'])
    environment = spectrum = None
    if version == 2 and data['environment'] is not None:
        env = _object(data['environment'], (f.name for f in fields(ProfileEnvironment)), 'environment')
        env['linewidth_rates_per_us'] = _tuple(env['linewidth_rates_per_us'], 'linewidth rates')
        environment = ProfileEnvironment(**env)
        if environment.rate_source == 'thermal_budget' and data['execution'] is None:
            raise PhysicsConfigurationError('LAB-007: thermal rate budgeting requires execution.')
        if data['spectrum'] is not None:
            raw = data['spectrum']
            if not isinstance(raw, dict) or (('probe_offsets_mhz' in raw) == ('probe_scan' in raw)):
                raise PhysicsConfigurationError('LAB-007: spectrum requires exactly one of probe_offsets_mhz or probe_scan.')
            grid_key = 'probe_scan' if 'probe_scan' in raw else 'probe_offsets_mhz'
            keys = {f.name for f in fields(ProfileSpectrum)} - {'probe_offsets_mhz', 'steady_state_backend'}
            if 'steady_state_backend' in raw:
                keys.add('steady_state_backend')
            spec = _object(raw, keys | {grid_key}, 'spectrum')
            if grid_key == 'probe_scan':
                declaration = spec.pop('probe_scan')
                if isinstance(declaration, dict) and 'anchor_offsets_mhz' in declaration:
                    template = _object(declaration, (f.name for f in fields(NonuniformProbeScan)), 'probe_scan')
                    template['anchor_offsets_mhz'] = _tuple(template['anchor_offsets_mhz'], 'probe_scan anchors')
                    scan = NonuniformProbeScan(**template)
                else:
                    scan = _construct(UniformProbeScan, declaration, 'probe_scan')
                spec['probe_offsets_mhz'] = scan.offsets_mhz()
            else:
                spec['probe_offsets_mhz'] = _tuple(spec['probe_offsets_mhz'], 'spectrum offsets')
            spectrum = ProfileSpectrum(**spec)
    applied = _object(data['fields'], ('probe', 'coupling', 'lo', 'signal'), 'fields')
    probe = _construct(AppliedLaser, applied['probe'], 'probe')
    coupling = _construct(AppliedLaser, applied['coupling'], 'coupling')
    lo = _construct(AppliedRF, applied['lo'], 'lo')
    signal = _construct(AppliedRF, applied['signal'], 'signal')
    if lo.polarization != signal.polarization:
        raise PhysicsConfigurationError('LAB-006: LO/signal must declare the same q.')
    try:
        policy = OpticalEnergyConvention(data['optical_energy_convention'])
    except (ValueError, TypeError) as exc:
        raise PhysicsConfigurationError('LAB-006: unknown optical energy convention.') from exc
    if mode == 'experimental':
        prep = _object(data['preparation'], (f.name for f in fields(PreparedChannel)), 'preparation')
        prep['ground'] = _construct(PreparedHyperfineState, prep['ground'], 'ground')
        prep['excited'] = _construct(PreparedHyperfineState, prep['excited'], 'excited')
        prep['quantization_axis'] = _tuple(prep['quantization_axis'], 'quantization_axis')
        preparation = PreparedChannel(**prep)
        bounds = _object(data['bounds'], (f.name for f in fields(LabScanBounds)), 'bounds')
        for name in ('probe_window_nm', 'coupling_window_nm', 'rf_window_hz'):
            bounds[name] = _tuple(bounds[name], name)
        request = ExperimentalLabRequest(probe, coupling, lo, signal, preparation,
                                         LabScanBounds(**bounds), policy)
        axis = preparation.quantization_axis
    else:
        if policy != OpticalEnergyConvention.FS_CENTROID:
            raise PhysicsConfigurationError('LAB-006: prescribed diagnostics retain the existing fs_centroid detuning contract.')
        ladder = _object(data['ladder'], ('ground', 'excited', 'rydberg1', 'rydberg2'), 'ladder')
        # Reject coercible nonnumeric declarations before the legacy validator.
        for role, value in ladder.items():
            keys = ('n', 'l', 'j', 'f', 'mf') if role in ('ground', 'excited') else ('n', 'l', 'j', 'mj')
            level = _object(value, keys, role)
            level['n'] = _integer(level['n'], f'{role}.n', minimum=1)
            level['l'] = _integer(level['l'], f'{role}.l')
            for key in set(keys) - {'n', 'l'}:
                level[key] = _real(level[key], f'{role}.{key}')
            n, l, j = level['n'], level['l'], level['j']
            if l >= n or 2*j not in ((2*l-1, 2*l+1) if l else (1,)):
                raise PhysicsConfigurationError(f'LAB-006: invalid alkali n,l,j in {role}.')
            if role.startswith('rydberg') and (abs(level['mj']) > j or not (j-level['mj']).is_integer()):
                raise PhysicsConfigurationError(f'LAB-006: invalid declared {role}.mJ.')
            ladder[role] = level
        # Reuse explicit axis validation, with the declared HFS/projections.
        preparation = PreparedChannel(PreparedHyperfineState(**ladder['ground']),
                                      PreparedHyperfineState(**ladder['excited']),
                                      ladder['rydberg1']['mj'], ladder['rydberg2']['mj'],
                                      _tuple(data['quantization_axis'], 'quantization_axis'))
        if (preparation.excited.mf - preparation.ground.mf != probe.polarization or
                preparation.rydberg2_mj - preparation.rydberg1_mj != lo.polarization):
            raise PhysicsConfigurationError('LAB-006: prescribed projections disagree with applied q.')
        for source, destination in (('ground', 'excited'), ('excited', 'rydberg1'), ('rydberg1', 'rydberg2')):
            a, b = ladder[source], ladder[destination]
            if abs(a['l'] - b['l']) != 1 or abs(a['j'] - b['j']) > 1:
                raise PhysicsConfigurationError('LAB-006: prescribed ladder violates E1 orbital rules.')
        g, e = preparation.ground, preparation.excited
        if abs(g.f-e.f) > 1 or g.f == e.f == 0:
            raise PhysicsConfigurationError('LAB-006: prescribed probe violates E1 hyperfine rules.')
        request = DeclaredLadder(tuple((role, tuple(level.items())) for role, level in ladder.items()))
        run_config, axis = None, preparation.quantization_axis
    if data.get('execution') is not None:
        thermal = environment is not None and environment.rate_source == 'thermal_budget'
        execution_keys = {'duration_us', 'sample_count', 'positions_m', 'target_azimuth_deg',
                      'transient_delay_us', 'sensor_probe_offset_mhz', 'processes',
                      'doa_method', 'cycles_per_snapshot'}
        if not thermal:
            execution_keys |= {'dephasing_rates_per_us', 'transit_rate_per_us'}
        if isinstance(data['execution'], dict) and 'doa_options' in data['execution']:
            execution_keys.add('doa_options')
        execution = _object(data['execution'], execution_keys, 'execution')
        if 'doa_options' in execution:
            from .doa_options import parse_doa_options
            method = execution['doa_method']
            if not isinstance(method, str):
                raise PhysicsConfigurationError('DOA-PROFILE-001: doa_method requires a string.')
            execution['doa_options'] = parse_doa_options(execution['doa_options'], method.strip().lower())
        duration = _real(execution.pop('duration_us'), 'duration [us]', positive=True)
        count = _integer(execution.pop('sample_count'), 'sample count', minimum=10)
        if count > np.iinfo(np.intp).max:
            raise PhysicsConfigurationError('LAB-006: sample count exceeds platform array-index capacity.')
        try:
            execution['tlist_us'] = tuple(np.linspace(0., duration, count))
        except (OverflowError, ValueError, TypeError, MemoryError) as exc:
            raise PhysicsConfigurationError('LAB-006: cannot construct the declared time grid; no shorter-grid substitute.') from exc
        execution['positions_m'] = tuple(_tuple(row, 'position [m]') for row in _tuple(execution['positions_m'], 'positions'))
        if not thermal:
            execution['dephasing_rates_per_us'] = _tuple(execution['dephasing_rates_per_us'], 'pure dephasing')
        if measurement is not None:
            execution['measurement_mode'] = measurement.mode
        if numerics is not None:
            execution['temporal_ode'] = numerics.temporal_ode
        run_config = (ThermalRunPlan if thermal else LaboratoryRunConfig)(**execution)
        run_config.validate_sampling((signal.frequency_hz - lo.frequency_hz) / 1e6)
        # Method lookup performs no initialization or worker dispatch.
        from ..doa.registry import get_doa_estimator
        try:
            if run_config.measurement_mode == 'aoa':
                get_doa_estimator(run_config.doa_method)
        except (ValueError, TypeError) as exc:
            raise PhysicsConfigurationError('LAB-006: unknown DoA method; no estimator substitute.') from exc
    temporal_doppler = None
    if 'temporal_doppler' in data:
        from .temporal_doppler import parse_temporal_doppler
        temporal_doppler = parse_temporal_doppler(data['temporal_doppler'])
        if temporal_doppler.enabled and (version != 2 or environment is None):
            raise PhysicsConfigurationError('DOP-TD-001: temporal ensemble requires explicit schema-2 environment.')
    profile = RunProfile(mode, probe, coupling, lo, signal, request, run_config, axis, environment, spectrum, noise,
                         schema_version=version)
    if numerics is not None:
        from dataclasses import replace
        profile = replace(profile,numerics=numerics)
    from .run_numerics import validate_profile_numerics
    validate_profile_numerics(profile)
    if 'atom' in data:
        from dataclasses import replace
        from .atom_species import AtomSpec
        profile = replace(profile, atom=_construct(AtomSpec, data['atom'], 'atom'))
    if temporal_doppler is not None:
        from dataclasses import replace
        profile = replace(profile, temporal_doppler=temporal_doppler)
    if provenance is not None:
        from dataclasses import replace
        profile = replace(profile,provenance=provenance)
    if measurement is not None:
        from dataclasses import replace
        profile = replace(profile,measurement=measurement)
    if 'study' in data:
        from dataclasses import replace
        from .study import parse_study, validate_study_profile
        profile = replace(profile, study=parse_study(data['study']))
        validate_study_profile(profile)
    from .measurement import validate_measurement_profile
    if measurement is not None:
        validate_measurement_profile(profile)
    if 'radiative' in data:
        from dataclasses import replace
        from .profile_radiative import RadiativeOptions
        profile = replace(profile, radiative=_construct(RadiativeOptions, data['radiative'], 'radiative'))
    from .profile_radiative import validate_profile_radiative
    validate_profile_radiative(profile)
    if 'magnetic_core' in data:
        from dataclasses import replace
        from .magnetic_core import MagneticCoreOptions, CoreTemporalDiagnostic, CorePeriodicControls
        core_data=dict(data['magnetic_core']) if isinstance(data['magnetic_core'],dict) else data['magnetic_core']
        if isinstance(core_data,dict):core_data.setdefault('temporal_diagnostic',None)
        if isinstance(core_data,dict) and core_data.get('temporal_diagnostic') is not None:
            if isinstance(core_data['temporal_diagnostic'],dict):
                core_data['temporal_diagnostic']=dict(core_data['temporal_diagnostic'])
                for key,value in dict(backend='qutip',initial_state='prepared',extract_if=False,transient_delay_us=0.,periodic=None).items():
                    core_data['temporal_diagnostic'].setdefault(key,value)
                if core_data['temporal_diagnostic']['periodic'] is not None:
                    core_data['temporal_diagnostic']['periodic']=_construct(CorePeriodicControls,core_data['temporal_diagnostic']['periodic'],'magnetic_core.temporal_diagnostic.periodic')
            core_data['temporal_diagnostic']=_construct(CoreTemporalDiagnostic,core_data['temporal_diagnostic'],'magnetic_core.temporal_diagnostic')
        profile = replace(profile, magnetic_core=_construct(MagneticCoreOptions, core_data, 'magnetic_core'))
    from .magnetic_core import validate_magnetic_core
    validate_magnetic_core(profile)
    from .lo_phase_reference import parse_lo_phase_reference,validate_lo_phase_reference
    if 'lo_phase_reference' in data:
        from dataclasses import replace
        profile=replace(profile,lo_phase_reference=parse_lo_phase_reference(data['lo_phase_reference']))
    validate_lo_phase_reference(profile)
    return profile


def load_run_profile(path):
    try:
        document = json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise PhysicsConfigurationError(f'LAB-006: cannot read profile {path!s}.') from exc
    return parse_run_profile(document)
