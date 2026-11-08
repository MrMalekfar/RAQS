"""Canonical editable JSON representation of profiles, including user studies."""
from dataclasses import asdict
import numpy as np
from ..exceptions import PhysicsConfigurationError


def noise_document(noise):
    data = {k: getattr(noise, k) for k in ('schema_version','enabled','seed','trials')}
    data['sources'] = {s.name: asdict(s.specification) for s in noise.sources}
    if noise.schema_version == 2:
        data['detector'] = asdict(noise.detector)
    if noise.schema_version == 3:
        data.update(drive=asdict(noise.drive), readout=noise_document(noise.readout) if noise.readout else None)
    return data


def profile_document(profile):
    """Return standard JSON-compatible data; no ARC or state writes.

    Reconstructs effective grids, never an approximate shorter scan. JSON
    profiles contain uniform time/DoA grids; unsupported programmatic grids
    fail rather than being changed during persistence.
    """
    version = 2 if profile.environment is not None else 1
    data = {'schema_version': version, 'mode': profile.mode,
            'fields': {k: asdict(getattr(profile,k)) for k in ('probe','coupling','lo','signal')},
            'optical_energy_convention': 'fs_centroid'}
    from .lo_phase_reference import validate_lo_phase_reference,lo_phase_document
    validate_lo_phase_reference(profile)
    if profile.lo_phase_reference is not None:
        data['lo_phase_reference']=lo_phase_document(profile.lo_phase_reference)
    if profile.atom is not None:
        data['atom'] = asdict(profile.atom)
    from .run_numerics import validate_profile_numerics
    validate_profile_numerics(profile)
    if profile.numerics is not None:
        data['numerics'] = asdict(profile.numerics)
    from .profile_radiative import validate_profile_radiative
    validate_profile_radiative(profile)
    if profile.radiative is not None:
        data['radiative'] = asdict(profile.radiative)
    from .magnetic_core import validate_magnetic_core
    validate_magnetic_core(profile)
    if profile.magnetic_core is not None:
        data['magnetic_core'] = asdict(profile.magnetic_core)
        if profile.magnetic_core.temporal_diagnostic is None:
            del data['magnetic_core']['temporal_diagnostic']
        elif profile.magnetic_core.temporal_diagnostic.periodic is None:
            del data['magnetic_core']['temporal_diagnostic']['periodic']
    if profile.mode == 'experimental':
        data.update(preparation=asdict(profile.request.preparation), bounds=asdict(profile.request.bounds),
                    optical_energy_convention=profile.request.optical_energy_convention.value)
    else:
        data.update(ladder=profile.request.level_mappings(), quantization_axis=profile.quantization_axis)
    if version == 2:
        data.update(environment=asdict(profile.environment),
                    spectrum=asdict(profile.spectrum) if profile.spectrum is not None else None)
        if profile.spectrum is not None and profile.spectrum.steady_state_backend is None:
            del data['spectrum']['steady_state_backend']
    if profile.run_config is not None:
        if profile.run_config.temporal_doppler is not None and profile.temporal_doppler is None:
            raise PhysicsConfigurationError('DOP-TD-001: cannot export resolved ensemble without its profile declaration.')
        execution = asdict(profile.run_config)
        if getattr(profile.run_config, 'radiative', None) is not None:
            from .profile_radiative import ResolvedProfileRadiative
            receipt = profile.run_config.radiative
            if not isinstance(receipt, ResolvedProfileRadiative) or receipt.declaration != profile.radiative:
                raise PhysicsConfigurationError('RELAX-002: cannot export a runtime graph without its matching declaration.')
        execution.pop('radiative', None)
        if profile.environment is not None and profile.environment.rate_source == 'thermal_budget':
            execution.pop('dephasing_rates_per_us', None)
            execution.pop('transit_rate_per_us', None)
        execution.pop('measurement_mode',None)
        execution.pop('temporal_doppler',None)
        execution.pop('temporal_ode',None)
        times = execution.pop('tlist_us')
        if not np.array_equal(times, np.linspace(0.,times[-1],len(times))):
            raise PhysicsConfigurationError('NOISE-008: cannot export nonuniform programmatic time grid to current profile schema.')
        execution.update(duration_us=float(times[-1]), sample_count=len(times))
        options = profile.run_config.doa_options
        execution.pop('doa_options',None)
        if options is not None:
            obj = {}
            if options.scan_azimuth_deg is not None:
                grid = options.scan_azimuth_deg
                if not np.array_equal(grid,np.linspace(grid[0],grid[-1],len(grid))):
                    raise PhysicsConfigurationError('NOISE-008: nonuniform DoA grid has no current JSON representation; no grid substitution.')
                obj['azimuth_scan'] = {'min_deg':float(grid[0]),'max_deg':float(grid[-1]),'point_count':len(grid)}
            for k in ('n_sources','loading_rel'):
                if getattr(options,k) is not None:
                    obj[k] = getattr(options,k)
            execution['doa_options'] = obj
        data['execution'] = execution
    if profile.temporal_doppler is not None:
        data['temporal_doppler'] = asdict(profile.temporal_doppler)
    if profile.noise is not None:
        data['noise'] = noise_document(profile.noise)
    if profile.provenance is not None:
        data['provenance'] = asdict(profile.provenance)
    if profile.measurement is not None:
        data['measurement'] = asdict(profile.measurement)
    if profile.study is not None:
        data['study'] = asdict(profile.study)
        from .noise_convergence import NoiseConvergenceStudy
        if isinstance(profile.study,NoiseConvergenceStudy):
            data['study']['budgets'] = dict(profile.study.budgets)
    # JSON arrays, including tuples nested in frozen dataclasses.
    def lists(value):
        if isinstance(value,dict): return {k:lists(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)): return [lists(v) for v in value]
        return value
    return lists(data)
