"""Wire migration preserves effective profile, noise, and RNG declarations."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.config.profile_document import noise_document, profile_document, unified_noise_document
from rydberg_aoa.config.run_profile import load_run_profile, parse_run_profile
from rydberg_aoa.exceptions import PhysicsConfigurationError
from rydberg_aoa.simulation.noise_random import NoiseRandomPlan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['experimental_reference', 'experimental_near_resonant', 'prescribed_reference'])
def test_lab006_migration_preserves_effective_inputs(name):
    filename = name + '.json'
    original = load_run_profile(ROOT / 'tests/fixtures/profile_schema_v1' / filename)
    migrated = load_run_profile(ROOT / 'examples/lab006' / filename)
    assert original.schema_version == 1 and migrated.schema_version == 2
    assert replace(original, schema_version=2) == migrated
    encoded = profile_document(migrated)
    assert encoded['environment'] is None and encoded['spectrum'] is None
    assert (encoded['execution'] is None) == (name == 'prescribed_reference')
    assert parse_run_profile(encoded) == migrated


def test_missing_execution_and_missing_spectrum_environment_are_not_inferred():
    experimental = profile_document(load_run_profile(ROOT / 'examples/lab006/experimental_reference.json'))
    experimental['execution'] = None
    with pytest.raises(PhysicsConfigurationError, match='require an execution object'):
        parse_run_profile(experimental)
    prescribed = profile_document(load_run_profile(ROOT / 'examples/lab006/prescribed_reference.json'))
    prescribed['spectrum'] = {}
    with pytest.raises(PhysicsConfigurationError, match='requires explicit environment and execution'):
        parse_run_profile(prescribed)


@pytest.mark.parametrize('name', [
    'noise002/prescribed_gaussian.json', 'noise004/prescribed_pin.json',
    'noise005/prescribed_rf_pin.json', 'noise006/prescribed_laser_frequency.json',
    'noise007/prescribed_rin_pin.json',
])
def test_unified_noise_keeps_typed_contract_and_streams(name):
    profile = load_run_profile(ROOT / 'examples' / name)
    canonical = profile_document(profile)['noise']
    legacy = parse_noise_config(noise_document(profile.noise))
    migrated = parse_noise_config(canonical)
    assert canonical['schema_version'] == 4
    assert migrated == legacy
    assert parse_noise_config(unified_noise_document(noise_document(migrated))) == migrated
    for source in migrated.active_sources:
        old = NoiseRandomPlan(legacy).generator(source, trial=0, path=0)
        new = NoiseRandomPlan(migrated).generator(source, trial=0, path=0)
        np.testing.assert_array_equal(old.random(8), new.random(8))


def test_drive_and_readout_switches_are_independent():
    profile = load_run_profile(ROOT / 'examples/noise005/prescribed_rf_pin.json')
    document = profile_document(profile)['noise']
    assert document['readout']['enabled']
    assert profile.noise.drive_selected and profile.noise.physical_readout_enabled
    document['readout']['enabled'] = False
    drive_only = parse_noise_config(document)
    assert drive_only.drive_selected and not drive_only.readout_enabled
    document['readout']['enabled'] = True
    for specification in document['drive']['sources'].values():
        specification['enabled'] = False
    readout_only = parse_noise_config(document)
    assert not readout_only.drive_selected and readout_only.physical_readout_enabled
    for specification in document['readout']['sources'].values():
        specification['enabled'] = False
    noiseless_pin = parse_noise_config(document)
    assert noiseless_pin.physical_readout_enabled and not noiseless_pin.active_sources
    document['readout']['model'] = 'gaussian'
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(document)
