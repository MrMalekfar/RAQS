"""Literature declarations remain honest and valid at the public profile boundary."""
import json
import math
from pathlib import Path

import pytest

from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document

DIRECTORY = Path(__file__).resolve().parents[1] / 'examples/literature001'
CASES = [
    ('holloway2014_17ghz', 50, (51, 1, 1.5), 17.04e9, .000175, 30.),
    ('holloway2014_68ghz', 32, (33, 1, 1.5), 68.64e9, .000250, 24.),
    ('holloway2014_104ghz', 28, (29, 1, 1.5), 104.77e9, .000120, 22.),
    ('psn2026_pi_channel_proxy', 40, (39, 3, 3.5), 37.07104e9, .006, 30.),
]


@pytest.mark.parametrize('name,n,r2,frequency,probe,coupling', CASES)
def test_reported_inputs_survive_public_round_trip(name, n, r2, frequency, probe, coupling):
    original = json.loads((DIRECTORY / (name + '.json')).read_text())
    exported = profile_document(parse_run_profile(original))
    reparsed = parse_run_profile(exported)
    assert profile_document(reparsed) == exported
    assert exported['mode'] == 'prescribed'
    assert exported['fields']['lo']['frequency_hz'] == frequency
    assert exported['fields']['probe']['power_mw'] == probe
    assert exported['fields']['coupling']['power_mw'] == coupling
    assert exported['ladder']['rydberg1']['n'] == n
    assert tuple(exported['ladder']['rydberg2'][k] for k in ('n', 'l', 'j')) == r2
    for field in exported['fields'].values():
        assert field['polarization'] == 0
    assert exported['measurement']['mode'] == 'if_only'
    assert exported['execution']['positions_m'] == [[0., 0., 0.]]
    assert exported['spectrum']['doppler_enabled'] is True
    assert exported['temporal_doppler']['enabled'] is name.startswith('psn')
    assert exported['measurement']['signal_enabled'] is name.startswith('psn')


@pytest.mark.parametrize('name', [item[0] for item in CASES[:3]])
def test_nist_fwhm_is_converted_to_radius_not_half_diameter(name):
    d = json.loads((DIRECTORY / (name + '.json')).read_text())
    for key, diameter in [('probe', .080), ('coupling', .100)]:
        assert d['fields'][key]['beam_waist_mm'] == pytest.approx(diameter / math.sqrt(2 * math.log(2)))
    assert d['environment']['cell_length_m'] == .075


def test_proxy_reports_actual_seam_calibration_and_not_experimental_validation():
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    assert manifest['exact_reproduction'] is False
    for row in manifest['cases']:
        assert row['numerical_campaign_validated'] is False
        assert row['published_curve_digitized'] is False
        assert row['initialization_checks']['initialized'] is True
        values = row['initialization_checks']['detunings_rad_per_us']
        assert all(math.isfinite(v) for v in values)
        # Rounded reported carriers retain nonzero ARC mismatch, not auto resonance.
        assert values[2] != 0.
        assert row['assumptions']
    proxy = manifest['cases'][-1]
    assert proxy['initialization_checks']['rabi_rad_per_us'][2] / (2 * math.pi) == pytest.approx(3.7)
    assert any('orthogonal' in item for item in proxy['assumptions'])
    d = json.loads((DIRECTORY / proxy['profile']).read_text())
    assert d['fields']['signal']['frequency_hz'] - d['fields']['lo']['frequency_hz'] == 40000.
    assert d['fields']['lo']['electric_field_vm'] == proxy['derived_model_inputs']['rf_field_peak_v_per_m']
