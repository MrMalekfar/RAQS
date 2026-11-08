"""Display regressions: measured signs, geometry, provenance and no mutation."""
import copy
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rydberg_aoa.exceptions import PhysicsCalculationError
from rydberg_aoa.visualization.profile_figures import create_profile_figures, save_profile_figures


@pytest.fixture
def data():
    levels = {key: dict(n=n, l=l, j=j, f=f, mf=0 if f else None, mj=None if f else -.5)
              for key, n, l, j, f in [('ground', 5, 0, .5, 3), ('excited', 5, 1, 1.5, 4),
                                     ('rydberg1', 26, 2, 2.5, None), ('rydberg2', 27, 1, 1.5, None)]}
    report = dict(levels=levels, rf_orientation=dict(signed_frequency_hz=-132.649e9),
                  applied_fields=dict(probe=dict(wavelength_nm=781.123456), coupling=dict(wavelength_nm=484.654321),
                                      lo=dict(frequency_hz=132.649e9), signal=dict(frequency_hz=132.6491e9)),
                  signed_if_mhz=.1, rabi_rad_per_us=[1., 2., 12., .01],
                  spectrum_declaration=dict(doppler_enabled=True),
                  resolved_rate_budget_per_us=dict(transit_reset=1.),
                  sensing=dict(execution=dict(sensor_probe_offset_mhz=.25)))
    result = dict(tlist=np.array([0., 100., 200.]),
                  time_traces=np.array([[-2., 1., 4.], [1., 5., 0.], [2., 0., 1.]]),
                  complex_responses=np.exp(1j*np.deg2rad([10., -12.5, -35.])),
                  array_positions=np.array([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.]]),
                  true_aoa=-30., estimated_aoa=-30.,
                  doa_result=SimpleNamespace(algorithm='music', azimuth_deg=-30., elevation_deg=0.),
                  angles=np.array([-90., -30., 0., 90.]), power=np.array([1., 5., 10., 1.]),
                  spectrum=dict(probe_offsets_mhz=np.array([-1., 0., 1.]), rf_off=np.array([.2, .4, .2]),
                                rf_on=np.array([.3, .2, .3]), rf_off_analytical=np.array([.1, .9, .1]),
                                rf_on_analytical=np.array([.8, .1, .8])))
    yield report, result
    plt.close('all')


def test_time_plot_preserves_signed_observable_all_paths_and_full_grid(data):
    report, result = data
    original = copy.deepcopy(result)
    style = plt.rcParams.copy()
    figures = create_profile_figures(report, result)
    ax = figures['fig_time_domain'].axes[0]
    assert len(ax.lines) == 3 and ax.get_xlim() == (0., 200.)
    for line, trace in zip(ax.lines, result['time_traces']):
        np.testing.assert_array_equal(line.get_xdata(), result['tlist'])
        np.testing.assert_array_equal(line.get_ydata(), trace - trace.mean())
    for key in ('time_traces', 'complex_responses', 'power'):
        np.testing.assert_array_equal(result[key], original[key])
    assert plt.rcParams == style


def test_readout_noise_labels_measured_and_clean_stationary_figures(data):
    report, result = data
    result['readout_noise'] = {'noise_applied': True}
    figures = create_profile_figures(report, result)
    for name in ('fig_time_domain', 'fig_phase_progression', 'fig_aoa_spectrum'):
        assert any('Measured data' in text.get_text() for text in figures[name].texts)
    for name in ('fig_spectrum', 'fig_analytical'):
        assert any('no measurement noise' in text.get_text() for text in figures[name].texts)


def test_normalized_if_phase_not_ideal_geometry_and_selected_angle_marker(data):
    report, result = data
    figures = create_profile_figures(report, result)
    phase = figures['fig_phase_progression'].axes[0]
    np.testing.assert_allclose(phase.lines[0].get_ydata(), [10., -12.5, -35.], atol=1e-12)
    assert '-22.50' in phase.lines[1].get_label()
    doa = figures['fig_aoa_spectrum'].axes[0]
    assert 'music' in doa.get_title() and 'Beamforming' not in doa.get_title()
    np.testing.assert_allclose(doa.lines[0].get_ydata(), [-10., 10*np.log10(.5), 0., -10.])
    assert doa.get_ylabel() == 'Normalized spectrum [dB]'
    assert doa.lines[2].get_xdata()[0] == -30.  # selected answer, not argmax at 0
    assert doa.lines[2].get_ydata()[0] == pytest.approx(10*np.log10(.5))


def test_db_display_preserves_zeros_and_extreme_dynamic_range(data):
    report, result = data
    result['power'] = np.array([0., 1e-300, 1e300, 1.])
    original = result['power'].copy()
    ax = create_profile_figures(report, result)['fig_aoa_spectrum'].axes[0]
    values = ax.lines[0].get_ydata()
    assert np.isneginf(values[0])
    np.testing.assert_allclose(values[1:], [-6000., 0., -3000.])
    assert any('not drawn' in text.get_text() for text in ax.texts)
    np.testing.assert_array_equal(result['power'], original)
    ax.figure.canvas.draw()


def test_db_display_is_invariant_to_spectrum_scale(data):
    report, result = data
    first = create_profile_figures(report, result)['fig_aoa_spectrum'].axes[0].lines[0].get_ydata()
    result['power'] = result['power'] * 1000
    second = create_profile_figures(report, result)['fig_aoa_spectrum'].axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_zero_at_selected_angle_is_not_replaced_by_display_floor(data):
    report, result = data
    result['power'][1] = 0.
    ax = create_profile_figures(report, result)['fig_aoa_spectrum'].axes[0]
    assert ax.lines[2].get_xdata()[0] == -30.
    assert np.isneginf(ax.lines[2].get_ydata()[0])


@pytest.mark.parametrize('positions', [ [[0,0,0],[1,0,0],[3,0,0]], [[0,0,0],[1,0,0],[1,1,0]],
                                       [[0,0,0],[2,0,0],[1,0,0]] ])
def test_no_index_fit_for_nonuniform_or_unordered_geometry(data, positions):
    report, result = data
    result['array_positions'] = np.asarray(positions)
    ax = create_profile_figures(report, result)['fig_phase_progression'].axes[0]
    assert len(ax.lines) == 1 and 'no index fit' in ax.texts[0].get_text()


@pytest.mark.parametrize('direction', [-1, 1])
def test_energy_order_and_applied_labels_use_report_only(data, monkeypatch, direction):
    from rydberg_aoa.config import atomic_config as ac
    monkeypatch.setattr(ac, 'require_atomic_physics_initialized', lambda: pytest.fail('global init consulted'))
    report, result = data
    report['rf_orientation']['signed_frequency_hz'] *= -direction
    ax = create_profile_figures(report, result)['energy_levels'].axes[0]
    heights = {text.get_text(): text.get_position()[1] for text in ax.texts}
    assert (heights[r'$|4\rangle$'] > heights[r'$|3\rangle$']) == (direction > 0)
    texts = '\n'.join(heights)
    assert '781.123456' in texts and '484.654321' in texts and '132.6491' in texts
    assert 'mJ=-0.5' in texts and 'mF=0' in texts


@pytest.mark.parametrize('algorithm', ['esprit', 'root_music'])
def test_search_free_has_result_panel_no_fabricated_spectrum(data, algorithm):
    report, result = data
    result['angles'] = result['power'] = None
    result['doa_result'].algorithm = algorithm
    ax = create_profile_figures(report, result)['fig_aoa_spectrum'].axes[0]
    assert not ax.lines and algorithm in ax.get_title() and 'no angular spectrum' in ax.get_title()


def test_spectrum_arrays_are_separate_unmodified_and_approximation_labeled(data):
    report, result = data
    original = copy.deepcopy(result['spectrum'])
    figures = create_profile_figures(report, result)
    for name, suffix in [('fig_spectrum', ''), ('fig_analytical', '_analytical')]:
        ax = figures[name].axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), original['rf_off'+suffix])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), original['rf_on'+suffix])
        assert 'offset from applied center' in ax.get_xlabel()
    assert 'excluded' in figures['fig_analytical'].axes[0].get_title()
    assert 'EIT' not in figures['fig_spectrum']._suptitle.get_text()
    for key in original:
        np.testing.assert_array_equal(result['spectrum'][key], original[key])


def test_optional_curves_are_omitted_and_publication_files_saved(data, tmp_path):
    report, result = data
    result['spectrum']['rf_off_analytical'] = result['spectrum']['rf_on_analytical'] = None
    figures = create_profile_figures(report, result)
    assert 'fig_analytical' not in figures
    save_profile_figures(tmp_path, figures)
    for name in figures:
        assert (tmp_path / f'{name}.png').read_bytes().startswith(b'\x89PNG')
        assert (tmp_path / f'{name}.pdf').read_bytes().startswith(b'%PDF')
    result['spectrum'] = None
    figures = create_profile_figures(report, result)
    assert set(figures) == {'fig_time_domain', 'fig_phase_progression', 'fig_aoa_spectrum', 'energy_levels'}


@pytest.mark.parametrize('key,value', [('time_traces', np.ones((2,3))),
                                      ('time_traces', np.ones((3,3), dtype=complex)),
                                      ('complex_responses', np.zeros(3)),
                                      ('power', np.zeros(4)), ('power', np.array([1,np.nan,1,1])),
                                      ('angles', np.array([0,1,1,2]))])
def test_invalid_data_fail_before_any_figure(data, key, value):
    report, result = data
    result[key] = value
    existing = plt.get_fignums()
    with pytest.raises(PhysicsCalculationError, match='LAB-008'):
        create_profile_figures(report, result)
    assert plt.get_fignums() == existing


def test_missing_applied_wavelength_is_not_inferred_from_energy(data):
    report, result = data
    del report['applied_fields']['coupling']['wavelength_nm']
    report['levels']['excited']['energy'] = -2.
    report['levels']['rydberg1']['energy'] = -1.
    with pytest.raises(PhysicsCalculationError, match='provenance') as caught:
        create_profile_figures(report, result)
    assert isinstance(caught.value.__cause__, KeyError)
    assert not plt.get_fignums()


def test_high_l_label_is_not_clamped_to_i():
    from rydberg_aoa.visualization.plot_energy_levels import _term_label
    assert 'K' in _term_label(dict(n=30, l=7, j=7.5), 3, False)


@pytest.mark.parametrize('key,value', [('signed_if_mhz', -.1), ('rabi_rad_per_us', [1., 2., np.nan, 1.])])
def test_invalid_label_metadata_is_rejected(data, key, value):
    report, result = data
    report[key] = value
    with pytest.raises(PhysicsCalculationError, match='LAB-008'):
        create_profile_figures(report, result)
    assert not plt.get_fignums()
