"""Standalone publication figures from explicit profile outputs (LAB-008).

No ARC queries, solver calls, global parameter reads or reconstructed curves.
Mean subtraction and logarithmic peak normalization affect display copies only.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..exceptions import PhysicsCalculationError
from .plot_phase_ramp import plot_phase_ramp
from .plot_phase_progression import plot_phase_progression
from .plot_aoa_spectrum import plot_aoa_spectrum
from .plot_doa_summary import plot_doa_summary
from .style import COLOR_PALETTE_MODERN


PUBLICATION_STYLE = {
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 1.1, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True,
    'ytick.minor.visible': True, 'xtick.major.size': 7, 'ytick.major.size': 7,
    'xtick.minor.size': 4, 'ytick.minor.size': 4,
    'axes.titleweight': 'bold', 'savefig.facecolor': 'white',
}


def _array(value, name, *, ndim, complex_allowed=False):
    try:
        data = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhysicsCalculationError(f'LAB-008: invalid {name}; cannot form numeric data.') from exc
    kinds = 'iufc' if complex_allowed else 'iuf'
    if data.dtype.kind not in kinds or data.ndim != ndim or data.size == 0 or not np.isfinite(data).all():
        raise PhysicsCalculationError(f'LAB-008: invalid {name}; finite numeric {ndim}-D data required.')
    return data


def _uniform_ordered_paths(positions):
    """An index fit has spatial meaning only for equally spaced ordered paths.

    Tolerance is solely for floating representation of declared positions;
    no geometry, phase, or angle is repaired or inferred from this decision.
    """
    steps = np.diff(positions, axis=0)
    scale = np.linalg.norm(steps[0])
    return bool(scale > 0 and np.allclose(steps, steps[0], rtol=1e-9, atol=scale*1e-12))


def _validate(report, result):
    times = _array(result['tlist'], 'time grid [us]', ndim=1)
    traces = _array(result['time_traces'], 'worker observable traces', ndim=2)
    responses = _array(result['complex_responses'], 'normalized IF responses', ndim=1, complex_allowed=True)
    positions = _array(result['array_positions'], 'path positions [m]', ndim=2)
    if (len(times) < 2 or not np.all(np.diff(times) > 0) or
            traces.shape != (len(responses), len(times)) or positions.shape != (len(responses), 3)
            or len(responses) < 2 or np.any(np.abs(responses) == 0)):
        raise PhysicsCalculationError('LAB-008: incompatible time/path shapes or undefined IF phase.')
    for key in ('true_aoa', 'estimated_aoa'):
        value = _array(result[key], key, ndim=0)
        if value.size != 1:
            raise PhysicsCalculationError(f'LAB-008: invalid {key}.')
    doa = result['doa_result']
    if not isinstance(doa.algorithm, str) or not doa.algorithm:
        raise PhysicsCalculationError('LAB-008: missing actual estimator name.')
    if float(doa.azimuth_deg) != float(result['estimated_aoa']):
        raise PhysicsCalculationError('LAB-008: estimator and displayed angle disagree.')
    if (result['power'] is None) != (result['angles'] is None):
        raise PhysicsCalculationError('LAB-008: incomplete DoA spectrum/grid pair.')
    if result['power'] is not None:
        angles = _array(result['angles'], 'DoA scan angles', ndim=1)
        power = _array(result['power'], 'DoA spectrum', ndim=1)
        if (power.shape != angles.shape or len(angles) < 2 or not np.all(np.diff(angles) > 0)
                or np.any(power < 0) or power.max() <= 0
                or not angles[0] <= result['estimated_aoa'] <= angles[-1]):
            raise PhysicsCalculationError('LAB-008: invalid angular spectrum; no normalization substitute.')
    spectrum = result.get('spectrum')
    if spectrum is not None:
        x = _array(spectrum['probe_offsets_mhz'], 'probe offsets [MHz]', ndim=1)
        if len(x) < 2 or not np.all(np.diff(x) > 0):
            raise PhysicsCalculationError('LAB-008: spectrum grid must be increasing.')
        for key in ('rf_off', 'rf_on', 'rf_off_analytical', 'rf_on_analytical'):
            curve = spectrum[key]
            if curve is None and key.endswith('_analytical'):
                continue
            if _array(curve, key, ndim=1).shape != x.shape:
                raise PhysicsCalculationError(f'LAB-008: mismatched {key} curve/grid.')
        if (spectrum['rf_off_analytical'] is None) != (spectrum['rf_on_analytical'] is None):
            raise PhysicsCalculationError('LAB-008: incomplete analytical curve pair.')
    # Report is the frozen provenance of labels, not mutable global config.
    signed = _array(report['rf_orientation']['signed_frequency_hz'], 'signed atomic RF frequency', ndim=0)
    if signed == 0:
        raise PhysicsCalculationError('LAB-008: zero atomic RF frequency has no energy orientation.')
    for name in ('probe', 'coupling'):
        if _array(report['applied_fields'][name]['wavelength_nm'], f'{name} wavelength [nm]', ndim=0) <= 0:
            raise PhysicsCalculationError(f'LAB-008: nonpositive {name} wavelength.')
    for name in ('lo', 'signal'):
        if _array(report['applied_fields'][name]['frequency_hz'], f'{name} carrier [Hz]', ndim=0) <= 0:
            raise PhysicsCalculationError(f'LAB-008: nonpositive {name} carrier.')
    signed_if = _array(report['signed_if_mhz'], 'signed IF [MHz]', ndim=0)
    fields = report['applied_fields']
    if signed_if != (fields['signal']['frequency_hz'] - fields['lo']['frequency_hz']) / 1e6:
        raise PhysicsCalculationError('LAB-008: IF label disagrees with applied carriers.')
    if spectrum is not None:
        if _array(report['rabi_rad_per_us'], 'Rabi frequencies [rad/us]', ndim=1).shape != (4,):
            raise PhysicsCalculationError('LAB-008: four Rabi frequencies required for plotting provenance.')
        _array(report['sensing']['execution']['sensor_probe_offset_mhz'], 'sensor probe offset [MHz]', ndim=0)
        if _array(report['resolved_rate_budget_per_us']['transit_reset'], 'transit rate [1/us]', ndim=0) < 0:
            raise PhysicsCalculationError('LAB-008: negative transit rate.')
    return times, traces, responses, positions


def _spectrum_figure(report, result, *, analytical):
    spectrum = result['spectrum']
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    x = spectrum['probe_offsets_mhz']
    suffix = '_analytical' if analytical else ''
    omega_mhz = report['rabi_rad_per_us'][2] / (2*np.pi)
    for key, color, label in (('rf_off', 'blue', 'RF off'),
                              ('rf_on', 'red', rf'LO on, $\Omega_{{LO}}/(2\pi)={omega_mhz:.2f}$ MHz')):
        ax.plot(x, spectrum[key + suffix], 'o-', color=color, linewidth=2, alpha=.8, label=label)
    fig.suptitle('Reduced Analytical Sensor Spectrum' if analytical else 'Numerical Sensor Spectrum',
                 fontsize=15, fontweight='bold', y=.98)
    if analytical:
        subtitle = 'Zero velocity; Doppler, transit and full upper-state dephasing excluded'
    else:
        declaration = report['spectrum_declaration']
        subtitle = ('Doppler coherence average' if declaration['doppler_enabled'] else 'Zero velocity')
        subtitle += '; transit reset included' if report['resolved_rate_budget_per_us']['transit_reset'] > 0 else '; transit reset off'
        offset = report['sensing']['execution']['sensor_probe_offset_mhz']
        ax.axvline(offset, color='#555555', linestyle='--', linewidth=1.2,
                   label=f'Sensor probe offset ({offset:+g} MHz)')
    ax.set_title(subtitle, fontsize=10, style='italic', pad=12)
    ax.set(xlabel='Probe offset from applied center [MHz]', ylabel='Cell transmission', xlim=(x[0], x[-1]))
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(.98, .91), bbox_transform=fig.transFigure)
    ax.grid(True, alpha=.3)
    fig.subplots_adjust(top=.73, bottom=.13, left=.12, right=.97)
    return fig


def create_profile_figures(report, result):
    """Return named figures using measured arrays and selected estimator only.

    Optional spectra are omitted when not computed. A search-free method gets
    a result panel in fig_aoa_spectrum, explicitly without an angular curve.
    """
    if result is not None and result.get('measurement_mode') == 'if_only':
        from .if_only_figures import create_if_only_figures
        return create_if_only_figures(report,result)
    try:
        times, traces, responses, positions = _validate(report, result)
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise PhysicsCalculationError('LAB-008: incomplete or invalid plotting provenance.') from exc
    figures = {}
    existing = set(plt.get_fignums())
    try:
        with plt.rc_context(PUBLICATION_STYLE):
            fig = plt.figure(figsize=(9.2, 5.5))
            ax = plot_phase_ramp(times, traces, colors=COLOR_PALETTE_MODERN)
            ax.set_ylabel(r'$\mathrm{Im}\langle\sigma_{21}\rangle(t)-\overline{\mathrm{Im}\langle\sigma_{21}\rangle}$')
            if 'detector_arrays' in result:
                ax.set_ylabel(f"Detector output minus mean [{result['readout_noise']['units']}]")
            figures['fig_time_domain'] = fig
            # Responses already carry the RF-direction normalization from the
            # production boundary. Never conjugate again or use ideal phases.
            phase_fig, _, _ = plot_phase_progression(
                np.rad2deg(np.unwrap(np.angle(responses))), result['true_aoa'], result['estimated_aoa'],
                fit_linear=_uniform_ordered_paths(positions))
            figures['fig_phase_progression'] = phase_fig
            fig = plt.figure(figsize=(8.4, 5.5))
            if result['power'] is None:
                plot_doa_summary(result['doa_result'], result['true_aoa'])
            else:
                power = np.asarray(result['power'])
                ax = plot_aoa_spectrum(result['angles'], power, result['true_aoa'],
                                       result['estimated_aoa'], algorithm=result['doa_result'].algorithm)
                # Broad lobes can occupy the traditional upper-left box.
                # Put the annotation in the lower band, leaving the peak clear.
                if np.max(power[:max(1, len(power)//4)]) / power.max() > .6:
                    ax.texts[0].set_position((.04, .28))
            figures['fig_aoa_spectrum'] = fig
            if 'readout_noise' in result or 'rf_drive_campaign' in result:
                campaign = result.get('noise_campaign', result.get('rf_drive_campaign'))
                readout = result.get('readout_noise', {'noise_applied': False})
                label = ('Measured data: effective Gaussian readout noise'
                         if readout['noise_applied'] else 'Zero-sigma readout: clean data')
                if 'detector_arrays' in result:
                    label = ('PIN detector: shot/electronic readout' if result['readout_noise']['noise_applied']
                             else 'PIN detector: noiseless readout')
                if 'rf_drive_campaign' in result:
                    label = ('Shared colored drive noise; ' if result['rf_drive_campaign']['drive_noise_included']
                             else 'Zero drive noise; ')+('with readout noise' if readout['noise_applied'] else 'no readout noise')
                    if result['rf_drive_campaign']['laser_frequency_noise_included']:
                        label += '; residual laser frequency noise'
                    if result['rf_drive_campaign']['laser_intensity_noise_included']:
                        label += '; laser intensity noise'
                if campaign is not None and campaign['trials_completed'] > 1:
                    label += f"; trial 0 of {campaign['trials_completed']} (not campaign mean)"
                for name in ('fig_time_domain', 'fig_phase_progression', 'fig_aoa_spectrum'):
                    figures[name].text(.5, .01, label,
                                       ha='center', fontsize=9)
            campaign = result.get('noise_campaign', result.get('rf_drive_campaign'))
            if campaign is not None and campaign['trials_completed'] > 1:
                fig, ax = plt.subplots(figsize=(9, 5))
                rows = campaign['trials']
                ax.plot([row['trial_index'] for row in rows], [row['error_deg'] for row in rows], 'o-')
                ax.axhline(0, color='gray', linestyle=':')
                ax.set(xlabel='Noise trial index', ylabel='Wrapped azimuth error [deg]',
                       title=f"Noise campaign ({result['doa_result'].algorithm})")
                ax.text(.04, .96, f"RMSE: {campaign['rmse_deg']:.6g} deg\nBias: {campaign['bias_deg']:.6g} deg",
                        transform=ax.transAxes, va='top')
                fig.tight_layout()
                figures['fig_noise_campaign'] = fig
            if 'rf_drive_realizations' in result and result['rf_drive_realizations'][0] is not None:
                realization = result['rf_drive_realizations'][0]
                fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
                for ax, values, label in zip(axes, (realization.lo_amplitude, realization.signal_amplitude,
                        realization.relative_phase_rad), ('LO amplitude multiplier', 'Signal amplitude multiplier', 'Relative envelope phase [rad]')):
                    ax.plot(realization.times_us, values)
                    ax.set_ylabel(label, fontsize=9)
                    ax.grid(True, alpha=.3)
                axes[-1].set(xlabel='Time [us]', xlim=(times[0], times[-1]))
                fig.suptitle('Common RF drive realization, trial 0')
                fig.tight_layout()
                figures['fig_rf_drive_noise'] = fig
                if result['rf_drive_campaign']['laser_frequency_noise_included']:
                    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                    for ax, values, label in zip(axes,
                            (realization.probe_frequency_hz, realization.coupling_frequency_hz),
                            ('Probe frequency offset [Hz]', 'Coupling frequency offset [Hz]')):
                        ax.plot(realization.times_us, values)
                        ax.set_ylabel(label)
                        ax.grid(True, alpha=.3)
                    axes[-1].set(xlabel='Time [us]', xlim=(times[0], times[-1]))
                    fig.suptitle('Common residual laser frequency realization, trial 0')
                    fig.tight_layout()
                    figures['fig_laser_frequency_noise'] = fig
                if result['rf_drive_campaign']['laser_intensity_noise_included']:
                    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                    for ax, values, label in zip(axes,
                            (realization.probe_intensity, realization.coupling_intensity),
                            ('Probe intensity / nominal', 'Coupling intensity / nominal')):
                        ax.plot(realization.times_us, values)
                        ax.axhline(1., color='gray', linestyle=':')
                        ax.set_ylabel(label)
                        ax.grid(True, alpha=.3)
                    axes[-1].set(xlabel='Time [us]', xlim=(times[0], times[-1]))
                    fig.suptitle('Common laser intensity realization, trial 0')
                    fig.tight_layout()
                    figures['fig_laser_intensity_noise'] = fig
            # Lazy import: plotting other figures does not import atomic config.
            from .plot_energy_levels import plot_energy_levels
            figures['energy_levels'] = plot_energy_levels(report=report)
            if result.get('spectrum') is not None:
                figures['fig_spectrum'] = _spectrum_figure(report, result, analytical=False)
                if result['spectrum']['rf_off_analytical'] is not None:
                    figures['fig_analytical'] = _spectrum_figure(report, result, analytical=True)
                if result.get('readout_noise', {}).get('noise_applied', False) or 'rf_drive_campaign' in result:
                    for name in ('fig_spectrum', 'fig_analytical'):
                        if name in figures:
                            figures[name].text(.5, .01, 'Clean stationary model: no measurement noise; no colored drive noise',
                                               ha='center', fontsize=9)
        return figures
    except Exception:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
        raise


def save_profile_figures(directory, figures):
    """Save standalone PNG and vector PDF without overwriting root figures."""
    from pathlib import Path
    directory = Path(directory)
    for name, figure in figures.items():
        for extension in ('png', 'pdf'):
            figure.savefig(directory / f'{name}.{extension}', dpi=300, bbox_inches='tight')
