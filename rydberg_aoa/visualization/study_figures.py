"""User-study plots from stored metrics; no physics or fabricated undefined data."""
import numpy as np
import matplotlib.pyplot as plt
from ..config.study import axis_spec, axis_label, METRIC_LABELS
from .profile_figures import PUBLICATION_STYLE


def create_study_figures(study):
    from ..config.operating_study import OperatingStudy
    if isinstance(study.definition,OperatingStudy):
        figures = {}
        with plt.rc_context(PUBLICATION_STYLE):
            for metric,label in [('predicted_if_snr_db','Predicted conditional IF SNR [dB]'),('mean_transmission','Mean optical transmission')]:
                fig,ax = plt.subplots(figsize=(9,5))
                for length in study.definition.cell_lengths_m:
                    points = [p for p in study.points if p.value[1]==length]
                    ax.plot([p.value[0] for p in points],
                        [np.nan if p.summary[metric][study.definition.path_index] is None else p.summary[metric][study.definition.path_index] for p in points],
                        'o-',label=f'L={length*1000:g} mm')
                ax.set(xlabel='Applied probe detuning offset [MHz]',ylabel=label,title=study.definition.name)
                ax.legend();ax.grid(True,alpha=.3)
                fig.text(.5,.01,'Sampled operating candidates; detector assumptions; no detection/sensitivity certification',ha='center',fontsize=9)
                figures['study_operating_'+metric]=fig
        return figures
    from ..config.detection_study import DetectionStudy
    if isinstance(study.definition,DetectionStudy):
        from .detection_figures import create_detection_figures
        report=study.report();figures=create_detection_figures(report)
        if 'numerical_validation' in report:
            from .detection_numerical_figures import create_detection_numerical_figures
            figures.update(create_detection_numerical_figures(report))
        return figures
    from ..config.field_study import SignalFieldStudy
    if isinstance(study.definition,SignalFieldStudy):
        return create_field_study_figures(study.report())
    from ..config.noise_convergence import NoiseConvergenceStudy
    if isinstance(study.definition,NoiseConvergenceStudy):
        return create_convergence_figures(study.report())
    figures = {}
    if not study.points:
        return figures
    uncertainty = study.report().get('uncertainty') if study.definition.statistics is not None else None
    with plt.rc_context(PUBLICATION_STYLE):
        x = np.asarray([p.value for p in study.points])
        for metric in study.definition.metrics:
            y = [p.summary['metrics'][metric] for p in study.points]
            fig,ax = plt.subplots(figsize=(9,5))
            if isinstance(y[0],list):
                for path in range(len(y[0])):
                    values = [row[path] for row in y]
                    reference = metric == 'relative_if_phase_rmse_deg_per_path' and path == study.definition.phase_reference_path
                    ax.plot(x,[np.nan if v is None else v for v in values],'o-',
                            label=f'Path {path+1}' + (' (reference)' if reference else ''))
                    if uncertainty:
                        entries = [p['metrics'][metric][path] for p in uncertainty['checkpoints'][-1]['points']]
                        _band(ax,x,entries,ax.lines[-1].get_color())
                ax.legend()
            else:
                ax.plot(x,y,'o-')
                if uncertainty:
                    entries = [p['metrics'][metric] for p in uncertainty['checkpoints'][-1]['points']]
                    _band(ax,x,entries,ax.lines[-1].get_color())
            label = axis_label(study.definition.source,study.definition.parameter)
            unit = axis_spec(study.definition.source,study.definition.parameter)[0]
            ax.set(xlabel=f'{label} [{unit}]', ylabel=METRIC_LABELS[metric])
            ax.set_title(study.definition.name,parse_math=False)
            ax.grid(True,alpha=.3)
            undefined = any(v is None for row in y for v in (row if isinstance(row,list) else [row]))
            caption = f'{study.status}; {study.definition.trials_per_point} trials/point; no convergence claim'
            caption += ('; empirical percentile bootstrap intervals' if uncertainty else '; no confidence intervals')
            if undefined: caption += '; gaps denote undefined values'
            if metric == 'relative_if_phase_rmse_deg_per_path':
                caption += f'; reference path {study.definition.phase_reference_path+1} is zero by definition'
            fig.text(.5,.01,caption,ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,1))
            figures['study_'+metric] = fig
    if uncertainty:
        figures.update(create_statistics_figures(uncertainty))
    return figures


def create_field_study_figures(report):
    """All measured fields retained; no implicit crop to a successful region."""
    if not report['points']:
        return {}
    figures = {}
    x = np.array([p['value'] for p in report['points']])
    amplitudes = np.array([p['summary']['if_amplitude_per_path'] for p in report['points']])
    unit = report['points'][0]['summary']['response_unit']
    analysis = report['analysis']
    with plt.rc_context(PUBLICATION_STYLE):
        names = ['field_if_amplitude'] + ([] if analysis is None else ['field_responsivity','field_linearity_deviation','field_derivative_spacing'])
        for name in names:
            fig, ax = plt.subplots(figsize=(9,5))
            if name == 'field_if_amplitude':
                for path in range(amplitudes.shape[1]):
                    line = ax.plot(x,amplitudes[:,path],'o-',label=f'Path {path+1}')[0]
                    if analysis:
                        ax.plot(x,np.asarray(analysis['linearity']['predicted_amplitude_per_path'])[:,path],
                                '--',alpha=.6,color=line.get_color(),label=f'Path {path+1} affine fit')
                ax.set_ylabel(f'IF peak amplitude [{unit}]')
            elif name == 'field_responsivity':
                for path in range(amplitudes.shape[1]):
                    line = ax.plot(x,np.asarray(analysis['responsivity_refined_per_path'])[:,path],'o-',label=f'Path {path+1} refined')[0]
                    ax.plot(x[::2],np.asarray(analysis['responsivity_base_per_path'])[:,path],'x--',color=line.get_color(),label=f'Path {path+1} base')
                ax.set_ylabel(f'd|A_IF|/dE [{unit}/(V/m)]')
            elif name == 'field_linearity_deviation':
                for path in range(amplitudes.shape[1]):
                    y = [row[path] for row in analysis['linearity']['relative_deviation_per_path']]
                    ax.plot(x,[np.nan if v is None else v for v in y],'o-',label=f'Path {path+1}')
                ax.axhline(analysis['linearity']['relative_budget'],color='black',linestyle='--',label='Declared budget (inside region)')
                lo,hi = analysis['linearity']['region_v_per_m']
                ax.axvspan(lo,hi,alpha=.1,color='green',label='Declared fit/check region')
                ax.set_ylabel('Relative deviation from affine fit')
            else:
                for path in range(amplitudes.shape[1]):
                    ax.plot(x[::2],np.asarray(analysis['derivative_comparison']['absolute_difference_at_base_per_path'])[:,path],
                            'o-',label=f'Path {path+1}')
                ax.set_ylabel(f'|Refined minus base derivative| [{unit}/(V/m)]')
            ax.set_xlabel('Applied RF signal field [V/m]')
            ax.set_title(report['study']['name'],parse_math=False)
            ax.grid(True,alpha=.3); ax.legend()
            fig.text(.5,.01,f"{report['status']}; stochastic sources off; no MDF/NEF or hardware calibration; gaps = undefined",
                     ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,1))
            figures[name] = fig
    return figures


def create_convergence_figures(report):
    """Offline renderer of actual case discrepancies, not reconstructed spectra."""
    figures={}
    if not report['points']:
        return figures
    labels={'if_response_absolute':'Maximum absolute IF response difference [trace units]',
            'if_response_relative':'Maximum relative complex IF difference',
            'if_phase_deg':'Maximum wrapped IF phase difference [deg]',
            'if_amplitude_relative':'Maximum relative IF amplitude difference',
            'aoa_deg':'Maximum azimuth difference [deg]'}
    with plt.rc_context(PUBLICATION_STYLE):
        for metric in report['study']['metrics']:
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            undefined=False
            for ax,axis in zip(axes,('output_samples','noise_step_us')):
                points=[p for p in report['points'] if p['axis'] in (axis,'reference')]
                points=sorted(points,key=lambda p:p[axis])
                values=[p['comparison']['metrics'][metric]['maximum'] for p in points]
                undefined = undefined or any(v is None for v in values)
                ax.plot([p[axis] for p in points],[np.nan if v is None else v for v in values],'o-')
                budget=report['study']['budgets'].get(metric)
                if budget is not None:ax.axhline(budget,color='red',linestyle='--',label='Declared budget');ax.legend()
                ax.set(xlabel='Output samples' if axis=='output_samples' else 'Noise step [us]',ylabel=labels[metric])
                ax.grid(True,alpha=.3)
            fig.suptitle(report['study']['name'],parse_math=False)
            caption=f"{report['status']}; finest-case self-comparison is not a convergence test; no continuum/adequacy claim"
            if undefined:caption+='; gaps denote undefined values'
            fig.text(.5,.01,caption,ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.95))
            figures['study_grid_'+metric]=fig
    return figures


def _band(ax,x,entries,color):
    bounds = [e['interval'] for e in entries]
    ax.fill_between(x,[np.nan if b is None else b[0] for b in bounds],
                    [np.nan if b is None else b[1] for b in bounds],color=color,alpha=.18)


def create_statistics_figures(analysis):
    """Pure renderer for offline analysis: estimates versus reused trial prefixes."""
    figures = {}
    checks = analysis['checkpoints']
    if not checks or not checks[-1]['points']:
        return figures
    with plt.rc_context(PUBLICATION_STYLE):
        for metric in checks[-1]['points'][0]['metrics']:
            fig,ax = plt.subplots(figsize=(9,5))
            x = [c['trials'] for c in checks]
            for point in checks[-1]['points']:
                final = point['metrics'][metric]
                width = len(final) if isinstance(final,list) else 1
                for path in range(width):
                    entries = [c['points'][point['index']]['metrics'][metric] for c in checks]
                    if isinstance(final,list): entries = [e[path] for e in entries]
                    label = f"Value {point['value']:g} [{analysis['axis']['unit']}]" + (f', path {path+1}' if isinstance(final,list) else '')
                    ax.plot(x,[np.nan if e['estimate'] is None else e['estimate'] for e in entries],'o-',label=label)
                    _band(ax,x,entries,ax.lines[-1].get_color())
            ax.set(xlabel='Trials in reused prefix',ylabel=METRIC_LABELS[metric],title='Monte Carlo sampling uncertainty')
            ax.grid(True,alpha=.3)
            ax.legend(fontsize=9)
            level = analysis['configuration']['confidence_level']
            fig.text(.5,.01,f'{100*level:g}% empirical percentile intervals; no adequacy or numerical-convergence claim',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,1))
            figures['study_trials_'+metric] = fig
    return figures
