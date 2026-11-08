"""Measured IF detection probabilities and intervals, without extrapolation."""
import numpy as np
import matplotlib.pyplot as plt
from .profile_figures import PUBLICATION_STYLE


def create_detection_figures(report):
    figures = {}
    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=(9,5))
        points = report['points']
        for point in points:
            p = point['summary']['probability']
            if p['probability'] is not None:
                lo,hi = p['confidence_interval']; value = p['probability']
                ax.errorbar(point['value'],value,yerr=[[value-lo],[hi-value]],fmt='o',capsize=4,color='tab:blue')
        ax.axhline(report['study']['p_detection'],linestyle='--',color='gray',label='Target P_D')
        ax.set(xlabel='Applied signal field [V/m]',ylabel='Detection probability',ylim=(0,1))
        ax.set_title(report['study']['name'],parse_math=False)
        ax.legend();ax.grid(True,alpha=.3)
        caption = report['status']+'; exact simultaneous intervals; no interpolation or minimum-field claim'
        sensitivity = report.get('sampled_sensitivity')
        if sensitivity is not None and sensitivity.get('status') != 'disabled':
            ax.set_ylim(-.03,1.03)  # Keep endpoint markers/interval caps visible.
            # Recompute from raw recorded evidence; never trust a cached minimum.
            from ..simulation.sampled_sensitivity import analyze_sampled_sensitivity
            audited = analyze_sampled_sensitivity(report)
            field = audited['minimum_supported_field_v_per_m']
            if field is not None:
                ax.axvline(field,color='tab:red',linestyle=':',label=f'Smallest supported sampled field: {field:g} V/m')
                ax.legend()
            caption = report['status']+'; '+audited['status']+'; sampled grid only; numerical convergence not certified'
        if report['threshold'] is None or report['threshold']['value'] is None:
            caption += '; threshold undefined: no probability points'
        fig.text(.5,.01,caption,ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,1))
        figures['study_detection_probability'] = fig
        calibration = report['calibration']
        if calibration is not None:
            unit = calibration['summary']['evidence']['statistic_units']
            fig,ax = plt.subplots(figsize=(9,5))
            for label,record in [('Null calibration',calibration),('Held-out null',report['null_validation'])]:
                if record is not None:
                    scores = np.sort(record['summary']['evidence']['scores'])
                    ax.step(scores,np.arange(1,len(scores)+1)/len(scores),where='post',label=label)
            threshold = report['threshold']
            if threshold is not None and threshold['value'] is not None:
                ax.axvline(threshold['value'],color='tab:red',linestyle='--',label='Frozen threshold')
            ax.set(xlabel=f'Squared IF peak magnitude [{unit}]',ylabel='Empirical cumulative fraction',ylim=(0,1),title='Independent null campaigns')
            ax.legend();ax.grid(True,alpha=.3);fig.tight_layout()
            figures['study_detection_null_distribution'] = fig
    return figures
