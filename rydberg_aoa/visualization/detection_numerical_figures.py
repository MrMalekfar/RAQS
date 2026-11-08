"""Paired numerical errors; undefined decisions remain plotting gaps."""
import numpy as np
import matplotlib.pyplot as plt
from .profile_figures import PUBLICATION_STYLE


def create_detection_numerical_figures(report):
    data=report.get('numerical_validation');figures={}
    if data is None or not data['comparisons']:return figures
    cases=data['comparisons'];x=np.arange(len(cases));labels=[c['name'] for c in cases]
    unit=report['calibration']['summary']['evidence']['trace_units']
    roles=[r['role'] for r in cases[0]['comparison']['jobs']]
    with plt.rc_context(PUBLICATION_STYLE):
        fig,ax=plt.subplots(figsize=(10,5))
        for j,role in enumerate(roles):
            ax.plot(x,[c['comparison']['jobs'][j]['maximum_if_absolute_error'] for c in cases],'o-',label=role)
        ax.set_xticks(x,labels);ax.set(xlabel='Declared diagnostic case',ylabel=f'Maximum paired |delta A_IF| [{unit}]',
            title='IF discrepancies against coupled baseline')
        ax.set_ylim(bottom=0)
        ax.grid(True,alpha=.3);ax.legend(fontsize=8)
        fig.text(.5,.01,'Absolute errors shown; acceptance uses the declared mixed absolute/relative budget. No continuum bound.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));figures['study_numerical_if_error']=fig
        fig,ax=plt.subplots(figsize=(10,5))
        for j,role in enumerate(roles):
            if role=='calibration':continue
            for key,style,label in [('decision_disagreement_fraction','-','own calibration'),
                ('frozen_baseline_threshold_disagreement_fraction','--','frozen baseline threshold')]:
                y=[c['comparison']['jobs'][j][key] for c in cases]
                ax.plot(x,[np.nan if v is None else v for v in y],'o'+style,label=role+'; '+label)
        ax.set_xticks(x,labels);ax.set(xlabel='Declared diagnostic case',ylabel='Paired decision disagreement fraction',ylim=(-.03,1.03),
            title='Detection decision sensitivity to numerical settings')
        ax.grid(True,alpha=.3);ax.legend(fontsize=7)
        fig.text(.5,.01,'Gaps denote undefined thresholds; paired numerical cases are not independent trial replicates.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));figures['study_numerical_decision_error']=fig
    return figures
