"""Publication style receiver figures; no angle or phase plots for IF-only."""
import numpy as np
import matplotlib.pyplot as plt
from ..simulation.if_only import summarize_if_result
from .profile_figures import PUBLICATION_STYLE


def create_if_only_figures(report,result):
    stats=summarize_if_result(result)
    units=stats['trace_units']; figures={}
    traces=np.asarray(result['time_traces'])
    values=np.asarray([t['if_real_per_path'] for t in stats['trials']])+1j*np.asarray([t['if_imag_per_path'] for t in stats['trials']])
    label='Signal ON' if result['signal_enabled'] else 'Signal OFF reference'
    with plt.rc_context(PUBLICATION_STYLE):
        fig,ax=plt.subplots(figsize=(9,5))
        for path,trace in enumerate(traces):
            ax.plot(result['tlist'],trace-np.mean(trace),label=f'Path {path+1}')
        ax.set(xlabel='Time [us]',ylabel=f'AC display copy [{units}]',title=label+' — receiver trace')
        ax.legend();ax.grid(True,alpha=.3);fig.tight_layout();figures['if_only_time_domain']=fig
        fig,ax=plt.subplots(figsize=(9,5))
        for path in range(values.shape[1]):
            ax.plot(np.arange(len(values)),np.abs(values[:,path]),'o-',label=f'Path {path+1}')
        ax.set(xlabel='Recorded trial index',ylabel=f'IF peak amplitude [{units}]',title=label+' — measured IF magnitudes')
        ax.legend();ax.grid(True,alpha=.3);fig.tight_layout();figures['if_only_trial_amplitude']=fig
        fig,ax=plt.subplots(figsize=(7,6))
        for path in range(values.shape[1]):
            ax.plot(values[:,path].real,values[:,path].imag,'o',label=f'Path {path+1}')
        ax.set(xlabel=f'Re(A_IF) [{units}]',ylabel=f'Im(A_IF) [{units}]',title=label+' — raw demodulation I/Q')
        ax.legend();ax.grid(True,alpha=.3);fig.tight_layout();figures['if_only_iq']=fig
    return figures
