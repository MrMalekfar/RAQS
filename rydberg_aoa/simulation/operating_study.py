"""Canonical operating-point runs and conditional PIN noise-budget analysis.

No new atomic solution, transfer model, detector generator or solver settings.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.constants import elementary_charge, Boltzmann
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from ..config.operating_study import validate_operating_profile, operating_points, operating_profile
from ..utils.numerical import if_fit_noise_weights


def point_budget(profile, report, result):
    """Exact conditional IF variance for the existing independent PIN samples.

    Each source remains explicitly enabled/disabled. White electronic/Johnson
    PSDs are one-sided. Counting variance is q I/tau. Multiplication by squared
    IF weights gives mean-square complex peak amplitude (not variance of |A|).
    """
    try:
        if (report['applied_fields'] != {k:asdict(getattr(profile,k)) for k in ('probe','coupling','lo','signal')} or
                report['noise_declaration'] != asdict(profile.noise) or
                report['measurement'] != asdict(profile.measurement) or report['mode'] != profile.mode):
            raise PhysicsCalculationError('NOISE-018: actual applied fields/noise/receiver differ from the planned candidate.')
        d = profile.noise.detector
        data = result['detector_arrays']
        y = np.asarray(result['clean_time_traces'])
        a = np.asarray(result['clean_complex_responses'])
        power = np.asarray(data['clean_optical_power_w'])
        if (y.ndim != 2 or power.shape != y.shape or a.shape != (len(y),) or
                any(v.dtype.kind not in 'iufc' or not np.isfinite(v).all() for v in (y,a,power)) or
                np.iscomplexobj(y) or np.iscomplexobj(power) or np.any(power < 0) or
                not np.array_equal(result['tlist'],profile.run_config.tlist_us)):
            raise PhysicsCalculationError('NOISE-018: finite clean PIN traces/IF/power on declared grid required.')
        selected, weights, fit = if_fit_noise_weights(result['tlist'],profile.signed_if_mhz,profile.run_config.transient_delay_us)
        gain = d.transimpedance_v_per_a if d.output_units == 'V' else 1.
        tau = d.integration_time_us*1e-6
        contributions = {}
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for source in profile.noise.sources:
                spec = source.specification
                variance = np.zeros(power.shape)
                if profile.noise.enabled and spec.enabled:
                    if source.name == 'photon_shot': variance = elementary_charge*d.responsivity_a_per_w*power/tau
                    elif source.name == 'dark_shot': variance.fill(elementary_charge*d.dark_current_a/tau)
                    elif source.name == 'electronic_current': variance.fill(spec.asd_a_per_sqrt_hz**2/(2*tau))
                    elif source.name == 'johnson_current': variance.fill(4*Boltzmann*spec.temperature_k/spec.shunt_resistance_ohm/(2*tau))
                    else: raise PhysicsCalculationError('NOISE-018: unsupported PIN budget source; no approximation.')
                contributions[source.name] = np.sum(variance[:,selected]*np.abs(weights)[None,:]**2,axis=1)*gain**2
            total = np.sum(list(contributions.values()),axis=0)
            signal = np.abs(a)**2
            snr = [None if s==0 or v==0 else float(10*(np.log10(s)-np.log10(v))) for s,v in zip(signal,total)]
        if (not np.isfinite(total).all() or not np.isfinite(signal).all() or
                any(v is not None and not math.isfinite(v) for v in snr)):
            raise PhysicsCalculationError('NOISE-018: budget overflow; no repaired SNR.')
        if report['detector_calibration']['status'] != 'same_declared_probe_channel_verified':
            raise PhysicsCalculationError('NOISE-018: calibration witness required.')
        return {'clean_if_real':a.real.tolist(),'clean_if_imag':a.imag.tolist(),
                'clean_if_peak':np.abs(a).tolist(),'clean_if_power':signal.tolist(),
                'if_noise_power_by_source':{k:v.tolist() for k,v in contributions.items()},
                'total_if_noise_power':total.tolist(),'predicted_if_snr_db':snr,'unit':d.output_units,
                'mean_optical_power_w':np.mean(power[:,selected],axis=1).tolist(),
                'mean_transmission':np.mean(data['clean_detector_transmission'][:,selected],axis=1).tolist(),
                'fit':fit,'noise_budget_scope':'conditional variance of existing independent PIN samples; no experimental PSD claim',
                'undefined_snr':'zero clean or noise IF power; never fabricated inf or zero'}
    except PhysicsCalculationError: raise
    except Exception as exc: raise PhysicsCalculationError('NOISE-018: canonical point budget failed; no substitute metric.') from exc


@dataclass(frozen=True)
class OperatingResult:
    status: str
    definition: object
    points: tuple
    base_profile: object
    selected_index: object
    nominal_profile: object = None
    nominal_report: object = None
    nominal_result: object = None

    def report(self):
        return {'schema_version':1,'study':asdict(self.definition),'status':self.status,
                'points_requested':len(operating_points(self.base_profile)),'points_completed':len(self.points),
                'selection':{'status':'sampled_candidate' if self.selected_index is not None else 'not_established',
                             'selected_index':self.selected_index,'method':self.definition.selection,
                             'tie_break':'declared Cartesian index','sensitivity_certified':False},
                'axis_unit':'(MHz,m)','points':[{'index':p.index,'value':p.value,'summary':p.summary,'run_report':p.report} for p in self.points],
                'hardware_calibration_verified':False,'physical_model_changed':False,
                'independent_detection_validation_required':True}


def execute_operating_study(profile, *, on_progress=None, should_cancel=None):
    from .study_execution import StudyProgress, StudyPointResult, _execute_isolated
    validate_operating_profile(profile)
    for hook in (on_progress, should_cancel):
        if hook is not None and not callable(hook): raise PhysicsConfigurationError('NOISE-018: callable hooks required.')
    design = operating_points(profile); points=[]
    def emit(state, i=None):
        if on_progress is not None:
            on_progress(StudyProgress(state,len(points),len(design),i,None if i is None else design[i],points[-1] if state=='point_completed' else None))
    for i, value in enumerate(design):
        if should_cancel is not None:
            answer = should_cancel()
            if type(answer) is not bool: raise PhysicsConfigurationError('NOISE-018: cancellation must be boolean.')
            if answer:
                emit('cancelled',i); return OperatingResult('cancelled',profile.study,tuple(points),profile,None)
        point = operating_profile(profile,i); emit('point_started',i)
        try:
            report,result = _execute_isolated(point)
            budget = point_budget(point,report,result)
            if points:
                first = points[0].report
                for key in ('levels','rf_orientation','rabi_rad_per_us','decay_rates_per_us','resolved_rate_budget_per_us','quantization_axis'):
                    if first[key] != report[key]: raise PhysicsCalculationError(f'NOISE-018: fixed {key} changed; no retuning/channel switching.')
                initial, current = first['detunings_rad_per_us'], report['detunings_rad_per_us']
                expected_shift = 2*math.pi*(value[0]-points[0].value[0])
                if initial[1:] != current[1:] or not math.isclose(current[0]-initial[0],expected_shift,rel_tol=1e-12,abs_tol=1e-12):
                    raise PhysicsCalculationError('NOISE-018: detunings changed beyond declared probe offset.')
            points.append(StudyPointResult(i,value,point,report,result,budget))
        except Exception as exc:
            emit('failed',i)
            raise PhysicsCalculationError(f'NOISE-018: operating point {i} failed; no dropped point or alternate physics.') from exc
        emit('point_completed',i)
    eligible = [(p.summary['predicted_if_snr_db'][profile.study.path_index],p.index) for p in points]
    eligible = [(v,i) for v,i in eligible if v is not None]
    selected = None if not eligible else max(eligible,key=lambda row:(row[0],-row[1]))[1]
    emit('completed')
    return OperatingResult('completed',profile.study,tuple(points),profile,selected)
