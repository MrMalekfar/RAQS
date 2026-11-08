"""Certified optical velocity classes, then ensemble coherence and one IF fit.

Each class retains the same RF/laser drive realization and its own LO-only
steady state. No atom transport, RF Doppler or Maxwell-Bloch propagation.
"""
from dataclasses import dataclass, asdict
import math
from time import perf_counter
import numpy as np
from ..exceptions import PhysicsCalculationError, PhysicsError


@dataclass(frozen=True)
class DopplerWorkerResult:
    record: tuple
    report: dict


def ensemble_coherence(plan, times, signed_if_mhz, delay_us, solve):
    """Refine Maxwell quadrature until two consecutive trace/AC/IF checks pass.

Uniform nodes use trapezoidal weights on the truncated Maxwell density;
Gauss-Hermite nodes use the full Maxwell distribution. Weights are validated
before arithmetic. Refinement is declared quadrature, never a solver retry.
"""
    from ..config.temporal_doppler import ResolvedTemporalDoppler
    from ..physics.doppler import generate_velocity_classes, get_doppler_shifts
    from ..physics.doppler.velocity_classes import get_velocity_std
    from ..utils.numerical import extract_if_component
    if not isinstance(plan, ResolvedTemporalDoppler):
        raise PhysicsCalculationError('DOP-TD-001: resolved ensemble required.')
    from ..config.temporal_doppler import AdaptiveTemporalDoppler
    if isinstance(plan.controls, AdaptiveTemporalDoppler):
        raise PhysicsCalculationError('DOP-PROFILE-001: adaptive plans require task-level resonance hints; use run_temporal_ensemble.')
    times = np.asarray(times)
    c = plan.controls
    previous = None
    consecutive = 0
    cache = {}
    history = []
    class_costs = []
    retained = times >= delay_us
    n = c.initial_classes
    while n <= c.max_classes:
        refinement_start = perf_counter()
        prior_solves = len(cache)
        velocities, weights = generate_velocity_classes(n, plan.temperature_k, plan.atom_mass_kg,
                                                       c.method, c.n_sigma,
                                                       **({'velocity_scale_ms':c.velocity_scale_ms} if c.method == 'sinh' else {}))
        if c.method == 'uniform':
            sigma = get_velocity_std(plan.temperature_k, plan.atom_mass_kg)
            weights = np.exp(-.5*(velocities/sigma)**2)
            weights[[0, -1]] *= .5
            weights /= weights.sum()
        if (not np.isfinite(velocities).all() or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-14)):
            raise PhysicsCalculationError('DOP-TD-001: invalid Maxwell nodes/weights; no repair.')
        average = np.zeros(len(times), dtype=complex)
        for v, w in zip(velocities, weights):
            key = float(v)
            if key not in cache:
                shifts = get_doppler_shifts(key, plan.probe_wavelength_m,
                                           plan.coupling_wavelength_m, c.counter_propagating)
                try:
                    solve_start = perf_counter()
                    trace = np.asarray(solve(*shifts))
                except PhysicsError as exc:
                    failure = PhysicsCalculationError(f'DOP-TD-001: velocity {key!r} m/s failed; no dropped class.')
                    failure.velocity_diagnostics = {'history':history,'completed_class_costs':class_costs,
                        'failed_velocity_ms':key,'failed_class_wall_seconds':perf_counter()-solve_start}
                    raise failure from exc
                except Exception as exc:
                    raise PhysicsCalculationError(f'DOP-TD-001: velocity {key!r} m/s calculation failed; no substitute.') from exc
                if trace.shape != times.shape or trace.dtype.kind not in 'fc' or not np.isfinite(trace).all():
                    raise PhysicsCalculationError(f'DOP-TD-001: invalid coherence at velocity {key!r} m/s.')
                cache[key] = trace
                class_costs.append({'velocity_ms':key,'wall_seconds':perf_counter()-solve_start})
            average += w*cache[key]
        if not np.isfinite(average).all():
            raise PhysicsCalculationError('DOP-TD-001: nonfinite ensemble coherence.')
        metrics = {'classes': n,'new_class_solves':len(cache)-prior_solves,
                   'total_class_solves':len(cache),'refinement_wall_seconds':perf_counter()-refinement_start}
        if previous is not None:
            a = average[retained]; b = previous[retained]
            ac_a = a-a.mean(); ac_b = b-b.mean()
            # Componentwise full trace plus AC norm and actual downstream IF.
            full_ok = np.all(np.abs(average-previous) <= c.atol+c.rtol*np.abs(average))
            ac_error = float(np.max(np.abs(ac_a-ac_b)))
            ac_scale = float(np.max(np.abs(ac_a)))
            amp_a = extract_if_component(np.imag(average), times, signed_if_mhz, delay_us)[0]
            amp_b = extract_if_component(np.imag(previous), times, signed_if_mhz, delay_us)[0]
            if_error = float(abs(amp_a-amp_b))
            passed = bool(full_ok and ac_error <= c.atol+c.rtol*ac_scale
                          and if_error <= c.atol+c.rtol*abs(amp_a))
            consecutive = consecutive+1 if passed else 0
            metrics.update(full_trace_passed=bool(full_ok), ac_error=ac_error,
                           if_error=if_error, passed=passed)
            if consecutive == 2:
                history.append(metrics)
                return average, {'enabled': True, 'plan': asdict(plan), 'method': c.method, 'classes': n,
                    'class_solves': len(cache), 'history': history, 'converged': True,
                    'class_costs':class_costs,'solve_wall_seconds':sum(row['wall_seconds'] for row in class_costs),
                    'rf_doppler_included': False, 'velocity_dimension': 1,
                    'truncated_tail_mass': 0. if c.method == 'gauss_hermite' else math.erfc(c.n_sigma/math.sqrt(2)),
                    'normalization': 'full Maxwell' if c.method == 'gauss_hermite' else 'conditional truncated Maxwell',
                    'acceptance': 'two consecutive full-trace, AC and complex-IF refinements'}
        history.append(metrics)
        previous = average
        n = 2*n-1
    failure = PhysicsCalculationError(f'DOP-TD-001: velocity quadrature did not converge within max_classes={c.max_classes}; history={history!r}. No v=0 fallback.')
    failure.velocity_diagnostics = {'history':history,'completed_class_costs':class_costs,
        'total_class_solves':len(cache),'solve_wall_seconds':sum(row['wall_seconds'] for row in class_costs)}
    raise failure


def run_temporal_ensemble(task, times, invoke):
    from ..utils.numerical import extract_if_component
    from ..config.temporal_doppler import AdaptiveTemporalDoppler
    if isinstance(task.temporal_doppler.controls, AdaptiveTemporalDoppler):
        from .adaptive_temporal_doppler import adaptive_ensemble, task_hints
        coherence, report = adaptive_ensemble(task.temporal_doppler, times,
            task.signed_if_mhz, task.transient_delay_us, invoke, task_hints(task),
            measurement_mode=task.measurement_mode)
    else:
        coherence, report = ensemble_coherence(task.temporal_doppler, times, task.signed_if_mhz,
                                             task.transient_delay_us, invoke)
    trace = np.imag(coherence)
    amp, xf, yf = extract_if_component(trace, times, task.signed_if_mhz, task.transient_delay_us)
    return DopplerWorkerResult((amp, xf, yf, task.element_index, trace),
                              dict(report, element_index=task.element_index))
