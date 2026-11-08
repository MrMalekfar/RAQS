"""DOP-PROFILE-001 orchestration: bounded exact-node solves, then one IF.

The cache is local to this invocation/sensor/noise realization, never shared
between changed fields, ODE settings or profiles. A failed stage cannot
publish an ensemble. Wall limits are checked around, not inside, native ODE.
"""
from dataclasses import asdict
import math
from time import perf_counter
import numpy as np

from ..exceptions import PhysicsCalculationError
from ..physics.doppler import get_doppler_shifts, resonance_velocity_hints
from ..physics.doppler.velocity_classes import get_velocity_std
from ..physics.doppler.temporal_quadrature import adapt, legendre_rule, compare_traces, validate_measurement_mode


def adaptive_ensemble(plan, times, frequency, delay, solve, hints, *, measurement_mode='aoa'):
    validate_measurement_mode(measurement_mode)
    c = plan.controls
    times = np.asarray(times)
    sigma = get_velocity_std(plan.temperature_k, plan.atom_mass_kg)
    scale = c.velocity_scale_ms
    extent = float(np.arcsinh(c.n_sigma*sigma/scale))
    if not math.isfinite(extent) or extent <= 0:
        raise PhysicsCalculationError('DOP-PROFILE-001: unrepresentable transformed domain.')
    # Same arithmetic as the frozen diagnostic for exact replay at n_sigma=5.
    boundaries = list(np.arange(-c.initial_y_intervals//2, c.initial_y_intervals//2+1)
                      * (extent/(c.initial_y_intervals//2)))
    boundaries[0], boundaries[-1] = -extent, extent
    for center in hints:
        if not math.isfinite(center):
            raise PhysicsCalculationError('DOP-PROFILE-001: nonfinite resonance sampling hint.')
        for offset in c.hint_offsets_ms:
            v = center+offset
            if -c.n_sigma*sigma < v < c.n_sigma*sigma:
                boundaries.append(float(np.arcsinh(v/scale)))
    boundaries = np.array(sorted(set(boundaries)))
    if len(boundaries)-1+2 > c.max_panels:
        raise PhysicsCalculationError('DOP-PROFILE-001: initial hints leave no room for two refinements.')
    cache, costs, history = {}, [], []
    start = perf_counter()

    def failure(message):
        exc = PhysicsCalculationError('DOP-PROFILE-001: '+message+'; no fallback or partial result.')
        exc.velocity_diagnostics = {'history': list(history), 'completed_class_costs': list(costs),
                                    'total_class_solves': len(cache)}
        return exc

    def check_wall():
        if perf_counter()-start >= c.max_wall_seconds:
            raise failure('declared per-path wall budget exhausted')

    def native(v):
        check_wall()
        key = float(v).hex()
        if key not in cache:
            if len(cache) >= c.max_new_solves:
                raise failure('declared per-path solve budget exhausted')
            shifts = get_doppler_shifts(v, plan.probe_wavelength_m,
                                       plan.coupling_wavelength_m, c.counter_propagating)
            began = perf_counter()
            try:
                z = np.asarray(solve(*shifts))
            except Exception as exc:
                raise failure(f'velocity {v!r} m/s failed') from exc
            if z.shape != times.shape or z.dtype.kind not in 'fc' or not np.isfinite(z).all():
                raise failure(f'invalid complete coherence at velocity {v!r} m/s')
            check_wall()
            cache[key] = z.copy()
            costs.append({'velocity_ms': float(v), 'wall_seconds': perf_counter()-began})
        return cache[key]

    def checkpoint(state, *unused):
        history[:] = state['history']

    budgets = asdict(c.budgets)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            state, average, nodes, weights = adapt(boundaries, native, times, frequency, delay,
                budgets, sigma, scale, max_panels=c.max_panels,
                mass_tolerance=c.mass_relative_tolerance, n_sigma=c.n_sigma, checkpoint=checkpoint,
                measurement_mode=measurement_mode)
        if not state['accepted']:
            raise failure('adaptive velocity refinement did not converge within max_panels')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            other, other_nodes, other_weights, mass = legendre_rule(state['bounds_y'], native,
                sigma, scale, c.independent_legendre_order, c.n_sigma)
            comparison = compare_traces(average, other, times, frequency, delay, budgets,
                                        measurement_mode=measurement_mode)
        mass_error = abs(mass-math.erf(c.n_sigma/math.sqrt(2)))/math.erf(c.n_sigma/math.sqrt(2))
        check_wall()
        if not comparison['accepted'] or mass_error > c.mass_relative_tolerance:
            raise failure('independent Gauss-Legendre check failed')
    except PhysicsCalculationError as exc:
        if not hasattr(exc, 'velocity_diagnostics'):
            exc.velocity_diagnostics = {'history': history, 'completed_class_costs': costs}
        raise
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
        raise failure('unrepresentable quadrature or IF calculation') from exc
    final_if = complex(state['history'][-1]['if']['real'], state['history'][-1]['if']['imag'])
    phase_defined = abs(final_if) > c.budgets.phase_amplitude_floor
    return average, {'enabled': True, 'converged': True, 'plan': asdict(plan),
        'measurement_mode': measurement_mode,
        'acceptance_target': 'if_phase' if measurement_mode == 'aoa' else 'complex_if_amplitude_with_phase_when_defined',
        'phase_defined': phase_defined, 'phase_rad': float(np.angle(final_if)) if phase_defined else None,
        'phase_undefined_reason': None if phase_defined else 'IF at or below declared phase_amplitude_floor',
        'phase_gate_applied': True if measurement_mode == 'aoa' else comparison['phase_gate_applied'],
        'local_phase_gate_applied': True if measurement_mode == 'aoa' else state['history'][-1]['phase_gate_applied'],
        'if_amplitude_converged': comparison['numerical_bounds_passed'],
        'method': c.method, 'classes': len(nodes), 'class_solves': len(cache),
        'class_costs': costs, 'solve_wall_seconds': sum(row['wall_seconds'] for row in costs),
        'wall_seconds': perf_counter()-start, 'history': history,
        'nodes_ms': nodes.tolist(), 'weights': weights.tolist(), 'adaptive': state,
        'independent': {'nodes_ms': other_nodes.tolist(), 'weights': other_weights.tolist(),
                        'mass_relative_error': mass_error, 'comparison': comparison},
        'sampling_hints_ms': list(hints), 'rf_doppler_included': False, 'velocity_dimension': 1,
        'normalization': 'conditional truncated Maxwell',
        'truncated_tail_mass': math.erfc(c.n_sigma/math.sqrt(2)),
        'acceptance': 'local defects, two global passes, independent leaf Gauss-Legendre',
        'full_reference_qualified': False,
        'qualification_scope': 'declared finite velocity domain, ODE and output time grid only'}


def task_hints(task):
    """Sampling aids only; signed RF detuning follows existing orientation."""
    p = task.temporal_doppler
    direction = 1 if task.rf_orientation is None else task.rf_orientation.direction
    return resonance_velocity_hints(*task.detunings_rad_per_us[:2], 0.,
        task.detunings_rad_per_us[2]*direction, task.omegas_rad_per_us[2],
        p.probe_wavelength_m, p.coupling_wavelength_m, p.controls.counter_propagating)
