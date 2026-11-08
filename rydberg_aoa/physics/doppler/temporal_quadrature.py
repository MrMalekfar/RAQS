"""Positive temporal velocity quadrature, promoted from frozen DOP-GRID-001.

Finite conditional Gaussian measure in v=scale*sinh(y). Raw local defects
are refinement indicators, not rigorous error estimates. Physics is supplied
by the canonical certified worker callback; no alternative physical model.
"""
from dataclasses import dataclass
import math
import numpy as np
from ...exceptions import PhysicsCalculationError


def validate_measurement_mode(measurement_mode):
    if not isinstance(measurement_mode, str) or measurement_mode not in ('aoa', 'if_only'):
        raise PhysicsCalculationError('DOP-H0-001: explicit aoa/if_only measurement mode required.')


def compare_traces(left, right, times, signed_if_mhz, delay_us, budgets, *, measurement_mode='aoa'):
    """Compare identical grids using existing ATOM-002 error budgets.

    Weak phase is undefined. Only IF-only may accept it on amplitude/trace
    budgets; AoA and strong IF retain the existing phase requirement.
    Full-complex and AC trace checks prevent DC from hiding a beat error.
    """
    from ...utils.numerical import extract_if_component
    validate_measurement_mode(measurement_mode)
    t = np.asarray(times)
    a, b = np.asarray(left), np.asarray(right)
    if any(x.shape != t.shape or x.dtype.kind not in 'iufc'
           or not np.isfinite(x).all() for x in (a, b)):
        raise PhysicsCalculationError('DOP-PROFILE-001: complete finite matching traces required.')
    x = complex(extract_if_component(a.imag, t, signed_if_mhz, delay_us)[0])
    y = complex(extract_if_component(b.imag, t, signed_if_mhz, delay_us)[0])
    retained = t >= delay_us
    ac_a, ac_b = a[retained]-a[retained].mean(), b[retained]-b[retained].mean()
    trace_error = float(np.max(np.abs(a-b)))
    ac_error = float(np.max(np.abs(ac_a-ac_b)))
    trace_bound = budgets['trace_absolute']+budgets['trace_relative']*float(np.max(np.abs(b)))
    ac_bound = budgets['trace_absolute']+budgets['trace_relative']*float(np.max(np.abs(ac_b)))
    error = abs(x-y)
    weak = min(abs(x), abs(y)) <= budgets['phase_amplitude_floor']
    phase = None if weak else float(abs(np.angle(x/y)))
    relative = None if abs(y) == 0 else float(error/abs(y))
    absolute_bound = budgets['if_absolute']+budgets['if_relative']*abs(y)
    metrics = [trace_error, ac_error, trace_bound, ac_bound, error, absolute_bound]
    metrics.extend(v for v in (phase, relative) if v is not None)
    if not all(math.isfinite(v) for v in metrics):
        raise PhysicsCalculationError('DOP-PROFILE-001: nonfinite comparison metric; no acceptance.')
    phase_ok = (measurement_mode == 'if_only') if weak else phase <= budgets['if_phase_rad']
    result = {'if_absolute_error': error, 'if_relative_error': relative,
        'if_phase_error_rad': phase, 'trace_error': trace_error, 'ac_trace_error': ac_error,
        'if_bound': absolute_bound, 'trace_bound': trace_bound, 'ac_bound': ac_bound,
        'left_if': {'real': x.real, 'imag': x.imag},
        'right_if': {'real': y.real, 'imag': y.imag},
        'phase_defined': not weak,
        'numerical_bounds_passed': bool(error <= absolute_bound and trace_error <= trace_bound and ac_error <= ac_bound),
        'accepted': bool(phase_ok and error <= absolute_bound and trace_error <= trace_bound
                         and ac_error <= ac_bound)}
    if measurement_mode == 'if_only':
        result.update(measurement_mode=measurement_mode, phase_gate_applied=not weak,
                      if_amplitude_accepted=bool(error <= absolute_bound))
    return result


@dataclass
class Panel:
    left: float
    right: float
    nodes: np.ndarray
    raw_weights: np.ndarray
    numerator: np.ndarray
    mass: float
    defect_numerator: np.ndarray
    defect_mass: float


def transformed_nodes(y, scale, sigma, extent, n_sigma):
    """v [m/s] and normalized Gaussian density times dv/dy [m/s]."""
    y = np.asarray(y, dtype=float)
    v = scale*np.sinh(y)
    v = np.where(y == -extent, -n_sigma*sigma, v)
    v = np.where(y == extent, n_sigma*sigma, v)
    density = np.exp(-.5*(v/sigma)**2)/(sigma*np.sqrt(2*np.pi))*scale*np.cosh(y)
    return v, density


def make_panel(left, right, solve, scale, sigma, extent, n_sigma):
    if not math.isfinite(left+right) or right <= left:
        raise PhysicsCalculationError('DOP-PROFILE-001: ordered finite panel boundaries required.')
    y = np.linspace(left, right, 5)
    nodes, density = transformed_nodes(y, scale, sigma, extent, n_sigma)
    traces = np.stack([solve(float(v)) for v in nodes])
    if traces.ndim != 2 or traces.dtype.kind not in 'fc' or not np.isfinite(traces).all():
        raise PhysicsCalculationError('DOP-PROFILE-001: invalid panel trace; no missing-class quadrature.')
    fine = (right-left)/12*np.array([1., 4., 2., 4., 1.])*density
    coarse = (right-left)/6*np.array([1., 4., 1.])*density[::2]
    numerator = np.sum(fine[:, None]*traces, axis=0)
    mass = float(fine.sum())
    return Panel(float(left), float(right), nodes, fine, numerator, mass,
        numerator-np.sum(coarse[:, None]*traces[::2], axis=0), mass-float(coarse.sum()))


def assembled_rule(panels):
    """Merge exact shared nodes and normalize the declared positive ratio rule."""
    weights = {}
    for panel in panels:
        for v, weight in zip(panel.nodes, panel.raw_weights):
            weights[float(v)] = weights.get(float(v), 0.)+float(weight)
    nodes = np.array(sorted(weights))
    raw = np.array([weights[float(v)] for v in nodes])
    mass = float(raw.sum())
    if not np.isfinite(raw).all() or np.any(raw <= 0) or not math.isfinite(mass) or mass <= 0:
        raise PhysicsCalculationError('DOP-PROFILE-001: invalid positive quadrature weights.')
    average = sum((p.numerator for p in panels), np.zeros_like(panels[0].numerator))/mass
    if not np.isfinite(average).all():
        raise PhysicsCalculationError('DOP-PROFILE-001: nonfinite weighted mean.')
    return average, nodes, raw/mass, mass


def adapt(boundaries, solve, times, frequency, delay, budgets, sigma, scale,
          *, max_panels, mass_tolerance, n_sigma, checkpoint=lambda *args: None,
          measurement_mode='aoa'):
    """Positive nested Simpson rule; raw coarse/fine defects are not error proofs.

    All leaf-panel defects must fit global full-trace/AC/complex-IF budgets.
    Two actual global refinement comparisons are also required. No Richardson
    extrapolation, negative weights or omitted tails. IF-only weak responses
    use the declared complex-amplitude bound; their phase stays undefined.
    """
    from ...utils.numerical import if_fit_noise_weights
    validate_measurement_mode(measurement_mode)
    selected, fit_weights, _ = if_fit_noise_weights(times, frequency, delay)
    amp = lambda z: complex(np.dot(fit_weights, z.imag[selected]))
    extent = float(np.arcsinh(n_sigma*sigma/scale))
    boundaries = np.asarray(boundaries, dtype=float)
    if (boundaries.ndim != 1 or len(boundaries) < 2 or not np.isfinite(boundaries).all()
            or boundaries[0] != -extent or boundaries[-1] != extent
            or np.any(np.diff(boundaries) <= 0)):
        raise PhysicsCalculationError('DOP-PROFILE-001: panels must cover the entire declared domain.')
    panels = [make_panel(a, b, solve, scale, sigma, extent, n_sigma) for a, b in zip(boundaries[:-1], boundaries[1:])]
    history, previous, consecutive = [], None, 0
    target_mass = math.erf(n_sigma/math.sqrt(2))
    while True:
        average, nodes, weights, mass = assembled_rule(panels)
        amplitude = amp(average)
        full_bound = budgets['trace_absolute']+budgets['trace_relative']*abs(average)
        ac = average[selected]-average[selected].mean()
        ac_bound = budgets['trace_absolute']+budgets['trace_relative']*float(max(abs(ac)))
        phase_defined = abs(amplitude) > budgets['phase_amplitude_floor']
        amplitude_only = measurement_mode == 'if_only' and not phase_defined
        if_bound = budgets['if_absolute']+budgets['if_relative']*abs(amplitude)
        if not amplitude_only:
            if_bound = min(if_bound, budgets['if_phase_rad']*abs(amplitude))
        errors = [(p.defect_numerator-average*p.defect_mass)/mass for p in panels]
        full = np.sum(np.abs(errors), axis=0)
        ac_error = [float(max(abs(e[selected]-e[selected].mean()))) for e in errors]
        if_error = [abs(amp(e)) for e in errors]
        full_ratio = float(max(full/full_bound))
        ac_ratio = sum(ac_error)/ac_bound
        if_ratio = None if if_bound == 0 else sum(if_error)/if_bound
        mass_error = abs(mass-target_mass)/target_mass
        indicator_passed = bool((phase_defined or amplitude_only)
            and full_ratio <= 1 and ac_ratio <= 1 and if_ratio is not None and if_ratio <= 1
            and mass_error <= mass_tolerance)
        comparison = None if previous is None else compare_traces(previous, average, times, frequency, delay, budgets,
                                                                  measurement_mode=measurement_mode)
        consecutive = consecutive+1 if indicator_passed and comparison is not None and comparison['accepted'] else 0
        row = {'panels': len(panels), 'nodes': len(nodes), 'full_defect_ratio': full_ratio,
            'ac_defect_ratio': ac_ratio, 'if_defect_ratio': if_ratio, 'mass_relative_error': mass_error,
            'indicator_passed': indicator_passed, 'consecutive_global_passes': consecutive,
            'if': {'real': amplitude.real, 'imag': amplitude.imag}, 'comparison': comparison}
        if measurement_mode == 'if_only':
            row.update(phase_defined=phase_defined, phase_gate_applied=not amplitude_only,
                       if_defect_bound=if_bound, measurement_mode=measurement_mode)
        history.append(row)
        state = {'history': history, 'accepted': consecutive >= 2,
                 'bounds_y': [[p.left, p.right] for p in panels],
                 'mass': mass, 'normalization': 'positive Simpson numerator / Simpson Gaussian mass',
                 'defect_interpretation': 'raw coarse/fine diagnostic, no rigorous error bound or Richardson extrapolation'}
        checkpoint(state, average, nodes, weights)
        if state['accepted'] or len(panels) >= max_panels:
            return state, average, nodes, weights
        priorities = [max(float(max(abs(e)/full_bound)), ac_error[i]/ac_bound,
                         if_error[i]/if_bound if if_bound else 0.,
                         abs(panels[i].defect_mass)/(target_mass*mass_tolerance)) for i, e in enumerate(errors)]
        index = min(range(len(panels)), key=lambda i: (-priorities[i], panels[i].left))
        parent = panels[index]
        mid = (parent.left+parent.right)/2
        # Evaluate both children before replacing a leaf, so failure never
        # creates a partially covered domain or a successful partial mean.
        children = [make_panel(parent.left, mid, solve, scale, sigma, extent, n_sigma),
                    make_panel(mid, parent.right, solve, scale, sigma, extent, n_sigma)]
        previous = average
        panels[index:index+1] = children


def legendre_rule(bounds_y, solve, sigma, scale, order, n_sigma):
    """Independent Gauss-Legendre nodes inside all final leaf intervals."""
    x, w = np.polynomial.legendre.leggauss(order)
    extent = float(np.arcsinh(n_sigma*sigma/scale))
    nodes, raw_weights, traces = [], [], []
    for left, right in bounds_y:
        y = (left+right)/2+(right-left)/2*x
        v, density = transformed_nodes(y, scale, sigma, extent, n_sigma)
        for vi, wi in zip(v, w*(right-left)/2*density):
            nodes.append(float(vi)); raw_weights.append(float(wi)); traces.append(solve(float(vi)))
    raw_weights = np.asarray(raw_weights)
    mass = float(raw_weights.sum())
    if (not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0)
            or not math.isfinite(mass) or mass <= 0):
        raise PhysicsCalculationError('DOP-PROFILE-001: invalid independent Gaussian weights.')
    weights = raw_weights/mass
    average = np.sum(weights[:, None]*np.stack(traces), axis=0)
    if not np.isfinite(average).all():
        raise PhysicsCalculationError('DOP-PROFILE-001: nonfinite independent average.')
    return average, np.asarray(nodes), weights, mass
