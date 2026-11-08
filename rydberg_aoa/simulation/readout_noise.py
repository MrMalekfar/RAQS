"""Effective Gaussian measurement noise, after certified atomic evolution.

Sigma is per real output sample of Im<sigma21>, not photocurrent or a PSD.
Atomic Hamiltonian, density states and RF selection are never modified here.
"""

import numpy as np

from ..config.noise import require_noise_execution_supported
from ..exceptions import PhysicsCalculationError, PhysicsError
from .noise_random import NoiseRandomPlan


def apply_gaussian_readout(traces, config, *, trial=0):
    """Return (measured, additive_noise, metadata), with strict disabled identity.

    Input must be a validated real M-by-N batch. Independent path streams use
    the explicit trial and path=row index; the runner sorts element IDs first.
    Disabled returns the input object and no noise/metadata, without any RNG.
    Enabled sigma=0 also avoids RNG and preserves the input object exactly.
    """
    require_noise_execution_supported(config)
    if config is not None and config.schema_version != 1:
        from ..exceptions import PhysicsConfigurationError
        raise PhysicsConfigurationError('NOISE-004: Gaussian helper accepts schema 1 only; use the physical PIN readout API.')
    from ..config.laboratory import _integer
    trial = _integer(trial, 'NOISE-003 trial')
    if config is not None and trial >= config.trials:
        from ..exceptions import PhysicsConfigurationError
        raise PhysicsConfigurationError('NOISE-003: trial outside declared campaign.')
    if config is None or not config.active_sources:
        return traces, None, None
    try:
        arr = np.asarray(traces)
    except (ValueError, TypeError, OverflowError) as exc:
        raise PhysicsCalculationError('NOISE-002: malformed clean trace batch.') from exc
    if (arr.ndim != 2 or not all(arr.shape) or
            arr.dtype.kind not in 'iuf' or not np.isfinite(arr).all()):
        raise PhysicsCalculationError('NOISE-002: clean traces must be finite real numeric M-by-N data.')
    sigma = config.sources[0].specification.sigma_observable
    plan = NoiseRandomPlan(config)
    noise = np.zeros(arr.shape, dtype=float)
    if sigma > 0:
        try:
            with np.errstate(over='raise', invalid='raise'):
                for path in range(arr.shape[0]):
                    noise[path] = sigma * plan.generator('readout_gaussian', trial=trial, path=path).standard_normal(arr.shape[1])
                measured = arr + noise
            if not np.isfinite(measured).all() or not np.isfinite(noise).all():
                raise PhysicsCalculationError('NOISE-002: nonfinite noisy trace; no clipping or replacement.')
        except PhysicsError:
            raise
        except Exception as exc:
            raise PhysicsCalculationError('NOISE-002: readout generation failed; no clean-trace fallback.') from exc
    else:
        measured = traces
    metadata = {'model': 'effective_gaussian_readout', 'units': 'Im<sigma21> per real sample',
                'sigma_observable': sigma, 'noise_applied': sigma > 0,
                'trial_index': trial, 'trials_executed': 1,
                'correlation': 'independent paths and samples',
                'spectrum_readout_noise_applied': False, 'random_plan': plan.metadata()}
    return measured, noise, metadata
