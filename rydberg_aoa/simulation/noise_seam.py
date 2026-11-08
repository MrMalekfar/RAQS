"""
Noise seam for the sensor-model → quantum-solver boundary (RF-004).

ARCHITECTURAL INSERTION POINT
-----------------------------
create_sensor_model(...)          # pure physics: H, c_ops, σ₂₁ (physics package)
        ↓
apply_noise_seam(H, c_ops, obs=σ₂₁, noise_model=None)     ← THIS MODULE
        ↓
steadystate(H, c_ops)  |  mesolve(H_t, ρ₀, tlist, c_ops, ...)

The seam is deliberately minimal and non-speculative: it implements no noise
physics and makes no assumption about whether future Noise Modeling will be
stochastic, Lindblad-only, time-dependent, correlated, amplitude-, phase- or
measurement-based. It provides ONE well-defined hook so future noise work can
be attached at a single boundary without rewriting `create_sensor_model`
(physics layer) or either solver consumer (`steady_state`, `time_domain`).

NO-OP DEFAULT CONTRACT
----------------------
With ``noise_model=None`` the seam returns the *same* H and c_ops objects —
no copies, no reordering, no container rebuild, no numerical edits. The
Lindblad generator reaching the solver,

    𝓛[ρ] = -(i/ħ)[H, ρ] + Σ_k ( C_k ρ C_k† − ½{C_k† C_k, ρ} ),

is therefore identical object-for-object to the pre-RF-004 model, and because
QuTiP's `steadystate`/`mesolve` are deterministic, identical inputs imply
bit-identical outputs: the Phase 0 frozen physics baseline is preserved by
construction, not merely by coincidence of tuning.
"""

from typing import Callable, Optional


def apply_noise_seam(
    H,
    c_ops,
    obs=None,
    noise_model: Optional[Callable] = None,
):
    """
    Return the (H, c_ops) pair to be handed to the quantum solver.

    This is the single sanctioned extension point between model construction
    and time evolution. The default path is a strict identity, so existing
    production callers that omit ``noise_model`` reproduce legacy behavior
    exactly.

    Parameters
    ----------
    H : qt.Qobj
        Total rotating-frame Hamiltonian Ĥ = Ĥ_detuning + Ĥ_int of the
        4-level cascade [rad/μs], as produced by ``create_sensor_model``.
    c_ops : list[qt.Qobj]
        Collapse operators {C_k} appearing in the Lindblad dissipator
        Σ_k 𝒟[C_k]ρ = Σ_k (C_k ρ C_k† − ½{C_k†C_k, ρ}). List order is
        meaningful for reproducibility and is never altered here.
    obs : qt.Qobj or sequence, optional
        Observable operator(s), currently σ₂₁ = |2⟩⟨1| (probe coherence,
        χ ∝ ⟨σ₂₁⟩/Ω_p per Bussey et al. Eq. 8). Read-only context forwarded
        verbatim to an injected model; entirely ignored when no model is
        supplied.
    noise_model : callable, optional
        Future NoiseModel attachment point. If provided, MUST be callable as
        ``model(H, c_ops, obs) -> (H, c_ops)``. It may inspect the inputs or
        augment the Liouvillian (e.g., additional collapse operators adding
        further Σ_k 𝒟[C_k] terms, or extended/modulated Hamiltonian drive
        terms). Any such physics belongs to the future model — never here.

    Returns
    -------
    tuple
        ``(H, c_ops)`` exactly as passed in when ``noise_model`` is None;
        otherwise the injected model's shape-validated return value.

    Raises
    ------
    TypeError
        If ``noise_model`` is not callable, or returns something that is not
        a 2-tuple ``(H, c_ops)`` — failing loudly rather than silently
        feeding a malformed generator to QuTiP.
    """
    # Default path: strict identity. Handing back the very objects produced
    # by the physics layer guarantees the solver sees byte-identical inputs
    # and hence bit-identical numerical results (baseline preservation).
    if noise_model is None:
        return H, c_ops

    # Fail-loud on malformed configuration (RF-003 philosophy applied at the
    # seam): a silent wrong-type fallback could corrupt the Liouvillian and
    # yield plausible-but-wrong sensor readings.
    if not callable(noise_model):
        raise TypeError(
            "noise_model must be callable as model(H, c_ops, obs) -> (H, c_ops); "
            f"got {type(noise_model).__name__!r}"
        )

    result = noise_model(H, c_ops, obs)

    # Validate output shape BEFORE it reaches the solver: the master equation
    # requires exactly (Ĥ, {C_k}). An explicit tuple-type + length check is
    # mandatory here rather than bare unpacking — Qobj exposes legacy
    # element-wise iteration, so `H_out, c_out = result` would silently
    # decompose a returned Hamiltonian Qobj into its two matrix rows,
    # feeding a structurally meaningless Liouvillian to QuTiP.
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "noise model must return a 2-tuple (H, c_ops); got "
            f"{type(result).__name__!r}"
        )
    H_out, c_ops_out = result

    return H_out, c_ops_out