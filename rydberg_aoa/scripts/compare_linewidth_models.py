"""Compare the two laser-linewidth dephasing models on a real profile.

This is an ANALYSIS script.  It changes no default and no profile schema:
the profile pipeline keeps the current model.

Models
------
``per_level`` (current pipeline)
    The probe and coupling linewidths enter the pure-dephasing budget of
    one level each, ``[g_probe, g_coupling, g_rf] + collision``, i.e.
    ``sqrt(g_k) |k><k|`` on levels 2, 3, 4.

``field_phase_diffusion``
    The probe and coupling linewidths enter ONCE, through the existing
    ``create_laser_phase_diffusion_ops`` (probe phase carried by levels
    2-4, coupling phase by levels 3-4).  The per-level budget then keeps
    only what is not optical linewidth: collisions and the RF linewidth.

Method
------
1. Run the unmodified pipeline with the requested linewidths: this is the
   ``per_level`` result, taken from the pipeline itself.
2. Run the unmodified pipeline with the optical linewidths set to zero and
   record the exact solver calls it makes (spectrum calls and sensing
   worker tasks).  Those calls carry the collision/RF-only budget.
3. Replay the recorded calls through the same solver functions with the
   field phase-diffusion operators attached at the sanctioned noise seam
   (``simulation.noise_seam``).  Nothing else differs.

A replay of step 1's own calls without any noise model must reproduce the
pipeline arrays exactly; the script checks that before reporting.

Usage
-----
    python -m rydberg_aoa.scripts.compare_linewidth_models \
        --profile examples/lab007/prescribed_full.json \
        --fwhm-mhz profile 0.5 --output-dir out/linewidth

``profile`` keeps the linewidths declared in the profile; a number sets
probe and coupling FWHM to that many MHz (rate ``2*pi*FWHM`` per us).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# A replay of the pipeline's own calls must reproduce the pipeline to this level.
REPLAY_TOLERANCE = 1e-12

FOUR_LEVEL_BASIS = tuple(SimpleNamespace(role=role)
                         for role in ("ground", "excited", "rydberg1", "rydberg2"))


class FieldPhaseDiffusion:
    """Noise-seam model: append the existing field phase-diffusion operators."""

    def __init__(self, probe_rate_per_us, coupling_rate_per_us):
        import qutip as qt
        from ..physics.lindblad.laser_phase_diffusion import create_laser_phase_diffusion_ops
        self.probe_rate_per_us = float(probe_rate_per_us)
        self.coupling_rate_per_us = float(coupling_rate_per_us)
        sparse_ops = create_laser_phase_diffusion_ops(
            FOUR_LEVEL_BASIS,
            probe_fwhm_mhz=self.probe_rate_per_us / (2.0 * math.pi),
            coupling_fwhm_mhz=self.coupling_rate_per_us / (2.0 * math.pi))
        self.operators = tuple(qt.Qobj(op.toarray()) for op in sparse_ops)

    def __call__(self, H, c_ops, obs=None):
        return H, list(c_ops) + list(self.operators)


def _reset_physics():
    from ..config import atomic_config
    atomic_config.reset_atomic_physics()


def run_pipeline_recording(profile_data):
    """Run the unmodified pipeline; return (report, output, recorded calls)."""
    from ..config.run_profile import parse_run_profile
    from . import _linewidth_recording as recording
    from ..simulation.profile_execution import execute_profile
    _reset_physics()
    with recording.record_solver_calls() as calls:
        report, output = execute_profile(parse_run_profile(copy.deepcopy(profile_data)))
        calls["rf_orientation"] = recording.published_rf_orientation()
    return report, output, calls


def replay_spectrum(calls, noise_model=None):
    """Re-run the recorded spectrum calls; returns {'rf_off': T, 'rf_on': T}."""
    from ..simulation.steady_state import simulate_spectrum
    curves = {}
    for name, (args, kwargs) in zip(("rf_off", "rf_on"), calls["spectrum"]):
        kwargs = dict(kwargs, analytical_enabled=False)
        if noise_model is not None:
            kwargs["noise_model"] = noise_model
        _, numerical, _ = simulate_spectrum(*args, **kwargs)
        curves[name] = np.asarray(numerical, dtype=float)
    return curves


def replay_sensing(calls, noise_model=None):
    """Re-run the recorded sensing tasks in-process; returns complex IF per path."""
    from ..config import parameters as params
    from ..simulation.time_domain import run_single_heterodyne_worker
    from ..simulation.time_domain import atomic_if_polarity
    responses = []
    for task in calls["tasks"]:
        if task.temporal_doppler is not None or task.rf_drive is not None or task.solver_diagnostic is not None:
            raise ValueError("comparison supports plain sensing tasks only "
                             "(no temporal Doppler, RF drive realization or solver diagnostic).")
        options = {"temporal_ode": task.temporal_ode} if task.temporal_ode is not None else {}
        previous = params.ATOMIC_TRANSIENT_DELAY_US
        try:
            params.ATOMIC_TRANSIENT_DELAY_US = task.transient_delay_us
            record = run_single_heterodyne_worker(
                task.element_index, task.omega_signal_complex, task.tlist_us,
                task.signed_if_mhz, *task.omegas_rad_per_us, *task.detunings_rad_per_us,
                task.decay_rates_per_us, task.dephasing_rates_per_us,
                noise_model=noise_model, transit_rate_per_us=task.transit_rate_per_us,
                rf_orientation=task.rf_orientation, **options)
        finally:
            params.ATOMIC_TRANSIENT_DELAY_US = previous
        responses.append(complex(record[0]))
    responses = np.asarray(responses, dtype=complex)
    # Same single conjugation the runner applies for an energy-downward RF edge.
    if calls["tasks"] and atomic_if_polarity(calls["tasks"][0].rf_orientation) < 0:
        responses = np.conj(responses)
    return responses


def estimate_angle(responses, output):
    from ..doa.geometry import ArrayGeometry
    from ..doa.registry import get_doa_estimator
    result = get_doa_estimator("phase_beamforming").estimate(
        np.asarray(responses, dtype=complex), ArrayGeometry(np.asarray(output["array_positions"])),
        {"frequency_hz": float(output["signal_carrier_hz"])})
    return float(result.azimuth_deg)


def _with_linewidths(profile_data, rates_per_us):
    data = copy.deepcopy(profile_data)
    environment = data.get("environment")
    if not isinstance(environment, dict) or "linewidth_rates_per_us" not in environment:
        raise ValueError("profile declares no environment.linewidth_rates_per_us.")
    environment["linewidth_rates_per_us"] = [float(v) for v in rates_per_us]
    environment["linewidth_enabled"] = True
    return data


def compare(profile_data, fwhm_mhz=None):
    """Return a JSON-ready comparison for one linewidth setting.

    ``fwhm_mhz=None`` keeps the profile's declared linewidth rates.
    """
    declared = [float(v) for v in profile_data["environment"]["linewidth_rates_per_us"]]
    if fwhm_mhz is None:
        rates = declared
    else:
        rates = [2.0 * math.pi * float(fwhm_mhz), 2.0 * math.pi * float(fwhm_mhz), declared[2]]

    # 1. current model, from the pipeline itself
    report_a, output_a, calls_a = run_pipeline_recording(_with_linewidths(profile_data, rates))
    # 2. the same pipeline with the optical linewidths removed from the budget
    report_b, output_b, calls_b = run_pipeline_recording(
        _with_linewidths(profile_data, [0.0, 0.0, rates[2]]))
    model = FieldPhaseDiffusion(rates[0], rates[1])

    result = {
        "linewidth_rates_per_us": rates,
        "linewidth_fwhm_mhz": [rates[0] / (2 * math.pi), rates[1] / (2 * math.pi)],
        "per_level_dephasing_rates_per_us": list(report_a["resolved_rate_budget_per_us"]["pure_dephasing"]),
        "field_model_remaining_dephasing_rates_per_us":
            list(report_b["resolved_rate_budget_per_us"]["pure_dephasing"]),
        "field_model_operator_count": len(model.operators),
    }

    if calls_a["spectrum"]:
        pipeline = {name: np.asarray(output_a["spectrum"][name], dtype=float) for name in ("rf_off", "rf_on")}
        replayed = replay_spectrum(calls_a)
        harness_error = max(float(np.max(np.abs(replayed[n] - pipeline[n]))) for n in pipeline)
        if harness_error > REPLAY_TOLERANCE:
            raise RuntimeError("replay of the pipeline's own spectrum calls does not reproduce the pipeline spectrum.")
        field = replay_spectrum(calls_b, model)
        offsets = np.asarray(output_a["spectrum"]["probe_offsets_mhz"], dtype=float)
        result["spectrum"] = {
            "probe_offsets_mhz": offsets.tolist(),
            "replay_of_pipeline_max_abs_error": harness_error,
            "per_level": {n: pipeline[n].tolist() for n in pipeline},
            "field_phase_diffusion": {n: field[n].tolist() for n in field},
            "max_abs_transmission_difference": {
                n: float(np.max(np.abs(field[n] - pipeline[n]))) for n in pipeline},
        }

    if calls_a["tasks"]:
        pipeline_if = np.asarray(output_a["complex_responses"], dtype=complex)
        replayed_if = replay_sensing(calls_a)
        if float(np.max(np.abs(replayed_if - pipeline_if))) > REPLAY_TOLERANCE:
            raise RuntimeError("replay of the pipeline's own sensing calls does not reproduce the "
                               "pipeline IF responses; this profile needs post-processing the "
                               "comparison does not model (e.g. physical readout).")
        field_if = replay_sensing(calls_b, model)
        ideal = np.exp(1j * np.asarray(output_a["element_phases_rad"], dtype=float))

        def relative_phase_error_deg(values):
            relative = values / values[0] / (ideal / ideal[0])
            return np.degrees(np.angle(relative)).tolist()

        result["if_output"] = {
            "replay_of_pipeline_max_abs_error": float(np.max(np.abs(replayed_if - pipeline_if))),
            "per_level_amplitude": np.abs(pipeline_if).tolist(),
            "field_phase_diffusion_amplitude": np.abs(field_if).tolist(),
            "amplitude_ratio_field_over_per_level": (np.abs(field_if) / np.abs(pipeline_if)).tolist(),
            "common_phase_shift_deg": np.degrees(np.angle(field_if / pipeline_if)).tolist(),
            "per_level_relative_phase_error_deg": relative_phase_error_deg(pipeline_if),
            "field_phase_diffusion_relative_phase_error_deg": relative_phase_error_deg(field_if),
            "true_aoa_deg": float(output_a["true_aoa"]),
            "per_level_pipeline_estimate_deg": float(output_a["estimated_aoa"]),
            "per_level_estimate_deg": estimate_angle(pipeline_if, output_a),
            "field_phase_diffusion_estimate_deg": estimate_angle(field_if, output_a),
        }
    _reset_physics()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--fwhm-mhz", nargs="+", default=["profile"],
                        help="'profile' and/or FWHM values in MHz for probe and coupling")
    parser.add_argument("--probe-offsets-mhz", nargs="+", type=float, default=None,
                        help="optional explicit spectrum offsets for this comparison only")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    profile_data = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    if args.probe_offsets_mhz is not None:
        if not isinstance(profile_data.get("spectrum"), dict):
            raise SystemExit("profile declares no spectrum.")
        profile_data["spectrum"].pop("probe_scan", None)
        profile_data["spectrum"]["probe_offsets_mhz"] = list(args.probe_offsets_mhz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"profile": str(args.profile), "cases": []}
    for value in args.fwhm_mhz:
        fwhm = None if value == "profile" else float(value)
        case = compare(profile_data, fwhm)
        case["case"] = "profile linewidths" if fwhm is None else f"{fwhm} MHz"
        results["cases"].append(case)
    target = output_dir / "linewidth_model_comparison.json"
    target.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {target}")
    return results


if __name__ == "__main__":
    main()
