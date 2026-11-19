"""Analysis harness comparing the per-level and field phase-diffusion linewidth models.

The pipeline default is not changed by this harness; these tests pin what
the comparison itself assumes: the existing phase-diffusion operators give
the expected coherence damping, the optical linewidth enters the compared
model exactly once, and a replay of the pipeline's own solver calls
reproduces the pipeline.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

from rydberg_aoa.physics.hamiltonian import create_sensor_model
from rydberg_aoa.scripts.compare_linewidth_models import FieldPhaseDiffusion, compare
from rydberg_aoa.simulation.noise_seam import apply_noise_seam
from tests.test_lab007_profile_integration import isolated  # noqa: F401 - explicit full state isolation

ROOT = Path(__file__).resolve().parents[1]
GAMMAS = [38.0, 0.03, 0.01]
OMEGAS = (1.0, 20.0, 0.0)


def _damping(dephasing, model=None):
    H, c_ops, obs = create_sensor_model(*OMEGAS, 0.0, 0.0, 0.0, GAMMAS, dephasing, transit_rate_per_us=0.0)
    H, c_ops = apply_noise_seam(H, c_ops, obs=obs, noise_model=model)
    liouvillian = qt.liouvillian(H, c_ops).full()
    return lambda row, column: -liouvillian[column * 4 + row, column * 4 + row].real


def test_existing_phase_diffusion_operators_damp_each_coherence_once():
    g_probe, g_coupling = 0.6, 1.0
    base = _damping([0.0, 0.0, 0.0])
    field = _damping([0.0, 0.0, 0.0], FieldPhaseDiffusion(g_probe, g_coupling))
    level = _damping([g_probe, g_coupling, 0.0])
    added = lambda damping, i, j: damping(i, j) - base(i, j)
    assert added(field, 0, 1) == pytest.approx(g_probe / 2)                  # rho_12: probe phase
    assert added(level, 0, 1) == pytest.approx(g_probe / 2)
    assert added(field, 0, 2) == pytest.approx((g_probe + g_coupling) / 2)   # rho_13: both lasers
    assert added(level, 0, 2) == pytest.approx(g_coupling / 2)               # per-level: coupling only
    assert added(field, 1, 2) == pytest.approx(g_coupling / 2)               # rho_23: coupling phase
    assert added(field, 2, 3) == pytest.approx(0.0, abs=1e-12)               # rho_34: common optical phase
    assert added(level, 2, 3) == pytest.approx(g_coupling / 2)


def test_zero_linewidth_adds_no_operator():
    assert FieldPhaseDiffusion(0.0, 0.0).operators == ()


def _lab007_without_spectrum():
    data = json.loads((ROOT / 'examples/lab007/prescribed_full.json').read_text())
    data['spectrum'] = None
    return data


def test_linewidth_enters_the_field_model_exactly_once(isolated):  # noqa: F811
    rate = 2 * math.pi * 0.5
    result = compare(_lab007_without_spectrum(), 0.5)
    per_level = np.asarray(result['per_level_dephasing_rates_per_us'])
    remaining = np.asarray(result['field_model_remaining_dephasing_rates_per_us'])
    # the per-level budget carries the optical linewidth; what is left for
    # the field model is the same budget without it (collisions, RF linewidth)
    assert per_level[:2] - remaining[:2] == pytest.approx([rate, rate], rel=1e-12)
    assert per_level[2] == remaining[2]
    assert np.all(remaining < 1e-3)
    assert result['field_model_operator_count'] == 2
    output = result['if_output']
    assert output['replay_of_pipeline_max_abs_error'] == 0.0
    assert output['per_level_estimate_deg'] == pytest.approx(-30.0, abs=1e-3)
    assert output['field_phase_diffusion_estimate_deg'] == pytest.approx(-30.0, abs=1e-3)
    ratio = np.asarray(output['amplitude_ratio_field_over_per_level'])
    assert np.all(np.abs(ratio - 1.0) > 1e-3)        # the models differ at 0.5 MHz


def test_models_coincide_without_optical_linewidth(isolated):  # noqa: F811
    result = compare(_lab007_without_spectrum(), 0.0)
    assert result['field_model_operator_count'] == 0
    assert result['per_level_dephasing_rates_per_us'] == result['field_model_remaining_dephasing_rates_per_us']
    output = result['if_output']
    assert output['amplitude_ratio_field_over_per_level'] == [1.0, 1.0, 1.0]
    assert output['field_phase_diffusion_estimate_deg'] == output['per_level_estimate_deg']
