"""RF-first level-finding contract tests (Phase-1 pipeline, asserted).

Replaces the former print-only Phase-1 script that executed the optimizer at
collection time inside a false-green try/except. The assertions mirror that
script's success criteria: RF-first scanning must surface RF-matched
transition candidates, the selected Rydberg pair must be distinct, and
(AUTO-001A-R2) the result must be FS-only — no legacy maximum-channel dipole
matrix, no fabricated transition object, and both Rydberg mJ fields still
None.
"""
import pytest

from rydberg_aoa.arc import LaserParams, RFParams
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.level_finder import SingleStageLevelOptimizer

_REFERENCE_RF_HZ = 10.0e9  # 10 GHz reference carrier of the preserved run


def _optimize_at_10ghz():
    """Run the Phase-1 RF-first optimization at the reference 10 GHz drive."""
    # Laser/RF parameter triple of the historical terminal-log run that this
    # regression preserves: λ_p = 780.2412 nm, λ_c = 483.4 nm, f_RF = 10 GHz.
    probe = LaserParams(wavelength_nm=780.2412, power_mw=0.001,
                        beam_waist_mm=0.5, detuning_mhz=0.0, polarization=0)
    coupling = LaserParams(wavelength_nm=483.4, power_mw=10.0,
                           beam_waist_mm=0.5, detuning_mhz=0.0, polarization=0)
    rf = RFParams(frequency_hz=_REFERENCE_RF_HZ, electric_field_vm=0.5,
                  polarization=0)
    optimizer = SingleStageLevelOptimizer(arc_atom=acfg.ATOM)
    return optimizer.optimize(probe, coupling, rf)


def test_single_stage_optimizer_initializes(configured_atom):
    """SingleStageLevelOptimizer must construct against the shared ARC atom."""
    optimizer = SingleStageLevelOptimizer(arc_atom=acfg.ATOM)
    assert optimizer is not None


@pytest.mark.slow
def test_rf_first_finds_transitions_and_levels(configured_atom):
    """RF-first scan at 10 GHz must find transitions and select a ladder."""
    result = _optimize_at_10ghz()
    assert result, "optimizer returned no result"

    # RF-first contract: the (n1, n2) scan must surface transition candidates
    rf_pairs = result.get('rf_transitions_found')
    assert rf_pairs, "RF-first scanning found no transitions at 10 GHz"

    # Best candidate gate: Rydberg-grid quantization (n, l, j lattice at
    # n ~ 100) bounds the achievable match; the observed floor is ~84 MHz
    # (~0.84% of the 10 GHz carrier), so gate on a 1% RELATIVE match rather
    # than an absolute bandwidth.
    best = min(rf_pairs, key=lambda p: abs(p['rf_frequency_error_hz']))
    assert abs(best['rf_frequency_error_hz']) < 0.01 * _REFERENCE_RF_HZ

    # AUTO-001A-R1: level selection is fine-structure-only — scan candidates
    # never carry dipoles (they attach to the declared winner only AFTER
    # selection, through the channel-dependent QN-001B/C dipole-matrix path).
    assert all('dipole_rf' not in p for p in rf_pairs), \
        "scan candidates must stay dipole-free (FS-only selection)"

    # Final selection must pick two DISTINCT Rydberg levels for the ladder
    levels = result['levels']
    assert levels['rydberg1'].label != levels['rydberg2'].label

    # AUTO-001A: the selected manifold carries its RF-first selection evidence
    # and the energy orientation resolved from ARC's SIGNED frequency at
    # selection time (RF-DIR-001) — never re-inferred downstream.
    selected = result['selected_rf_pair']
    orientation = selected['rf_orientation']
    assert orientation.direction in (-1, +1)
    r1, r2 = levels['rydberg1'], levels['rydberg2']
    signed = acfg.ATOM.getTransitionFrequency(r1.n, r1.l, r1.j, r2.n, r2.l, r2.j)
    assert orientation.signed_frequency_hz == pytest.approx(signed)
    assert orientation.carrier_hz == pytest.approx(abs(signed))

    metrics = result['optimization_metrics']
    assert metrics['rf_energy_direction'] == orientation.direction
    assert metrics['rf_carrier_hz'] == pytest.approx(orientation.carrier_hz)
    assert metrics['signed_rf_frequency_hz'] == pytest.approx(signed)
    # The reported carrier error must equal | |nu| - f_RF | of the same ARC nu.
    assert metrics['rf_frequency_error_hz'] == pytest.approx(
        abs(abs(signed) - _REFERENCE_RF_HZ))
    assert metrics['mismatch_hz'] == pytest.approx(
        metrics['rf_frequency_error_hz'])
    assert metrics['rf_applied_carrier_hz'] == pytest.approx(_REFERENCE_RF_HZ)

    # AUTO-001A-R2 FS-only result boundary: selection produces no legacy
    # maximum-channel dipole, no fabricated transition object and no metrics
    # dipole claim (an assertion demanding one would reach for an undeclared
    # magnetic channel); both selected Rydberg mJ fields stay None and the
    # declared-channel Rabi boundary remains explicit and untouched.
    assert 'dipole_matrix' not in result
    assert 'transitions' not in result
    assert 'dipole_rf' not in selected
    assert 'dipole_rf' not in metrics
    assert selected['from_level'].mj is None
    assert selected['to_level'].mj is None

