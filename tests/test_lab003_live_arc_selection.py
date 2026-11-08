"""LAB-003 independent, live ARC oracle; no dependency skips or runtime init."""

from dataclasses import replace
import math

import pytest
from arc import Rubidium85

from rydberg_aoa.config.constants import C_LIGHT
from rydberg_aoa.config.laboratory import OpticalEnergyConvention
from rydberg_aoa.level_finder.laboratory import select_laboratory_manifolds
from rydberg_aoa.exceptions import PhysicsCalculationError
from tests.test_lab002_laboratory_request import reference_request
from tests.test_lab003_optical_feasibility import configuration_snapshot


def independent_oracle(atom, req):
    """Own loops, E1 rules, ARC calls and ordering; no production scanner."""
    b, e, p = req.bounds, req.preparation.excited, req.preparation
    shift = 0.
    if req.optical_energy_convention is OpticalEnergyConvention.LOW_LEVEL_HFS:
        shift = atom.getHFSEnergyShift(e.j, e.f, *atom.getHFSCoefficients(e.n, e.l, e.j))
    found = []
    for n1 in range(b.n_min, b.n_max + 1):
        for l1 in range(min(n1, b.max_l + 1)):
            if abs(l1 - e.l) != 1:
                continue
            for j1 in ([.5] if l1 == 0 else [l1 - .5, l1 + .5]):
                if abs(j1 - e.j) > 1 or abs(p.rydberg1_mj) > j1:
                    continue
                frequency = atom.getTransitionFrequency(e.n, e.l, e.j, n1, l1, j1) - shift
                if frequency <= 0:
                    continue
                proposed = C_LIGHT / (frequency - req.coupling.detuning_mhz * 1e6) * 1e9
                if not b.coupling_window_nm[0] <= proposed <= b.coupling_window_nm[1]:
                    continue
                for n2 in range(b.n_min, b.n_max + 1):
                    for l2 in range(min(n2, b.max_l + 1)):
                        if abs(l1 - l2) != 1:
                            continue
                        for j2 in ([.5] if l2 == 0 else [l2 - .5, l2 + .5]):
                            if abs(j1 - j2) > 1 or abs(p.rydberg2_mj) > j2:
                                continue
                            signed = atom.getTransitionFrequency(n1, l1, j1, n2, l2, j2)
                            assert math.isfinite(signed) and signed != 0
                            mismatch = abs(abs(signed) - req.lo.frequency_hz)
                            if mismatch <= b.rf_tolerance_hz:
                                found.append((mismatch, n1, n2, l1, l2, j1, j2, signed, proposed))
    return sorted(found)


@pytest.fixture(scope='module')
def live():
    atom, req = Rubidium85(), reference_request()
    before = configuration_snapshot()
    result = select_laboratory_manifolds(atom, req)
    oracle = independent_oracle(atom, req)
    assert configuration_snapshot() == before
    return atom, req, result, oracle


def test_live_entire_feasible_set_and_winner_match_independent_oracle(live):
    _, req, result, oracle = live
    observed = [(c.mismatch_hz, c.rydberg1.n, c.rydberg2.n,
                 c.rydberg1.l, c.rydberg2.l, c.rydberg1.j, c.rydberg2.j,
                 c.signed_frequency_hz, c.coupling.proposed_wavelength_nm)
                for c in result.candidates]
    assert observed == oracle and oracle
    assert result.winner == result.candidates[0]
    assert result.winner.carrier_hz == abs(oracle[0][7])
    assert result.winner.mismatch_hz == abs(result.winner.carrier_hz - req.lo.frequency_hz)
    print('\nLAB-003 live top-5: (mismatch,n1,n2,l1,l2,j1,j2,signed,proposed_nm)')
    for row in oracle[:5]:
        print(row)


def test_live_optical_applied_resonance_and_proposal_remain_distinct(live):
    _, req, result, _ = live
    optical = result.winner.coupling
    assert optical.applied_wavelength_nm == req.coupling.wavelength_nm
    assert optical.applied_wavelength_nm != optical.proposed_wavelength_nm
    expected = C_LIGHT / (req.coupling.wavelength_nm * 1e-9) - optical.resonant_frequency_hz
    assert optical.detuning_hz == expected
    assert optical.detuning_rad_per_us == 2 * math.pi * expected / 1e6
    assert result.level_objects()['rydberg1'].mj is None


def test_live_hfs_shifts_both_arms_without_changing_rf_or_applied_fields(live):
    atom, req, centroid, _ = live
    req = replace(req, optical_energy_convention=OpticalEnergyConvention.LOW_LEVEL_HFS)
    result = select_laboratory_manifolds(atom, req)
    g, e = req.preparation.ground, req.preparation.excited
    ground = atom.getHFSEnergyShift(g.j, g.f, *atom.getHFSCoefficients(g.n, g.l, g.j))
    excited = atom.getHFSEnergyShift(e.j, e.f, *atom.getHFSCoefficients(e.n, e.l, e.j))
    assert result.ground_hfs_shift_hz == ground
    assert result.excited_hfs_shift_hz == excited
    assert result.probe.resonant_frequency_hz == centroid.probe.resonant_frequency_hz + excited - ground
    assert result.winner.coupling.resonant_frequency_hz == centroid.winner.coupling.resonant_frequency_hz - excited
    assert result.winner.signed_frequency_hz == centroid.winner.signed_frequency_hz
    assert req.lo == centroid.request.lo and req.coupling == centroid.request.coupling
    print('\nHFS shifts [Hz]:', ground, excited,
          'proposed coupling/probe [nm]:', result.winner.coupling.proposed_wavelength_nm,
          result.probe.proposed_wavelength_nm)


def test_live_inaccessible_resonance_does_not_widen_hardware_domain(live):
    atom, req, _, _ = live
    req = replace(req, bounds=replace(req.bounds, coupling_window_nm=(483.35, 483.45)))
    before = configuration_snapshot()
    with pytest.raises(PhysicsCalculationError, match='No feasible RF-first'):
        select_laboratory_manifolds(atom, req)
    assert configuration_snapshot() == before
