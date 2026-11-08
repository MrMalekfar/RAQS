"""LAB-003: physical-domain filtering precedes RF ranking, never tuning."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from rydberg_aoa.config.constants import C_LIGHT
from rydberg_aoa.config.laboratory import OpticalEnergyConvention
from rydberg_aoa.config import laser_rf_config as lr, parameters as params
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.level_finder import SingleStageLevelOptimizer
from rydberg_aoa.level_finder.laboratory import select_laboratory_manifolds
from tests.test_lab002_laboratory_request import reference_request


def request():
    ref = reference_request()
    return replace(ref, bounds=replace(ref.bounds, n_min=26, n_max=27, rf_tolerance_hz=1000.))


class Atom:
    I = 2.5
    groundStateN = 5

    def __init__(self):
        self.calls = []

    def getTransitionFrequency(self, n1, l1, j1, n2, l2, j2):
        self.calls.append((n1, l1, j1, n2, l2, j2))
        if n1 == 5 and l1 == 0:
            return C_LIGHT / (780.3e-9)
        if n1 == 5:
            return C_LIGHT / ((500 if n2 == 26 else 483) * 1e-9)
        return -(132.649e9 + (0 if n1 == 26 else 100))

    def getHFSCoefficients(self, n, l, j):
        return (200e6 if l == 0 else 40e6), 0.

    def getHFSEnergyShift(self, j, f, a, b):
        return .5 * a * (f * (f + 1) - self.I * (self.I + 1) - j * (j + 1))

    def __getattr__(self, name):
        if 'Dipole' in name or 'Rabi' in name or name == 'getTransitionWavelength':
            pytest.fail(f'LAB-003 must not reach legacy/maximum-channel helper {name}')
        raise AttributeError(name)


def configuration_snapshot():
    return copy.deepcopy((lr.LASER_PARAMS, lr.RF_PARAMS, lr.LO_RF_PARAMS, lr.SIGNAL_RF_PARAMS,
                          params.LEVEL_SELECTION_TOLERANCES, params.SI_Constants.LAMBDA_P,
                          params.OMEGA_S_BASE, params.OMEGA_LO_BASE))


def test_optical_feasibility_is_applied_before_rf_ranking():
    req, atom = request(), Atom()
    before = configuration_snapshot()
    result = SingleStageLevelOptimizer(atom).optimize_laboratory(req)
    # The exact RF match at n1=26 needs an unavailable 500-nm laser.
    assert result.winner.rydberg1.n == 27
    assert result.winner.mismatch_hz == 100.
    assert result.optical_exclusions > 0
    assert result.rf_candidates_count > len(result.candidates)
    assert all(c.rydberg1.n == 27 for c in result.candidates)
    assert result.winner.rydberg1.mj is result.winner.rydberg2.mj is None
    assert result.rf_orientation.direction == -1
    assert result.winner.signed_frequency_hz < 0
    assert result.winner.carrier_hz == abs(result.winner.signed_frequency_hz)
    assert req.coupling.wavelength_nm == result.winner.coupling.applied_wavelength_nm == 483.4
    assert result.winner.coupling.proposed_wavelength_nm == pytest.approx(483.)
    assert configuration_snapshot() == before
    assert result.request is req


def test_rf_mismatch_still_dominates_inside_feasible_domain():
    req = request()
    req = replace(req, bounds=replace(req.bounds, coupling_window_nm=(480., 510.)))
    result = select_laboratory_manifolds(Atom(), req)
    # The 500-nm candidate wins despite worse applied optical detuning.
    assert result.winner.rydberg1.n == 26
    assert result.winner.mismatch_hz == 0
    assert abs(result.winner.coupling.detuning_hz) > 1e12


def test_full_domain_and_same_n_are_allowed_and_order_is_deterministic():
    first, second = select_laboratory_manifolds(Atom(), request()), select_laboratory_manifolds(Atom(), request())
    assert first == second
    assert first.winner.rydberg1.n == 27 and first.winner.rydberg2.n == 26
    assert any(c.rydberg1.n == c.rydberg2.n for c in first.candidates)
    keys = [(c.mismatch_hz, c.rydberg1.n, c.rydberg2.n, c.rydberg1.l,
             c.rydberg2.l, c.rydberg1.j, c.rydberg2.j) for c in first.candidates]
    assert keys == sorted(keys)


def test_fixed_projection_support_filters_j_without_scan_or_dipole():
    req = request()
    req = replace(req, preparation=replace(req.preparation, rydberg1_mj=1.5, rydberg2_mj=1.5))
    result = select_laboratory_manifolds(Atom(), req)
    assert result.projection_exclusions > 0
    assert all(c.rydberg1.j >= 1.5 and c.rydberg2.j >= 1.5 for c in result.candidates)
    levels = result.level_objects()
    assert levels['excited'].mf == req.preparation.excited.mf
    assert levels['rydberg1'].mj is levels['rydberg2'].mj is None
    levels['rydberg1'].n = 99
    assert result.winner.rydberg1.n == 27


def test_no_feasible_pair_is_fail_loud_without_retry_or_widening():
    req = request()
    req = replace(req, bounds=replace(req.bounds, coupling_window_nm=(483.35, 483.45)))
    before = configuration_snapshot()
    with pytest.raises(PhysicsCalculationError, match='No feasible RF-first') as error:
        select_laboratory_manifolds(Atom(), req)
    assert 'optical exclusions=' in str(error.value) and 'No retry' in str(error.value)
    assert configuration_snapshot() == before


def test_inaccessible_prepared_probe_never_selects_another_excited_state():
    req = request()
    req = replace(req, bounds=replace(req.bounds, probe_window_nm=(780.24, 780.25)))
    atom = Atom()
    with pytest.raises(PhysicsCalculationError, match='prepared probe.*inaccessible'):
        select_laboratory_manifolds(atom, req)
    assert atom.calls == [(5, 0, .5, 5, 1, 1.5)]


def test_declared_hfs_policy_changes_both_optical_resonances_with_correct_signs():
    req = replace(request(), optical_energy_convention=OpticalEnergyConvention.LOW_LEVEL_HFS)
    result = select_laboratory_manifolds(Atom(), req)
    assert result.ground_hfs_shift_hz == 250e6
    assert result.excited_hfs_shift_hz == 150e6
    assert result.probe.resonant_frequency_hz == C_LIGHT / 780.3e-9 + 150e6 - 250e6
    assert result.winner.coupling.resonant_frequency_hz == C_LIGHT / (483 * 1e-9) - 150e6
    assert result.winner.signed_frequency_hz == -(132.649e9 + 100)


def test_centroid_policy_does_not_query_hfs_or_claim_zero_measured_hfs():
    class NoHFSAtom(Atom):
        def getHFSCoefficients(self, *args):
            pytest.fail('explicit centroid policy must not require HFS data')
    result = select_laboratory_manifolds(NoHFSAtom(), request())
    assert result.request.optical_energy_convention is OpticalEnergyConvention.FS_CENTROID


@pytest.mark.parametrize('bad', [None, 'bad', (1.,), (np.nan, 0), (1j, 0), (True, 0)])
def test_invalid_hfs_coefficients_do_not_become_centroid_selection(bad):
    class BadHFS(Atom):
        def getHFSCoefficients(self, *args):
            return bad
    req = replace(request(), optical_energy_convention=OpticalEnergyConvention.LOW_LEVEL_HFS)
    with pytest.raises(PhysicsCalculationError):
        select_laboratory_manifolds(BadHFS(), req)


def test_missing_hfs_data_preserves_cause_without_policy_switch():
    failure = ValueError('missing measured HFS')
    class BadHFS(Atom):
        def getHFSCoefficients(self, *args):
            raise failure
    req = replace(request(), optical_energy_convention=OpticalEnergyConvention.LOW_LEVEL_HFS)
    with pytest.raises(PhysicsCalculationError) as error:
        select_laboratory_manifolds(BadHFS(), req)
    assert error.value.__cause__ is failure
    assert req.optical_energy_convention is OpticalEnergyConvention.LOW_LEVEL_HFS


@pytest.mark.parametrize('bad', [np.nan, np.inf, None, 'bad', np.complex128(1), np.array([1.])])
def test_invalid_optical_arc_data_aborts_instead_of_excluding_candidate(bad):
    class BadOptical(Atom):
        def getTransitionFrequency(self, *args):
            return bad if args[0] == 5 and args[1] == 1 else super().getTransitionFrequency(*args)
    with pytest.raises(PhysicsCalculationError):
        select_laboratory_manifolds(BadOptical(), request())


def test_valid_negative_coupling_energy_is_an_absorption_domain_exclusion():
    class DownwardOptical(Atom):
        def getTransitionFrequency(self, *args):
            value = super().getTransitionFrequency(*args)
            return -value if args[0] == 5 and args[1] == 1 and args[3] == 26 else value
    result = select_laboratory_manifolds(DownwardOptical(), request())
    assert result.winner.rydberg1.n == 27


def test_optical_offset_is_honored_once_and_proposal_is_not_applied():
    req = request()
    req = replace(req, coupling=replace(req.coupling, detuning_mhz=20.))
    optical = select_laboratory_manifolds(Atom(), req).winner.coupling
    resonance = C_LIGHT / (483 * 1e-9)
    expected = C_LIGHT / (483.4e-9) + 20e6 - resonance
    assert optical.detuning_hz == expected
    assert optical.detuning_rad_per_us == 2 * np.pi * expected / 1e6
    assert optical.proposed_wavelength_nm == C_LIGHT / (resonance - 20e6) * 1e9
    assert req.coupling.wavelength_nm == 483.4


@pytest.mark.parametrize('policy', ['unknown', None, True, {}])
def test_unknown_energy_policy_is_not_defaulted(policy):
    with pytest.raises(PhysicsConfigurationError, match='energy convention'):
        replace(request(), optical_energy_convention=policy)


def test_changed_arc_frequency_cannot_publish_stale_ranking():
    class ChangingAtom(Atom):
        def __init__(self):
            super().__init__()
            self.seen = set()

        def getTransitionFrequency(self, *args):
            value = super().getTransitionFrequency(*args)
            if args[0] != 5 and args in self.seen:
                return value - 1.  # Within the legacy resolver's relative tolerance.
            self.seen.add(args)
            return value
    with pytest.raises(PhysicsCalculationError, match='ranking is stale'):
        select_laboratory_manifolds(ChangingAtom(), request())


def test_wavelength_conversion_underflow_is_a_project_error():
    from rydberg_aoa.level_finder.laboratory import _optical_access
    laser = replace(request().probe, wavelength_nm=5e-324)
    with pytest.raises(PhysicsConfigurationError, match='represented in metres'):
        _optical_access(laser, 1e14, 'probe')
