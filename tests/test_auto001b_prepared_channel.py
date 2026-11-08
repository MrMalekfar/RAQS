"""AUTO-001B fixed-channel domain, RF priority, units and fail-loud boundaries."""

from dataclasses import replace
import math

import numpy as np
import pytest

from rydberg_aoa.arc.dipole import EA0_TO_CM
from rydberg_aoa.config.constants import C_LIGHT, EPSILON_0, HBAR
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.level_finder import SingleStageLevelOptimizer
from rydberg_aoa.level_finder.prepared_channel import select_prepared_laboratory_channel
from tests.test_lab003_optical_feasibility import Atom as OpticalAtom, request as optical_request, configuration_snapshot


def request():
    req = optical_request()
    return replace(req, bounds=replace(req.bounds, coupling_window_nm=(480., 510.)))


class Atom(OpticalAtom):
    def __init__(self):
        super().__init__()
        self.dipole_calls = []
        self.probe = -2.
        self.coupling = lambda args: -.02 if args[5] == 26 else 200.
        self.rf = lambda args: -3.

    def getDipoleMatrixElementHFS(self, *args):
        self.dipole_calls.append(('probe', args))
        return self.probe

    def getDipoleMatrixElementHFStoFS(self, *args):
        self.dipole_calls.append(('coupling', args))
        return self.coupling(args)

    def getDipoleMatrixElement(self, *args):
        self.dipole_calls.append(('rf', args))
        return self.rf(args)


def test_rf_first_not_strongest_dipole_and_fixed_channel_is_resolved():
    atom, req = Atom(), request()
    before = configuration_snapshot()
    result = SingleStageLevelOptimizer(atom).optimize_prepared_laboratory(req)
    assert result.winner.rydberg1.n == 26
    assert result.winner.manifold.mismatch_hz == 0
    assert result.winner.coupling_ea0 == -.02 and result.winner.rf_ea0 == -3
    assert result.probe_ea0 == -2
    assert any(c.coupling_ea0 == 200 for c in result.candidates)
    assert result.request is req
    assert result.implied_excited_mj == -.5 and result.nuclear_projection == .5
    assert result.rf_orientation.direction == -1
    assert configuration_snapshot() == before
    levels = result.level_objects()
    assert levels['ground'].mj is levels['excited'].mj is None
    assert levels['ground'].mf == levels['excited'].mf == 0
    assert levels['rydberg1'].mj == levels['rydberg2'].mj == -.5
    levels['rydberg1'].mj = 1.5
    assert result.winner.rydberg1.mj == -.5


def test_every_call_has_declared_projections_q_and_no_magnetic_scan():
    atom = Atom()
    result = select_prepared_laboratory_channel(atom, request())
    probe_calls = [a for kind, a in atom.dipole_calls if kind == 'probe']
    assert probe_calls == [(5, 0, .5, 3., 0., 5, 1, 1.5, 4., 0., 0)]
    coupling = [a for kind, a in atom.dipole_calls if kind == 'coupling']
    assert all(a[:5] == (5, 1, 1.5, 4., 0.) and a[-2:] == (-.5, 0) for a in coupling)
    assert len(coupling) == len(set(a[5:9] for a in coupling))  # One call per R1.
    rf = [a for kind, a in atom.dipole_calls if kind == 'rf']
    assert all(a[3] == a[7] == -.5 and a[8] == 0 for a in rf)
    assert len(rf) == len(result.candidates)


@pytest.mark.parametrize('edge', ['coupling', 'rf'])
def test_valid_zero_excludes_pair_in_same_channel_then_rf_ranks(edge):
    atom = Atom()
    if edge == 'coupling':
        atom.coupling = lambda a: 0. if a[5] == 26 else -1.
    else:
        atom.rf = lambda a: 0. if a[0] == 26 else -1.
    result = select_prepared_laboratory_channel(atom, request())
    assert result.optical_selection.winner.rydberg1.n == 26
    assert result.winner.rydberg1.n == 27
    assert result.winner.manifold.mismatch_hz == 100
    assert result.dark_exclusions and all(why == edge for _, why in result.dark_exclusions)
    assert all(c.rydberg1.mj == c.rydberg2.mj == -.5 for c in result.candidates)


def test_all_dark_is_failure_without_substituted_channel_or_floor():
    atom = Atom()
    atom.coupling = lambda a: 0.
    before = configuration_snapshot()
    with pytest.raises(PhysicsConfigurationError, match='no driven candidate'):
        select_prepared_laboratory_channel(atom, request())
    assert all(kind != 'rf' for kind, _ in atom.dipole_calls)
    assert configuration_snapshot() == before


def test_dark_prepared_probe_aborts_before_fs_scan():
    atom = Atom()
    atom.probe = 0.
    with pytest.raises(PhysicsConfigurationError, match='probe has zero ARC dipole'):
        select_prepared_laboratory_channel(atom, request())
    assert atom.calls == []
    assert len(atom.dipole_calls) == 1


def test_exact_probe_angular_zero_does_not_accept_backend_roundoff():
    req = request()
    req = replace(req, preparation=replace(req.preparation,
                  excited=replace(req.preparation.excited, f=3.)))
    atom = Atom()  # Backend would return a finite nonzero value for this dark channel.
    with pytest.raises(PhysicsConfigurationError, match='exact zero angular coefficient'):
        select_prepared_laboratory_channel(atom, req)
    assert atom.dipole_calls == atom.calls == []


def test_impossible_implied_electronic_projection_fails_before_arc():
    req = request()
    req = replace(req, preparation=replace(req.preparation, rydberg1_mj=-2.5, rydberg2_mj=-2.5))
    atom = Atom()
    with pytest.raises(PhysicsConfigurationError, match='implied excited mJ'):
        select_prepared_laboratory_channel(atom, req)
    assert atom.dipole_calls == atom.calls == []


BAD = [None, True, np.bool_(False), '1', b'1', 1j, np.complex128(1+0j),
       np.array(1.), np.array([1.]), np.nan, np.inf, 10**1000]


@pytest.mark.parametrize('edge', ['probe', 'coupling', 'rf'])
@pytest.mark.parametrize('bad', BAD)
def test_invalid_arc_dipole_is_not_a_dark_domain_exclusion(edge, bad):
    atom = Atom()
    if edge == 'probe':
        atom.probe = bad
    else:
        setattr(atom, edge, lambda a: bad)
    with pytest.raises(PhysicsCalculationError, match='dipole'):
        select_prepared_laboratory_channel(atom, request())


@pytest.mark.parametrize('method', ['getDipoleMatrixElementHFS', 'getDipoleMatrixElementHFStoFS', 'getDipoleMatrixElement'])
def test_arc_exception_preserves_cause_and_never_continues(method):
    atom = Atom()
    failure = ValueError('ARC database failed')
    def fail(*args):
        raise failure
    setattr(atom, method, fail)
    with pytest.raises(PhysicsCalculationError) as error:
        select_prepared_laboratory_channel(atom, request())
    assert error.value.__cause__ is failure


def test_existing_physics_error_identity_is_preserved():
    atom = Atom()
    failure = PhysicsCalculationError('original physics error')
    def fail(*args):
        raise failure
    atom.getDipoleMatrixElementHFS = fail
    with pytest.raises(PhysicsCalculationError) as error:
        select_prepared_laboratory_channel(atom, request())
    assert error.value is failure


def test_rabi_units_against_independent_gaussian_and_mu_e_over_hbar():
    req = request()
    result = select_prepared_laboratory_channel(Atom(), req)
    def expected(d, laser):
        intensity = 2 * laser.power_mw * 1e-3 / (math.pi * (laser.beam_waist_mm * 1e-3)**2)
        field = math.sqrt(2 * intensity / (EPSILON_0 * C_LIGHT))
        return abs(d) * EA0_TO_CM * field / HBAR / 1e6
    assert result.rabi.probe == pytest.approx(expected(-2., req.probe), rel=2e-15)
    assert result.rabi.coupling == pytest.approx(expected(-.02, req.coupling), rel=2e-15)
    assert result.rabi.lo == abs(-3.) * EA0_TO_CM * req.lo.electric_field_vm / HBAR / 1e6
    assert result.rabi.signal / result.rabi.lo == pytest.approx(req.signal.electric_field_vm / req.lo.electric_field_vm, rel=2e-15)


def test_zero_applied_fields_turn_drives_off_without_zeroing_dipoles():
    req = request()
    req = replace(req, probe=replace(req.probe, power_mw=0), coupling=replace(req.coupling, power_mw=0),
                  lo=replace(req.lo, electric_field_vm=0), signal=replace(req.signal, electric_field_vm=0))
    result = select_prepared_laboratory_channel(Atom(), req)
    assert (result.rabi.probe, result.rabi.coupling, result.rabi.lo, result.rabi.signal) == (0, 0, 0, 0)
    assert result.probe_ea0 == -2 and result.winner.rf_ea0 == -3


def test_small_nonzero_dipole_is_not_clipped_or_rejected_as_dark():
    atom = Atom()
    atom.rf = lambda a: -1e-12
    result = select_prepared_laboratory_channel(atom, request())
    assert result.winner.rf_ea0 == -1e-12
    assert result.rabi.lo > 0 and not result.dark_exclusions


def test_frequency_changes_during_channel_evaluation_invalidate_ranking():
    class ChangingAtom(Atom):
        def getTransitionFrequency(self, *args):
            value = super().getTransitionFrequency(*args)
            return value - 1 if args[0] != 5 and any(k == 'rf' for k, _ in self.dipole_calls) else value
    with pytest.raises(PhysicsCalculationError, match='ranking is stale'):
        select_prepared_laboratory_channel(ChangingAtom(), request())


def test_nonrequest_is_a_project_configuration_error():
    with pytest.raises(PhysicsConfigurationError, match='explicit ExperimentalLabRequest'):
        select_prepared_laboratory_channel(Atom(), None)


@pytest.mark.parametrize('q', [-1, 1])
def test_sigma_rf_keeps_declared_ladder_projections_and_downward_energy(q):
    req = request()
    req = replace(req, preparation=replace(req.preparation, rydberg2_mj=-.5 + q),
                  lo=replace(req.lo, polarization=q), signal=replace(req.signal, polarization=q))
    atom = Atom()
    result = select_prepared_laboratory_channel(atom, req)
    assert result.winner.rydberg1.mj == -.5
    assert result.winner.rydberg2.mj == -.5 + q
    assert result.rf_orientation.direction == -1
    assert all(a[7] - a[3] == a[8] == q for kind, a in atom.dipole_calls if kind == 'rf')


def test_error_on_losing_candidate_cannot_be_hidden_by_an_earlier_valid_winner():
    atom = Atom()
    failure = ValueError('later candidate ARC failure')
    def coupling(args):
        if args[5] == 27:
            raise failure
        return -1.
    atom.coupling = coupling
    with pytest.raises(PhysicsCalculationError) as error:
        select_prepared_laboratory_channel(atom, request())
    assert error.value.__cause__ is failure
    assert any(kind == 'rf' for kind, _ in atom.dipole_calls)


@pytest.mark.parametrize('change', [{'power_mw': 5e-324}, {'beam_waist_mm': 5e-324}])
def test_gaussian_unit_conversion_underflow_is_not_a_zero_drive(change):
    req = request()
    req = replace(req, probe=replace(req.probe, **change))
    with pytest.raises(PhysicsConfigurationError, match='conversion underflow'):
        select_prepared_laboratory_channel(Atom(), req)


def test_nonzero_arc_dipole_that_underflows_in_si_is_a_visible_failure():
    atom = Atom()
    atom.probe = -5e-324
    with pytest.raises(PhysicsCalculationError, match='dipole SI conversion underflow'):
        select_prepared_laboratory_channel(atom, request())
