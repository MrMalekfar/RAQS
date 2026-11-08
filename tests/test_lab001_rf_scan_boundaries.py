"""LAB-001: RF-first selection has an explicit domain and no rescue search."""

import copy
import warnings

import numpy as np
import pytest

from rydberg_aoa.arc.ground_states import find_excited_state, find_ground_state
from rydberg_aoa.arc.types import AtomicLevel, LaserParams, RFParams
from rydberg_aoa.config import atomic_config as ac, laser_rf_config as lr
from rydberg_aoa.config.parameters import LEVEL_SELECTION_TOLERANCES as limits
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.level_finder import SingleStageLevelOptimizer
from rydberg_aoa.level_finder.arc_level_finder import (
    ARCOptimizer, resolve_rf_first_manifold, scan_fs_rf_candidates,
)


F_RF = 120e9
EXCITED = AtomicLevel(n=5, l=1, j=1.5, f=4, mf=0)


class Atom:
    groundStateN = 5
    I = 2.5
    elementName = "Rb85"

    def __init__(self, frequency=-F_RF):
        self.frequency = frequency
        self.queries = []

    def getTransitionFrequency(self, *qns):
        self.queries.append(qns)
        return 600e12 if qns[0] < 10 else self.frequency

    def getTransitionWavelength(self, *qns):
        return 780e-9

    def getEnergy(self, *qns):
        return -0.1


def _params():
    return (LaserParams(**lr.LASER_PARAMS["probe"]),
            LaserParams(**lr.LASER_PARAMS["coupling"]),
            RFParams(frequency_hz=F_RF, electric_field_vm=1., polarization=0))


BAD_REAL = [True, np.bool_(False), "120", b"120", 1j,
            np.complex128(120 + 0j), np.array(120.), np.array([120.]),
            np.nan, np.inf, 10**1000]


@pytest.mark.parametrize("bad", BAD_REAL)
@pytest.mark.parametrize("field", ["carrier", "tolerance"])
def test_scan_rejects_invalid_scalar_before_querying_arc(field, bad):
    atom = Atom()
    carrier, tolerance = (bad, 0.) if field == "carrier" else (F_RF, bad)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PhysicsConfigurationError):
            scan_fs_rf_candidates(atom, (20, 22), EXCITED, carrier, tolerance)
    assert not atom.queries


@pytest.mark.parametrize("bad", BAD_REAL + [0.])
def test_invalid_arc_frequency_is_a_calculation_error_never_a_candidate(bad):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PhysicsCalculationError):
            scan_fs_rf_candidates(Atom(bad), (20, 22), EXCITED, F_RF, F_RF)


@pytest.mark.parametrize("bounds", [(20.8, 22), (20, 22.8), (True, 22),
                                        (0, 22), (-1, 22), (22, 20),
                                        (20,), (20, 22, 24), None])
def test_scan_rejects_invalid_integer_domain(bounds):
    atom = Atom()
    with pytest.raises(PhysicsConfigurationError):
        scan_fs_rf_candidates(atom, bounds, EXCITED, F_RF, 0)
    assert not atom.queries


@pytest.mark.parametrize("cap", [True, -1, 2.5, "3", np.nan])
def test_scan_rejects_invalid_l_cap(cap):
    with pytest.raises(PhysicsConfigurationError):
        scan_fs_rf_candidates(Atom(), (20, 22), EXCITED, F_RF, 0, max_l=cap)


@pytest.mark.parametrize("carrier", BAD_REAL)
def test_resolver_rejects_invalid_carrier(carrier):
    candidates, _ = scan_fs_rf_candidates(Atom(), (20, 21), EXCITED, F_RF, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PhysicsConfigurationError):
            resolve_rf_first_manifold(candidates, Atom(), carrier)


@pytest.mark.parametrize("frequency", [np.nan, -F_RF, "120", np.complex128(F_RF + 2j)])
def test_resolver_rejects_bad_stored_carrier_before_ranking(frequency):
    candidates, _ = scan_fs_rf_candidates(Atom(), (20, 21), EXCITED, F_RF, 0)
    candidates[0]["transition_freq_hz"] = frequency
    with pytest.raises(PhysicsCalculationError):
        resolve_rf_first_manifold(candidates, Atom(), F_RF)


def test_configured_inclusive_bounds_control_both_independent_n_axes(monkeypatch):
    monkeypatch.setitem(limits, "n_range_min", np.int64(30))
    monkeypatch.setitem(limits, "n_range_max", np.int64(31))
    monkeypatch.setitem(limits, "frequency_tolerance_hz", 0.)
    atom = Atom()
    result = SingleStageLevelOptimizer(atom).optimize(*_params())
    assert result["optimization_info"]["success"] is True
    rf_queries = [q for q in atom.queries if q[0] >= 30]
    assert {q[0] for q in rf_queries} == {30, 31}
    assert {q[3] for q in rf_queries} == {30, 31}
    assert any(q[0] == q[3] for q in rf_queries)
    assert result["levels"]["rydberg1"].mj is None
    assert result["levels"]["rydberg2"].mj is None


def test_no_candidate_error_reports_domain_without_changing_inputs(monkeypatch):
    monkeypatch.setitem(limits, "n_range_min", 30)
    monkeypatch.setitem(limits, "n_range_max", 30)
    monkeypatch.setitem(limits, "frequency_tolerance_hz", 0.)
    before = copy.deepcopy(limits)
    atom = Atom(-F_RF - 1e6)
    with pytest.raises(PhysicsCalculationError, match="No RF.*candidate") as error:
        SingleStageLevelOptimizer(atom).optimize(*_params())
    assert "30" in str(error.value) and "tolerance" in str(error.value)
    assert limits == before


@pytest.mark.parametrize("key,value", [("n_range_min", 30.5), ("n_range_max", True),
                                       ("n_range_min", 0), ("n_range_max", 29)])
def test_optimizer_rejects_invalid_configured_bounds_before_arc(monkeypatch, key, value):
    monkeypatch.setitem(limits, "n_range_min", 30)
    monkeypatch.setitem(limits, "n_range_max", 31)
    monkeypatch.setitem(limits, key, value)
    atom = Atom()
    with pytest.raises(PhysicsConfigurationError):
        SingleStageLevelOptimizer(atom).optimize(*_params())
    assert not atom.queries


def test_selected_energies_come_only_from_injected_atom(monkeypatch):
    monkeypatch.setitem(limits, "n_range_min", 30)
    monkeypatch.setitem(limits, "n_range_max", 30)
    monkeypatch.setitem(limits, "frequency_tolerance_hz", 0.)
    calls = []
    class EnergyAtom(Atom):
        def getEnergy(self, *qns):
            calls.append(qns)
            return -0.125
    result = SingleStageLevelOptimizer(EnergyAtom()).optimize(*_params())
    assert len(calls) == 4
    assert all(level.energy == -0.125 for level in result["levels"].values())


def test_arc_energy_failure_keeps_cause_and_stops_selection():
    failure = RuntimeError("energy database failure")
    class BrokenEnergyAtom(Atom):
        def getEnergy(self, *qns):
            raise failure
    atom = BrokenEnergyAtom()
    with pytest.raises(PhysicsCalculationError) as error:
        SingleStageLevelOptimizer(atom).optimize(*_params())
    assert error.value.__cause__ is failure
    assert not atom.queries


def test_empty_optimizer_result_causes_one_attempt_without_tolerance_widening(monkeypatch):
    import rydberg_aoa.level_finder as finder
    calls = []
    class EmptyOptimizer:
        def __init__(self, **kwargs):
            pass
        def optimize(self, *args, **kwargs):
            calls.append(copy.deepcopy(limits))
            return {}
    monkeypatch.setattr(finder, "SingleStageLevelOptimizer", EmptyOptimizer)
    before = copy.deepcopy(limits)
    with pytest.raises(PhysicsCalculationError, match="Cannot determine atomic levels"):
        ac._initialize_atomic_levels()
    assert calls == [before]
    assert limits == before


def test_optimizer_exception_identity_is_preserved_without_retry(monkeypatch):
    import rydberg_aoa.level_finder as finder
    failure = PhysicsCalculationError("sentinel ARC failure")
    calls = []
    class BrokenOptimizer:
        def __init__(self, **kwargs):
            pass
        def optimize(self, *args, **kwargs):
            calls.append(True)
            raise failure
    monkeypatch.setattr(finder, "SingleStageLevelOptimizer", BrokenOptimizer)
    with pytest.raises(PhysicsCalculationError) as error:
        ac._initialize_atomic_levels()
    assert error.value is failure and len(calls) == 1


@pytest.mark.parametrize("operation", [find_ground_state, lambda a: find_excited_state(a, 780.)])
def test_missing_atom_is_not_replaced_by_rubidium_levels(operation):
    with pytest.raises(PhysicsConfigurationError):
        operation(None)


def test_excited_state_arc_failure_is_chained_and_d1_is_not_tried():
    failure = RuntimeError("D2 database failure")
    calls = []
    class BrokenAtom(Atom):
        def getTransitionWavelength(self, *qns):
            calls.append(qns)
            if qns[-1] == 1.5:
                raise failure
            return 795e-9
    with pytest.raises(PhysicsCalculationError) as error:
        find_excited_state(BrokenAtom(), 795.)
    assert error.value.__cause__ is failure
    assert len(calls) == 1


def test_valid_d2_domain_exclusion_allows_d1_search():
    class D1Atom(Atom):
        def getTransitionWavelength(self, *qns):
            return (780 if qns[-1] == 1.5 else 795) * 1e-9
    assert find_excited_state(D1Atom(), 795., tolerance_nm=0.01).j == 0.5


@pytest.mark.parametrize("value", [np.nan, np.inf, -780e-9, 0., "780", np.complex128(780e-9)])
def test_invalid_arc_optical_wavelength_is_not_skipped(value):
    class BadAtom(Atom):
        def getTransitionWavelength(self, *qns):
            return value
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PhysicsCalculationError):
            find_excited_state(BadAtom(), 780.)
