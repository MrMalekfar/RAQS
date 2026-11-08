"""Live fixed-channel evidence with independent angular × reduced-J oracles."""

import math

from arc import Rubidium85
import pytest
from sympy import Rational as R, sqrt
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j

from rydberg_aoa.arc.dipole import EA0_TO_CM
from rydberg_aoa.config.constants import HBAR
from rydberg_aoa.config.rf_configuration import build_configured_lab_request
from rydberg_aoa.level_finder.prepared_channel import select_prepared_laboratory_channel
from tests.test_lab002_laboratory_request import reference_request
from tests.test_lab003_optical_feasibility import configuration_snapshot


def predicted_elements(atom, req, candidate):
    """No full ARC DME API: signed angular algebra × reduced electronic DME."""
    g, e, p = req.preparation.ground, req.preparation.excited, req.preparation
    a, b = candidate.rydberg1, candidate.rydberg2
    i, jg, je, fg, fe, mg, me = map(R, (atom.I, g.j, e.j, g.f, e.f, g.mf, e.mf))
    qp, qc, qr = map(R, (req.probe.polarization, req.coupling.polarization, req.lo.polarization))
    j1, j2, m1, m2 = map(R, (a.j, b.j, p.rydberg1_mj, p.rydberg2_mj))
    angular_probe = ((-1)**(fg - mg) * wigner_3j(fg, 1, fe, -mg, -qp, me)
                     * (-1)**(jg + i + fe + 1) * sqrt((2*fg + 1)*(2*fe + 1))
                     * wigner_6j(fg, 1, fe, je, i, jg))
    mj_e = m1 - qc
    angular_coupling = ((-1)**(je - mj_e) * clebsch_gordan(je, i, fe, mj_e, me - mj_e, me)
                        * wigner_3j(je, 1, j1, -mj_e, -qc, m1))
    angular_rf = (-1)**(j1 - m1) * wigner_3j(j1, 1, j2, -m1, -qr, m2)
    return (float(angular_probe) * atom.getReducedMatrixElementJ(g.n, g.l, g.j, e.n, e.l, e.j),
            float(angular_coupling) * atom.getReducedMatrixElementJ(e.n, e.l, e.j, a.n, a.l, a.j),
            float(angular_rf) * atom.getReducedMatrixElementJ(a.n, a.l, a.j, b.n, b.l, b.j))


@pytest.fixture(scope='module')
def live():
    atom, template = Rubidium85(), reference_request()
    req = build_configured_lab_request(preparation=template.preparation, bounds=template.bounds, atom=atom)
    before = configuration_snapshot()
    result = select_prepared_laboratory_channel(atom, req)
    assert configuration_snapshot() == before
    return atom, req, result


def test_live_all_candidates_match_signed_independent_oracles(live):
    atom, req, result = live
    allowed = []
    for fs in result.optical_selection.candidates:
        probe, coupling, rf = predicted_elements(atom, req, fs)
        if probe != 0 and coupling != 0 and rf != 0:
            allowed.append(fs)
    assert [c.manifold for c in result.candidates] == allowed
    assert result.winner.manifold == allowed[0]
    for c in result.candidates:
        expected = predicted_elements(atom, req, c.manifold)
        assert (result.probe_ea0, c.coupling_ea0, c.rf_ea0) == pytest.approx(expected, rel=2e-13, abs=0)
    assert result.winner.rydberg1.mj == result.winner.rydberg2.mj == -.5
    assert result.nuclear_projection == .5
    assert result.rf_orientation.signed_frequency_hz == result.winner.manifold.signed_frequency_hz
    print('\nAUTO-001B live winner:', result.winner)
    print('probe [ea0]:', result.probe_ea0, 'Rabi [rad/us]:', result.rabi)


def test_live_lo_signal_use_identical_channel_and_correct_angular_units(live):
    _, req, result = live
    expected_lo = abs(result.winner.rf_ea0) * EA0_TO_CM * req.lo.electric_field_vm / HBAR / 1e6
    assert result.rabi.lo == expected_lo
    assert math.isclose(result.rabi.signal / result.rabi.lo,
                        req.signal.electric_field_vm / req.lo.electric_field_vm, rel_tol=2e-15)
    # FS selection boundary stays unmodified by channel materialization.
    assert result.optical_selection.winner.rydberg1.mj is None
    assert result.level_objects()['rydberg1'].mj == req.preparation.rydberg1_mj
