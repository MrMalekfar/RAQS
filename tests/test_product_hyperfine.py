"""Angular-momentum spectrum oracle independent of product-basis construction."""
import numpy as np
import pytest
from rydberg_aoa.physics.magnetic_core import CoreState
from rydberg_aoa.physics.product_hyperfine import product_hyperfine_hz
from rydberg_aoa.exceptions import PhysicsConfigurationError


def test_product_operator_matches_total_F_eigenenergies_and_degeneracy():
    j=1.5;spin=3.5;a=23000.;b=-2700.
    states=tuple(CoreState('r',35,1,j,mj=m,mi=i) for m in np.arange(-j,j+1) for i in np.arange(-spin,spin+1))
    operator=product_hyperfine_hz(states,spin,a,b)
    energies=[]
    for f in np.arange(abs(j-spin),j+spin+1):
        c=f*(f+1)-spin*(spin+1)-j*(j+1)
        e=.5*a*c+b*(.75*c*(c+1)-spin*(spin+1)*j*(j+1))/(2*spin*(2*spin-1)*j*(2*j-1))
        energies.extend([e]*int(2*f+1))
    np.testing.assert_allclose(np.linalg.eigvalsh(operator),sorted(energies),atol=1e-9,rtol=0.)
    # BLAS evaluation of the quadratic term has roundoff at sub-nanohertz scale.
    np.testing.assert_allclose(operator,operator.T,atol=1e-9,rtol=0.)
    assert abs(np.trace(operator))<1e-8
    m=np.array([s.mj+s.mi for s in states])
    assert np.max(abs((m[:,None]-m[None,:])*operator))==0


def test_permuting_product_states_only_permutes_hamiltonian():
    states=tuple(CoreState('r',35,1,.5,mj=m,mi=i) for m in (-.5,.5) for i in (-.5,.5))
    h=product_hyperfine_hz(states,.5,12.,0.);order=[2,0,3,1]
    np.testing.assert_allclose(product_hyperfine_hz(tuple(states[i] for i in order),.5,12.,0.),h[np.ix_(order,order)])


def test_missing_mI_channel_is_not_inferred():
    with pytest.raises(PhysicsConfigurationError,match='Complete'):
        product_hyperfine_hz((CoreState('r',35,1,.5,mj=.5,mi=.5),),.5,1.,0.)


def test_hyperfine_variant_keeps_original_model_and_dipoles_unchanged():
    from dataclasses import dataclass
    from scipy import sparse
    from rydberg_aoa.physics.hyperfine_variant import with_product_hyperfine
    @dataclass
    class Model:
        basis:tuple
        hamiltonian:object
        report:dict
    states=tuple(CoreState('r',35,1,.5,mj=m,mi=i) for m in (-.5,.5) for i in (-.5,.5))
    model=Model(states,sparse.csr_matrix((4,4)),{'model':'test'})
    changed=with_product_hyperfine(model,(35,1,.5),nuclear_spin=.5,a_hz=12.,b_hz=0.,
        provenance=dict(source='independent test input',data_origin='synthetic oracle'))
    np.testing.assert_allclose(changed.hamiltonian.toarray(),-2*np.pi/1e6*product_hyperfine_hz(states,.5,12.,0.))
    assert model.hamiltonian.nnz==0 and model.report=={'model':'test'}
    assert changed.basis is model.basis and changed.report['added_HFS_changes_radiative_bath'] is False
    with pytest.raises(PhysicsConfigurationError,match='already added'):
        with_product_hyperfine(changed,(35,1,.5),nuclear_spin=.5,a_hz=12.,b_hz=0.,
            provenance=dict(source='test',data_origin='synthetic oracle'))
