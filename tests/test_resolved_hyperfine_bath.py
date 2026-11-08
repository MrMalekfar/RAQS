from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.constants import h,k
from rydberg_aoa.physics.resolved_hyperfine_bath import resolve_hyperfine_bath


def test_secular_jump_split_preserves_counted_probability_and_einstein_ratio():
    a=(2,1,.5);b=(1,0,.5)
    parts={a:(SimpleNamespace(f=None),)*2,b:(SimpleNamespace(f=None),)}
    shifts={a:{None:0.},b:{None:0.}};nu=1e9;t=300.
    nb=1/np.expm1(h*nu/(k*t));down=np.sqrt(1+nb)*np.array([[.2,.3]])
    up=np.sqrt(nb)*np.array([[.2],[.3]])
    jumps=[(a,b,sparse.csr_matrix(down)),(b,a,sparse.csr_matrix(up))]
    outgoing={a:down.T@down+.1*np.eye(2),b:up.T@up+.2*np.eye(1)}
    before={f:g.copy() for f,g in outgoing.items()}
    total,result,report=resolve_hyperfine_bath(parts,shifts,outgoing,jumps,a,np.diag([-1e6,1e6]),
        {(a,b):-nu,(b,a):nu},t)
    assert report['hyperfine_sectors']==2
    retained={f:np.zeros_like(g) for f,g in outgoing.items()}
    for source,target,j in result:retained[source]+=(j.getH()@j).toarray().real
    np.testing.assert_allclose(total[a]-retained[a],.1*np.eye(2),atol=1e-13)
    np.testing.assert_allclose(total[b]-retained[b],.2*np.eye(1),atol=1e-13)
    for i,delta in enumerate([-1e6,1e6]):
        down_rate=sum(abs(j.toarray()[0,i])**2 for s,_,j in result if s==a)
        up_rate=sum(abs(j.toarray()[i,0])**2 for s,_,j in result if s==b)
        assert abs(up_rate/down_rate-np.exp(-h*(nu+delta)/(k*t)))<1e-14
    for f in outgoing:np.testing.assert_array_equal(outgoing[f],before[f])


def test_degenerate_hfs_keeps_collective_coherence():
    a=(2,1,.5);b=(1,0,.5);parts={a:(SimpleNamespace(f=None),)*2,b:(SimpleNamespace(f=None),)}
    jump=sparse.csr_matrix([[.2,.3]]);outgoing={a:(jump.getH()@jump).toarray()+.1*np.eye(2),b:np.eye(1)}
    total,result,report=resolve_hyperfine_bath(parts,{a:{None:0.},b:{None:0.}},outgoing,[(a,b,jump)],
        a,np.zeros((2,2)),{(a,b):-1e9},300.)
    assert report['hyperfine_sectors']==1
    np.testing.assert_allclose(result[0][2].toarray(),jump.toarray(),atol=1e-15)
    np.testing.assert_allclose(total[a],outgoing[a],atol=1e-15)
