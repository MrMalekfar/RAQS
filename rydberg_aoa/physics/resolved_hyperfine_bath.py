"""Fully secular radiative-bath control for one explicit product-HFS manifold."""
import numpy as np
from numbers import Real
from scipy import sparse
from scipy.constants import h,k
from ..exceptions import PhysicsConfigurationError,PhysicsCalculationError


def resolve_hyperfine_bath(parts, shifts, outgoing, jumps, manifold_fs,
                            hyperfine_hz, signed_frequencies_hz, temperature_k):
    """Split incident jumps by HFS eigenspaces and update Einstein weights.

    This is a fully secular LIMIT, not an interpolation or a proof that the
    secular approximation applies. Other manifolds keep their declared basis
    and shifts. The unretained outgoing operator is secularized but its rate
    spectrum is not recalculated: unknown destinations remain first exits.
    Inputs are not mutated. Frequencies and hyperfine_hz are in Hz; existing
    jump amplitudes have units sqrt(1/us). No linewidth or reset is added.
    """
    fs=tuple(manifold_fs);hz=np.asarray(hyperfine_hz)
    if (fs not in parts or hz.shape!=(len(parts[fs]),)*2 or not np.isfinite(hz).all()
            or np.linalg.norm(hz-hz.conj().T)>1e-9
            or isinstance(temperature_k,(bool,np.bool_)) or not isinstance(temperature_k,Real)
            or not np.isfinite(temperature_k) or temperature_k<=0):
        raise PhysicsConfigurationError('Explicit Hermitian HFS and positive bath temperature required')
    eigen,vectors=np.linalg.eigh(hz);sectors=[]
    # Eigenvalues within 1e-6 Hz represent one exact F degeneracy.
    for ids in np.split(np.arange(len(eigen)),np.flatnonzero(np.diff(eigen)>1e-6)+1):
        sectors.append((float(eigen[ids].mean()),vectors[:,ids]@vectors[:,ids].conj().T))
    assert np.linalg.norm(sum(p for _,p in sectors)-np.eye(len(hz)))<1e-12
    retained={f:np.zeros_like(np.asarray(outgoing[f]),dtype=complex) for f in parts}
    for source,_,j in jumps:retained[source]+=(j.getH()@j).toarray()
    unknown={f:np.asarray(outgoing[f])-retained[f] for f in parts}
    unknown[fs]=sum(p@unknown[fs]@p for _,p in sectors)
    result=[];max_relative_rate_change=0.;incident=0
    def occupation(frequency):return 1/np.expm1(h*frequency/(k*temperature_k))
    for source,target,j in jumps:
        if fs not in (source,target):result.append((source,target,j));continue
        incident+=1;op=j.toarray();rows,columns=np.nonzero(op)
        src_shifts=np.array([shifts[source][s.f] for s in parts[source]])
        dst_shifts=np.array([shifts[target][s.f] for s in parts[target]])
        frequency=signed_frequencies_hz[(source,target)]+dst_shifts[rows]-src_shifts[columns]
        old=float(frequency.mean())
        if np.max(abs(frequency-old))>1e-10*abs(old):
            raise PhysicsConfigurationError('Each incident jump must have a single declared pre-splitting frequency')
        source_sectors=sectors if source==fs else [(0.,np.eye(op.shape[1]))]
        target_sectors=sectors if target==fs else [(0.,np.eye(op.shape[0]))]
        nb=occupation(abs(old))
        for es,ps in source_sectors:
            for et,pt in target_sectors:
                actual=old+et-es
                if old*actual<=0:raise PhysicsCalculationError('HFS changes the declared transition energy order')
                thermal=(1+occupation(abs(actual)))/(1+nb) if old<0 else occupation(abs(actual))/nb
                ratio=(abs(actual)/abs(old))**3*thermal
                max_relative_rate_change=max(max_relative_rate_change,abs(ratio-1))
                result.append((source,target,sparse.csr_matrix(np.sqrt(ratio)*(pt@op@ps))))
    totals={f:g.copy() for f,g in unknown.items()}
    for source,_,j in result:totals[source]+=(j.getH()@j).toarray()
    for g in unknown.values():
        if np.linalg.eigvalsh(g).min()<-1e-11:raise PhysicsCalculationError('Negative first-exit operator in bath control')
    return totals,result,dict(hyperfine_sectors=len(sectors),incident_jumps_before=incident,
        jumps_before=len(jumps),jumps_after=len(result),maximum_relative_Einstein_rate_change=max_relative_rate_change,
        fully_secular_limit_only=True,unknown_destination_frequencies_recalculated=False)
