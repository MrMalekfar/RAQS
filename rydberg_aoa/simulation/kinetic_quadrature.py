"""Positive volume-Maxwell quadrature for ballistic histories; no flux weight."""
from numbers import Real
import numpy as np
from scipy.special import roots_laguerre,ndtr,ndtri
from ..exceptions import PhysicsConfigurationError,PhysicsCalculationError


def rayleigh_speed_rule(sigma,order):
    """Gauss rule directly in transverse speed, with volume density s*exp(-s²/2).

    Dimensionless moments are 2^(k/2) Gamma(1+k/2). Stieltjes recurrence and
    the Jacobi eigenproblem give positive weights, without flux weighting.
    High precision is used only to construct the small moment recurrence.
    """
    if (isinstance(sigma,(bool,np.bool_)) or not isinstance(sigma,Real) or not np.isfinite(sigma) or sigma<=0
            or type(order) is not int or not 1<=order<=64):
        raise PhysicsConfigurationError('Positive thermal sigma and Rayleigh order in 1..64 required')
    import mpmath as mp
    from scipy.linalg import eigh_tridiagonal
    with mp.workdps(max(60,8*order)):
        moments=[mp.power(2,mp.mpf(k)/2)*mp.gamma(1+mp.mpf(k)/2) for k in range(2*order+1)]
        def inner(p,q):return mp.fsum(a*b*moments[i+j] for i,a in enumerate(p) for j,b in enumerate(q))
        p=[mp.mpf(1)];previous=[];beta=mp.mpf(0);diagonal=[];off_diagonal=[]
        for j in range(order):
            xp=[mp.mpf(0)]+p;alpha=inner(xp,p);diagonal.append(float(alpha))
            if j+1==order:break
            residual=xp.copy()
            for i,a in enumerate(p):residual[i]-=alpha*a
            for i,a in enumerate(previous):residual[i]-=beta*a
            next_beta=mp.sqrt(inner(residual,residual));assert next_beta>0
            off_diagonal.append(float(next_beta));previous=p;p=[a/next_beta for a in residual];beta=next_beta
        # QR preserves the tiny quadrature weights needed by high moments.
        nodes,vectors=eigh_tridiagonal(diagonal,off_diagonal,lapack_driver='stev' if order>16 else 'auto')
    weights=vectors[0]**2
    assert np.all(nodes>0) and np.all(weights>0) and abs(weights.sum()-1)<1e-13
    return sigma*nodes,weights


def patterson_uniform_rule(order):
    """Positive nested rule on [-1,1], orders 3 through 255; mass two.

    Construct each added polynomial by moment orthogonality, then integrate
    its Lagrange basis. Patterson, Math. Comp. 22 (1968), 847-856.
    High precision is confined to this small rule construction.
    """
    if type(order) is not int or order not in (3,7,15,31,63,127,255):
        raise PhysicsConfigurationError('Patterson order in 3,7,15,31,63,127,255 required')
    import mpmath as mp
    precision=max(100,4*order)
    with mp.workdps(precision):
        def polynomial(roots):
            p=[mp.mpf(1)]
            for root in roots:
                q=[mp.mpf(0)]*(len(p)+1)
                for j,a in enumerate(p):q[j]-=root*a;q[j+1]+=a
                p=q
            return p
        def moment(j):return mp.mpf(0) if j%2 else mp.mpf(2)/(j+1)
        nodes=[-mp.sqrt(mp.mpf(3)/5),mp.mpf(0),mp.sqrt(mp.mpf(3)/5)]
        tolerance=mp.power(10,-precision//2)
        while len(nodes)<order:
            n=len(nodes);p=polynomial(nodes)
            moments=[moment(j) for j in range(3*n+2)]
            integrals=[mp.fsum(a*moments[i+j] for i,a in enumerate(p)) for j in range(2*n+2)]
            matrix=mp.matrix([[integrals[i+j] for j in range(n+1)] for i in range(n+1)])
            try:e=list(mp.lu_solve(matrix,mp.matrix([-integrals[i+n+1] for i in range(n+1)])))+[mp.mpf(1)]
            except (ArithmeticError,ValueError) as exc:
                raise PhysicsCalculationError(f'Patterson moment solve failed at order {n}') from exc
            derivative=[j*e[j] for j in range(1,len(e))];added=[]
            edges=[mp.mpf(-1)]+nodes+[mp.mpf(1)]
            for left,right in zip(edges[:-1],edges[1:]):
                fleft=mp.polyval(e[::-1],left);fright=mp.polyval(e[::-1],right)
                if fleft*fright>=0:raise PhysicsCalculationError('Patterson roots do not interlace')
                x=(left+right)/2
                for iteration in range(256):
                    fx=mp.polyval(e[::-1],x);dx=mp.polyval(derivative[::-1],x)
                    if fx==0:break
                    if fleft*fx>0:left=x;fleft=fx
                    else:right=x;fright=fx
                    candidate=x-fx/dx if dx else (left+right)/2
                    if not left<candidate<right:candidate=(left+right)/2
                    if abs(candidate-x)<tolerance:x=candidate;break
                    x=candidate
                else:raise PhysicsCalculationError('Patterson root refinement did not converge')
                added.append(x)
            nodes=sorted(nodes+added)
        p=polynomial(nodes);derivative=[j*p[j] for j in range(1,len(p))];weights=[]
        for node in nodes:
            divided=[mp.mpf(0)]*order;divided[-1]=p[-1]
            for j in range(order-2,-1,-1):divided[j]=p[j+1]+node*divided[j+1]
            weights.append(mp.fsum(a*moment(j) for j,a in enumerate(divided))/mp.polyval(derivative[::-1],node))
        x=np.array([float(a) for a in nodes]);w=np.array([float(a) for a in weights])
    if not (np.all(abs(x)<1) and np.all(w>0) and abs(sum(w)-2)<1e-13):
        raise PhysicsCalculationError('Patterson rule lost positive mass')
    return x,w


def transverse_maxwell_rule(sigma,radial_order,angular_order):
    """vx,vy and raw weights: q=v_perp²/(2sigma²), density exp(-q)dq dtheta/(2pi)."""
    if (isinstance(sigma,(bool,np.bool_)) or not isinstance(sigma,Real) or not np.isfinite(sigma) or sigma<=0
            or any(type(n) is not int or n<1 for n in (radial_order,angular_order))):
        raise PhysicsConfigurationError('Positive thermal sigma and integer radial/angular orders required')
    q,w=roots_laguerre(radial_order);theta=(np.arange(angular_order)+.5)*2*np.pi/angular_order
    speed=sigma*np.sqrt(2*q)
    velocities=np.stack((speed[:,None]*np.cos(theta),speed[:,None]*np.sin(theta)),axis=-1).reshape(-1,2)
    weights=np.repeat(w/angular_order,angular_order)
    return velocities,weights


def axial_maxwell_rule(sigma,boundaries,order):
    """Gauss-Legendre in Gaussian CDF on explicit velocity panels, without renormalization.

    Weights sum to Phi(vmax/sigma)-Phi(vmin/sigma) by construction. The
    unrepresented tails remain missing probability, not reweighted atoms.
    """
    edges=np.asarray(boundaries)
    if (isinstance(sigma,(bool,np.bool_)) or not isinstance(sigma,Real) or not np.isfinite(sigma) or sigma<=0
            or type(order) is not int or order<1 or edges.ndim!=1 or len(edges)<2
            or edges.dtype.kind not in 'iuf' or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0)):
        raise PhysicsConfigurationError('Positive sigma, ordered velocity panels and positive integer order required')
    z,w=np.polynomial.legendre.leggauss(order);cdf=ndtr(edges/sigma);nodes=[];weights=[]
    if np.any(np.diff(cdf)<=0) or cdf[0]<=0 or cdf[-1]>=1:
        raise PhysicsConfigurationError('Velocity CDF panels must have positive representable finite mass')
    for a,b in zip(cdf[:-1],cdf[1:]):
        p=a+(b-a)*(z+1)/2
        if np.any((p<=0)|(p>=1)):raise PhysicsConfigurationError('Unrepresentable Gaussian quantile node')
        nodes.extend(sigma*ndtri(p));weights.extend((b-a)*w/2)
    return np.asarray(nodes),np.asarray(weights)
