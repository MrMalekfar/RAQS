"""Spawn-local BDF option instrumentation and exact nested output-grid checks."""
import numpy as np
from ..config.detection_numerics import SolverDiagnostic
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError


def diagnostic_grid(times, subdivision):
    from ..config.laboratory import _integer
    from ..config.detection_numerics import validate_grid_subdivision
    subdivision=_integer(subdivision,'NOISE-016 subdivision',minimum=1)
    validate_grid_subdivision(times,subdivision)
    values=[]
    for a,b in zip(times,times[1:]):
        values.extend([a]+[a+(b-a)*i/subdivision for i in range(1,subdivision)])
    values.append(times[-1])
    if not np.all(np.diff(values)>0) or not np.array_equal(np.asarray(values)[::subdivision],times):
        raise PhysicsConfigurationError('NOISE-016: requested subdivision has no exact representable nested grid.')
    return tuple(values)


def run_diagnostic_worker(task, invoke):
    """Called only in a serial spawn worker; restore mesolve even on failure."""
    from . import time_domain as td
    options=task.solver_diagnostic
    if not isinstance(options,SolverDiagnostic):
        raise PhysicsConfigurationError('NOISE-016: typed solver diagnostic required.')
    grid=diagnostic_grid(task.tlist_us,options.output_subdivision)
    original=td.mesolve; calls=[]
    def instrumented(*args,**kwargs):
        merged=dict(kwargs['options'])
        baseline = getattr(task,'temporal_ode',None)
        expected = {'method':'bdf','nsteps':5000,'normalize_output':False} if baseline is None else baseline.options()
        if any(merged.get(key) != value for key,value in expected.items()):
            raise PhysicsCalculationError('NOISE-016: canonical BDF/certification policy changed; no diagnostic override.')
        # An explicit diagnostic work limit is independent of local error
        # tolerances. The canonical production options above stay unchanged.
        merged.update(rtol=options.rtol,atol=options.atol,nsteps=options.nsteps)
        kwargs['options']=merged;calls.append(merged)
        return original(*args,**kwargs)
    td.mesolve=instrumented
    try:
        record=invoke(grid)
    finally:
        td.mesolve=original
    from .temporal_doppler import DopplerWorkerResult
    receipt = record if isinstance(record, DopplerWorkerResult) else None
    expected_calls = receipt.report['class_solves'] if receipt is not None else 1
    if len(calls) != expected_calls:
        raise PhysicsCalculationError(f'NOISE-016: expected {expected_calls} instrumented mesolve calls, got {len(calls)}; no silent default case.')
    if options.output_subdivision==1:return record
    if receipt is not None:record = receipt.record
    trace=np.asarray(record[4])
    if len(trace)!=len(grid):raise PhysicsCalculationError('NOISE-016: dense worker returned incomplete certification grid.')
    selected=trace[::options.output_subdivision]
    from ..utils.numerical import extract_if_component
    amplitude,xf,yf=extract_if_component(selected,task.tlist_us,task.signed_if_mhz,task.transient_delay_us)
    coarse = (amplitude,xf,yf,record[3],selected)
    return DopplerWorkerResult(coarse, receipt.report) if receipt is not None else coarse
