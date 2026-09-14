"""Invoke the upstream public L-BFGS interface; do not implement its solver."""
import time
import numpy as np


def run_lbfgs(runtime, feed, *, maxiter=10000, maxfun=10000):
    """One atomic training phase; resume restarts an uncommitted phase.

    The upstream interface returns after assigning the final variables but does
    not return SciPy's OptimizeResult. Report counters, not inferred convergence.
    """
    optimizer = runtime.model.optimizer_BFGS_Pre
    optimizer.optimizer_kwargs["options"].update(maxiter=maxiter, maxfun=maxfun)
    counts = {"iterations": 0, "function_evaluations": 0}
    def step_callback(_):
        counts["iterations"] += 1
    def loss_callback(loss):
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite native L-BFGS loss")
        counts["function_evaluations"] += 1
    started = time.perf_counter()
    optimizer.minimize(runtime.session, feed_dict=feed, fetches=[runtime.model.loss],
                       step_callback=step_callback, loss_callback=loss_callback)
    return {**counts, "seconds": time.perf_counter()-started,
            "final_loss": runtime.metrics(feed)["loss"],
            "termination": "upstream_interface_returned; convergence_not_exposed",
            "resume_boundary": "phase_start_if_interrupted"}
