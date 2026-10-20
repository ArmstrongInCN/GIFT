"""Invoke the upstream public L-BFGS interface; do not implement its solver."""
import time
import numpy as np


def run_lbfgs(runtime, feed, *, maxiter=10000, maxfun=10000):
    """One atomic training phase; resume restarts an uncommitted phase.

    The repaired outside runtime exposes the native SciPy OptimizeResult as
    metadata. Callback counts and actual termination are both retained; reaching
    a configured limit is not described as convergence. A generic interface
    without that metadata can still be diagnosed but is explicitly unverified.
    """
    optimizer = runtime.model.optimizer_BFGS_Pre
    optimizer.optimizer_kwargs["options"].update(maxiter=maxiter, maxfun=maxfun)
    initial_loss = runtime.metrics(feed)["loss"] if callable(getattr(runtime, "metrics", None)) else None
    before = runtime.state() if callable(getattr(runtime, "state", None)) else None
    counts = {"iterations": 0, "function_evaluations": 0}
    def step_callback(_):
        """Count native optimizer iteration callbacks without inferring convergence."""
        counts["iterations"] += 1
    def loss_callback(loss):
        """Count finite objective evaluations; reject a nonfinite loss before reporting success."""
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite native L-BFGS loss")
        counts["function_evaluations"] += 1
    started = time.perf_counter()
    optimizer.minimize(runtime.session, feed_dict=feed, fetches=[runtime.model.loss],
                       step_callback=step_callback, loss_callback=loss_callback)
    seconds = time.perf_counter()-started
    final_loss = runtime.metrics(feed)["loss"]
    result = {**counts, "seconds": seconds, "initial_loss": initial_loss,
              "final_loss": final_loss, "loss_change": None if initial_loss is None else final_loss-initial_loss,
              "termination": "upstream_interface_returned; convergence_not_exposed",
              "resume_boundary": "phase_start_if_interrupted"}
    if before is not None:
        after = runtime.state()
        result["max_state_array_change"] = max(float(np.max(np.abs(left-right)))
            for left, right in zip(before["variables"], after["variables"]))
    if callable(getattr(runtime, "lbfgs_result", None)):
        native = runtime.lbfgs_result()
        if any(not np.isfinite(native[key]) for key in ("objective", "gradient_inf_norm")):
            raise FloatingPointError("Nonfinite native L-BFGS terminal objective or gradient")
        result.update(scipy_result=native, termination=native["message"],
                      convergence_reported_by_scipy=bool(native["success"]),
                      callback_counts_match_scipy=(counts["iterations"] == native["iterations"]
                          and counts["function_evaluations"] == native["function_evaluations"]))
    return result
