"""Run exactly one experiment with an explicit child-only inference environment.

This wrapper does not train models, schedule future tasks or select checkpoints.
Arguments after the experiment name go to its independent numerical entry point.
The default current protocol dispatches the N300/single-initial-state W1/W2
evaluation entries. Select the reference protocol explicitly for the independent
N1000/46-history prediction experiments. Neither route trains or selects models.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from experiments.paper_nomenclature import EXPERIMENT_ID_CHOICES, resolve_experiment_id

ROOT = Path(__file__).resolve().parents[1]
# Maps each experiment code to its formal module so this wrapper launches the
# correct independent numerical entry point without selecting checkpoints.
MODULES = dict(W3="w3_equation_identification", W1="w1_recursive_prediction",
               W2="w2_cross_resolution", S1="s1_high_frequency_branch",
               S2="s2_recursive_local_correction", S3="s3_initial_distribution")
CURRENT_MODULES = {"W1": "experiments.additional.w1_current_evaluate",
                   "W2": "experiments.additional.w2_single_evaluate"}


def child_environment(experiment, arguments, device, parent=None):
    """Build child-only paths and thread defaults while preserving user overrides.

    CPU mode hides CUDA in the child. GPU visibility and existing environment
    settings are otherwise retained; the parent process environment is not
    rewritten.
    """
    parent = os.environ if parent is None else parent
    env = dict(parent)
    method = next((item.split("=", 1)[1] for item in arguments if item.startswith("--method=")), None)
    if "--method" in arguments and arguments.index("--method") + 1 < len(arguments):
        method = arguments[arguments.index("--method") + 1]
    # Default to one thread for W3's non-GIFT reference numerics and two for the
    # other data pipelines. setdefault preserves explicit user thread settings.
    threads = "1" if experiment == "W3" and method != "GIFT" else "2"
    for key, value in {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                       "OPENBLAS_NUM_THREADS": "1"}.items():
        env.setdefault(key, value)
    env["CUDA_VISIBLE_DEVICES"] = "-1" if device == "cpu" else parent.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), str(ROOT), parent.get("PYTHONPATH"))))
    return env


def main(argv=None):
    """Launch exactly one experiment as an argument-list subprocess and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("experiment", choices=EXPERIMENT_ID_CHOICES)
    parser.add_argument("--protocol", choices=("current", "reference"), default="current",
                        help="current selects the N300/single-state W1/W2 protocol; reference is the independent N1000 protocol")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        source_experiment = resolve_experiment_id(args.experiment)
    except ValueError as error:
        parser.error(str(error))
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if any(value == "--device" or value.startswith("--device=") for value in arguments):
        parser.error("select --device before the experiment name")
    if any(value == "--protocol" or value.startswith("--protocol=") for value in arguments):
        parser.error("select --protocol before the experiment name")
    module = (CURRENT_MODULES[source_experiment] if args.protocol == "current" and source_experiment in CURRENT_MODULES
              else "experiments.formal." + MODULES[source_experiment] + ".run")
    command = [args.python, "-B", "-m", module,
               "--device", args.device, *arguments]
    result = subprocess.run(command, cwd=ROOT,
        env=child_environment(source_experiment, arguments, args.device))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
