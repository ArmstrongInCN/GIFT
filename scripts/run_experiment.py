"""Run exactly one experiment with an explicit child-only inference environment.

This wrapper does not train models, schedule future tasks or select checkpoints.
Arguments after the experiment name go to its independent numerical entry point.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Maps each experiment code to its formal module so this wrapper launches the
# correct independent numerical entry point without selecting checkpoints.
MODULES = dict(M1="m1_equation_identification", M2="m2_recursive_prediction",
               M3="m3_cross_resolution", S1="s1_high_frequency_branch",
               S2="s2_recursive_local_correction", S3="s3_seed_stability", S4="s4_initial_distribution")


def child_environment(experiment, arguments, device, parent=None):
    parent = os.environ if parent is None else parent
    controlled = {"PYTHONPATH", "PYTHONHOME", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                  "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE"}
    env = {name: value for name, value in parent.items() if name.upper() not in controlled}
    method = next((item.split("=", 1)[1] for item in arguments if item.startswith("--method=")), None)
    if "--method" in arguments and arguments.index("--method") + 1 < len(arguments):
        method = arguments[arguments.index("--method") + 1]
    # M1 in non-GIFT mode is single-threaded to keep its reference numerics
    # reproducible; other experiments use two threads for the data pipeline.
    threads = "1" if experiment == "M1" and method != "GIFT" else "2"
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="0" if device == "cuda" else "-1", PYTHONDONTWRITEBYTECODE="1")
    if experiment != "M1":
        # Resolve the prediction modules from this checkout, including when a
        # separately installed package is present in the selected environment.
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("experiment", choices=tuple(MODULES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if any(value == "--device" or value.startswith("--device=") for value in arguments):
        parser.error("select --device before the experiment name")
    command = [args.python, "-B", "-m", "experiments.formal." + MODULES[args.experiment] + ".run",
               "--device", args.device, *arguments]
    result = subprocess.run(command, cwd=ROOT,
        env=child_environment(args.experiment, arguments, args.device))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
