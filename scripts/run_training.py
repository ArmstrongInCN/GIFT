"""Launch one model's training command with suggested thread settings."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Maps each baseline/GIFT model to its training module so this wrapper launches
# the right run with the recorded child environment and no checkpoint selection.
MODULES = {name: "training.train_"+name for name in ("fno2d", "fno3d", "uno", "unet", "pinn")}
MODULES.update(gift_low="experiments.formal.train_low_generator",
               gift_branch="experiments.formal.train_gift_branches",
               gift_generator="training.train_gift_generator",
               gift_predictor="training.train_gift_predictor")


def child_environment(model, python, device, parent=None):
    """Set model-specific child environment defaults without changing the parent.

    Project imports precede the inherited PYTHONPATH. Thread values and visible
    GPU mappings supplied by the user remain effective; CPU selection explicitly
    hides CUDA only in the launched training process.
    """
    env = dict(os.environ if parent is None else parent)
    threads = "1" if model in ("unet", "pinn") else "16"
    for key, value in {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                       "OPENBLAS_NUM_THREADS": "1"}.items():
        env.setdefault(key, value)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH"))))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if model.startswith("gift_"):
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if model == "uno":
        env["OMP_NUM_THREADS"] = (os.environ if parent is None else parent).get("OMP_NUM_THREADS", "24")
    elif model == "unet":
        from training.unet_environment import FIXED_ENVIRONMENT
        for key, value in FIXED_ENVIRONMENT.items():
            env.setdefault(key, value)
    elif model == "pinn":
        for key, value in {"TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0",
                           "TF_DETERMINISTIC_OPS": "1", "TF_CUDNN_DETERMINISTIC": "1"}.items():
            env.setdefault(key, value)
        library = Path(python).resolve().parent / "Library" / "bin"
        if os.name == "nt" and library.is_dir():
            env["PATH"] = str(library) + os.pathsep + env.get("PATH", "")
    return env


def main(argv=None):
    """Dispatch one training entry with explicit device and forwarded model options."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("model", choices=tuple(MODULES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cuda")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if any(value == "--device" or value.startswith("--device=") for value in arguments):
        parser.error("select --device before the model, not inside forwarded arguments")
    device = ("gpu" if args.device == "cuda" else args.device) if args.model == "pinn" else args.device
    if args.model != "pinn" and device == "gpu":
        parser.error("PyTorch models use device cuda or cpu")
    command = [args.python, "-B", "-m", MODULES[args.model], "--device", device, *arguments]
    result = subprocess.run(command, cwd=ROOT, env=child_environment(args.model, args.python, device))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
