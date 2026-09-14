"""Launch one independent training command with its recorded child environment.

Only the child receives these settings. No shell is used and no popup is opened.
This is an optional convenience wrapper, not an all-experiment scheduler.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = {name: "training.train_"+name for name in ("fno2d", "fno3d", "uno", "unet", "pinn")}
MODULES.update(gift_low="experiments.formal.train_low_generator",
               gift_branch="experiments.formal.train_gift_branches")


def child_environment(model, python, device, parent=None):
    env = dict(os.environ if parent is None else parent)
    for name in ("PYTHONPATH", "PYTHONHOME", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        env.pop(name, None)
    threads = "1" if model in ("unet", "pinn") else "16"
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if model not in ("unet", "pinn"):
        env["CUDA_VISIBLE_DEVICES"] = "-1" if device == "cpu" else "0"
    if model in ("fno2d", "fno3d", "gift_branch"):
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if model == "uno":
        # Match the recorded native launch; train_uno sets Torch intra-op=1.
        absent = {"MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"}
        env = {key: value for key, value in env.items() if key.upper() not in absent | {"OMP_NUM_THREADS"}}
        env["OMP_NUM_THREADS"] = "24"
    elif model == "unet":
        from training.unet_environment import ABSENT_ENVIRONMENT, FIXED_ENVIRONMENT
        controlled = set(ABSENT_ENVIRONMENT) | set(FIXED_ENVIRONMENT)
        env = {key: value for key, value in env.items() if key.upper() not in controlled}
        env.update(FIXED_ENVIRONMENT)
    elif model == "pinn":
        env.update(CUDA_VISIBLE_DEVICES="0" if device == "gpu" else "-1",
                   TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0",
                   TF_DETERMINISTIC_OPS="1", TF_CUDNN_DETERMINISTIC="1")
        library = Path(python).resolve().parent / "Library" / "bin"
        if os.name == "nt" and library.is_dir():
            env["PATH"] = str(library) + os.pathsep + env.get("PATH", "")
    return env


def main(argv=None):
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
