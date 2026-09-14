"""Stdlib-only, read-only startup checks for the single-GPU U-Net profile.

These checks select no device, seed, backend flag, or training budget. Configure
the documented environment in the child process before starting Python. A
successful gate is not a scientific reproduction or full-training certificate.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping


ABSENT_ENVIRONMENT = (
    "PYTHONHOME", "PYTHONPATH", "PYTHONPYCACHEPREFIX", "PYTHONSTARTUP",
    "PYTHONINSPECT", "PYTHONUSERBASE", "PYTHONWARNINGS", "PYTHONOPTIMIZE",
    "PYTHONSAFEPATH", "PYTHONCASEOK", "CUDA_VISIBLE_DEVICES",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TF_XLA_FLAGS", "XLA_FLAGS",
    "KMP_AFFINITY", "KMP_BLOCKTIME", "KMP_DUPLICATE_LIB_OK", "KMP_SETTINGS",
    "ATEN_CPU_CAPABILITY", "NPY_DISABLE_CPU_FEATURES", "MKL_CBWR",
    "MKL_ENABLE_INSTRUCTIONS", "CUDA_DEVICE_MAX_CONNECTIONS",
    "TORCH_CUDNN_V8_API_DISABLED", "CUBLASLT_WORKSPACE_SIZE",
    "TF_USE_LEGACY_KERAS", "NVIDIA_TF32_OVERRIDE",
)
FIXED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    "PYTHONBREAKPOINT": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1", "MKL_DYNAMIC": "FALSE", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0", "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "TF_DETERMINISTIC_OPS": "1", "TF_CUDNN_DETERMINISTIC": "1",
}
_PREIMPORT_VALIDATED = False


def validate_environment(environment: Mapping[str, str] | None = None) -> None:
    """Reject missing, mismatched or case-variant profile keys; never mutate.

    Only known profile names appear in an error: arbitrary environment values
    and unrelated keys are never printed. Presence differs from an empty value.
    Canonical uppercase spelling is required on case-sensitive hosts as well.
    """
    environment = os.environ if environment is None else environment
    names: dict[str, list[str]] = {}
    for key in environment:
        names.setdefault(key.upper(), []).append(key)
    present = [key for key in ABSENT_ENVIRONMENT if key in names]
    mismatched = [key for key, value in FIXED_ENVIRONMENT.items()
                  if names.get(key) != [key] or environment.get(key) != value]
    if present or mismatched:
        details = []
        if present:
            details.append("must be absent: " + ", ".join(present))
        if mismatched:
            details.append("fixed value/canonical key required: " + ", ".join(mismatched))
        raise ValueError("U-Net environment mismatch; " + "; ".join(details)
                         + ". Use scripts.run_training or docs/SETUP.md; no environment was changed.")


def validate_before_import() -> None:
    """Bind the formal startup check to a process with no scientific imports."""
    global _PREIMPORT_VALIDATED
    validate_environment()
    if any(name.split(".", 1)[0] in {"numpy", "torch", "h5py"} for name in sys.modules):
        raise ValueError("U-Net formal startup must run before NumPy/Torch/HDF5 imports; use training.train_unet in a new process")
    _PREIMPORT_VALIDATED = True


def require_validated_startup() -> None:
    """Reject internal-controller bypass and environment changes after preflight."""
    if not _PREIMPORT_VALIDATED:
        raise ValueError("U-Net formal execution requires the pre-import training.train_unet entry point")
    validate_environment()


def preflight(argv: list[str]) -> bool:
    """Return whether to dispatch; only exact --tiny opts out of formal checks.

    The downstream complete parser also disables abbreviation. Unknown options
    are left for that parser, never interpreted here as a nonformal opt-out.
    This small help does not import the scientific runtime or read any inputs.
    """
    parser = argparse.ArgumentParser(prog="python -m training.train_unet", add_help=False,
                                     allow_abbrev=False)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    selected, _ = parser.parse_known_args(argv)
    if selected.help:
        print("Independent U-Net training / same-run resume\n"
              "Usage: python -m training.train_unet [--dry-run | --run-training] [options]\n"
              "Formal startup requires the child environment described in docs/SETUP.md.\n"
              "Options: --output PATH, --device DEVICE (default cuda), --resume,\n"
              "  --data-file RELATIVE_PATH, --data-profile released|regenerated,\n"
              "  --stop-after-epoch N, --checkpoint-interval N (default 10).\n"
              "Explicit NONFORMAL tests only: --tiny [--device cpu] [--tiny-epochs N]\n"
              "  [--tiny-trajectories N] [--tiny-batch N] [--tiny-rollout N].\n"
              "Long-option abbreviations are not accepted. No runtime/data is loaded by --help.")
        return False
    if not selected.tiny:
        validate_before_import()
    return True
