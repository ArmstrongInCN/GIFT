"""Independent FNO-3D training/resume entry point."""
from pathlib import Path
import sys

# Allow running as a script (no package context) by putting the repo root on
# the path before importing the shared baseline controller.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.baseline_control import main

if __name__ == "__main__":
    main("fno3d")
