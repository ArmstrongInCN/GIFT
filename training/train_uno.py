"""Independent U-NO training/resume entry point."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.baseline_control import main

if __name__ == "__main__":
    import torch
    # The measured native U-NO training profile uses one intra-op thread.
    # This controls CPU scheduling, not the external optimizer or network.
    torch.set_num_threads(1)
    main("uno")
