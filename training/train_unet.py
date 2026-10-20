"""Independent U-Net training entry point."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    """Delegate U-Net CLI parsing, training and resume to the shared baseline controller."""
    from training.baseline_control import main as train
    train("unet", argv)


if __name__ == "__main__":
    main()
