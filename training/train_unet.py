"""Independent U-Net entry; formal environment checks precede scientific imports."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.unet_environment import preflight


def main(argv=None):
    """Keep --help and rejected formal launches free of runtime/data imports."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        dispatch = preflight(arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if dispatch:
        from training.baseline_control import main as train
        train("unet", arguments)

if __name__ == "__main__":
    main()
