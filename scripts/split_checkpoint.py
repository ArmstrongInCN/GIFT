"""Create lossless checkpoint parts for repository distribution; no training."""
import argparse
from pathlib import Path

from training.weight_files import split_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    # The 40 MiB default keeps each part under GitHub's 100 MiB ordinary limit;
    # split_weights preserves every byte and records an index for reconstruction.
    parser.add_argument("--part-size-mib", type=int, default=40)
    args = parser.parse_args(argv)
    print(split_weights(args.input, args.output, part_bytes=args.part_size_mib * 1024 ** 2))


if __name__ == "__main__":
    main()
