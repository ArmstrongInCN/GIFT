"""Render saved ANIE measurements into a new figure directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.additional.w1_anie.run import draw_comparison
from experiments.formal._shared.figure_evidence import bind_result


def create_figures(source, output):
    """Render committed comparison measurements into a separate new figure directory.

    The saved result is read-only and model prediction is never repeated.
    """
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Saved measurements and new figures must use separate directories")
    evidence = bind_result(source, "W1-ANIE", ("summary/comparison.csv",))
    output.mkdir(parents=True, exist_ok=False)
    draw_comparison(source, output / "figures")
    return {"status": "complete", "output": str(output),
            "source_report_sha256": evidence["report_sha256"],
            "predictions_recomputed": False, "original_files_modified": False}


def main(argv=None):
    """Print a saved-figure plan or render it when explicitly executed."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = (create_figures(args.source, args.output) if args.execute else
              {"status": "PLAN_ONLY", "source": str(args.source), "output": str(args.output)})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
