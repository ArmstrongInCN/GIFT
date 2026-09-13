"""Persist numerical evidence before running optional figure subprocesses."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Sequence

from .common import finish_output, write_json_new


def publish_numeric_then_plot(output: Path, report: dict[str, Any], commands: Sequence[Sequence[str]]) -> None:
    """A plot failure cannot erase a completed numeric report or raw outputs.

    The final manifest honestly distinguishes full completion from a plotting
    failure. Figures can subsequently be rendered with their independent CLIs
    into a new figure directory without rerunning model inference.
    """
    numeric_report = {**report, "status": "numerical_complete", "plots": "not_run"}
    write_json_new(output / "numeric_report.json", numeric_report)
    completed = 0
    try:
        for command in commands:
            subprocess.run(list(command), check=True)
            completed += 1
    except (OSError, subprocess.CalledProcessError) as error:
        failed_report = {**report, "status": "numerical_complete_plot_failed",
                         "plot_execution": {"completed_commands": completed,
                                            "error": str(error)}}
        finish_output(output, failed_report)
        raise RuntimeError(
            f"Numerical results are safely saved in {output}; optional plotting failed. "
            "Re-render figures separately; do not rerun training."
        ) from error
    finish_output(output, {**report, "plot_execution": {
        "status": "complete" if commands else "skipped_by_request",
        "completed_commands": completed,
    }})
