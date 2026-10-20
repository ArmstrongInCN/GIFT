"""Canonical paper experiment identities, ordering and method display names.

Display names are deliberately one-way: the current N50 model and the
independent auxiliary GIFT-Lite model share a paper label, not a checkpoint.
Experiment identities are the same in code, reports, figures and command lines.
"""
from __future__ import annotations

PAPER_EXPERIMENT_ORDER = ("W1", "W2", "W3", "S1", "S2", "S3")
EXPERIMENT_ID_CHOICES = PAPER_EXPERIMENT_ORDER
PAPER_TITLES_ZH = {"W1": "流场演化预测", "W2": "跨分辨率预测", "W3": "方程参数识别",
                   "S1": "高频支路有效性", "S2": "递归局部修正有效性", "S3": "初值分布对照"}
PAPER_FIGURE_NUMBERS = {"W1": 2, "W2": 3, "W3": 4}

CURRENT_PREDICTION_METHOD_ORDER = ("GIFT", "GIFT-N50", "fno2d", "fno3d", "uno", "unet", "anie_ns")
PARAMETER_METHOD_ORDER = ("GIFT", "PDE-FIND", "PDE-FIND-KC", "PINN-SR", "PINN-SR-KC")
METHOD_DISPLAY_NAMES = {"GIFT": "GIFT", "GIFT-N50": "GIFT-Lite", "GIFT-Lite": "GIFT-Lite",
                        "fno2d": "FNO-2D", "FNO2D": "FNO-2D", "FNO-2D": "FNO-2D",
                        "fno3d": "FNO-3D", "FNO3D": "FNO-3D", "FNO-3D": "FNO-3D",
                        "uno": "U-NO", "U-NO": "U-NO", "unet": "U-Net", "U-Net": "U-Net",
                        "anie_ns": "ANIE", "ANIE": "ANIE", "PDE-FIND": "PDE-FIND",
                        "PDE-FIND-KC": "PDE-FIND-KC", "PINN-SR": "PINN-SR", "PINN-SR-KC": "PINN-SR-KC"}


def resolve_experiment_id(identifier: str) -> str:
    """Validate an identity without translating or accepting alternate numbering."""
    if identifier not in EXPERIMENT_ID_CHOICES:
        raise ValueError(f"unknown experiment {identifier!r}; choose {', '.join(EXPERIMENT_ID_CHOICES)}")
    return identifier


def method_display_name(source_method: str) -> str:
    """Format a method label while preserving its independent source key."""
    return METHOD_DISPLAY_NAMES.get(source_method, source_method)


def prediction_method_order(*, include_anie: bool = True) -> tuple[str, ...]:
    """Return source keys in presentation order; ANIE participates only in W1."""
    return tuple(method for method in CURRENT_PREDICTION_METHOD_ORDER if include_anie or method != "anie_ns")
