"""Read completed native PINN coefficients without TensorFlow or retraining.

The same terminal state also belongs to its original resumable training journal.
Reading a checkpoint here does not itself execute or certify a training campaign.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "pinn"
CONDITIONS = ("noise_000", "noise_001", "noise_010")
METHODS = {"open": "PINN-SR", "known": "PINN-SR-KC"}
BUDGET = {"pre_nadam": 5000, "rounds": 6, "ado_nadam": 1000,
          "post_nadam": 20000, "lbf_maxiter": 10000, "lbf_maxfun": 10000}


def digest(raw):
    return hashlib.sha256(raw).hexdigest().upper()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def expected_library(mode):
    """Names/order of the declared physical library; no network code is loaded."""
    if mode == "known":
        return ["w", "advection", "laplacian", "q"]
    polynomials = ["", "u**1", "u**2", "v**1", "v**2", "w**1", "w**2",
                   "uv", "uw", "vw", "q", "q^2", "u*q", "v*q", "w*q"]
    derivatives = ["", "w_{x}", "w_{y}", "w_{xx}", "w_{xy}", "w_{yy}"]
    return [polynomial + derivative for polynomial in polynomials for derivative in derivatives]


def expected_layout(columns):
    """Tensor names/shapes of the fixed 3–60×8–3 network and native NAdam state."""
    layout, weights, biases = [], [], []
    for index in range(9):
        suffix = "_%d" % index if index else ""
        weight = {"name": "W" + suffix + ":0", "shape": [3 if index == 0 else 60, 3 if index == 8 else 60], "dtype": "float32"}
        bias = {"name": "b" + suffix + ":0", "shape": [1, 3 if index == 8 else 60], "dtype": "float32"}
        layout.extend([weight, bias])
        weights.append(weight)
        biases.append(bias)
    coefficient = {"name": "Variable:0", "shape": [columns, 1], "dtype": "float32"}
    layout.append(coefficient)
    layout.extend({"name": name + ":0", "shape": [], "dtype": "float32"}
                  for name in ("beta1_power", "beta2_power"))
    for item in biases + weights + [coefficient]:
        for suffix in ("/Adam:0", "/Adam_1:0"):
            layout.append({"name": item["name"][:-2] + suffix, "shape": item["shape"], "dtype": "float32"})
    return layout


def read_checkpoint(result_path, condition, mode):
    if condition not in CONDITIONS or mode not in METHODS:
        raise ValueError("unknown condition or library")
    path = Path(result_path).resolve(strict=True)
    if path.name != "result.json":
        raise ValueError("select the completed PINN run's result.json")
    raw = path.read_bytes()
    result = json.loads(raw, object_pairs_hook=unique_object)
    completion = json.loads((path.parent / "COMPLETED.json").read_bytes(), object_pairs_hook=unique_object)
    if (result.get("schema") != "gift.pinn-terminal.v1"
            or completion.get("schema") != "gift.pinn-completion.v1"
            or result.get("run_id") != completion.get("run_id")
            or completion.get("result_sha256", "").upper() != digest(raw)
            or result.get("mode") != mode or result.get("training_completed") is not True
            or result.get("diagnostic_test_only") is not False
            or completion.get("diagnostic_test_only") is not False
            or result.get("eligible_for_formal_M1") is not True
            or result.get("requested_budget") != BUDGET):
        raise ValueError("not a completed full-budget native PINN run")
    snapshot = result["terminal_snapshot"]
    identity, state = snapshot["identity"], snapshot["fsm"]
    inputs = identity["backend"]["inputs"]
    if (inputs["condition"] != condition or identity.get("mode") != mode
            or identity.get("budget") != BUDGET
            or identity.get("diagnostic_test_only") is not False
            or inputs["selection"].get("diagnostic_subset") is not False
            or inputs["selection"].get("train_rows") != 24000
            or inputs["selection"].get("physics_rows") != 84000
            or state.get("phase") != "complete" or state.get("nadam_total") != 31000
            or snapshot.get("model_iteration") != 31000 or state.get("round") != 5
            or len(state.get("stridge_records", [])) != 6 or state.get("final_mask") is not True):
        raise ValueError("checkpoint schedule or input population is incomplete")
    names = result["library"]
    columns = 4 if mode == "known" else 90
    if names != expected_library(mode):
        raise ValueError("coefficient library terms or order differ")
    archive_path = path.parent / "terminal_state.npz"
    archive_bytes = archive_path.read_bytes()
    checksum = digest(archive_bytes)
    if checksum != result["terminal_state_sha256"].upper() or checksum != completion["terminal_state_sha256"].upper():
        raise ValueError("terminal NPZ checksum mismatch")
    layout = snapshot["model_layout"]
    if layout != expected_layout(columns):
        raise ValueError("expected fixed 59-variable native TensorFlow layout")
    if result["training_cost"] != state.get("cost") or result["training_cost"].get("nadam_updates") != 31000:
        raise ValueError("training cost does not match the completed state")
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        expected = {"model_%03d" % index for index in range(59)}
        expected.update({"model_mask", "raw_coefficients", "effective_coefficients"})
        if set(archive.files) != expected:
            raise ValueError("terminal state array inventory differs")
        for index, item in enumerate(layout):
            value = archive["model_%03d" % index]
            if (value.dtype != np.dtype("float32") or list(value.shape) != item["shape"]
                    or not np.isfinite(value).all()):
                raise ValueError("invalid native state tensor")
        if archive["model_mask"].shape != (columns, 1):
            raise ValueError("native mask shape differs")
        mask = archive["model_mask"].reshape(-1)
        coefficients = archive["raw_coefficients"]
        effective = archive["effective_coefficients"]
        if (any(value.dtype != np.dtype("float32") for value in (mask, coefficients, effective))
                or mask.shape != (columns,) or coefficients.shape != (columns,) or effective.shape != (columns,)
                or not np.isin(mask, [0., 1.]).all() or not np.isfinite(coefficients).all()
                or not np.array_equal(effective, coefficients * mask)):
            raise ValueError("sparse coefficient and mask mismatch")
        if not np.array_equal(coefficients, archive["model_018"].reshape(-1)):
            raise ValueError("readout coefficients differ from native lambda_w")
        if (not np.array_equal(coefficients, np.asarray(result["raw_coefficients"], np.float32))
                or not np.array_equal(mask, np.asarray(result["mask"], np.float32))
                or not np.array_equal(effective, np.asarray(result["effective_coefficients"], np.float32))):
            raise ValueError("JSON and NPZ coefficient values differ")
    coefficients = dict(zip(names, effective.astype(np.float64).tolist()))
    parameters = ({"nu": coefficients["laplacian"], "beta": -coefficients["advection"], "gamma": coefficients["q"]}
                  if mode == "known" else
                  {"nu": (coefficients["w_{xx}"] + coefficients["w_{yy}"])/2,
                   "beta": -(coefficients["u**1w_{x}"] + coefficients["v**1w_{y}"])/2,
                   "gamma": coefficients["q"]})
    return {"method": METHODS[mode], "condition": condition, "mode": mode,
            "parameters": parameters, "coefficient_names": names,
            "coefficients": effective.astype(np.float64).tolist(),
            "readout_scope": "trained_coefficients_readout", "training": False, "forward": False,
            "verified_tensor_count": 59, "result_sha256": digest(raw), "archive_sha256": checksum,
            "source_data_protocol": {"data_sha256": inputs["data"]["sha256"],
                                     "sampling_sha256": inputs["sampling"]["sha256"]},
            "training_provenance": identity, "training_cost": result["training_cost"],
            "sparse_structure_recovery_claimed": False}


def read_reference(condition, mode, reference_root=None):
    root = Path(reference_root or REFERENCE_ROOT)
    return read_checkpoint(root / condition / mode / "result.json", condition, mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--mode", choices=tuple(METHODS), required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = (read_checkpoint(args.checkpoint, args.condition, args.mode) if args.checkpoint else
              read_reference(args.condition, args.mode))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
