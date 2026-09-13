"""Read released numeric PINN terminal coefficients; never train or run a network.

Only NumPy and the standard library are needed. Preserved optimizer tensors do
not make this reference archive a resume checkpoint for the new training code.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


MANIFEST_SHA256 = "e061310f528e864034fecf79b641ab65272f05ee51d8fccdb51b499e26955d15"
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "pinn_reference"
CONDITIONS = ("noise_000", "noise_001", "noise_010")
METHODS = {"open": "PINN-SR", "known": "PINN-SR-KC"}


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key: " + key)
        result[key] = value
    return result


def _expected_tensors(mode):
    """Archive schema only: names/shapes, not model construction or algorithms."""
    network = {"weight_0": [3, 60], "weight_out": [60, 3], "bias_out": [1, 3]}
    network.update({"weight_%d" % i: [60, 60] for i in range(1, 8)})
    network.update({"bias_%d" % i: [1, 60] for i in range(8)})
    count = 90 if mode == "open" else 4
    trainable = dict(network, coefficients=[count, 1])
    shapes = dict(trainable, coefficient_mask=[count, 1])
    for name, shape in trainable.items():
        shapes[name + "/Adam"] = shape
        shapes[name + "/Adam_1"] = shape
    gradient_shapes = []
    for i in range(8):
        gradient_shapes.extend([network["weight_%d" % i], network["bias_%d" % i]])
    gradient_shapes.extend([network["weight_out"], network["bias_out"], [count, 1]])
    shapes.update({"gradient_acc_%d" % i: shape for i, shape in enumerate(gradient_shapes)})
    shapes.update({name: [] for name in ("beta1_power", "beta2_power", "data_total", "physics_total", "l1_total")})
    return [{"key": "tensor_%03d" % i, "name": name, "shape": shapes[name], "dtype": "float32"}
            for i, name in enumerate(sorted(shapes))]


def _manifest(root):
    raw = (root / "manifest.json").read_bytes()
    if _sha256(raw) != MANIFEST_SHA256:
        raise ValueError("PINN reference manifest SHA256 mismatch")
    result = json.loads(raw, object_pairs_hook=_unique_object)
    if (result.get("schema") != "gift.pinn-reference-manifest.v1"
            or result.get("artifact_role") != "reference_not_resume"
            or result.get("is_resume_checkpoint") is not False
            or result.get("fresh_training") is not False
            or result.get("full_budget_retraining_verified") is not False):
        raise ValueError("Invalid reference-only catalog status")
    expected_pairs = {(condition, mode) for condition in CONDITIONS for mode in METHODS}
    rows = result.get("runs", [])
    if len(rows) != 6 or {(row.get("condition"), row.get("mode")) for row in rows} != expected_pairs:
        raise ValueError("Expected exactly six distinct reference conditions/modes")
    for row in rows:
        condition, mode = row["condition"], row["mode"]
        if (row.get("method") != METHODS[mode]
                or row.get("path") != "%s/%s/terminal_state.npz" % (condition, mode)
                or row.get("tensor_count") != 82
                or row.get("artifact_role") != "reference_not_resume"
                or row.get("is_resume_checkpoint") is not False
                or row.get("full_budget_retraining_verified") is not False):
            raise ValueError("Invalid reference condition/mode/path/status")
    for mode in METHODS:
        if result["tensor_schemas"][mode] != _expected_tensors(mode):
            raise ValueError("PINN terminal 82-tensor schema mismatch")
        names = result["coefficient_libraries"][mode]
        if len(names) != (90 if mode == "open" else 4) or len(set(names)) != len(names):
            raise ValueError("Invalid coefficient library identity")
        if mode == "known" and names != ["w", "u*w_x+v*w_y", "w_xx+w_yy", "q"]:
            raise ValueError("Incorrect KC coefficient order")
        if mode == "open" and not {"w_xx", "w_yy", "u*w_x", "v*w_y", "q"}.issubset(names):
            raise ValueError("Open library lacks the recorded physical terms")
    return result


def read_reference(condition, mode, reference_root=None):
    """Verify all 82 arrays, then project actual coefficient*mask values.

    ``reference_root`` may locate an unchanged copy of this complete catalog.
    A caller cannot supply a replacement manifest hash or arbitrary checkpoint.
    """
    if condition not in CONDITIONS or mode not in METHODS:
        raise ValueError("Choose a supported condition and open/known mode")
    root = Path(reference_root or REFERENCE_ROOT).resolve(strict=True)
    catalog = _manifest(root)
    record = next(row for row in catalog["runs"] if row["condition"] == condition and row["mode"] == mode)
    path = (root / record["path"]).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError("Reference archive must stay inside its catalog root")
    raw = path.read_bytes()
    if len(raw) != record["bytes"] or _sha256(raw) != record["sha256"]:
        raise ValueError("PINN reference archive size/SHA256 mismatch")
    tensors = catalog["tensor_schemas"][mode]
    expected_keys = {item["key"] for item in tensors}
    if set(record["array_sha256"]) != expected_keys:
        raise ValueError("Incomplete array checksum map")
    values = {}
    # Hash and parse the same bytes, without a second path read or pickle.
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != 82 or set(archive.files) != expected_keys:
            raise ValueError("PINN reference archive key set mismatch")
        for item in tensors:
            array = archive[item["key"]]
            if (array.dtype != np.dtype("float32") or list(array.shape) != item["shape"]
                    or not np.isfinite(array).all()):
                raise ValueError("PINN reference array dtype/shape/finite check failed")
            if _sha256(array.tobytes(order="C")) != record["array_sha256"][item["key"]]:
                raise ValueError("PINN reference tensor SHA256 mismatch")
            if item["name"] in ("coefficients", "coefficient_mask"):
                values[item["name"]] = array.copy()
    mask = values["coefficient_mask"]
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("PINN reference mask must contain only zero or one")
    effective = (values["coefficients"] * mask).reshape(-1)
    names = catalog["coefficient_libraries"][mode]
    coefficients = dict(zip(names, effective.astype(np.float64).tolist()))
    if mode == "open":
        parameters = {"nu": (coefficients["w_xx"] + coefficients["w_yy"]) / 2,
                      "beta": -(coefficients["u*w_x"] + coefficients["v*w_y"]) / 2,
                      "gamma": coefficients["q"]}
    else:
        parameters = {"nu": coefficients["w_xx+w_yy"],
                      "beta": -coefficients["u*w_x+v*w_y"], "gamma": coefficients["q"]}
    return {"schema": "gift.pinn-reference-readout.v1", "condition": condition, "mode": mode,
            "method": METHODS[mode], "readout_scope": "trained_coefficients_readout",
            "artifact_role": "reference_not_resume", "is_resume_checkpoint": False,
            "training": False, "forward": False, "fresh_training": False,
            "full_budget_retraining_verified": False, "verified_tensor_count": 82,
            "manifest_sha256": MANIFEST_SHA256, "archive_sha256": record["sha256"],
            "parameters": parameters, "coefficient_names": names,
            "coefficients": effective.astype(np.float64).tolist(),
            "nonzero_effective_coefficients": int(np.count_nonzero(effective)),
            "sparse_structure_recovery_claimed": False,
            "source_data_protocol": record["source_data_protocol"],
            "original_files": record["original_files"],
            "checkpoint_hash_scope": catalog["checkpoint_hash_scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--mode", choices=tuple(METHODS), required=True)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_reference(args.condition, args.mode, args.reference_root), indent=2))


if __name__ == "__main__":
    main()
