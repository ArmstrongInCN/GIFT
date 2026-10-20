"""Check real full-subset GIFT setup before measuring full and tail groups.

Failed disposable probes remain explicit evidence. They do not trigger native
gate changes, sample replacement, or a new formal training initialization.
"""
import argparse
import json
from pathlib import Path
import traceback

import h5py

from training.w1_v2_gift import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "RUN_PLAN.json").read_text(encoding="utf-8"))
    if tuple(plan[key] for key in ("training_count", "observation_points", "future_steps", "time_stride", "observation_dt")) != (300, 76, 75, 2, .04):
        raise ValueError("preflight requires the declared N30076 plan")
    with h5py.File(plan["window"], "r") as source:
        if (source["training/vorticity"].shape != (300, 76, 64, 64)
                or source["validation/vorticity"].shape != (20, 76, 64, 64)):
            raise ValueError("preflight dataset does not match the N30076 plan")
    failed = False
    for rows in (32, 12):
        path = args.root / f"PREFLIGHT_GIFT_FULL300_GROUP{rows}.json"
        if path.exists():
            raise FileExistsError("previous computation is preserved; do not overwrite a probe")
        try:
            result = preflight(plan["window"], device="cuda", seed=7, rows=rows, setup_rows=300)
        except Exception as error:
            failed = True
            result = dict(status="failed_disposable_probe", purpose="computation_feasibility_not_formal_training",
                          setup_rows=300, measurement_rows=rows, reason=repr(error), traceback=traceback.format_exc())
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({key: result[key] for key in ("status", "generator", "branch", "reason") if key in result}), flush=True)
    if failed:
        raise SystemExit("A full-subset disposable probe failed; inspect its native-gate evidence before full queue launch.")


if __name__ == "__main__":
    main()
