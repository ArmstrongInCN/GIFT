"""Check real full-subset GIFT setup before measuring full and tail groups.

Failed disposable probes remain explicit evidence. They do not trigger native
gate changes, sample replacement, or a new formal training initialization.
"""
import argparse
import json
from pathlib import Path
import traceback

from training.w1_v2_gift import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "RUN_PLAN.json").read_text(encoding="utf-8"))
    failed = False
    for rows in (32, 8):
        path = args.root / f"PREFLIGHT_GIFT_FULL200_GROUP{rows}.json"
        if path.exists():
            raise FileExistsError("previous computation is preserved; do not overwrite a probe")
        try:
            result = preflight(plan["window"], device="cuda", seed=7, rows=rows, setup_rows=200)
        except Exception as error:
            failed = True
            result = dict(status="failed_disposable_probe", purpose="computation_feasibility_not_formal_training",
                          setup_rows=200, measurement_rows=rows, reason=repr(error), traceback=traceback.format_exc())
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({key: result[key] for key in ("status", "generator", "branch", "reason") if key in result}), flush=True)
    if failed:
        raise SystemExit("A full-subset disposable probe failed; inspect its native-gate evidence before full queue launch.")


if __name__ == "__main__":
    main()
