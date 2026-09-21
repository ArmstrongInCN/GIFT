"""Small CPU-only adapter check. No data, training, rollout or GPU is used."""

from __future__ import annotations

import argparse
import json
import time

from .models import EXPECTED_PARAMETERS, build_model, load_checkpoint, source_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(EXPECTED_PARAMETERS), required=True)
    parser.add_argument("--external-root")
    parser.add_argument("--checkpoint", help="optional trusted terminal artifact; strict state-key check")
    parser.add_argument("--forward", action="store_true", help="one small synthetic CPU forward")
    args = parser.parse_args()
    started = time.perf_counter()
    import torch

    # One thread and a fixed seed keep this smoke check cheap and repeatable; it
    # measures nothing and must never be quoted as a timing.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.checkpoint:
        model, _ = load_checkpoint(args.model, args.checkpoint, external_root=args.external_root)
    else:
        model = build_model(args.model, external_root=args.external_root)
    model.eval()
    result = {"model": args.model, "source": source_record(args.model, args.external_root),
              "parameters": sum(p.numel() for p in model.parameters()),
              "state_entries": len(model.state_dict()), "checkpoint": args.checkpoint,
              "strict_state_load": bool(args.checkpoint), "device": "cpu",
              "training_performed": False, "reproduction_claim": False}
    if args.forward:
        # Input and output layouts differ per model and follow each upstream
        # implementation: (batch, x, y, channels) for fno2d and uno, an extra
        # time axis for fno3d, channels-first for unet.
        shape = {"fno2d": (1, 24, 24, 46), "fno3d": (1, 16, 16, 16, 49),
                 "uno": (1, 64, 64, 46), "unet": (1, 46, 64, 64)}[args.model]
        expected = {"fno2d": (1, 24, 24, 1), "fno3d": (1, 16, 16, 16, 1),
                    "uno": (1, 64, 64, 1), "unet": (1, 1, 64, 64)}[args.model]
        with torch.no_grad():
            prediction = model(torch.zeros(shape, dtype=torch.float32))
        if tuple(prediction.shape) != expected or not torch.isfinite(prediction).all().item():
            raise ValueError("synthetic forward shape/finite check failed")
        result.update(input_shape=list(shape), output_shape=list(prediction.shape), finite=True)
    result["seconds"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
