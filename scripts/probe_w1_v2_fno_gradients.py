"""Fresh-initialization gradient feasibility, using training fields only."""
import gc
import json
from pathlib import Path

import h5py
import torch

from adapters.models import fno_utilities
from training.w1_v2_train import build_network, seed_initialization, atomic_json


def main():
    torch.set_num_threads(4)
    from experiments.provenance import external_source
    root = external_source('prediction_probe')
    source = r"D:\2_GIFT\external_sources_v2"
    with h5py.File(root / "data/train_validation_5to8.h5") as data:
        truth = torch.tensor(data["training/vorticity"][:4], device="cuda")
    criterion = fno_utilities(source)[1](size_average=False)
    rows = []
    for horizon in (1, 5, 10, 20, 50, 100, 150):
        seed_initialization(7)
        model = build_network(dict(method="fno2d", history=1, source_root=source), "cuda")
        prediction = truth[:, 0].unsqueeze(-1)
        loss = 0
        for step in range(horizon):
            prediction = model(prediction)
            loss = loss + criterion(prediction.reshape(4, -1), truth[:, step + 1].reshape(4, -1)) / 4
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in grads)
        row = dict(horizon=horizon, loss=float(loss), finite_gradients=finite,
                   max_abs_gradient=max(float(gradient.abs().max()) for gradient in grads) if finite else None,
                   activation_checkpoint=False, model_seed=7, micro_batch=4)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, prediction, loss, grads
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(root / "fno2d_gradient_feasibility.json", dict(training_rows=[0, 1, 2, 3], measurements=rows,
                validation_or_test_used=False, explanation="No training updates; distinguishes native long-horizon gradient instability from memory checkpointing"))


if __name__ == "__main__":
    main()
