# Checkpoints: train, read or resume / 检查点使用

These are separate operations:

| Operation | Input | Training performed? |
| --- | --- | --- |
| Fresh training | Supplied observations and initialized model | Full declared budget |
| Quick evaluation | Packaged trained weights plus test inputs | No |
| Same-run continuation | That run's committed model, optimizer, random and phase state | Remaining uncommitted work |

Downloaded terminal weights are not inserted into an unrelated run's training
journal. See [SETUP.md](SETUP.md) for each model's independent command and the
special PINN L-BFGS phase-boundary recovery rule.

## GIFT training regimes

Full-data prediction GIFT uses `artifacts/gift_full/generator.pt` and
`artifacts/gift_full/gift_seed_SEED.pt`. Its generator and each branch are trained
on 1,000 trajectories for 500 epochs each; their budgets and times are reported
separately. Branch metadata bind the exact generator hash and declared seed.

The full-data generator and all three prediction branches are included with
their completed training histories, budget records and weight hashes. These
are terminal 500-epoch states, not intermediate training checkpoints. Both GIFT
regimes can be evaluated using the commands in EXPERIMENT_EXECUTION.md.

GIFT-Lite uses the preserved clean generator in `artifacts/fixed_k21_n64_unmasked/`
and the three branches in `artifacts/gift/`. It uses 50 training trajectories and
its own multi-stage schedule, not a smaller network. This is not a data-only
ablation with identical training budgets. M1 retains its own clean/noisy generator
weights and parameter-identification protocol.

## Ordinary and split weight files

`artifacts/CHECKPOINTS.json` records the available weights and SHA-256 hashes.
Most models use ordinary `.pt` or `.npz` files. U-NO uses `weights.json` and
numbered `.part` files in `artifacts/formal/uno/`; keep that folder together.

The parts contain the **exact bytes of one normal trained PyTorch checkpoint**,
not quantized or altered weights. The loader checks each part and the complete
reconstructed-file hash, then calls `torch.load(weights_only=True)`. Reconstruction
is in memory; it does not create another file or fetch data from the network.

U-NO 权重以无损分片保存，避免超过 GitHub 普通 Git 的单文件限制。运行时自动校验、
按顺序拼接后加载；模型参数、数值精度和训练算法均不改变。不要单独加载某个分片。

An independently trained ordinary `.pt` file is also accepted through M2's
`--uno-model /path/to/model.pt`. To prepare your own completed weights for
distribution, use a new output directory:

```shell
python -m scripts.split_checkpoint --input ../runs/uno/model.pt --output ../weights/uno
```

The default part size is 40 MiB. This storage operation performs no training and
does not establish the checkpoint's scientific accuracy. Formal experiment
readers separately require the declared 500-epoch prediction budget.

## Verification scope

File hash identity, identical tensors, successful same-run recovery and agreement
of experimental metrics are different checks. Runtime/rounding differences can
change generated fields and learned weights. A completed budget is not a claim
of low error or exact sparse-structure recovery. Model-specific timing scopes
are reported alongside the weights rather than pooled as identical measurements.
