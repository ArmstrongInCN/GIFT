# Training protocol / 训练协议

The four supervised prediction baselines use the same 1,000 training trajectories,
46-frame input history and **500 epochs**. Each epoch visits each trajectory once
and selects one reproducibly seeded training window per trajectory. Use the final
epoch, not test performance, to select the released baseline checkpoint.

四个同类预测网络使用相同的 1,000 条训练轨迹、46 帧输入历史和 500 epoch。
每个 epoch 将全部训练轨迹各访问一次，并以固定种子选择时间窗口。
各模型保持自己的预测方式和损失计算，不将相同 epoch 宣称为相同参数更新次数或计算量。

| Model | Epochs | Physical batch | Gradient accumulation | Expected parameter updates | Training prediction steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO-2D | 500 | 10 | 2 | 25,000 | 150 recursive |
| FNO-3D | 500 | 5 | 2 | 50,000 | 150 jointly predicted |
| U-NO | 500 | 16 | 1 | 31,500 | 20 recursive |
| U-Net | 500 | 20 | 1 | 25,000 | 4 recursive |

U-NO retains its final eight-trajectory batch in every epoch; no trajectory is
dropped. All four schedulers advance once per epoch (StepLR: 100 epochs, factor
0.5). The initial learning rate is 0.001. Weight decay is 0.00001 for U-NO and
0.0001 for the other three. Test trajectories are never used for updates,
normalization, scheduling or checkpoint selection.

## GIFT

GIFT retains its generator-pretraining and multi-stage correction design; it is
not included in the equal-epoch/equal-data comparison. Report its training and
validation trajectory counts, both pretraining phases, correction stages,
parameter updates and elapsed training times separately. In particular, the
low-frequency generator's random-minibatch updates are **steps**, not epochs.
Shared generator pretraining is reported once and its reuse across the three
correction-model seeds is explicit. Do not hide pretraining cost or describe
GIFT and the prediction baselines as compute-matched.

GIFT 不需要统一数据量和总 epoch：保留既有多阶段结构，单独完整报告成本。
低阶生成器的有放回随机小批量训练不能改名为 epoch。共享预训练成本计一次，
三个校正模型种子的后续训练分别报告；模型选择仍使用指定验证集，不使用测试结果。

The low generator uses 6,000 + 6,000 random-minibatch updates for each noise
condition (50 training / 20 validation trajectories). For each high-branch seed,
18 derivative epochs contain 313 batches each; two short-rollout epochs and two
single-epoch long-rollout stages each contain 50 batches. The full branch budget
is therefore 5,834 parameter updates, even when validation selects an earlier
model. These are budget counts; the completed history and cost records provide
the execution evidence. Local correction is an inference operation, not another
unreported training stage.

GIFT generator timing includes training, affine refits, validation, final
evaluation and intermediate checkpoint IO. Branch costs distinguish per-pass
training/validation timers from whole-run wall time. The packaged branch runs
provide whole-run time including data/RHS preparation, training, validation,
selection and weight export; per-stage times are unavailable. Shared generator
pretraining is excluded from branch times and reported separately. Each
`training_cost` states its exact timing scope; do not silently pool it with the
baselines' epoch timer or replace unavailable timings with estimated measurements.

## Independent runs, timing and continuation

### PINN equation identification

PINN is not an epoch-matched prediction baseline. Each of its six runs (three
noise conditions, known/open library) uses 24,000 observation rows and 84,000
physics rows with native Xavier initialization, seed 1234. The fixed schedule is:

1. 5,000 joint network/coefficient NAdam updates, learning rate 0.001,
   physics weight 1, coefficient L1 weight 0.0000001.
2. One native public L-BFGS phase, with `maxiter=10000`, `maxfun=10000`.
   These are upper limits, not a promise of 10,000 executed iterations.
3. Six alternating STRidge calls and 1,000 joint NAdam updates per call,
   learning rate 0.0001, physics weight 2 and no coefficient L1 penalty.
4. Freeze the exact-nonzero coefficient mask, then 20,000 joint NAdam updates
   at the same settings. No final L-BFGS phase is added.

The total is 31,000 NAdam updates, six STRidge calls and the recorded L-BFGS
callback counts. Save all native optimizer state, coefficients, mask and phase
counters together. The terminal reader verifies this completion and rejects
diagnostic subsets or reduced budgets. Report actual phase times and callbacks;
the upstream optimizer interface does not expose a complete convergence result.

M1 reads the learned physical coefficients without TensorFlow inference. In the
known library, viscosity is the Laplacian coefficient, transport strength is
minus the advection coefficient, and forcing strength is the `q` coefficient.
In the open library, average the two corresponding diffusion coefficients and
the two negated transport coefficients; forcing strength remains the coefficient
of `q`, not of `w`. Extra learned terms are retained in the coefficient output;
this three-parameter readout is not a claim of exact sparse-structure recovery.

### Prediction baselines

Train each model with its own command, for example:

```shell
python -m scripts.run_training fno2d --run-training --output ../runs/fno2d
python -m scripts.run_training fno3d --run-training --output ../runs/fno3d
python -m scripts.run_training uno --run-training --output ../runs/uno
python -m scripts.run_training unet --run-training --output ../runs/unet
```

The environment, dataset and external-source paths must be configured first.
To continue an interrupted run, add `--resume` and keep the same output directory.
The default committed boundary is every 10 completed epochs; an explicit
`--stop-after-epoch` also commits that boundary. Uncommitted epochs may need to
be repeated. Model, optimizer, scheduler and random state are restored together;
changing source code, data or training settings does not qualify as same-run continuation.

Every committed checkpoint records actual update counts and per-epoch seconds.
The final weights also contain `training_cost`: cumulative committed-epoch wall
time, including batch loading, forward/backward, updates and GPU synchronization.
It excludes setup, checkpoint writing, paused time and discarded unfinished work;
this is not an end-to-end installation or recovery-time measurement. Include this
definition and hardware/runtime details with the experimental cost table. Do not
invent a measured runtime from the expected update counts above.

U-NO retains the upstream bicubic-interpolation backward operation. PyTorch can
report that this CUDA operation is nondeterministic; its warning is not suppressed
and `warn_only=True` does not turn it into a deterministic implementation. Fixed
seeds and complete continuation state preserve the declared procedure, not a
guarantee of bitwise-identical weights across devices or independent CUDA runs.
