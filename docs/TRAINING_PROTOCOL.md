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

## GIFT prediction training / GIFT 预测训练

Two training regimes share the same GIFT architecture. **GIFT** uses all 1,000
prediction-training trajectories; its generator and high-frequency branch each
receive 500 trajectory epochs. This is **500 + 500**, not a 500-epoch total and
not a compute-matched claim. **GIFT-Lite** uses independently trained
50-trajectory models and its own multi-stage schedule; Lite denotes reduced
training data, not a smaller network. M1 identification keeps its separate
protocol below: its generator is trained on, and its coefficients are read from,
the single local training trajectory 0 that the four baselines also use.

GIFT 的生成元与高频支路各训练 500 epoch，分项报告。每个 epoch 对
全部 1,000 条训练轨迹各访问一次，从每条轨迹选择一个固定种子控制的时间窗口。
GIFT-Lite 使用 50 条训练轨迹、20 条验证轨迹及验证选模方式；网络结构不缩小。
两者的训练预算也不同，因此不能把性能差异仅归因于训练数据数量。

### Data and model selection

Prediction training IDs are 0–999; GIFT-Lite uses 0–49. Validation IDs are
1000–1039; IDs 1000–1019 contain the complete 0–10 trajectory interval and are
used for checkpoint-validation measurements. Independent test IDs are
1040–1219 (180 trajectories), paired across N64/N96/N128. Identity mapping is
fixed before model evaluation and does not depend on errors or failures.
The identification dataset used by M1 retains its separate local identifiers; all
five M1 configurations identify their parameters from local training trajectory 0.

Full-data GIFT uses **terminal stage states and terminal epoch weights**.
Validation monitors the declared procedure and does not select a model or
alter learning rates. Test observations are not used for parameter updates,
normalization, stage decisions or checkpoint selection.

### Full-data stages and budget

| Component / stage | Trajectory epochs | Physical batch | Parameter updates |
| --- | ---: | ---: | ---: |
| Generator, phase 1 | 250 | 16 (final batch 8) | 15,750 |
| Generator, phase 2 | 250 | 16 (final batch 8) | 15,750 |
| Branch, derivative supervision | 400 | 16 (final batch 8) | 25,200 |
| Branch, short rollout | 50 | 5 | 10,000 |
| Branch, long rollout stage 1 | 25 | 5 | 5,000 |
| Branch, long rollout stage 2 | 25 | 5 | 5,000 |

The generator has 31,500 gradient updates; each branch has 45,200. Its affine
constant/linear fit is a separate analytic operation: one initial fit and one
fit on the same sampled observations at the end of each generator epoch,
501 fits total. These fits are disclosed, not relabelled as gradient updates.
The generator is trained once and reused by the three independently initialized
branches with seeds 20260820, 20260821 and 20260822.

Generator architecture, rank 8, P21 support, four-point centred derivative
numerator on five observed frames, normalized MSE, gradient clipping at 5,
rank-aware affine solver and AdamW are unchanged mathematical components.
The two phase learning rates are 0.01 and 0.003, with within-phase cosine decay
to 2% of the initial value and weight decay 1e-8. The affine fit and the
gradient updates use training observations only.

Branch architecture, Q21 derivative loss, dual-track RK4, local correction,
10-step short rollout, 50-step long rollout with 10-step truncated gradient
segments, and gradient clipping at 1 remain the GIFT mathematical components.
Stage learning rates are 0.0015, 0.0002, 0.00008 and 0.00008; AdamW weight decay
is 1e-6. The derivative stage uses cosine decay to 0.0001. Each new stage
initializes its own optimizer; continuation within a stage restores that
optimizer's complete state. No terminal state is replaced by a better test or
validation checkpoint.

Generator derivative centres are sampled from 2–498. Branch derivative
centres use the same available observation stencil. Rollout anchors are
sampled from 0–450 so every 50-step target is a saved observation within 0–10.
The full-data sampling unit is a trajectory epoch. The reduced-data derivative
stage instead passes over its fixed set of derivative examples.

### Reduced-data GIFT-Lite

GIFT-Lite uses a generator trained for 6,000 + 6,000 random-minibatch
updates on 50 trajectories, with 20 validation trajectories. These updates
are steps, not trajectory epochs. Its three branch runs use 18 derivative
passes, two short-rollout passes and two single-pass long-rollout stages,
5,834 updates per seed. The packaged branches are the validation-selected
weights from their respective fresh-initialization runs.
The shared generator cost is reported once, separately from the three branches.

### Independent commands and continuation

```shell
python -m training.train_gift_generator --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m training.train_gift_predictor --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --low-model ../runs/gift_generator/model.pt --seed 20260820 --output ../runs/gift_branch_20260820
```

Run the branch command separately for each of the three declared seeds. Replace
`--run-training` with `--resume` to restore that run's committed model, optimizer,
scheduler, epoch/stage counters and random state. The default boundary is every
10 epochs and each stage end; `--stop-after-epoch` commits an explicit boundary.
Uncommitted epochs may be repeated after interruption. A published unrelated
generator or a reduced-data checkpoint is not accepted as the full-data training
prerequisite.

All four GIFT training entry points share `--execution auto`: fresh GPU training
uses CUDA graphs, CPU uses eager, and resume retains the journal's backend.
Use `--execution eager` for the explicit reference implementation. This does not
change the full-data 500 + 500 epoch budgets, the reduced-data selection protocol,
or any batch/update counts. Source and runtime checks remain strict. See
[execution efficiency and measured scope](PERFORMANCE.md).

Full-data timers sum committed epoch time: sampling, forward/backward,
parameter updates, generator affine refits / branch frozen-RHS preparation,
and scheduled validation. They exclude initial input loading/calibration,
checkpoint writing, paused time and discarded uncommitted work. Report the
measured times from each completed run; do not substitute estimated timings.

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
