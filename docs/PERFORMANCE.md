# GIFT execution efficiency / GIFT 计算效率

Every GIFT training entry point uses the shared `auto` execution policy. Fresh
GPU training uses `cuda-graph` to replay the same tensor operations with less
CPU dispatch overhead; CPU uses `eager`. `--execution eager` selects the explicit
reference training path. Prediction remains eager by default. No extra package
is needed. The formal GIFT-Lite branch command requires CUDA; CPU fallback applies
to entry points that support CPU.

所有 GIFT 独立训练入口共用 `auto` 执行策略：GPU 新训练默认使用 `cuda-graph`，
CPU 使用 `eager`；`--execution eager` 可明确选择参考训练路径。预测默认路径不变。
优化不改变模型、训练数据量、epoch 数、批大小、
AdamW、学习率日程、损失、RK4 步长或递归截断位置，也不启用混合精度。

## Use / 使用

Full-data GIFT and reduced-data GIFT-Lite use the same training engine. Each
model still has its own command, scientific schedule and complete-state journal:

```shell
python -m scripts.run_training gift_generator --run-training --dataset ../GIFT-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m scripts.run_training gift_predictor --run-training --dataset ../GIFT-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --low-model ../runs/gift_generator/model.pt --seed 20260820 --output ../runs/gift_branch_20260820
python -m scripts.run_training gift_low --run-training --condition noise_000 --output ../runs/gift_lite_generator
python -m scripts.run_training gift_branch --run-training --low-model ../runs/gift_lite_generator/gift_main.pt --seed 20260820 --output ../runs/gift_lite_branch_20260820
```

Continue with `--resume`, the same output directory, source, runtime and execution
backend. `auto` inherits the journal's backend when resuming; it does not silently
upgrade an eager run. Graphs are rebuilt from the saved model, optimizer, scheduler and RNG
state; the checkpoint never depends on a serialized CUDA graph. Changing source
or switching backends is **not** silently accepted as same-run continuation.
Existing parameter artifacts remain usable for inference.

断点恢复时保持输出目录、源代码、运行环境及执行后端一致。`gift_low`、
`gift_generator` 和 `gift_predictor` 将 `--run-training` 替换为 `--resume`；
`gift_branch` 则保留 `--run-training`，另加 `--resume`。
执行图在恢复后重建，不保存设备内存地址。旧源代码产生的
训练日志不能通过修改身份校验来冒充当前实现的同次续算；已有模型参数文件
仍可正常读取。首次图构建计入训练 epoch 耗时，并非免费的训练计算。

The reduced-data branch command retains `--run-training` when adding `--resume`.
The clean/noisy reduced-data generator keeps its best-validation phase selection,
random-minibatch sampling and affine refit schedule. GIFT-Lite branches keep their
derivative/short/long selection and optimizer/loader resets. These protocols are
not replaced by the full-data trajectory-epoch schedule. M1 evaluation, published
weights and scientific settings are not changed by execution selection.

For the full-data branch trainer, a reviewed source/backend change uses `--continue-from PARENT_RUN` instead of
`--resume`, with a new output directory and `--transition-record RECORD.json`.
The record binds the exact parent checkpoint SHA256, both source maps and
backends, and hashed numerical-equality evidence. All other scientific identity
fields and the numerical runtime must match. The complete committed model,
optimizer, scheduler, RNG and history are transferred without a training update;
the parent remains untouched. Subsequent ordinary resume is strict again.
Continuation provenance is retained in the journal and terminal artifact. This
is not permission to substitute another model's trained weights or to accept an
unverified numerical change.

经数值对照验证的执行方式切换须使用独立续算目录和明确的转换记录，不能放宽
普通断点校验。已提交 epoch 的训练状态全部保留；未落盘的工作从最近检查点
重算。训练耗时按各段实际记录累计，不把暂停等待时间算作计算时间。

M2/M3/S1/S2/S3 accept `--gift-execution cuda-graph`. This only changes GIFT
execution, including the GIFT-Lite inference arm. External algorithms and M1
are unaffected. Inference retains all correction reports, leakage audits,
per-trajectory failure reasons and RHS evaluation counts. Its reported elapsed
time includes graph construction. Very short predictions can be slower because
the initial graph cost is not yet amortized.
Reported RHS counts describe the simulated RK4 steps. Graph warmups are extra
discarded evaluations, not additional simulated steps or optimizer updates;
their time is included in graph setup and total elapsed time.

## Measurements / 实测

Hardware: RTX A5500 Laptop GPU (16 GiB), Intel i9-12950HX, Windows,
PyTorch 2.10.0 + CUDA 12.6; 16 Torch CPU threads. Original float32/complex64
arithmetic and TF32 settings are retained. All GIFT training entry points disable
cuDNN benchmarking, select deterministic cuDNN algorithms and fix the cuBLAS
workspace before tensor work. Seeding alone is insufficient for reproducible
convolution backward operations. Full-data/generator determinism policies are
retained; the reduced-data branch uses this same explicit runtime policy.
Journals bind these settings, and numerical agreement across different hardware
or library versions is not guaranteed to be bitwise. Affine-fit statistics
remain float64/complex128 with the original chunk size and rank threshold.

The following measurements visit 60 observed training trajectories per repeat,
with three repeats and alternating backend order. Each timing includes host-to-
device copies, forward/backward, clipping, AdamW and scalar loss reporting.
It excludes initial graph setup, data preparation, validation and affine refits.
These are component benchmarks, **not completed 500-epoch training times**.

| Stage / 阶段 | Eager median / 中位秒数 | Graph median / 中位秒数 | Speedup / 加速比 |
| --- | ---: | ---: | ---: |
| Generator gradient training / 生成元梯度训练 | 0.0759 | 0.0273 | 2.78× |
| Branch derivative training / 支路导数训练 | 0.0575 | 0.0338 | 1.70× |
| Branch short recursion / 支路短递归训练 | 6.1704 | 1.3059 | 4.73× |
| Branch long recursion / 支路长递归训练 | 31.5045 | 6.4316 | 4.90× |

Graph setup, including two warmups, took approximately 0.23, 0.32, 1.42 and
6.30 seconds respectively for the measured batch shapes. Results depend on
hardware, thermal state, batch shape and runtime; they are not universal ratios.
Final model parameters and aggregate losses matched bitwise in all repeats.

An additional seven-trajectory, 50-step inference check retained complete
diagnostics. With correction enabled, total elapsed-time speedups including
graph setup were 2.35× at N64, 1.92× at N96 and 1.50× at N128. N64 used observed
training fields; the larger grids used explicitly resized timing/safety fixtures,
not cross-resolution accuracy data. Predictions, failure locations/reasons,
trigger counts, leakage measurements and all scientific report fields matched
the eager reference. Artificial failures were also checked: a failure-heavy N128
case was about 6% slower, illustrating why this backend remains optional.

Additional numerical stress tests covered 1,000 real trajectory inputs per
branch stage (63 derivative updates; 200 short/long updates), comparing
raw gradients, clipped gradients, model parameters and complete optimizer state
after each update. The paired stress harness also checked identical controlled
scheduler states; it is not a newly completed formal epoch. Two further branch
initialization seeds were checked on 21 trajectories each, including single-row
tail batches. All comparisons were bitwise equal. Fresh-process continuation was checked separately through the
actual training entry points on explicitly marked, short-budget fixtures,
including phase changes and short final batches. This does not replace a full
500 + 500 epoch accuracy run.

The independent reduced-data entry points were also checked with clean, 1% and
10% noise fixtures: affine refits, phase-best selection, optimizer/scheduler and
random states matched the eager reference exactly. Branch checks covered the
derivative, short-recursion and long-recursion stages, including their separate
loader and optimizer resets. Eager and graph runs use the same deterministic
CUDA runtime for these comparisons. Independent-process interruptions were
inserted only after committed boundaries; resumed numerical histories and
selected weights matched uninterrupted runs. These are explicitly short-budget
control-flow tests, not substitutes for reported scientific experiments.

Complete-budget, from-scratch checks on the stated environment reproduced every
packaged generator tensor bitwise: 12,000 updates for each of the clean/1%/10%
noise generators, and 500 trajectory epochs (31,500 updates, 501 affine fits)
for the full-data generator. The three reduced-data runs recorded 117.52, 121.38
and 126.28 seconds respectively for training, refits, validation and intermediate
checkpoint I/O, including graph setup but excluding initial preparation and
terminal export. The full-data run recorded 266.84 committed epoch seconds,
including graph setup and excluding initial preparation and checkpoint writes.
These single-run checks have different timing scopes and are not additional
rows in the repeated component benchmark above. Tensor equality does not imply
identical serialized files: timestamps, source records and run metadata differ.

All three reduced-data branches were also trained from scratch for the complete
5,834-update schedule. Their committed training-plus-validation times were
205.49, 233.10 and 221.20 seconds for seeds 20260820, 20260821 and 20260822.
These figures include graph setup and omit shared generator pretraining, initial
data preparation and checkpoint/export I/O. An independent second full-budget
run of seed 20260820 matched the first run's selected parameters, optimizer,
scheduler, RNG and numerical history bitwise, excluding elapsed-time fields.
The repeat is a reproducibility check, not a fourth statistical seed.

## What is optimized / 优化范围

- Cache only constant frequency grids, masks and table indices. Learned
  generator tables and branch coefficient fields are recomputed after updates.
- Capture forward/backward tensor work separately for each phase/batch shape.
  Gradient clipping, AdamW and scheduler steps remain ordinary PyTorch calls.
  A short final batch has its own graph: no padding or changed sample weighting.
- For full training batches (16 derivative trajectories or 5 rollout trajectories),
  pack differentiable spectral-weight layouts once per backward segment, build
  disjoint Fourier output bands by concatenation, and batch the four coefficient
  projections. Parameters keep their ordinary storage and checkpoint format.
  Packed views are rebuilt every segment and never reused across updates. Other
  batch shapes use the reference graph because layout changes can affect the
  last bits of gradients. Default prediction and validation do not use these
  training-only kernels.
- Use the same local-correction tensor core in training and audited inference.
  Training omits diagnostic reports it does not consume and accumulates safety
  flags on the device. A failed batch is rejected **before** the optimizer step.
- Capture the complete projected inference RK4 step, including its eight
  leakage measurements. Correction, failure isolation and output sampling retain
  their original behavior. Retain at most two active batch shapes; additional
  failure-reduced shapes use eager execution without padding trajectories.

Data verification/loading was about 35 seconds once at startup on the measured
system. Per-epoch sampling was about 0.05–0.07 seconds; derivative target/frozen
RHS preparation about 0.32 seconds; a 1,000-state affine refit about 0.06 seconds.
These are not the dominant long-recursion bottleneck. Hash verification, observed
derivative construction, affine-fit precision and checkpoint durability are not
weakened for speed. Increasing batch size, reducing RK4 stages, enabling lower
precision or rewriting external algorithms is deliberately outside this backend.

训练专用布局不减少 RK4 步数、支路调用或反向段数。长递归一批仍包含 50 个 RK4
时间步、200 次支路前向和 5 段反向。尾批不填充、不丢弃，使用原图执行路径。
布局副本由普通自动微分连接到原参数；不分离权重，不实现自定义反向算子。

## Reproduce timing / 重做计时

Use the same configured environment and explicitly selected model files:

```shell
python -m scripts.benchmark_gift --dataset ../GIFT-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --generator ../runs/gift_generator/model.pt --branch ../runs/gift_branch_20260820/model.pt --rows 60 --repeats 3 --output ../benchmarks/gift.json
```

This writes a new JSON receipt and performs disposable optimizer updates only.
It does not overwrite weights, resume formal training, or update experiment
tables. An explicitly supplied same-run training checkpoint may also be used
for disposable timing. Benchmark speed never substitutes for accuracy evidence.

For graph buffer/RNG tests, set `GIFT_RUN_CUDA_TESTS=1` and run
`python -m pytest tests/test_gift_acceleration.py -q`. Ordinary tests exercise
CPU fallback and reference behavior without requiring a GPU.

See the [PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs)
for fixed-address/shape requirements and the
[profiler documentation](https://docs.pytorch.org/docs/2.10/profiler.html) for
operator profiling. Numerical equality measured on one configuration does not
guarantee bitwise equality across every GPU, library version or future run.
