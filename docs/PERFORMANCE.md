# GIFT execution efficiency / GIFT 计算效率

GIFT provides an optional `cuda-graph` execution backend. It replays the same
tensor operations with less CPU dispatch overhead. `eager` remains the default
reference backend; CPU execution remains available. No extra package is needed.

GIFT 提供可选的 `cuda-graph` 执行后端，以减少 CPU 逐个调度小算子的开销。
默认仍为 `eager` 参考路径。优化不改变模型、训练数据量、epoch 数、批大小、
AdamW、学习率日程、损失、RK4 步长或递归截断位置，也不启用混合精度。

## Use / 使用

Add `--execution cuda-graph` to either independent full-data training command:

```shell
python -m training.train_gift_generator --run-training --execution cuda-graph --dataset ../GIFT-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m training.train_gift_predictor --run-training --execution cuda-graph --dataset ../GIFT-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --low-model ../runs/gift_generator/model.pt --seed 20260820 --output ../runs/gift_branch_20260820
```

Continue with `--resume`, the same output directory, source, runtime and execution
backend. Graphs are rebuilt from the saved model, optimizer, scheduler and RNG
state; the checkpoint never depends on a serialized CUDA graph. Changing source
or switching backends is **not** silently accepted as same-run continuation.
Existing parameter artifacts remain usable for inference.

断点恢复时保持输出目录、源代码、运行环境及执行后端一致，将 `--run-training`
替换为 `--resume`。执行图在恢复后重建，不保存设备内存地址。旧源代码产生的
训练日志不能通过修改身份校验来冒充当前实现的同次续算；已有模型参数文件
仍可正常读取。首次图构建计入训练 epoch 耗时，并非免费的训练计算。

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
arithmetic and TF32/determinism settings are retained. Affine-fit statistics
remain float64/complex128 with the original chunk size and rank threshold.

The following measurements visit 60 observed training trajectories per repeat,
with three repeats and alternating backend order. Each timing includes host-to-
device copies, forward/backward, clipping, AdamW and scalar loss reporting.
It excludes initial graph setup, data preparation, validation and affine refits.
These are component benchmarks, **not completed 500-epoch training times**.

| Stage / 阶段 | Eager median / 中位秒数 | Graph median / 中位秒数 | Speedup / 加速比 |
| --- | ---: | ---: | ---: |
| Generator gradient training / 生成元梯度训练 | 0.0955 | 0.0283 | 3.38× |
| Branch derivative training / 支路导数训练 | 0.0580 | 0.0341 | 1.70× |
| Branch short recursion / 支路短递归训练 | 6.2972 | 1.4037 | 4.49× |
| Branch long recursion / 支路长递归训练 | 31.1887 | 6.9506 | 4.49× |

Graph setup, including two warmups, took approximately 0.24, 0.31, 1.65 and
7.20 seconds respectively for the measured batch shapes. Results depend on
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
training stage (63 generator/derivative updates; 200 short/long updates), comparing
raw gradients, clipped gradients, model parameters and complete optimizer state
after each update. The paired stress harness also checked identical controlled
scheduler states; it is not a newly completed formal epoch. Two further branch
initialization seeds were checked on 20 trajectories each. All comparisons were
bitwise equal. Fresh-process continuation was checked separately through the
actual training entry points on explicitly marked, short-budget fixtures,
including phase changes and short final batches. This does not replace a full
500 + 500 epoch accuracy run.

## What is optimized / 优化范围

- Cache only constant frequency grids, masks and table indices. Learned
  generator tables and branch coefficient fields are recomputed after updates.
- Capture forward/backward tensor work separately for each phase/batch shape.
  Gradient clipping, AdamW and scheduler steps remain ordinary PyTorch calls.
  A short final batch has its own graph: no padding or changed sample weighting.
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
