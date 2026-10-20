# GIFT execution efficiency / GIFT 计算效率

## W1/W2 配方与实测成本

[N300/76点协议](W1_W2_PROTOCOL.md) 的GIFT与GIFT-Lite训练采用eager执行，监督时距0.08/0.48秒、内部RK4 dt = 0.02、观测保存间隔0.04秒。N1000独立控制器的auto/cuda-graph策略和加速基准在下节单独说明。

| 方法 | 单模型训练进程（秒） | 分钟 |
| --- | ---: | ---: |
| FNO-2D | 5482.334 | 91.37 |
| FNO-3D | 6104.994 | 101.75 |
| GIFT N300，生成元与分支合计 | 10296.856 | 171.61 |
| ANIE，训练1029轮、MSE-best epoch29 | 21176.742 | 352.95 |
| U-Net N300，500轮 | 652.677 | 10.88 |
| U-NO N300，500轮 | 8332.408 | 138.87 |
| GIFT-Lite，生成元与分支合计 | 1911.591 | 31.86 |

七方法训练进程合计53957.603秒（14.9882小时）。时间包含各进程的数据准备、训练、内部validation、checkpoint写入和导出；GIFT两个组件已经计入其合计，不再加内部timer。U-NO/U-Net仅监督20/4步，其他方法使用各自原生loss和目标；N50的更新更少。因此该表描述固定配方成本，不是统一计算预算的架构效率排名。

| 测量范围 | 进程墙钟（秒） | 分钟 |
| --- | ---: | ---: |
| U-NO/U-Net/GIFT-Lite的两人口W1推理，以及七方法评分与作图 | 448.829 | 7.48 |
| N96/N128配对确认真值生成 | 774.489 | 12.91 |
| 六方法两人口W2原生推理、评分与图 | 2826.245 | 47.10 |

这些分项按声明的进程范围计时，不是全部W1/W2训练与评价的统一端到端时间。W2 N64指标来自对应W1预测的一秒前缀，不重复计预测时间；FNO-3D高网格完整输出75个未来token后评价前25点，所以其计算不限于25时间token。进程中的推理、验证和作图子计时不再重复相加。环境安装、来源获取及文档操作不计入模型训练时间。

成本来源为 `source-record:9b0778b4fa631783ad24`（SHA-256 `133fbb3022e0dbe3056d335360f0e78b03df3d990cbfb5a80232c67ec5ca6876`）与 `source-record:34daf08966d3f15edc14`（`d3607da57ec27cc6a51270060c5157d92694c5980d561e8ceb93de5477930787`）。ANIE训练成本来自 `source-record:0458d853302dcbaf7140`（SHA-256 `41aff903c363426402e2ebaa8e64acdc5358d95223c8be7421c55f3e3fa7d4b7`）。其他方法成本范围记录在原目录的 `FINAL_COST_CLOSURE.json`；CPU图像导出等操作另有 `CPU_EXECUTION_EVENTS.jsonl` 记录，不混入上述训练计时。机器为RTX A5500 Laptop GPU；浮点和随机算子不保证跨设备逐位一致。

## W3 无噪声50组参数识别实测成本

每种方法各完成50个参数case。下表为FINAL_COST_CLOSURE中的方法级worker墙钟合计及单槽均值；两机并行执行，累计worker时间不能解释为日历运行时长，也不是等计算预算效率排名。训练、回归、参数读出和辅助阶段分别记录；表中仅累计已完整计时的worker墙钟，中断尝试另报区间。

| 方法 | 50槽累计worker墙钟 | 单槽均值 |
|---|---:|---:|
| GIFT | 5,313.126 秒（1.48 小时） | 106.263 秒 |
| PDE-FIND | 12,552.202 秒（3.49 小时） | 251.044 秒 |
| PDE-FIND-KC | 160.025 秒（0.04 小时） | 3.200 秒 |
| PINN-SR | 409,616.955 秒（113.78 小时） | 8,192.339 秒 |
| PINN-SR-KC | 285,430.598 秒（79.29 小时） | 5,708.612 秒 |


两台主机worker墙钟合计：DQY_STATION 350,335.529秒，DQY_PC 362,737.377秒；跨机合计 713,072.906秒（198.08小时）。并行与调度使此数值不等于实验经历的日历时长。采样准备等辅助事件合计 278.298秒；这些准备事件不计入模型训练成本。

参数041的PINN-SR运行经历一次基础设施中断，随后从该槽自身检查点继续，未创建第二个科学槽、未重抽seed。中断尝试按证据计为 800.370–36924.215秒区间；具体区间与完整尝试清单见[成本闭合](../results/formal/W3_single_trajectory_20261010/summary/FINAL_COST_CLOSURE.json)。所有250个科学实例均完成；中断尝试只能报告上述区间，不能将它或包含它的总成本写成精确值。

锁定预算和真实计时边界见[W3协议](W3_PROTOCOL.md)、[逐例成本清单](../results/formal/W3_single_trajectory_20261010/summary/)与run审计。不同方法的目标、系数库、优化过程与实现环境不同，不能根据单一墙钟汇总宣称统一效率比较。


## Independent N1000 engine policy / 独立N1000执行策略

The N1000 GIFT entry points in `scripts.run_training` use the shared `auto` execution policy. Fresh
GPU training uses `cuda-graph` to replay the same tensor operations with less
CPU dispatch overhead; CPU uses `eager`. `--execution eager` selects the explicit
reference training path. Prediction remains eager by default. No extra package
is needed. CPU training uses the same model and declared budget with eager execution.

N1000协议的 `scripts.run_training` GIFT入口共用 `auto` 执行策略：GPU 新训练默认使用 `cuda-graph`，
CPU 使用 `eager`；`--execution eager` 可明确选择参考训练路径。预测默认路径不变。
优化不改变模型、训练数据量、epoch 数、批大小、
AdamW、学习率日程、损失、RK4 步长或递归截断位置，也不启用混合精度。

## Use / 使用

Full-data GIFT and reduced-data GIFT-Lite use the same training engine. Each
model still has its own command, scientific schedule and complete-state journal:

```shell
python -m scripts.run_training gift_generator --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m scripts.run_training gift_predictor --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --low-model ../runs/gift_generator/model.pt --seed 20260820 --output ../runs/gift_branch_20260820
python -m scripts.run_training gift_low --run-training --condition noise_000 --output ../runs/gift_lite_generator
python -m scripts.run_training gift_branch --run-training --low-model ../runs/gift_lite_generator/gift_main.pt --seed 20260820 --output ../runs/gift_lite_branch_20260820
```

Continue with `--resume` and the same output directory. The model, optimizer,
scheduler, RNG and progress are restored from the latest committed boundary.
Graphs are rebuilt; no CUDA graph or device address is serialized. Source and
environment records describe the run rather than prevent code maintenance.
Actual model, data and schedule compatibility is checked before continuing.

断点恢复使用原输出目录。`gift_low`、
`gift_generator` 和 `gift_predictor` 将 `--run-training` 替换为 `--resume`；
`gift_branch` 则保留 `--run-training`，另加 `--resume`。
执行图在恢复后重建，不保存设备内存地址。源码和环境变化不阻止兼容状态的恢复，
运行设置如实记录。首次图构建计入训练 epoch 耗时。

The reduced-data branch command retains `--run-training` when adding `--resume`.
The clean/noisy reduced-data generator keeps its best-validation phase selection,
random-minibatch sampling and affine refit schedule. GIFT-Lite branches keep their
derivative/short/long selection and optimizer/loader resets. These protocols are
not replaced by the full-data trajectory-epoch schedule. W3 evaluation, published
weights and scientific settings are not changed by execution selection.

Use `--continue-from PARENT_RUN` and a new output directory when intentionally
starting a separate continuation. It copies the committed model, optimizer,
scheduler, RNG and history without a training update. No transition certificate
is required. The parent remains unchanged. Uncommitted work restarts from the
last boundary; paused time is not counted as training time.

W1/W2/S1/S2/S3 accept `--gift-execution cuda-graph`. This only changes GIFT
execution, including the GIFT-Lite inference arm. External algorithms and W3
are unaffected. Inference retains per-trajectory failures, correction gates and
RHS evaluation counts; detailed leakage diagnostics are optional. Its reported elapsed
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

Independent short-budget continuation and complete-budget equality checks are engineering evidence for the execution backend. They do not replace the current W3 50-case noise-free measurements or contribute additional statistical training seeds. Detailed source versions and numerical check records are retained outside the repository.

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
python -m scripts.benchmark_gift --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --generator ../runs/gift_generator/model.pt --branch ../runs/gift_branch_20260820/model.pt --rows 60 --repeats 3 --output ../benchmarks/gift.json
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
