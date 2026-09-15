# Reading results / 阅读实验结果

There are three distinct ways to inspect the experiments:

1. Read the measured tables and SVGs included under `results/formal/`.
2. Run one experiment with the supplied trained weights and the external data
   package to recompute predictions and statistics, without training.
3. Train each model independently, then run the experiments with those weights.

项目内的表格、图像是已完成实验的数值记录。读取它们不等于重新进行推理，更不等于
从零训练。第二种方式用于快速验证；第三种方式验证完整训练与评价流程。各条命令
以及断点续算边界见 [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md) 和
[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)。

## Included files

Each compact result directory contains `summary/metrics.csv`, a JSON summary
when available, and `published.json`. The manifest records the source measurement,
execution settings, runtime, source hashes and data/model bindings, plus exact
hashes of every included file. Figure directories also include their own
`figure_manifest.json` and the small CSV/NPZ arrays used in the selected panels.
These arrays are derived figure evidence, not a training dataset.

Full prediction HDF5 archives, per-call execution journals and private machine
paths are not included. Run the corresponding experiment to generate a complete
output directory. The compact package is not a resumable experiment directory
and cannot be supplied in place of a full run to the figure-rendering commands.

GIFT exports store portable metadata. Their serialized file hashes can therefore
differ from the serialization used during measurement while the model tensors,
scaling and computational metadata remain identical. The input bindings state
which relation was verified; they do not equate metadata-only changes with a
new training run or claim that all serialized bytes are identical.

## Read-only integrity checks

For an included compact package:

```shell
python -m scripts.verify_published_results --experiment M3 --result-dir results/formal/M3_cross_resolution
```

For a full output generated on your own device:

```shell
python -m scripts.verify_results --experiment M3 --result-dir ../runs/M3
```

Both commands check file integrity, not numerical reproduction. Recompute
predictions using the experiment commands and compare the resulting metrics.
Hardware, numerical libraries and nondeterministic upstream GPU operations can
affect floating-point values; no cross-device bitwise equality is promised.
Non-finite predictions and large finite errors are retained and reported, not
removed to improve averages. In particular, S2 finite-only summaries must be
read together with their finite trajectory counts.

## Evaluation boundary

The prediction tables and SVGs contain full-data GIFT, GIFT-Lite and the
applicable baselines. Full-data GIFT uses the same 1,000 training trajectories
as the baselines, with 500 epochs for EACH of its generator and branch stages.
These are terminal weights; the 500 + 500 budget is reported separately and is
not a 500-epoch total or a compute-matched comparison.
GIFT-Lite uses 50 trajectories, its own training schedule and validation-based selection;
it is not a data-only ablation of the full-data training protocol.

Prediction IDs are 0–999 for training, 1000–1039 for validation, and 1040–1219
for the independent test set. GIFT-Lite uses training IDs 0–49 and full-interval
validation IDs 1000–1019. The same initial condition has the same ID across
resolutions. Every test trajectory is retained, regardless of its error or
numerical stability. M1 keeps its separate local IDs and evaluation protocol.

Prediction measurements are produced by independent experiment runs on these
180 test trajectories using the supplied weights. The compact-package assembly
only copies verified summaries and figures; it does not run additional inference
or training. Its manifests bind the native measurement and execution-identity
hashes, model/data inputs, executed source files, runtime and external sources.
S1 and S3 retain separate full-data/Lite runs, joined by an aggregation-only command.

In M2, full-data GIFT has the lowest mean error at all six reported positive
lead times. Its t=8 mean is 0.036488, compared with 0.088190 for GIFT-Lite.
All six methods retain 180/180 finite trajectories. Both GIFT regimes transfer
directly to the tested N96/N128 grids without target-grid training.

S1 shows lower full-field and Q21 state prediction errors with the branch
enabled. However, the instantaneous Q21 derivative mean is worse than the
branch-disabled value and is higher for full-data GIFT than for GIFT-Lite;
the long-tailed per-state ratios must be read with their quantiles.

S2 uses only this one independent test set. Full-data GIFT has no correction
triggers and remains finite with correction both enabled and disabled, so this
test shows no added stability benefit for that regime. Without local correction,
GIFT-Lite trajectory 1062 becomes non-finite at t=6.74 for all three seeds; the
later finite-only means use 179 rather than 180 trajectories and cannot by
themselves establish a precision advantage. Small average seed variation is
not a general numerical stability guarantee.

In M1, the open PINN-SR library retains 90/90 nonzero terms at all three noise
levels: its three-coefficient readout is not successful sparse-structure recovery.
L-BFGS callback counts do not establish convergence.

预测结果同时展示 GIFT、GIFT-Lite 和相应基线，统一统计独立测试集 1040–1219。
实测推理由各实验独立完成，精简发布包的整理不另行训练或推理。S2 仅使用这组测试：
修正在 GIFT-Lite 上避免一条轨迹失稳，但全数据 GIFT 未触发修正，不能把保护效果
概括到两个训练设置。瞬时 Q21 方程右端的长尾误差也完整保留，不只报告有利指标。
