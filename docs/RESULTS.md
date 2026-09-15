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

The prediction tables and SVGs currently contain completed GIFT-Lite models
(50 training trajectories) and the completed baselines. Full-data GIFT uses
1,000 training trajectories and 500 epochs for EACH of its two stages; its
generator is complete, but branch training and prediction evaluation remain
pending. No intermediate full-data checkpoint is represented as a final result.
GIFT-Lite also retains its own training schedule and validation-based selection;
it is not a data-only ablation of the full-data training protocol.

Prediction IDs are 0–999 for training, 1000–1039 for validation, and 1040–1219
for the independent test set. GIFT-Lite uses training IDs 0–49 and full-interval
validation IDs 1000–1019. The same initial condition has the same ID across
resolutions. Every test trajectory is retained, regardless of its error or
numerical stability. M1 keeps its separate local IDs and evaluation protocol.

The included prediction summaries were recomputed from completed per-trajectory
outputs on these 180 test trajectories; no new model inference was performed
for that aggregation. The manifests bind the measured source outputs, selection
rule and aggregation code. Independent experiment commands recompute predictions
with the supplied weights and the same fixed test set.

In M2, U-NO has the lowest mean error at t=5.5; GIFT-Lite has the lowest mean
error at the five reported times from t=6 to t=8. All five completed methods
retain 180/180 finite trajectories. S2 uses only this one independent test set.
Without local correction, trajectory 1062 becomes non-finite at t=6.74 for all
three seeds; the later finite-only means use 179 rather than 180 trajectories
and cannot by themselves establish a precision advantage. Small average seed
variation is not a general numerical stability guarantee.

In M1, the open PINN-SR library retains 90/90 nonzero terms at all three noise
levels: its three-coefficient readout is not successful sparse-structure recovery.
L-BFGS callback counts do not establish convergence.

当前预测结果只包含已完成的 GIFT-Lite 和基线，统一统计独立测试集 1040–1219。
这些统计由已完成的逐轨迹预测重新汇总，不称为一次新的推理或训练；完整数据 GIFT
须待训练和评价完成后加入。S2 只保留这一组测试，失稳轨迹及有限样本数均如实记录。
