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

In M2, U-NO has the lowest mean error at t=5.5; GIFT has the lowest mean error
at the five reported times from t=6 to t=8. All five methods retain 200/200
finite trajectories. In M1, the open PINN-SR library retains 90/90 nonzero
terms at all three noise levels: its three-coefficient readout is not successful
sparse-structure recovery. L-BFGS callback counts do not establish convergence.

S2 without local correction includes a non-finite trajectory and a large finite
error amplification. These cases are retained. Such unstable trajectories can
amplify small changes in learned weights; small average seed variation for the
corrected model is not a general numerical stability guarantee.

The experiment metadata identifies trajectories 1000–1019 as used for GIFT
architecture development. They are included in the reported 1000–1199 cohort,
although not used for gradient training or validation-based checkpoint selection.
Consequently, the 200-trajectory cohort is not wholly untouched by method
development. The additional 1100–1199 summaries are a 100-trajectory subset of
that cohort, not another independent 200-trajectory sample. S2's 1200–1399 cohort
is held out from GIFT training, but overlaps the prediction baselines' training
set; it is used only for the GIFT correction ablation.

运行元数据将 1000–1019 标为 GIFT 架构开发轨迹。主表保留完整的 200 条评价结果，
不将“未参与梯度训练”扩大表述为“从未用于方法开发”。这些边界限制了结果的解释，
不能由重新初始化模型或增加训练轮数消除。
