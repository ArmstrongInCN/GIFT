# Independent experiments / 独立实验运行

新机执行当前协议优先使用 [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md) 的 `scripts.reproduce_prediction`（准备→每方法训练→校验→W1/W2评价）和 [W3_REPRODUCTION.md](W3_REPRODUCTION.md)（全50输入gate→训练/测量→250槽聚合）。这两条流程直接消费正式数据包，不依赖旧运行目录；下文的本机来源登记保留为历史复核接口。

## W1/W2 运行接口与输入绑定

[N300/76点协议](W1_W2_PROTOCOL.md) 声明固定种子与超参数、每方法一个训练实例、单真实t5场。W1七方法预测75步，W2除ANIE外六方法在配对N64/N96/N128预测25步。主方法N300，GIFT-Lite使用有序名单前50条；U-NO/U-Net训练监督分别20/4步，GIFT采用0.08/0.48秒监督、内部dt0.02与保存dt0.04。

配置与模型/数据绑定目录为 `prediction_extension`；其 `EXTENSION_LOCK.json` 指向 `prediction_native/PROTOCOL_LOCK.json` 及各方法声明的训练或权重配置。模型路径与SHA见 [CHECKPOINTS.md](CHECKPOINTS.md)。当前W1评价入口读取七个完整模型及各自声明的绑定，不自动重训。

```powershell
python -m scripts.run_experiment W1 --help
python -m scripts.run_anie_coordinate_refresh --help
```

ANIE配置、模型选择和W1数值绑定于 `prediction_aligned`；其时空token排列见 [ANIE实现说明](ANIE_ADAPTATION.md)。其他方法与W2使用各自声明目录。

`scripts.run_anie_coordinate_refresh`提供显式坐标对齐的ANIE训练入口。`scripts.run_w1_single`的四方法参考队列和`scripts.run_w1_w2_single`的扩展队列使用各自机器JSON配方，前者的ANIE配置为`native_ns_single_v1`，后者复用该实例；它们不能代替当前七方法主表的`aligned_ns_single_v1`模型。逐模型训练必须读取对应模型的实际机器配置，而不能把通用协议文档当作另一队列的配方。

当前W1评价接口为`experiments.additional.w1_current_evaluate`，从仓库外来源登记加载七个固定模型，其中ANIE使用声明的时空对齐配置。输入参数明确指定真值文件、组、确认或探索人口及新输出目录；W2接口为`experiments.additional.w2_single_evaluate`，明确指定base/extension锁、模型与配对真值。锁内配置、数据及checkpoint身份必须一致。各入口的`--help`给出实际参数。输出目录须与输入数据及已有测量分开；恢复使用同一配置目录，保留已提交预测和进度，同一目录不能同时启动两个执行进程。科学失败保留方法和轨迹，不通过修改参数或重抽种子替换失败结果。

```shell
python -m scripts.run_experiment W1 --data-file <truth.h5> --group <cohort_group> --cohort confirmation --output ../runs/w1_confirmation --skip-plots
python -m scripts.run_experiment W2 --help
```

`scripts.run_experiment`默认使用当前N300单初态W1/W2协议。独立N1000、46帧历史接口仅在显式`--protocol reference`时使用；该入口用于对应参考协议，不能将其测量作为当前W1/W2主表。

确认IDs100000–100179与训练、验证独立，但两个评价人口的成绩已被查看，因此不是全部方法的盲测；探索IDs1040–1219另行报告。两人口分别产生完整raw预测、numeric指标和图像来源。结果目录为 [W1包](../results/formal/W1_flow_prediction_20261007/) 与 [W2包](../results/formal/W2_cross_resolution_20261007/)。原生网络/训练接口见 [实现说明](W1_W2_PROTOCOL.md)，W2网格与数据绑定见 [W2实现](W1_W2_PROTOCOL.md)，计时范围见 [PERFORMANCE.md](PERFORMANCE.md)。

代码和命令直接使用W1、W2、W3、S1、S2、S3，不另选编号空间。当前W1/W2计算使用上述N300单初态入口；S1–S3在各自独立N1000人口和模型上执行，不能互换权重或输入接口。

Use the prediction/GIFT environment and external roots from [SETUP.md](SETUP.md).
Each command below runs **one experiment**, reads selected trained weights and
writes to a new output directory. It does not train all models. The optional
`scripts.run_experiment` wrapper configures only its child process: two CPU
threads for prediction/GIFT inference, one for W3 regression/coefficient readout,
one visible GPU, and no global TF32 override. Place `--device cpu` before the
experiment name for an explicitly recorded CPU evaluation.

以下命令彼此独立。使用随项目提供的完整权重可跳过训练；使用自己从零训练得到的
权重时，显式指定相应路径。输出目录不能与数据、源码或已有结果重叠。

## W3: 50-case noise-free equation-parameter evaluation

The completed W3 campaign evaluates five methods on 50 fixed parameter combinations, each using its own single local ID0 trajectory. It has 250 method/case results and 750 parameter rows. Use the published numerical package to inspect or verify results; the package includes the frozen case list, per-slot outputs, supports, costs and file hashes, but intentionally contains no third-party algorithm source, training coordinator, or replacement single-case default model.

The run-specific dual-host coordinator and exact run/checkpoint state are retained outside the GIFT Git repository. To reproduce training, obtain each external algorithm from the cited upstream repository, install it outside the GIFT directory, apply the written [manual adaptation](EXTERNAL_MANUAL_ADAPTATION.md), and follow the fixed [W3 protocol](W3_PROTOCOL.md). Each training and parameter-readout command must bind the matching case dataset, true coefficients, and checkpoint.

The data/result package is [W3_single_trajectory_20261010](../results/formal/W3_single_trajectory_20261010/README.md). W3 Figure 4 contains three parameter-specific absolute percentage error distributions, each in a single full-range panel: [图4a：ν SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)（[PNG](../assets/figures/w3_parameter_nu_errors.png)） · [图4b：β SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_beta_errors.svg)（[PNG](../assets/figures/w3_parameter_beta_errors.png)） · [图4c：γ SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_gamma_errors.svg)（[PNG](../assets/figures/w3_parameter_gamma_errors.png)）. The [source manifest](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json) binds them to the saved parameter rows. Dots have a nominal diameter of 2.1 pt, and the seven lanes are separated by 2.7 pt for a compact point layer. Each real observation is rendered as 12 concentric, stroke-free, same-color transparent discs, using a radial Gaussian alpha profile for a soft edge (sigma 0.42 relative to the radius; nominal central alpha 0.95). This only softens each real marker; it does not apply KDE, smooth the data or add artificial observations. The shared renderer places all exact APE observations on seven fixed horizontal lanes over a Tukey boxplot: solid Q1–Q3 boxes, an extended median line passing through and beyond each box to about 2.1 times its height, and thin whiskers with short caps ending at the outermost actual observations within the inclusive fences [Q1 − 1.5 × IQR, Q3 + 1.5 × IQR], where IQR = Q3 − Q1. Observations outside the whiskers remain included in the same 50 points without extra duplicate flier symbols; means are not marked separately. Solid axes, bold black method names and saturated dark method colors for dots, boxes and whiskers are retained. Median lines use exactly the same color as the box, with a 1.6 pt width and their existing extended length. Horizontal ticks point upward into the plot, while their numerical labels stay below the axis. The bottom spine extends beyond the final tick by 3% of the axis span without changing measured x coordinates or tick positions. Plot text is limited to method names, ticks, the parameter-specific horizontal-axis label and a numerical Median value for each method; see [figure specifications](FIGURES.md).

To reproduce the figure on CPU, choose new output paths:

~~~shell
python -m experiments.formal.w3_equation_identification.plot_parameter_distributions --source results/formal/W3_single_trajectory_20261010/summary/parameter_results.csv --output-dir results/user/W3_parameters/figures --preview-dir results/user/W3_parameters/previews
~~~

This command renders existing numerical results and does not train a model or repeat parameter identification.

## S1 and S2: ablations

```shell
python -m scripts.run_experiment S1 --gift-regime GIFT --output ../runs/S1_GIFT
python -m scripts.run_experiment S1 --gift-regime GIFT-Lite --output ../runs/S1_GIFT_Lite
python -m scripts.combine_regime_results --experiment S1 --gift ../runs/S1_GIFT --gift-lite ../runs/S1_GIFT_Lite --output ../runs/S1
python -m scripts.run_experiment S2 --output ../runs/S2
```

S1 disables the high-frequency branch of each selected GIFT model; S2 compares
recursive local correction on/off. These commands do not start branch retraining. All accept `--low-model`
and three `--gift-model SEED=PATH` arguments for seeds20260820/20260821/20260822.
Use `--lite-low-model` and three `--lite-gift-model SEED=PATH` arguments for explicit
GIFT-Lite overrides. S1 retains independently resumable runs for each training regime;
the combination command verifies both completed reports and joins their tables
without retraining or rerunning inference. S2 evaluates both regimes on one
common independent test population. It preserves failures and
reports finite-only summaries separately; a finite-only mean is not a mean over
the full population. No training trajectories are used as supplementary tests.

## S3: initial-condition distribution

On the experiment machine, select the verified S3 observation mirror before this evaluation:

```powershell
$env:GIFT_DATA_ROOT = 'D:\2_GIFT\canonical_data'
```

S3 has a separate [initial-condition distribution protocol](S3_PROTOCOL.md),
Gaussian population and model directory. Use `scripts.run_experiment S3` to
evaluate those weights; its entry point accepts explicit model paths. Four-vortex weights
are not accepted as Gaussian-trained models. The S3 package contains both distributions'
N1000 numerical and fixed-trajectory evidence, independently of W1's N300 measurements.

## Recovery, numerical outputs and SVGs

W1/W2/S1/S2/S3 save checksummed numerical calls at method/seed/grid/cohort
boundaries. Add `--resume` to the identical command with its original output
directory. Completed calls are verified and reused; an interrupted call repeats.
Changed source, data, weights or numerical runtime are rejected. Resumption
prints the new aggregation directory inside the original run; use that directory
as the input to figure rendering. It does not overwrite the prior aggregation.

`summary/metrics.csv` contains summary values; `raw/` preserves arrays or fields.
Keep the completion record and file hashes together. W1/W2 can plot after numeric
completion when `--skip-plots` is omitted. To redraw the five report SVGs separately,
follow [FIGURES.md](FIGURES.md). A figure failure does not invalidate already
committed numerical outputs and does not require retraining.

Check one complete numerical output and, optionally, its separately rendered figures:

The compact results included in the repository are described in [RESULTS.md](RESULTS.md).
They have a separate integrity checker; they are not full experiment directories.

```shell
python -m scripts.verify_results --experiment W1 --result-dir ../runs/W1 --figures ../runs/figures_W1_curve --figures ../runs/figures_W1_keyframes
```

This read-only check verifies completion, file hashes, SVG structure and the
figure-to-report binding. It does not recompute metrics, inspect visual layout or
establish scientific accuracy; those checks remain separate.

数值结果、表格和 SVG 必须来自同一组已完成实验。保留非有限预测和失败记录，
不通过换检查点、删轨迹或改色限来维持预期的模型排名。
