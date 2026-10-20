# Setup and independent execution / 安装与独立运行

新机器的当前协议完整入口见 [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md)：从正式包直接准备N300/N50、七方法独立训练、模型校验和W1/W2评价。当前W3的完整准备/生成/训练/测量/聚合见 [W3_REPRODUCTION.md](W3_REPRODUCTION.md)。下文保留本机历史来源说明和独立辅助协议；新流程不需要复制作者的旧运行目录。

## W1/W2 配置与来源

单初态N300/76点实验配置见 [固定协议](W1_W2_PROTOCOL.md)。主方法使用300条训练、20条validation、t5–8每0.04秒76个真实节点和每方法一个固定种子模型；GIFT-Lite使用有序名单前50条。U-NO/U-Net分别监督20/4步，GIFT监督时距0.08/0.48秒、内部dt0.02、保存dt0.04。W1七方法，W2除ANIE外六方法；确认集此前四方法结果已被查看，非全方法盲测，探索人口另报。

本机配置目录为 `prediction_extension`，输入绑定保存在其 `EXTENSION_LOCK.json`。ANIE使用 `prediction_aligned` 中声明的坐标配置和训练权重。模型文件与SHA见 [CHECKPOINTS.md](CHECKPOINTS.md)，独立训练与评价参数见 [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md)。当前W1入口加载七个固定模型进行评价；安装环境或下载来源本身不执行训练。

```powershell
& 'D:\2_GIFT\anie-env\Scripts\python.exe' -m scripts.run_experiment W1 --help
```

公开U-NO/U-Net源码由 `external_sources.json` 固定提交与散列绑定，本机来源父目录为 `D:/2_GIFT/external_sources_v2`；ANIE官方来源和选定文件元数据记录在 `anie_sources.json`，本机 checkout 为 `D:/2_GIFT/research_sources/nie_anie`。这些官方源码与数据作为独立输入保持只读。外部算法运行接口保存在仓库外 `D:/2_GIFT/gift-external-runtime`，不随 GIFT Git 仓库分发；新使用者须按[手工适配说明](EXTERNAL_MANUAL_ADAPTATION.md)自行提供这些接口。W1/W2权重与完整预测存于声明的外部目录，Git中的 [W1包](../results/formal/W1_flow_prediction_20261007/) 和 [W2包](../results/formal/W2_cross_resolution_20261007/) 提供汇总、图像及选定数值证据，不提供可恢复训练状态。

下方逐模型训练命令采用独立N1000/46帧基线输入协议，以及W3/GIFT-Lite的专门数据配置。它们与N300单初态模型的张量接口和预算分别声明，不互换权重。

For W1/W2/W3/S1/S2/S3 evaluation commands and their independent recovery rules,
see [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md).

Run commands from the project root. Sibling directories are convenient for the
dataset and external repositories. External checkouts, adapted code and runtime
interfaces must be outside the GIFT repository folder, even when ignored by Git.
Keep downloaded data and upstream code out of commits. Installing dependencies is
separate from training. The commands below do not run all experiments at once.

## Python environments

The tested prediction/GIFT stack uses Python 3.10.19, PyTorch 2.10.0 with CUDA
12.6, NumPy 2.2.6 and the versions in `requirements-reproduction.txt`.

```shell
python -m venv ../gift-env
# Activate this environment using your shell's activation command.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .
```

PINN uses a **separate Python 3.8.20 environment**. Its native Windows GPU stack
was exercised with CUDA Toolkit 11.2.2, cuDNN 8.1.0.77 and the packages in
`requirements-pinn.txt`. Do not install the PyTorch requirements into this environment.
The project's PINN modules run from the source root; the Python-3.10 GIFT package
does not need to be installed here.

```shell
conda create -n gift-pinn -c conda-forge python=3.8.20 cudatoolkit=11.2.2 cudnn=8.1.0.77
conda activate gift-pinn
python -m pip install -r requirements-pinn.txt
```

当前W3实测GIFT/读出环境为Python3.12.14、PyTorch2.10.0+cu126、NumPy1.26.4和h5py3.16.0；PINN训练使用上述独立Python3.8.20/TensorFlow2.10.0环境。两套模型训练环境分别记录于各自权重来源，不混装。当前验证硬件为 NVIDIA RTX A5500 Laptop GPU（16 GB）。
小规模运行和断点续算已验证；完整实验的完成情况以对应结果记录为准。
不同硬件、驱动和数学库可能导致浮点差异，不能承诺跨设备逐比特一致。

## S3本机数据与权重身份

本机当前数据包为 `D:/GIFT/GIFT-dataset`，S3 使用其中的 `s3_gaussian/`。将 `GIFT_DATA_ROOT` 指向包根，代码按 `dataset.json` 的角色表定位数据。历史运行记录中的 `D:/2_GIFT/canonical_data` 保留为原来源路径。

```powershell
$env:GIFT_DATA_ROOT = 'D:\GIFT\GIFT-dataset'
python -m scripts.run_experiment S3 --output ../runs/S3_recheck
```

数据文件及模型使用一致的`gaussian_s3`身份，完整观测数组和模型训练状态与原测量输入逐位一致；当前文件SHA见权重清单，原测量来源SHA保留在原始证据中。这种元数据身份镜像不是重新生成或重新训练。新使用者可按[S3协议](S3_PROTOCOL.md)自行生成同一协议数据，或准备已核验的S3数据副本。

## Data and upstream sources

The current W3 experiment uses 50 noise-free parameter cases, each with one
local ID0 trajectory and no external validation trajectory. The reorganized
package contains them under `identification/w3/parameter_000` through
`parameter_049`, with each case's data receipt, truth parameters and sampling
files. GIFT uses the fixed terminal checkpoint. See
[W3_PROTOCOL.md](W3_PROTOCOL.md) and the
[current numerical package](../results/formal/W3_single_trajectory_20261010/README.md).
The case-specific models and complete training states remain in the corresponding
run's `training/parameter_*/` directories outside the Git repository. Their
original input directory, `D:/2_GIFT/runs/w3_single_trajectory_20261010/data`,
remains the historical source. W3 job entry points accept `--dataset-root` for
the packaged inputs; `--root` selects a separate run directory with the required
protocol/source locks, master gate and training states. Never use the data
package as a writable run directory. Inspect the interface with
`python -m scripts.run_w3_parameter_job --help`.

The existing dataset record is *GIFT Navier–Stokes input data*, ScienceDB,
DOI `10.57760/sciencedb.013lo` (CC BY 4.0). The 2026-10-20 reorganization is a
local package and has not been uploaded as an updated ScienceDB version. Its
layout and scope are described in [the package README](data_package/README.md):
current W1/W2 N300/N50 slices, separate N1000 auxiliary inputs, 50 W3 cases,
S3 Gaussian inputs and numerical source data for six formal experiments. It
contains no model weights or full prediction archive. The fields can also be
regenerated on your own device from the recorded
initial-condition parameters using the commands in [DATA_GENERATION.md](DATA_GENERATION.md),
so downloading this package is not required.

Point `GIFT_DATA_ROOT` at the package root. On the experiment machine:

```powershell
$env:GIFT_DATA_ROOT = 'D:\GIFT\GIFT-dataset'
$env:GIFT_EXTERNAL_ROOT = [IO.Path]::GetFullPath('../gift-external')
$env:GIFT_ANIE_SOURCE_ROOT = [IO.Path]::GetFullPath('../gift-external/anie')
$env:GIFT_EXTERNAL_ADAPTER_ROOT = [IO.Path]::GetFullPath('../gift-external-runtime')
python -m scripts.verify_data --root $env:GIFT_DATA_ROOT
python -m scripts.prepare_external --root $env:GIFT_EXTERNAL_ROOT
python -m scripts.prepare_external --root $env:GIFT_EXTERNAL_ROOT --verify-only
```

In bash, use `export GIFT_DATA_ROOT=/absolute/path/GIFT-dataset`,
`export GIFT_EXTERNAL_ROOT=/absolute/path/gift-external`,
`export GIFT_ANIE_SOURCE_ROOT=/absolute/path/gift-external/anie` and
`export GIFT_EXTERNAL_ADAPTER_ROOT=/absolute/path/gift-external-runtime`. Pass the
upstream root to the same download/verification command. Packages without
`dataset.json` retain their legacy path handling; explicit HDF5 arguments must
use the layout actually present. `--source fno` (or `uno`, `unet`, `pde_find`, `pinn_sr`,
`tensorflow115`) limits preparation to one upstream repository.

The helper fetches the commit in `external_sources.json` into a new directory,
with newline conversion disabled, and verifies source hashes. **Existing
directories are only checked, never modified.** A failed or mismatched existing
checkout is retained; select a fresh external directory to retry. Manual clones
are also supported if the same directory names, commits and file bytes match.

The helper only downloads and verifies sources; it does not adapt algorithms.
ANIE is downloaded separately from the repository and commit in `anie_sources.json`.
Users must perform the written operations in [EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md)
outside GIFT and prepare the stated external interfaces before baseline training
or inference. `GIFT_EXTERNAL_ADAPTER_ROOT` points to a directory containing an
`adapters/` package; its default is the sibling `gift-external-runtime/`. The
bridge rejects locations within the GIFT repository. If the interface is absent,
external-baseline entry points stop with a setup error rather than silently
substituting another implementation. GIFT-only computation and published result
reading do not acquire external algorithms.

On the experiment machine, the previously used implementations and their
executable adaptation registry are retained in `D:/2_GIFT/gift-external-runtime`,
outside GIFT; the official checkouts are unchanged. This preserves the existing
local computation while removing implementations from the distributable repository.
This local dependency is not supplied by a fresh GIFT clone. The TensorFlow checkout
supplies only two optimizer modules at runtime; importing the full source tree is
not required. Upstream software remains subject to its own terms; a source hash
or public download URL is not a licence grant.

For the existing experiment machine, select the preserved external runtime as follows:

```powershell
$env:GIFT_EXTERNAL_ADAPTER_ROOT = 'D:\2_GIFT\gift-external-runtime'
$env:GIFT_EXTERNAL_ROOT = 'D:\2_GIFT\external_sources_v2'
$env:GIFT_ANIE_SOURCE_ROOT = 'D:\2_GIFT\research_sources\nie_anie'
```

These paths select code dependencies only. They do not modify the declared
training data, weights, seeds, losses, schedules or evaluation populations.

## Independent N1000 prediction commands and W3 training

The optional launcher starts one child process, configures only that child's
environment, and waits for it. It creates no scheduled task and opens no window.
All arguments after the model name go to its independent training entry point.
Place launcher options (`--python`, `--device`) **before** the model name.
The explicit input paths below use this machine's `D:/GIFT/GIFT-dataset`;
replace that root when running on another machine.

```shell
python -m scripts.run_training fno2d --run-training --data-profile canonical --output ../runs/fno2d
python -m scripts.run_training fno3d --run-training --data-profile canonical --output ../runs/fno3d
python -m scripts.run_training uno --run-training --data-profile canonical --output ../runs/uno
python -m scripts.run_training unet --run-training --data-profile canonical --output ../runs/unet
python -m scripts.run_training gift_generator --run-training --dataset D:/GIFT/GIFT-dataset/prediction/four_vortex/train1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m scripts.run_training gift_predictor --run-training --dataset D:/GIFT/GIFT-dataset/prediction/four_vortex/train1000_n64_t0_t10_dt0p02.h5 --seed 20260820 --low-model ../runs/gift_generator/model.pt --output ../runs/gift_predictor_20260820
```

Repeat `gift_predictor` independently for seeds `20260821` and `20260822`, each
in a new output directory. Each exports `model.pt`. The generator and every
branch each train for 500 trajectory epochs; see [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md).

For reduced-data GIFT-Lite (50 training trajectories):

```shell
python -m scripts.run_training gift_low --run-training --condition noise_000 --output ../runs/gift_low
python -m scripts.run_training gift_branch --run-training --seed 20260820 --low-model ../runs/gift_low/gift_main.pt --output ../runs/gift_branch
```

With the new package selected, `noise_000` here resolves to the canonical
four-vortex `standard_n64.h5` and its auxiliary 50-trajectory prediction
protocol. It does not select the current W3 50-case experiment. Historical
`noise_001`/`noise_005` commands require a separate legacy or generated-input
collection; those noisy inputs are absent from the current package.

W3 uses the 50-case manifest and the corresponding case-specific inputs and models. The numerical package is [W3_single_trajectory_20261010](../results/formal/W3_single_trajectory_20261010/README.md); full training requires obtaining and adapting the external algorithms outside this project according to [W3_PROTOCOL.md](W3_PROTOCOL.md).


Torch intra-op thread counts are 16 for FNO and GIFT, and 1 for U-NO and U-Net;

Torch intra-op thread counts are 16 for FNO and GIFT, and 1 for U-NO and U-Net;
PINN uses one TensorFlow thread. U-NO's recorded launch sets `OMP_NUM_THREADS=24`,
leaves MKL/OpenBLAS/NumExpr thread overrides unset, then explicitly sets Torch
intra-op to 1 in `training.train_uno`. The environment value is therefore not a
claim that Torch training uses 24 threads. U-Net's suggested thread settings are defined in
`training/unet_environment.py`; direct module invocation is also supported.
PINN uses float32 with TF32 disabled. U-Net uses cuDNN TF32 and disables matmul
TF32. These are recorded numerical environments, not changes to the upstream
network. `--device cpu` before the model selects a CPU runtime. Full budgets and
custom runs may use either supported device; CPU executes eagerly. Environment
versions are recorded, and compatible checkpoints can be resumed after code or
environment maintenance. Cross-device floating-point values may differ.
FNO leaves `CUBLAS_WORKSPACE_CONFIG` unset; all GIFT training profiles set
`:4096:8` and deterministic cuDNN kernel selection. The launcher sets the workspace
before scientific imports, and GIFT trainers set kernel flags before tensor work.

## W3 numerical package

The current W3 release is a verified numerical package with 50 parameter cases, five methods and 750 parameter rows. Its manifest binds the case design, slot outputs, coefficient supports, costs and SHA256 identities. See [the W3 package README](../results/formal/W3_single_trajectory_20261010/README.md) and [results protocol](W3_PROTOCOL.md).

Each W3 readout is bound to its own case truth, input receipt and fitted model. [图4a：ν SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)（[PNG](../assets/figures/w3_parameter_nu_errors.png)） · [图4b：β SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_beta_errors.svg)（[PNG](../assets/figures/w3_parameter_beta_errors.png)） · [图4c：γ SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_gamma_errors.svg)（[PNG](../assets/figures/w3_parameter_gamma_errors.png)） show absolute percentage errors for ν, β and γ separately, using 100 × |estimate − truth| / |truth| (%). Each method retains all 50 observations per parameter. Horizontal point coordinates retain the exact individual APE values, without jitter, rounding or binning. Dots are deterministically arranged on seven equally spaced horizontal lanes, preferring an available central lane; dense values may still overlap. Lane position has no quantitative meaning. Dots have a nominal diameter of 2.1 pt, and the seven lanes are separated by 2.7 pt for a compact point layer. Each real observation is rendered as 12 concentric, stroke-free, same-color transparent discs, using a radial Gaussian alpha profile for a soft edge (sigma 0.42 relative to the radius; nominal central alpha 0.95). This only softens each real marker; it does not apply KDE, smooth the data or add artificial observations. The Tukey boxplot shows Q1–Q3 in each solid box, with an extended median line passing through and beyond the box and the same Median value at the right of the row. IQR = Q3 − Q1; thin whiskers and short caps end at the smallest and largest actual observations within the inclusive fences [Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]. All 50 actual observations, including those outside the whiskers, remain visible without additional duplicate flier symbols. Means are not marked separately. The plot shows an empirical distribution across parameter cases, not confidence intervals or variation across training seeds. Each figure contains one full-range panel with a common linear axis starting at zero for all five methods: 0–300% for ν and β, and 0–110% for γ. All finite values are retained. Method names are bold black; dots, solid boxes and whiskers use saturated dark method colors. Each median line extends through and beyond the box, to about 2.1 times its height, with a 1.6 pt width and exactly the same color as the box. Horizontal ticks point upward into the plot, their numerical labels remain below the axis, and the bottom spine extends beyond the final tick by 3% of the axis span without changing measured x coordinates or tick positions. Axes are solid black, whiskers thin and caps short. Inside each plot, text is limited to method names, coordinate ticks, the parameter-specific horizontal-axis label and each method's numerical Median value. Main tables use the same APE definition. File and data bindings are recorded in the [figure manifest](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json). Third-party source code and the dual-compute run controller are not included in this repository; download upstream implementations and adapt them outside the GIFT tree as described in [manual adaptation](EXTERNAL_MANUAL_ADAPTATION.md).

Source-level checks and small native continuation tests are documented in [VALIDATION.md](VALIDATION.md), separately from full scientific experiments.
