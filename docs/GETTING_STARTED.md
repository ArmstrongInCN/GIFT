# Getting started / 使用指南

This guide describes the candidate's supported paths. Consult
[validation status](REPRODUCIBILITY_STATUS.md) for what has actually passed.
本指南区分读取参照数据、检查点推理与从零训练，不将三者混为复现证据。

## Environment / 环境

The recorded main environment is 64-bit CPython 3.10, PyTorch 2.10.0+cu126,
CUDA runtime 12.6 and float32 observations. The commands below select that
PyTorch build explicitly for supported NVIDIA Windows/Linux systems; the
[official PyTorch 2.10 instructions](https://pytorch.org/get-started/previous-versions/#v2100)
list the cu126 package index. Use an official build appropriate for other
hardware/operating systems and validate it separately. Neither the same version
numbers nor a successful installation promises cross-device bitwise equality.
PINN-SR training requires its separate legacy environment; do not mix it into
this one. Recovered PINN coefficient readout needs NumPy, not TensorFlow/Torch/
SciPy or external clones; the module entry also imports h5py through its parent
package. PINN 训练使用独立旧环境；已训练系数快读无需该训练环境，见
[M1 quick-readout instructions](adapters/M1.md#9-已恢复训练终态的独立-pinn-快速读取).

Run installation and every `python -m ...` command below **from the root of the
cloned repository**. This candidate supports source-checkout execution: the
installed `gift` wheel contains the core library, not the `experiments`,
`training`, `adapters` or data-generation entry points and their source/assets.
Installing that wheel alone is not a complete experiment installation.
安装和下列模块命令均须在 clone 的仓库根目录执行；仅安装 `gift` wheel
不包含训练、实验、适配和数据生成入口，不等于获得完整运行工程。

```bash
# First ensure `python` is your 64-bit CPython 3.10 interpreter.
python -m venv .venv
# Activate the environment using your operating system's activation command.
python -m pip install --upgrade pip==26.0.1
python -m pip install "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

For the recorded build the last line should print `2.10.0+cu126 12.6`.
Install PyTorch **before** the requirements file: its `torch==2.10.0` constraint
accepts the installed `2.10.0+cu126` build; it does not by itself select a CUDA
index. Do not silently substitute a different build while claiming the same
numerical environment. `torchvision` and `torchaudio` are not used by this project.

On 2026-09-10, a fresh Windows AMD64 / CPython 3.10.19 venv with pip 26.0.1
successfully resolved all nine fixed requirements together with the explicit
cu126 build using `pip install --dry-run --ignore-installed --only-binary=:all:`:
32 distributions, no dependency conflict. The official index provided the
`torch-2.10.0+cu126-cp310-cp310-win_amd64.whl` and its separate metadata.
This was metadata-only resolution, not downloading/installing the training
packages or testing CUDA operation. Transitive dependencies are not all pinned
by the requirements file; resolution is not proof of an identical environment.
已完成上述 Windows/Python 3.10 固定依赖的元数据解析预检，未完成新环境整套安装
或 GPU 运行验收；其他设备应另行验证，不作逐位相同承诺。

Do not run training or inference inside an author's authoritative source folder.
无须访问作者电脑上的 `1_GIFT`，也不要向它写入缓存或输出。

## Inputs outside Git / 仓库外输入

Download and extract the separate dataset package once it is deposited.
No Zenodo DOI has been assigned in this candidate; no placeholder DOI is used.
The selected directory must directly contain `standard_ns_n64_full_spectrum.h5`,
`manifest.json` and `splits.json`.
数据包尚未取得 Zenodo DOI；待独立分发后解压，将 `GIFT_DATA_ROOT` 指向直接包含上述三个文件的目录。`GIFT_EXTERNAL_ROOT` 则指向仓库外固定上游 clone 的父目录，不能用最新分支替代锁定提交。

Linux/macOS:

```bash
export GIFT_DATA_ROOT="/absolute/path/to/gift-dataset"
export GIFT_EXTERNAL_ROOT="/absolute/path/to/external-checkouts"
```

PowerShell:

```powershell
$env:GIFT_DATA_ROOT = 'D:\datasets\gift-dataset'
$env:GIFT_EXTERNAL_ROOT = 'D:\external-checkouts'
```

The paths above are examples, not author's paths that must exist on your machine.
GIFT-only work does not require external baseline code. For comparisons, follow
the pinned [upstream guides](THIRD_PARTY_SOURCES.md); do not use the latest branch
as a substitute for the recorded revision.

## Three distinct workflows / 三条独立路径

1. **Reference tables / 查看参照表：** inspect `results/formal/*/summary/metrics.csv`
   and `EXPERIMENTS.md`. This performs no model inference or training.
2. **Checkpoint inference / 检查点推理：** run an experiment using the released
   trained weights in `artifacts/`. Each experiment uses a new output directory.
3. **From-scratch training / 从零训练：** train each desired model into its own
   output directory, then explicitly select those outputs for evaluation.
   A completed reference model is never an initialization for a fresh run.

For GIFT-only experiments, the independent modules are:
以下 S1–S3 只需 GIFT 权重与数据，不需要外部基线源码；各命令单独执行。

```bash
python -m experiments.formal.s1_high_frequency_branch.run --output runs/s1 --device cuda
python -m experiments.formal.s2_recursive_local_correction.run --output runs/s2 --device cuda
python -m experiments.formal.s3_seed_stability.run --output runs/s3 --device cuda
```

M2 additionally requires the pinned FNO, U-NO and U-Net sources and terminal
weights. U-NO is a separate asset whose download location is still pending in
`artifacts/CHECKPOINTS.json`; it is not included in a clean clone.
M2 需要 FNO、U-NO、U-Net 的固定源码及终点权重；U-NO 独立下载位置尚未提供，不能假定 clone 已带齐。

```bash
python -m experiments.formal.m2_recursive_prediction.run --output runs/m2 --device cuda
```

M3 uses only GIFT and FNO; it does not require U-NO or U-Net sources/weights.
M3 只需 GIFT 与 FNO，不受 U-NO 尚缺下载地址这一项阻断。

```bash
python -m experiments.formal.m3_cross_resolution.run --output runs/m3 --device cuda
```

Do not run these jobs simultaneously on a small GPU. They are separate commands,
not a mandatory all-in-one job. See the status document for completed and pending
acceptance, and the [checkpoint-inference evidence / 检查点推理证据](../validation/README.md)
for available portable comparison records. These records are not fresh training.
显存有限时不要并行启动这些任务；可分别执行，不要求一次跑完。随附新推理记录不等于从零训练验收。

To evaluate a fresh complete artifact layout, set `GIFT_CHECKPOINT_ROOT` to that
layout; it has the same subdirectories as `artifacts/`. For GIFT high-frequency
branches, individual paths may instead be selected with repeated
`--gift-model SEED=PATH` (all three formal seeds are required for these experiments).
评估自行训练的权重时，应显式选择该次输出；完整目录使用与 `artifacts/` 相同的布局。高频模型需提供三个种子的路径，并用 `--low-model PATH` 指定对应的冻结低频前置模型，不要无意混入默认参照权重。

## Resume / 断点续算

M2, M3 and S1–S3 save each completed numerical unit (method, seed, grid and/or
cohort) with checksums. Re-run the **same command and original output path** with
`--resume` after an interruption. Inputs, source, external implementations,
weights, batch sizes and numerical runtime must match. The currently unfinished
unit is recomputed; completed units are not inferred again. This is experiment
continuation, not loading a training checkpoint or borrowing another run's result.

```bash
python -m experiments.formal.m2_recursive_prediction.run --output runs/m2 --device cuda --resume
```

The first aggregation is under `runs/m2`; a resumed aggregation is under the
new `runs/m2/resumed_attempts/<id>` directory printed at startup. Earlier files
are never overwritten. Compare the `summary/metrics.csv` from that new directory.
`continuation.json` distinguishes recomputed and reused units. Do not delete or
edit `.resume` receipts while a job is running. Partial crash debris is retained,
not automatically cleaned. Only one process may own a given experiment run.

实验中断后，给原命令加 `--resume`，并保留原始 `--output` 路径。已完成单元
从本次运行的校验文件恢复；未完成单元重算。新汇总写入启动时打印的独立目录，
旧结果保留。小型控制测试与实际完整实验的验收范围分别见状态文档，不能由续算成功推定数值通过。

Independent training commands / 独立训练命令：

For GIFT low-generator training, set this 16-thread CPU reduction profile
**before starting Python**. On the checked host it reproduces the archived
pre-training normalization value; using two threads changes that float32 value.
This is part of the numerical profile, not a change in the model.

GIFT 低频训练需要在启动 Python 前设置线程配置；本机实测 2 与 16 线程会使
训练前的 float32 归一化值出现末位差异。恢复原训练时请勿改变线程配置。
完整训练及跨设备适用范围仍以复现状态中的实际验收为准。

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

PowerShell:

```powershell
$env:OMP_NUM_THREADS = '16'
$env:MKL_NUM_THREADS = '16'
```

```bash
python -m experiments.formal.train_low_generator --run-training --condition noise_000 --output runs/gift-low
python -m experiments.formal.train_gift_branches --run-training --seed 20260820 --low-model runs/gift-low/gift_main.pt --output runs/gift-high-20260820
python -m training.train_fno2d --run-training --output ../gift_runs/fno2d
python -m training.train_fno3d --run-training --output ../gift_runs/fno3d
python -m training.train_uno --run-training --output ../gift_runs/uno
python -m training.train_unet --run-training --output ../gift_runs/unet
```

These are independent jobs, not a script to launch all at once. Repeat the
high-branch command separately for seeds 20260821 and 20260822. For noise
experiments select `noise_001` or `noise_010` in the low-generator entry.
Baseline output directories must be outside the repository, data and external
source roots; the `../gift_runs/...` examples use a sibling directory, not the
repository's `runs/` directory. Keep that same external path when resuming.
See [baseline training details](adapters/TRAINING.md) before expensive runs,
and [data generation](DATA_GENERATION.md) for clean/noise/sampling commands.
以上是独立任务，不是同时启动的脚本。高频三个种子分别训练；噪声低频任务按条件另行运行。基线输出必须位于仓库、数据与外部源码目录之外，续算保持原输出路径。PINN 的 known/KC 独立训练与 open 门禁见 [M1 适配说明](adapters/M1.md)，不适用上述主环境训练命令；不要与无需训练环境的 PINN 系数快读混同。

Resume checkpoints are separate from inference-only terminal weights. A valid
resume includes the model, optimizer, scheduler, all random-number states,
data-order state, phase position, and best-model selection state. Resuming with
a different input hash, configuration, source revision or recorded numerical
profile is rejected. A crash may require repeating work since the last boundary.

Do not copy a terminal `.pt` into a checkpoint journal. Keep the original run
directory and use the individual trainer's `--resume` option when supported and
validated in the status document. Short continuous/resume tests have been
performed for each model type, but do not replace full-budget comparisons.
训练续算用法不同：低频 GIFT 将 `--run-training` 替换为 `--resume`；高频 GIFT 与四个基线保留 `--run-training` 再加 `--resume`。所有情况都保留同一输出目录与配置；不要将终点 `.pt` 改名放进 journal，也不要把小预算恢复测试当作完整训练证明。

If a baseline's complete final journal was saved but power failed before its
inference weight export finished, do not restart training. The independent
[terminal export recovery tool](CHECKPOINT_EXPORT_RECOVERY.md) can create a
new CPU-only inference artifact from an explicitly selected, SHA-bound complete
journal. It preserves uncertain provenance as unknown and never overwrites the
old file; recovery is not numerical acceptance.

若基线已保存完整终态 journal，却在导出推理权重前断电，无须重训。
使用[终态补导出工具](CHECKPOINT_EXPORT_RECOVERY.md)读取明确选择且核对 SHA 的
完整状态，另存新文件；原文件保留，缺失的来源信息不猜填，补导出成功也不等于实验数值通过。

## Reading validation results / 如何理解验收

The frozen scalar criterion used in the previous audit was
`abs(new - reference) <= 0.0005 + 0.05 * abs(reference)` for designated numeric
columns, with exact row keys and counts. It does not mean every prediction field
or checkpoint is byte-identical. Do not change thresholds, discard a divergent
trajectory, or enable a disabled ablation component to turn a failed run into a pass.

```bash
python scripts/compare_results.py M2 runs/m2/summary/metrics.csv --output runs/m2-comparison.json
```

This reports numeric agreement only. Input provenance and training freshness
must also be checked. It reports every mismatched row/value, not just an initial
sample; the default tolerance is not a tunable command-line option.
这条命令只比较表格，既不训练也不推理。它逐项报告全部超差，容差不能通过命令行放宽；数值通过还不等于输入来源或从零训练身份通过，更不等于预测场逐位相同。
