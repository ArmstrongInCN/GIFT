# Setup and independent execution / 安装与独立运行

For M1/M2/M3/S1/S2/S3 evaluation commands and their independent recovery rules,
see [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md).

Run commands from the project root. Keep the dataset and external repositories
in sibling directories, outside this Git repository. Installing dependencies is
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

这两套环境不要混装。当前验证硬件为 NVIDIA RTX A5500 Laptop GPU（16 GB）。
小规模运行和断点续算已验证；完整实验的完成情况以对应结果记录为准。
不同硬件、驱动和数学库可能导致浮点差异，不能承诺跨设备逐比特一致。

## Data and upstream sources

Download the separate data package. In PowerShell, for example:

```powershell
$env:GIFT_DATA_ROOT = (Resolve-Path '../gift-data').Path
$env:GIFT_EXTERNAL_ROOT = [IO.Path]::GetFullPath('../gift-external')
python -m scripts.verify_data --root $env:GIFT_DATA_ROOT
python -m scripts.prepare_external --root $env:GIFT_EXTERNAL_ROOT
python -m scripts.prepare_external --root $env:GIFT_EXTERNAL_ROOT --verify-only
```

In bash, use `export GIFT_DATA_ROOT=/absolute/path/gift-data` and
`export GIFT_EXTERNAL_ROOT=/absolute/path/gift-external`, then pass that root to
the same Python command. `--source fno` (or `uno`, `unet`, `pde_find`, `pinn_sr`,
`tensorflow115`) limits preparation to one upstream repository.

The helper fetches the commit in `external_sources.json` into a new directory,
with newline conversion disabled, and verifies source hashes. **Existing
directories are only checked, never modified.** A failed or mismatched existing
checkout is retained; select a fresh external directory to retry. Manual clones
are also supported if the same directory names, commits and file bytes match.

No manual editing of downloaded code is required. The project applies the small
in-memory operations listed in [EXTERNAL_ADAPTATIONS.md](EXTERNAL_ADAPTATIONS.md).
The TensorFlow checkout supplies only two optimizer modules at runtime; importing
the full TensorFlow source tree is not required. Upstream software remains subject
to its own terms; a source hash or public download URL is not a licence grant.

## One model per command

The optional launcher starts one child process, configures only that child's
environment, and waits for it. It creates no scheduled task and opens no window.
All arguments after the model name go to its independent training entry point.
Place launcher options (`--python`, `--device`) **before** the model name.

```shell
python -m scripts.run_training fno2d --run-training --data-profile canonical --output ../runs/fno2d
python -m scripts.run_training fno3d --run-training --data-profile canonical --output ../runs/fno3d
python -m scripts.run_training uno --run-training --data-profile canonical --output ../runs/uno
python -m scripts.run_training unet --run-training --data-profile canonical --output ../runs/unet
python -m scripts.run_training gift_generator --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m scripts.run_training gift_predictor --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --seed 20260820 --low-model ../runs/gift_generator/model.pt --output ../runs/gift_predictor_20260820
```

Repeat `gift_predictor` independently for seeds `20260821` and `20260822`, each
in a new output directory. Each exports `model.pt`. The generator and every
branch each train for 500 trajectory epochs; see [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md).

For reduced-data GIFT-Lite (50 training trajectories) and the M1 generators
(single training trajectory):

```shell
python -m scripts.run_training gift_low --run-training --condition noise_000 --output ../runs/gift_low
python -m scripts.run_training gift_low --run-training --condition noise_000 --training-trajectories 1 --output ../runs/m1_noise_000
python -m scripts.run_training gift_branch --run-training --seed 20260820 --low-model ../runs/gift_low/gift_main.pt --output ../runs/gift_branch
```

For a complete fresh M1 set, run `gift_low` with `--training-trajectories 1` for
`noise_000`, `noise_001` and `noise_010`, each in a separate output; the clean M1
generator is a separate artifact from the GIFT-Lite clean generator. For three-seed
GIFT-Lite experiments, run `gift_branch` independently for seeds `20260821` and
`20260822` in separate outputs, reusing the same completed clean low generator. A
branch exports `gift_seed_SEED.pt`; a low generator exports `gift_main.pt` for
clean data or `gift_noise_001.pt` / `gift_noise_010.pt` for the noisy conditions.

Use the PINN Python environment for the following independent command (one noise
condition and one library):

```shell
python -m scripts.run_training pinn --execute --mode known --condition noise_000 --output ../runs/pinn/noise_000/known
python -m scripts.run_training pinn --execute --mode open --condition noise_000 --output ../runs/pinn/noise_000/open
```

Repeat the PINN commands separately for `noise_001` and `noise_010`. Add `--resume`
to a command and retain its output directory to continue that same run. For
`gift_low`, `gift_generator` and `gift_predictor`, replace `--run-training` with `--resume` instead of combining them. A fresh
run requires a new output directory. PINN's L-BFGS recovery is phase-boundary
recovery, as explained in the adaptation guide. Diagnostic/tiny runs are not
formal experiment results.

Torch intra-op thread counts are 16 for FNO and GIFT, and 1 for U-NO and U-Net;
PINN uses one TensorFlow thread. U-NO's recorded launch sets `OMP_NUM_THREADS=24`,
leaves MKL/OpenBLAS/NumExpr thread overrides unset, then explicitly sets Torch
intra-op to 1 in `training.train_uno`. The environment value is therefore not a
claim that Torch training uses 24 threads. U-Net's explicit startup settings are defined in
`training/unet_environment.py`; direct invocation must satisfy the same gate.
PINN uses float32 with TF32 disabled. U-Net uses cuDNN TF32 and disables matmul
TF32. These are recorded numerical environments, not changes to the upstream
network. `--device cpu` before the model selects a CPU runtime. The external
prediction baselines permit CPU only with an explicit `--tiny` diagnostic budget;
their formal-budget gate requires CUDA. The formal GIFT-Lite branch command also
requires CUDA. CPU-capable GIFT entry points record separate runtime identities;
CPU and GPU training are not advertised as bitwise-identical.
FNO leaves `CUBLAS_WORKSPACE_CONFIG` unset; all GIFT training profiles set
`:4096:8` and deterministic cuDNN kernel selection. The launcher sets the workspace
before scientific imports, and GIFT trainers set kernel flags before tensor work.

## M1 quick readout, aggregation and SVG

Use the prediction/GIFT Python environment for M1 readout, including PINN's
NumPy-only coefficient reader. An explicit checkpoint requires its SHA-256.

```shell
python -m experiments.formal.m1_equation_identification.run --execute --method PINN-SR-KC --condition noise_000 --checkpoint ../runs/pinn/noise_000/known/result.json --checkpoint-sha256 ACTUAL_SHA256 --output ../runs/m1_jobs
python -m experiments.formal.m1_equation_identification.aggregate --jobs ../runs/m1_jobs --output ../runs/M1 --skip-plots
```

Aggregation requires all five methods under all three conditions (15 completed
jobs). It does not substitute missing measurements. Add `--use-results` instead
of `--execute` to an M1 method/condition command to read that job's committed
result without opening a training checkpoint. [FIGURES.md](FIGURES.md) lists the
independent SVG rendering commands; the numeric outputs and their completion
hashes must exist first.

Source-level checks and small native continuation tests are documented in
[VALIDATION.md](VALIDATION.md), separately from full scientific experiments.
