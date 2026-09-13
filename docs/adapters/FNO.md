# External FNO-2D / FNO-3D adapter

The implementation is not distributed here. Obtain the paper-era source from
[neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator/tree/01d2aeca407f85ce1c1e78b313edb749973fc051),
commit `01d2aeca407f85ce1c1e78b313edb749973fc051`. A current pip `neuraloperator`
model is not an equivalent replacement. Consult the upstream terms yourself;
this project's license does not license the external code.

From a shell, use a new external directory (never inside this repository):
在仓库外新目录取得下列固定提交；保留 LF 字节，然后回到本仓库根运行检查命令。`--forward` 仅做合成输入 CPU 检查，不训练，也不证明完整复现。

```powershell
$env:GIFT_EXTERNAL_ROOT = 'D:\gift_external'
git -c core.autocrlf=false clone https://github.com/neuraloperator/neuraloperator "$env:GIFT_EXTERNAL_ROOT\fno"
git -C "$env:GIFT_EXTERNAL_ROOT\fno" config core.autocrlf false
git -C "$env:GIFT_EXTERNAL_ROOT\fno" checkout --detach 01d2aeca407f85ce1c1e78b313edb749973fc051
python -B -m adapters.check --model fno2d --forward
python -B -m adapters.check --model fno3d --forward
```

Run the Python commands from this repository root. On POSIX, set the same
environment variable with `export GIFT_EXTERNAL_ROOT=/absolute/external/path`.
The loader checks byte hashes in `external_sources.json`; use LF files, not
Windows autocrlf conversion. It does not modify the checkout. The lock covers
`fourier_2d_time.py`, `fourier_3d.py` and `utilities3.py`.

## Exact adaptation boundary

The original scripts contain top-level data loading, CUDA training and output
writes. Importing them directly is unsafe for a loader. The adapter parses the
external file and compiles only `SpectralConv2d_fast` + `FNO2d`, or
`SpectralConv3d` + `FNO3d`, into memory. It supplies their existing torch/numpy
imports without running the original script's seed, data or training statements.
No generated upstream source is saved or shipped.

Exactly one constructor AST constant changes: FNO-2D's `fc0` input width
12 → 48 (46 history + 2 coordinates), or FNO-3D's 13 → 49 (46 + 3 coordinates).
This happens **before construction**, preserving initialization draw order for
the formal input size. Replacing `fc0` after constructing a 10-history model
would consume the wrong RNG sequence. No forward, FFT, gradient, BatchNorm,
coordinate or padding algorithm is replaced. FNO-2D uses four BatchNorm layers;
FNO-3D retains its unused BatchNorm state entries and uses no zero padding.
No cached-grid subclass from the former project is included.

Formal constructors are `(12,12,20)` with 466,437 parameters and
`(8,8,8,20)` with 3,282,457 parameters. Native input layouts are respectively
`[B,N,N,46]` and `[B,N,N,T,49]`. FNO-3D coordinates are external and ordered
`[x,y,t,history...]`; its output must be decoded using the original training-only
position statistics. The small CPU check supplies synthetic input, not a
scientific FNO-3D normalization/rollout test.

## Training and verification status

Independent CLIs are `python -B -m training.train_fno2d` and
`python -B -m training.train_fno3d`; see [training usage](TRAINING.md).
The formal protocol is 500 terminal epochs, 1,000 trajectories, 46 history frames,
150 future frames, 62 start windows, and one deterministic window per trajectory
per epoch. FNO-2D physical batch 10 with accumulation 2 is not interchangeable
with physical batch 20 because BatchNorm observes the physical batch. FNO-3D
physical batch 5 with accumulation 2 has effective batch 10. Preserve reference
normalizers and Adam/StepLR update order; do not infer training equivalence from
parameter counts or state-key compatibility.

The previous full fresh training diverged from published loss logs at epoch 1
for both models. Its exact historical cause remains unresolved. The candidate
has since passed a controlled same-seed comparison against the current formal
training functions: initialization, sampling, loss, gradients and first updated
state matched for FNO-2D's full 150-step objective and FNO-3D's 150-frame block.
The FNO-3D comparison reused verified training normalizers to isolate the update;
separately, only input/output position 0 were rebuilt from all 62,000 windows
and matched exactly (2 of 196 positions). These controlled checks alone were not full-budget training or
historical source/profile provenance. See [current evidence](../REPRODUCIBILITY_STATUS.md).
受控首更新检查已完成；该阶段只核对当前原函数，单独不足以证明历史完整训练可重建。不能把模型加载、首更新或两个统计位置的相同扩大成全预算、全部统计量或六实验通过。

Subsequently, both independent fresh runs completed two full epochs: epoch 1
followed by an actual new-process resume from that run's own checkpoint to epoch 2.
The four aggregate loss values per model exactly matched the original CSV rows;
source/input/runtime and saved-state checks passed. FNO-3D's four normalizer arrays
were bit-exact across its own resume, not thereby proven equal to every reference
statistic. See [FNO-2D two-epoch evidence](../../validation/fresh_20260910/fno2d_epoch2/README.md)
and [FNO-3D two-epoch evidence](../../validation/fresh_20260910/fno3d_epoch2/README.md).
Those two-epoch records remain valid historical boundaries; aggregate equality
alone does not prove reference gradient/weight/trajectory equality. The later
full-budget terminal audits are recorded separately below.

FNO-3D subsequently continued its own run to **500/500 epochs and 50,000 updates**.
The completed CPU terminal audit passed the source/input/runtime, final-budget,
saved-state and own-export integrity checks. All 1,000 recorded original-unit
loss comparisons (500 epochs x two metrics), all 50 published-reference model
state entries and all four published-reference normalizer arrays were bit-exact.
The exported model also matched all 50 state entries of its own final journal.
See [FNO-3D full-budget evidence](../../validation/fno3d_full_20260911/README.md).

FNO-2D also continued its own fresh run to **500/500 epochs and 25,000 updates**.
Its completed CPU terminal audit passed source/input/runtime, final-budget,
history, optimizer/RNG and own-export integrity checks. All 1,000 original-unit
loss comparisons (500 epochs x two metrics) and all 42 published-reference
model-state entries were bit-exact. Its export matched all 42 state entries
of its own final journal. No published weights were used for initialization
or continuation. See [FNO-2D full-budget evidence](../../validation/fno2d_full_20260913/README.md).

These checked-host full-training results do not establish equality of every intermediate gradient or trajectory,
or an uninterrupted-versus-resumed full-run comparison.

The subsequent independent experiments used both newly trained FNO exports.
[M2](../../validation/fresh_fno_M2_20260913/README.md), with fresh GIFT/U-NO and
reference U-Net, completed in 333.05 s: 216/245 numeric comparisons passed;
all 29 failures belong to U-NO, with no failed FNO item and all 98 counts matching.
[M3](../../validation/fresh_fno_M3_20260913/README.md), with fresh GIFT/FNO only,
completed in 686.55 s: all 9,350 numeric and 3,740 count comparisons passed across
1,870 rows. The original tolerance was retained; inference PASS is not a claim
of bit-exact inference or whole-project acceptance.

FNO-2D 与 FNO-3D 均已完成 500 轮自身续算及终态审计：各自记录的 1,000 项损失，以及分别 42/50 项参照模型状态均逐位相同；FNO-3D 的 4 项参照归一化数组也相同。使用新 FNO 的独立 M3 已通过；M2 的 29 项失败全部属于 U-NO，FNO 无新增失败项。原容差与失败记录均保留，不能宣称全项目通过。

`python -B -m adapters.check --model fno2d --checkpoint /absolute/terminal.pt --forward`
optionally verifies strict state keys/shapes and one CPU call. It never resumes
training. No automatic download of weights, training or checkpoint selection
occurs. Verification outcomes are recorded separately, not assumed by this file.

On 2026-09-10, the new adapter passed strict loading of the existing published
FNO-2D and FNO-3D terminal artifacts (42/50 state entries), the parameter-count
checks above, and one finite synthetic CPU forward each (1.63/1.68 seconds in
the existing Python 3.10 / PyTorch 2.10 environment). Official external Git
checkouts were used; no training or scientific rollout was performed.
