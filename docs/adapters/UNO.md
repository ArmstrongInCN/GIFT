# External U-NO adapter

Use [ashiq24/UNO](https://github.com/ashiq24/UNO/tree/19462d82729ef64ef7b9e97056ddcaaf3044ad47)
at commit `19462d82729ef64ef7b9e97056ddcaaf3044ad47` (upstream BSD-2-Clause).
No upstream architecture, optimizer, utility, training script or source patch is
distributed here. Its license is separate from the project's license.
在仓库外取得下列固定提交，回到本仓库根运行检查；不要替换上游版本或插值反向算法。`--forward` 只执行一次合成 CPU 前向；独立训练和自身续算见文末 TRAINING 链接。

```powershell
$env:GIFT_EXTERNAL_ROOT = 'D:\gift_external'
git -c core.autocrlf=false clone https://github.com/ashiq24/UNO "$env:GIFT_EXTERNAL_ROOT\uno"
git -C "$env:GIFT_EXTERNAL_ROOT\uno" config core.autocrlf false
git -C "$env:GIFT_EXTERNAL_ROOT\uno" checkout --detach 19462d82729ef64ef7b9e97056ddcaaf3044ad47
python -B -m adapters.check --model uno --forward
```

Run from the release root. On POSIX use the equivalent `export` for the absolute
external root. All seven locked source/license files in `external_sources.json`
must match exact LF bytes. Dependencies are installed separately using the
project environment instructions; no upstream pip package substitution is made.
The old project stored CRLF copies; the lock records their hashes separately as
`legacy_crlf_sha256`. All seven official LF blobs were checked against those
copies and differ only in line endings. Only the canonical LF `sha256` is
accepted at runtime; the loader does not silently normalize files.

The loader imports the verified external integral operators, `Adam` and
`navier_stokes_uno2d`; from external `utilities3` it compiles only the unchanged
`LpLoss` class with torch available. The unused MatReader/scipy import is not
executed: it otherwise sets `KMP_DUPLICATE_LIB_OK=True` via its dependency chain.
No loss class body is rewritten. It then constructs native
`UNO(in_width=50,width=32,pad=0,factor=0.75)`: 15,290,641 parameters,
46 history + 4 internal periodic-coordinate channels. The native input is
`[B,64,64,46]`, output `[B,64,64,1]`. A separate process per model prevents
upstream generic module names such as `utilities3` from colliding; the loader
refuses an already-imported conflicting module. Import does not create pyc.

The upstream module seeds torch/numpy at import. The loader omits only the two
module-level `torch.manual_seed(0)` / `np.random.seed(0)` AST statements in memory,
so importing dependencies does not override a caller's explicit seed or queue
a delayed CUDA seed. No source is rewritten on disk. The caller must explicitly
set its training seed before construction; model initialization is unchanged.
The adapter does not change TF32, deterministic flags, precision, interpolation,
forward or backward. **Native bicubic/antialias backward remains native.**

## Training boundary and unresolved equivalence

The independent training CLI is `python -B -m training.train_uno`. The formal protocol
is 150 terminal epochs, seed 0, physical batch 16 (last batch 8), 20-step
closed-loop loss without teacher forcing/detach, original upstream Adam
(lr 0.001, weight decay 0.00001), StepLR(100,0.5), and 9,450 updates. Training
uses `checkpoint(..., use_reentrant=False)` and no normalization. Evaluation
extends to 150 recursive steps with batch 4; it is not the 20-step training loss.

The published trainer used `use_deterministic_algorithms(True,warn_only=True)`
with native backward. The previous staging trainer imposed strict determinism
and substituted a custom antialiased-bicubic backward. That variant is **not**
included here. Its loss already differed at epoch 1, before interruption or
resume; missing old global-RNG snapshots do not explain that earlier difference.
The earlier successful 101–108 resume log comparison does not establish that
the custom backward matches published training. Neither TF32 nor backward has
been isolated as the unique cause of the full training discrepancy.

Do not require strict deterministic algorithms and then silently substitute a
gradient to make training run. Preserve the published profile and record warnings.
The controlled native first-update check found a maximum parameter difference
of about 4.75e-6 between candidate and formal functions; the formal-native
self-control differed by up to 5.84e-6. This does not isolate the cause of the
older full-training discrepancy or establish bitwise CUDA repeatability.
The independent controller now saves model/optimizer/scheduler, epoch/update
cursor, all RNG states, data/source hashes and actual numerical profile; old
missing RNG state is not invented. Exact continuation is a separate claim from
starting from zero. See [current evidence](../REPRODUCIBILITY_STATUS.md).
保留原生反向；同源自对照也有微小首更新差异，不能为了逐位通过而换梯度。新控制器已有自身状态保存/恢复，但短 CPU 续算通过不等于完整 CUDA 训练可逐位重现。

Optional strict state loading:
`python -B -m adapters.check --model uno --checkpoint /absolute/terminal.pt --forward`.
This performs only one synthetic CPU call; it does not prove training, CUDA,
recursive rollout or whole-experiment equivalence. Validation outcomes are
recorded separately; this loading command does not establish from-zero or
whole-experiment numerical reproducibility.

On 2026-09-10, the final loader passed strict loading of the existing published
epoch-150 artifact (36 state entries), the parameter-count check above and one
finite synthetic CPU forward (2.33 seconds). A separate CPU import check with
nonzero caller seeds confirmed torch/numpy RNG preservation and no initialized
CUDA context. These are bounded adapter tests, not training-equivalence tests.

The new controller also passed an explicit nonformal two-epoch CPU continuation
test on 2026-09-10: continuous execution versus a fresh-process resume after
epoch 1 produced exact model, optimizer, scheduler, RNG and scientific-history
matches. Each epoch used one synthetic trajectory and one native-backward step;
this is not a formal 150-epoch equivalence claim. See [training usage](TRAINING.md).

## Actual formal first epoch (historical boundary)

The current controller (`7CDA8F5E...`) subsequently completed one full epoch of
its unchanged150-epoch budget with native backward:1000 trajectories,20-step
rollout,batch16 (last8),63 updates; process66.069s,epoch54.298s. A separate CPU
audit passed own epoch0→1 source/data/runtime/journal,optimizer,scheduler and RNG
checks. The initial36 tensors match the earlier fresh initialization hashes.
This historical record establishes a valid saved pause, not by itself full-budget
acceptance or an executed GPU resume; the later full-run evidence is below.

The one reference training loss is reported per trajectory **and step**.
The candidate's saved20-step per-trajectory sum11.766310146331787 becomes
0.5883155073165893 after division by20; the original CSV has0.5883155637741089
(absolute difference5.6457519548303026e-8). This is not exact reconstruction of
an unsaved accumulator, not two independent loss comparisons, and not a new
tolerance-based training acceptance. The native nondeterminism warning is
preserved. See [first-epoch evidence](../../validation/uno_epoch1_20260911/README.md).
实际首轮完整执行且自身状态核验通过，但损失并非逐位相同；保留微差与原生警告，
不修改反向算法或门槛。此首轮历史记录不能代替后续完整训练或实验验收。

## Actual complete 150-epoch run

The same current controller subsequently resumed its own epoch-1 journal and
completed the original 150-epoch budget: 9,450 updates, native backward and no
published checkpoint used for training. The completed CPU terminal audit reports
`valid_full_run=true / VALID_FULL_RUN_REFERENCE_DIFFERENCES`: source/data/runtime,
history, optimizer, scheduler, RNG and terminal bindings passed. The exported
model matches its own immutable terminal journal in **36/36 tensors**, while
the published terminal comparison is **0/36 exact**. All 150 independent original
training-loss comparisons are nonexact, using only the reporting-unit `/20`
conversion above and no added tolerance. This does not identify a unique cause
of the discrepancy or establish bitwise native CUDA repeatability.

The resumed process took 8,401.320 s, in addition to the separate first-epoch
process's 66.069 s. Their approximately 8,467.389 s sum excludes pauses and audits.
This is actual same-run GPU continuation, not an uninterrupted-versus-resumed
full-run bitwise test. See the [full-training evidence](../../validation/uno_full_20260911/README.md).
M2 numerical acceptance of this new terminal is a separate experiment; completion
and a valid fresh terminal are not report-reproduction acceptance. Published
weights and the checkpoint catalogue are unchanged.
完整 150 轮及自身 GPU 续算已完成，完整性通过；对原发布结果的权重与训练损失差异
仍保留。新终态是否通过 M2 数值检验须独立判断，不因完整训练而宣称报告已复现。

The subsequent full M2 run **failed 29/245 numerical comparisons**, all in U-NO,
while all row keys/counts matched. At t=8, mean relative error was 0.6600432
versus the archived 0.4160649. Compared with the earlier successful M2 using
the same fresh GIFT models and reference baselines, only the U-NO model
selection/path/hash changed; source, data, external code and runtime identities
matched. This localizes the changed experimental input, not the first differing
floating-point operation. See [full M2 FAIL evidence](../../validation/fresh_uno_M2_20260911/README.md).
No reference weights or tolerance were replaced to remove this failure.

后续完整 M2 已实际运行，29/245 项超差均属于 U-NO；其余方法及行键、计数通过。
对照此前 M2，仅 U-NO 权重选择改变，其他执行身份一致。差异如实保留，不能将
完整训练成功写成原报告通过，也不据此武断认定某个算子是全部误差的原因。
