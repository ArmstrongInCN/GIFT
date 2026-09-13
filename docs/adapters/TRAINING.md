# Independent baseline training and continuation

Each model has its own CLI: `training.train_fno2d`, `training.train_fno3d`,
`training.train_uno`, `training.train_unet`. None starts another model. The
shared controller is project-owned experiment/data/checkpoint glue, not a copy
of the upstream training script. Install pinned external sources first.

From the release root, with `GIFT_DATA_ROOT` pointing to the separate data
package and `GIFT_EXTERNAL_ROOT` pointing to the external checkouts:

```powershell
python -B -m training.train_uno --dry-run
python -B -m training.train_uno --run-training --device cuda --output D:\gift_runs\uno_001
python -B -m training.train_uno --run-training --device cuda --output D:\gift_runs\uno_001 --resume
```

Replace `uno` with the desired model; do not run both training commands at once.
在仓库根执行，先配置仓库外的数据与固定源码。将 `uno` 换成目标模型；`--dry-run` 只检查计划，`--run-training` 才训练。中断后保留原命令和输出目录，加 `--resume`；不要同时启动示例中的两条训练命令。
For U-Net, also follow the [U-Net-only child-process environment](UNET.md#u-net-only-child-process-environment)
before Python starts. It requires the TF32 override to be completely unset and
thread counts of 1; it is not the FNO-2D thread-16 profile. The current CLI does
not itself enforce the entire historical B7 pre-import environment gate.
U-Net 初训与续算均须使用该专用启动环境；说明补全不代表训练代码的门禁已修复，
也不能把旧控制器的两步首更新诊断当作当前控制器的正式四步或500轮验收。
The first command does not build a model, train or write output. It validates
source hashes and data metadata, explicitly reports that data bytes have not
yet been hashed. Training fully checks the data hash before use. Relative
`--data-file` paths resolve strictly below `GIFT_DATA_ROOT`; the formal default
is `fno/fno1000_n64_t0_t10_dt0p02.h5`. Outputs must be outside code/data/upstream
roots, and a fresh run refuses an existing output directory.

## Released and regenerated input profiles

All four independent CLIs accept `--data-profile released|regenerated`. The
default is `released`: formal training still requires the exact original
training H5 SHA-256 `1322ad0a5be67bccab2a93c55a5a5748f792f80f3a839032bce888959d54e3ee`.
Selecting another profile does not change any model, loss, batch, seed or budget.
默认 `released` 必须匹配原数据 SHA；只有完成独立生成和组装、具备完整来源记录的新集合才可选 `regenerated`。切换 profile 不会改变模型或预算，也不能把已发布权重说成由新集合训练。

For a **newly generated, separately assembled** collection, set `GIFT_DATA_ROOT`
to its root and explicitly select the new profile:

```powershell
$env:GIFT_DATA_ROOT = 'D:\gift_data\generated_collection_001'
python -B -m training.train_uno --data-profile regenerated --dry-run
python -B -m training.train_uno --data-profile regenerated --run-training --device cuda --output D:\gift_runs\uno_regenerated_001
```

See [DATA_GENERATION.md](../DATA_GENERATION.md) for independent generation and
create-only assembly. The baseline consumer requires the canonical
`fno/fno1000_n64_t0_t10_dt0p02.h5`, the collection `manifest.json`, `splits.json`
and `provenance/fno-training/{run,COMPLETE}.json`. A collection may omit unrelated
generation jobs; this does not claim that all six experiments have their inputs.
The input gate checks unique in-root file paths/sizes, the collection status,
current assembler/generator/solver source hashes, completion/binding identity,
the original PDE/numerical protocol, full populations, exact time axes, dtype,
shape and initial-condition parameter hashes. Execution additionally verifies
the consumed H5 bytes and scans every training/validation field for finite values.
Pilot, subset, missing receipts, mock markers, external/virtual fields and
unwritten NaN frames cannot qualify as regenerated training input. Dry-run
does not hash the large H5 or scan its full values: its success is **not** a
scientific input acceptance or training result.

The actual new file hashes, collection manifest hash, generation attempt/binding,
generation source hashes, profile and extra-950 parameter origin are included in
the training journal identity and exported model provenance. Extra 950 trajectories
remain generation from specified initial conditions with unknown historical seed,
not a recovered from-seed claim. Resume must use this same identity. No command
imports published weights as a training initialization or relabels published
weights as trained on regenerated containers. Existing published weights may be
used for a separately described inference comparison, never as proof that the
new dataset was used to train them. `--tiny` remains test-only; it cannot be used
to turn a mock into a formal regenerated input.

CPU profile tests cover metadata/plumbing success and rejection of altered
manifests, sources, PDE, IDs, times, parameters, receipts, mock and unwritten data.
The successful plumbing fixture explicitly mocks the scientific-validator boundary;
it is not a real generated trajectory set. Run with a **new external** basetemp:

```sh
python -B -m pytest tests/test_baseline_data_profiles.py tests/test_generated_data_assembly.py tests/test_baseline_control.py::ProtocolTests -v -p no:cacheprovider --basetemp /new/profile-test-output
```

The regenerated switch is implemented, but full-population real generation,
real collection assembly and full-budget training with that collection have
**not** been executed/accepted. The profile change did not train any model or
establish numerical agreement with published experiment results.

The formal constants are fixed. No seed, physical batch, accumulation or
training-rollout override silently changes a formal run. Original closed-loop
loss and terminal-epoch selection are retained; no validation/test selection is
added. U-NO uses native upstream Adam and interpolation gradients. FNO utility
classes are compiled only from the external pinned `utilities3.py`. FNO-3D
normalization includes all 1,000 × 62 training windows and preserves the
row-major reduction order; it requires enough host RAM for the dense training
array and working position samples, as the original trainer did.

## Same-run resume

The controller saves an initial random state at epoch 0, then an immutable
checkpoint every 10 completed epochs by default. Each independent CLI accepts
`--checkpoint-interval N`; tiny tests, an explicit stopping epoch and the terminal
epoch always save. No old checkpoint is automatically deleted. `LATEST.json` names the last complete
boundary. Each checkpoint includes model, optimizer, scheduler, all RNG states,
normalizers, epoch/update cursor, scientific history and exact configuration,
data/source/runtime identity. `--resume` accepts only this run's own journal,
not a published terminal artifact. Source, seed, budget, data or recorded runtime changes
are rejected rather than silently mixed. Restore occurs after constructing and
loading the model/optimizer. A crash loses work since the previous saved
boundary: with the default interval this may include nine completed unsaved
epochs plus the in-progress tenth epoch. It does not claim mid-minibatch
continuation. Saving every epoch is available, but 500 full U-Net optimizer
states may require roughly 150 GB; the default ten-epoch interval avoids that
storage requirement while preserving resumability.

The baseline journal currently records library versions and precision flags,
but does **not** automatically compare CPU thread counts or `OMP_NUM_THREADS` /
`MKL_NUM_THREADS`. Keep those settings fixed yourself across initial training
and continuation. The checked FNO-2D first-epoch run used both environment
variables set to `16` before Python started; this is a tested profile, not a
recovered complete historical runtime record. Configure the checkpoint interval
at the start of a run and retain it when resuming.

基线续算校验目前不自动拦截 CPU 线程数及 OMP/MKL 环境变量变化；初训和续算必须
保持这些设置一致。已检验的 FNO-2D 首轮在启动 Python 前将两项均设为 `16`。
这不代表历史运行环境已完整恢复，也不证明后续 499 轮通过；保存间隔亦需保持不变。

`--stop-after-epoch 1` provides an explicit test/pause boundary without changing
the total training budget. Use the same command without this option plus
`--resume` to continue. Do not start two processes against the same output.
The shared journal rejects concurrent conflicting saves.

Completed runs export `model.pt` with the new schema
`gift.independent-baseline-weights.v1`, top-level state dict, normalization,
method/configuration, actual run provenance, terminal epoch and resume flag.
It contains no optimizer and never pretends to be an original published file.
Incomplete runs do not produce it. Loading weights is not optimizer resume.

## Bounded tests, not a reproduction certificate

Explicit `--tiny` enables nonformal synthetic/subset testing. Only in this mode
do `--tiny-epochs`, `--tiny-trajectories`, `--tiny-batch` and `--tiny-rollout`
apply; defaults are 2 epochs, 2 trajectories, physical batch 1 and rollout 1
(FNO-3D 16 to retain its native temporal modes). Tiny inputs still have 64×64
grids and the original architecture/history width. The dataset content hash is
recorded but is not confused with the formal dataset hash. All tiny checkpoints
are marked nonformal and exported weights have `artifact_role="test_only"`.

Fast tests: `python -B -m unittest tests.test_baseline_control.ProtocolTests -v`.
Real CPU continuation tests are opt-in: set `GIFT_BASELINE_TEST_ROOT` to a **new**
external directory, then run `python -B -m unittest tests.test_baseline_control -v`.
They create small synthetic HDF5 files and retain all evidence. No data or prior
result is overwritten or deleted. Each real-model test compares two epochs
continuously against one epoch + fresh-process resume, including exact final
model/optimizer/scheduler/RNG and scientific history fields.

On 2026-09-10, the four real-model continuation tests plus four fast protocol
tests passed in 61.957 seconds. The U-NO loader was then narrowed to avoid its
unused SciPy import, and checkpoint interval 10 was added. A clean-child check
confirmed no KMP environment mutation, unchanged torch/numpy RNG and no CUDA
initialization; all five fast protocol tests and the U-NO real continuation
regression then passed in 22.213 seconds. No full-budget training was run for
these changes. The initial four-model test and this targeted later regression
are distinct evidence, not a claim that all four full trainings were repeated.

CPU resume equality does not establish CUDA full-budget reproducibility. The
controlled FNO first-update comparisons now match the current formal functions,
including FNO-2D's 150-step objective and FNO-3D's 150-frame block. This does not
resolve the incomplete historical source/profile binding or reproduce full
500-epoch checkpoints. U-NO native versus former custom-backward contributions
remain unresolved; native CUDA self-comparison is not bit-repeatable. U-Net's
existing exact full training belongs to the earlier campaign, not this controller.
See [current evidence](../REPRODUCIBILITY_STATUS.md) and each model's adapter note.
No claim that all six experiments now pass is made.
已完成受控首更新和短续算测试，不表示新控制器已完成全预算训练。所有 tiny 输出仍为 test-only，不能放入正式评估；保持原数值定义和容差，不用重跑或改算法来掩盖来源缺口。
