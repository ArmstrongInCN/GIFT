# Checkpoints: train, read or resume / 检查点使用

当前七方法推理模型与两份配套生成元可通过 `scripts.package_current_models` 导出独立公开副本、全量校验并在新机器激活，流程见 [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md)。[源码元数据清理版 Release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020-source-free)提供[参考模型包](https://github.com/ArmstrongInCN/GIFT/releases/download/current-protocol-20261020-source-free/GIFT-current-models-20261020-source-free.zip)，[归档清单](CURRENT_MODEL_BUNDLE.json)记录其下载地址、文件大小和 SHA256。此前[旧 Release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020)的包仍含 ANIE 的四段源码元数据（441字符），现已由新包取代，旧附件作为来源记录保留。公开副本清理源码元数据，并逐张量验证数值完全不变；原权重保持只读，原测量文件、当前来源文件与公开副本各自保存文件哈希。当前W3可由正式输入包新建自己的运行gate并从零训练，见 [W3_REPRODUCTION.md](W3_REPRODUCTION.md)。

元数据清理会改变 ANIE 公开副本的序列化字节及 SHA256，不能宣称它与原文件逐字节相同，也不能据此取得 ANIE 的使用、修改或权重分发授权。来源及相关文献见 [README](../README.md#references)，外部依赖范围见[边界说明](EXTERNAL_REPOSITORY_POLICY.md)。下表保留当前来源文件身份，不以公开副本的新哈希覆盖历史科学结果的来源绑定。

## W1/W2 模型文件与来源

[N300/76点协议](W1_W2_PROTOCOL.md) 使用单真实初态、固定超参数与每方法一个模型实例。主方法N300，GIFT-Lite为固定有序名单前50条。其余六方法的模型身份、配方、种子和数据绑定存于 `prediction_extension/EXTENSION_LOCK.json` 与同目录 `FORMAL.json`；ANIE训练配置与模型身份存于 `prediction_aligned/PROTOCOL_LOCK.json` 及 `prediction_aligned/training/anie_ns_seed7/ATTEMPT.json`。以下为外部实验文件，不包含在Git结果包中。

| 方法 | 来源键及相对路径 | 当前文件SHA-256 | 选择规则 |
| --- | --- | --- | --- |
| GIFT | `prediction_native/training/GIFT_seed20260820/branch/model.pt` | `fe40059ce9863df79a31e6f3c16e890c577b277519e2fc0a234e4f7014cbf478` | epoch500终态 |
| FNO-2D | `prediction_native/training/fno2d_seed0/terminal_weights.pt` | `6dc9197637184db3f7c412aa82d0630be8eafaa79054cf0401a876fd04ca3b05` | epoch500终态 |
| FNO-3D | `prediction_native/training/fno3d_seed0/terminal_weights.pt` | `286aeea4c54c6810a43ab572b66309c5c3db16fb9c8323286ad581d3968fc0a1` | epoch500终态 |
| ANIE | `prediction_aligned/training/anie_ns_seed7/best.pt` | `3adbe09b412c1a3a214ac06bebde51079298018799b80a685d8cfde4f93abb35` | validation-MSE-best epoch29；仅W1 |
| U-NO | `prediction_extension/training/uno_seed0/terminal_weights.pt` | `75334c44d832fab4d5ef525af68c3848ed1392594e1c32885f3c539179919068` | epoch500终态 |
| U-Net | `prediction_extension/training/unet_seed0/terminal_weights.pt` | `ee6918cba6991b904cd6ccaa6738dd992325ad8691ffce5919438e32307eb34c` | epoch500终态 |
| GIFT-Lite | `prediction_extension/training/GIFT-N50_seed20260820/branch/model.pt` | `fe78259d166941e7cab84859669a0a508a0f580f19c7a2a908809d4adeb3ec93` | epoch500终态 |

表中来源键由仓库外登记解析为当前可加载文件。模型数值、optimizer、RNG、normalizer及观测数组与测量时状态逐位一致；当前文件的规范化身份元数据使序列化SHA不同。原测量实际权重的SHA仍保留在科学结果清单中，不改写原预测的来源绑定。对应独立核验见仓库外`CANONICAL_ACTIVE_MIRRORS.json`和`CANONICAL_CHECKPOINT_BOUNDARY.json`。

GIFT分支与对应生成元绑定，组件不能跨配置互换：

- GIFT生成元：`prediction_native/training/GIFT_seed20260820/generator/model.pt`，SHA-256 `a4489e094c1edd74cabc8ead557a7535ddcdbf45cb67f74aa164af86a3c66c31`。
- GIFT-Lite生成元：`prediction_extension/training/GIFT-N50_seed20260820/generator/model.pt`，SHA-256 `4b4f95124604e43c6d880fa10dea48d997f576168321b0f9b2e7313ffed7400c`。

FNO-2D/FNO-3D/U-NO/U-Net均选择500轮终态；ANIE依原NS validation-MSE-best规则选第29轮，按原早停规则训练至1029轮。GIFT及GIFT-Lite生成元/分支各500轮终态，采用0.08/0.48秒监督、内部dt0.02、保存dt0.04。U-NO/U-Net监督20/4步，N50总更新少于N300。ANIE仅参加W1；W2六方法冻结其N64训练权重。

确认人口的两个评价人口的成绩已被查看，故不视为全部方法盲测；确认与探索分开评价。W1/W2结果包保存数值汇总与图像证据，不是训练续算文件。下方现有 `artifacts/` 权重用于独立S1–S3及N1000辅助协议，具体metadata和哈希见各自清单。W3按50组参数分别保存GIFT及两种PINN的150个正式终态模型，实际模型文件、训练状态与逐case身份绑定保存在仓库外的运行目录，范围见下方W3章节。训练恢复需要同一实验的模型、optimizer、scheduler、RNG和阶段状态，最终推理权重不等于完整恢复状态。

These are separate operations:

| Operation | Input | Training performed? |
| --- | --- | --- |
| Fresh training | Supplied observations and initialized model | Full declared budget |
| Quick evaluation | Packaged trained weights plus test inputs | No |
| Same-run continuation | That run's committed model, optimizer, random and phase state | Remaining uncommitted work |

Downloaded terminal weights are not inserted into an unrelated run's training
journal. See [SETUP.md](SETUP.md) for each model's independent command and the
special PINN L-BFGS phase-boundary recovery rule.

## External implementation location

External-baseline loading requires the user-prepared runtime outside GIFT. Set
`GIFT_EXTERNAL_ADAPTER_ROOT` to its root, such as a sibling
`gift-external-runtime` directory. Official source locations remain as recorded
in the weights and run metadata. Moving the connecting implementation outside
the repository does not change model state keys, architecture, profile or
checkpoint-selection rules. Follow [EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md)
when preparing a new external runtime; matching filenames alone is insufficient
to establish computational equivalence. Existing weight and measurement hashes
remain historical evidence and must not be rewritten for a path migration.

## Training resume

Use the original output directory with `--resume`. Resume restores the model,
optimizer, scheduler, RNG and committed progress, then checks actual model,
data and schedule compatibility. Source hashes, library patch versions, paths,
thread settings and checkpoint intervals do not prevent compatible recovery.
Original checkpoints remain read-only and readable. Publication copies may omit
source-snippet metadata; their separate hashes and exact tensor-equivalence
records distinguish that cleanup from a change to a scientific model.

A complete final checkpoint can also finish an interrupted weights export.
This does not repeat optimization steps. Existing different weights are never
overwritten. To export a baseline checkpoint independently:

```shell
python -m scripts.recover_baseline_export --execute --checkpoint ../runs/uno/checkpoints/epoch_0500_ID.pt --output ../exports/uno.pt
```

ATTEMPT metadata and SHA checks are optional for this file operation. Ordinary
inference uses exported weights and model configuration, without the original
training dataset or optimizer journal.

## Independent N1000 GIFT regimes / 辅助协议权重

Full-data prediction GIFT uses `artifacts/gift_full/generator.pt` and
`artifacts/gift_full/gift_seed_SEED.pt`. Its generator and each branch are trained
on 1,000 trajectories for 500 epochs each; their budgets and times are reported
separately. Branch metadata bind the exact generator hash and declared seed.

The full-data generator and all three prediction branches are included with
their completed training histories, budget records and weight hashes. These
are terminal 500-epoch states, not intermediate training checkpoints. Both GIFT
regimes can be evaluated using the commands in EXPERIMENT_EXECUTION.md.

GIFT-Lite uses the preserved clean generator in `artifacts/fixed_k21_n64_unmasked/`
and the three branches in `artifacts/gift/`. It uses 50 training trajectories and
its own multi-stage schedule, not a smaller network. This is not a data-only
ablation with identical training budgets. The current W3 numerical sweep has 50 parameter cases and five methods; it is separate from prediction and GIFT-Lite models. Its fitted models are bound individually to the corresponding case truth and input data.

## W3 50组无噪声参数识别结果

当前W3数值包包含50组不同真值、每组一条ID0轨迹和五种方法各50次拟合。它按case保存系数读出、完整系数/支持、真实耗时与输入/来源SHA，不将某一case伪装为所有参数组合的单一默认模型。公开包见[逐例数值](../results/formal/W3_single_trajectory_20261010/README.md)。

W3结果包不包含训练用checkpoint。50组参数各自对应一份GIFT终态与两种PINN终态，共150份正式神经模型；实际检查点、优化器、随机状态与阶段记录保存在仓库外的 `w3_single_trajectory_20261010/training/parameter_*/` 运行目录，并由逐case输入与模型身份清单绑定，不随GIFT项目仓库分发。

外部 PINN-SR、PDE-FIND 与其它上游实现仍由使用者从[登记的外部仓库](EXTERNAL_REPOSITORY_POLICY.md)下载并在项目外适配。仓库内只保留说明、接口和引用，不含这些外部算法实现。


## Gaussian initial-condition regime (S3)

S3 repeats the same protocol on a separate smooth Gaussian random-field population
and has its own weights under `artifacts/s3_gaussian/`: `gift_full/generator.pt`
and `gift_full/gift_seed_SEED.pt` for full-data GIFT, `gift_lite/generator.pt` and
`gift_lite/gift_seed_SEED.pt` for GIFT-Lite, plus `fno2d/`, `fno3d/`, `unet/` and
`uno/` for the baselines. These are terminal models trained from scratch on the
Gaussian trajectories, not the four-vortex weights. The S3 evaluation rejects any
model whose recorded input population, freshness or frozen-generator binding
differs, so an W1 model cannot silently stand in for an S3 model.

S3 uses the current `gaussian_s3` cohort identity in the separate observation package selected by `GIFT_DATA_ROOT`. Its released model identity metadata is paired to that observation mirror. All learned tensors, optimizer/RNG values and observation arrays retain their original values. `artifacts/CHECKPOINTS.json` lists the actual released file bytes and hashes; the measurement checkpoint hashes in the scientific result provenance still identify the original measured files. The two are separate evidence identities, with equivalence proved outside the repository in `S3_CANONICAL_DATA_MIRROR.json` and `S3_PACKAGED_METADATA_NORMALIZATION.json`.

`artifacts/s3_gaussian/TRAINING_COSTS.json` reports each run's declared budget and
its measured committed interval with an explicit timing scope; those scopes differ
between model families and are not pooled into a single comparable measurement.

## Ordinary and split weight files

`artifacts/CHECKPOINTS.json` records the available weights and SHA-256 hashes.
Most models use ordinary `.pt` or `.npz` files. U-NO uses `weights.json` and
numbered `.part` files in `artifacts/formal/uno/` and in
`artifacts/s3_gaussian/uno/`; keep each of those folders together. U-Net is
95.31 MiB in both regimes and stays a single ordinary file. Only checkpoints that
cannot be distributed as one file within GitHub's per-file limit are split.

The parts contain the **exact bytes of one normal trained PyTorch checkpoint**,
not quantized or altered weights. The loader checks each part and the complete
reconstructed-file hash, then calls `torch.load(weights_only=True)`. Reconstruction
is in memory; it does not create another file or fetch data from the network.

U-NO 权重以无损分片保存，避免超过 GitHub 普通 Git 的单文件限制。运行时自动校验、
按顺序拼接后加载；模型参数、数值精度和训练算法均不改变。不要单独加载某个分片。

An independently trained ordinary `.pt` file is also accepted through W1's
`--uno-model /path/to/model.pt`. To prepare your own completed weights for
distribution, use a new output directory:

```shell
python -m scripts.split_checkpoint --input ../runs/uno/model.pt --output ../weights/uno
```

The default part size is 40 MiB. This storage operation performs no training and
does not establish the checkpoint's scientific accuracy. Reference experiment
commands use their documented budgets; ordinary model loading accepts a compatible
completed custom run and records its actual settings.

## Verification scope

File hash identity, identical tensors, successful same-run recovery and agreement
of experimental metrics are different checks. Runtime/rounding differences can
change generated fields and learned weights. A completed budget is not a claim
of low error or exact sparse-structure recovery. Model-specific timing scopes
are reported alongside the weights rather than pooled as identical measurements.
