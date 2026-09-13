# 新 standard 部分集合装配 / New-standard partial collection

2026-09-13 实际完成：**真实数据装配及三个完整 GIFT 输入检查均通过**。这是新生成数据的部分集合证据，不是 mock、tiny 测试或训练结果。

本次装配使用已完成的新 standard、由它派生的成对噪声，以及 clean/1%/10% 三份采样文件。相关来源见 [standard 核验](../full_standard_20260913/README.md) 和 [新 standard 噪声核验](../new_standard_noise_20260913/README.md)。

| 实际检查 / Executed check | 结果 / Result |
|---|---|
| 装配子进程 / Assembly subprocess | 返回码 0；7.703 秒 |
| 装配及后续检查 / Assembly plus checks | 25.078 秒，总耗时不含先前数据生成 |
| 集合清单 / Collection manifest | 17 个条目；1,689,077,180 字节，不含 manifest 本身 |
| 全清单哈希 / All manifest hashes | 全部精确一致；源码及父运行收据未变 |
| 三个 GIFT 输入检查 / Three GIFT input gates | `noise_000`、`noise_001`、`noise_010`；`profile=regenerated`，`full=True`，全部通过 |
| 训练及 CUDA / Training and CUDA | 未训练，未初始化 CUDA；未包含 PINN 初始化权重 |

集合 manifest SHA256：`b3fb1f601c209ce6163ffce61dc786284027abf683c94be0aa350ca9dec5b2e2`。

## 明确保留的边界 / Explicit limits

- 本次集合未纳入五个槽位：`cross-resolution`、`fno-test`、`fno-training`、`fno-training-coarse`、`short-test`。这是装配时的快照，不推断其他独立任务的当前进度；未使用发布版真值填补缺口。
- 三个 GIFT 检查是实际完整输入检查，不是训练、前向预测或实验数字验收。采样 NPZ 被纳入集合并通过装配检查，不等于 PINN 训练或初始化已通过。
- 原集合 manifest 的 `training_profile_integration_verified=false` 保持原样；其后的三个 GIFT 输入检查结果另记于 audit，不推广为所有训练接口均已验证。
- 新 standard 相对发布版 clean 已记录的数值差异仍保留。`complete_six_experiment_collection=false` 和 `experiment_numeric_acceptance=false` 均未改变。

## 证据格式 / Evidence format

[result.json](result.json) 保留原 audit 和集合 manifest 的全部解析字段与数值，并绑定原 `START.json`、`END.json`、`audit.json`、`assemble_and_check.py` 和集合清单的 SHA256。仅 START 命令中的机器绝对路径替换为 `<WORK>` / `<PROJECT>` 逻辑标签，并统一路径分隔符；这是结构化便携导出，不是原 JSON 的逐字节副本。原始运行 UUID 保留。

本目录只有 README、结果和 [manifest.json](manifest.json)。数据数组、源码、检查点及原始日志不在此证据包内。本次整理只核对小型收据、清单及七份源码哈希，没有重复装配、输入检查或大数组哈希计算。

English summary: Real newly generated standard/noise/three-sampling data were assembled, and all three full regenerated GIFT input gates passed. The collection contains 17 manifest entries and still explicitly lacks five job slots. No training, CUDA initialization, PINN initialization, full six-experiment collection, or numerical experiment acceptance is claimed. The portable result preserves the recorded audit and manifest values; only machine-specific START command paths are relabeled.
