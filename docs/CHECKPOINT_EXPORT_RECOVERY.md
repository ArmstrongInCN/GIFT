# Checkpoint export recovery / 终态补导出

## 中文

独立 baseline 训练先提交完整 checkpoint，再写 `model.pt`。若后一步被断电中断，可能已保存 complete journal，却没有可用导出；训练 `--resume` 会拒绝重复执行已完成任务。此工具只补导出，不训练、不推进或修改 journal。

在项目根执行。先确认原训练进程退出，从可信记录核定**明确的不可变终态 checkpoint** 与同目录 `ATTEMPT.json` 的 SHA256；不要猜测最新文件。工具不读取 `LATEST.json`，不需要私有 wrapper 或 START/END。

```text
python -B scripts/recover_baseline_export.py
python -B scripts/recover_baseline_export.py --execute --checkpoint <own-terminal.pt> --checkpoint-sha256 <SHA256> --attempt <same-run/ATTEMPT.json> --attempt-sha256 <SHA256> --output <new-directory/model_recovered.pt>
```

第一条只输出计划，不读文件、不导入Torch、不改变CUDA/线程环境。第二条要求输出父目录已存在、文件不存在。保留旧 `model.pt`（包括残缺文件），显式指定新文件。仅处理本项目正式 UNO 150轮、FNO2D/FNO3D/U-Net 500轮的 `gift.training-boundary.v1`。

- 验证输入SHA、同目录、identity/runtime一致、完整预算/历史、基本optimizer状态/step、scheduler及RNG字段；只读CPU张量检查。CVD=-1、threads1；已初始化CUDA的调用进程会被拒绝，不构造模型、不恢复优化器/RNG、不forward。
- 写全新同目录临时文件，flush/fsync并CPU读回核对，复核输入SHA后通过硬链接原子、create-only发布；不覆盖任何目标。失败保留临时文件。需本地同文件系统硬链接支持（如NTFS）；不保证硬件断电或网络文件系统的额外持久性语义。
- 使用单独的 `gift.recovered-baseline-weights.v1` / `recovered_terminal`，不是原 fresh 导出。`fresh_training`、`same_run_resume_used` 未记录，保留 `null`；初始化文本只说明来源绑定的独立训练协议。来源与数据SHA原样保留，不重新读取数据/上游，也不伪造receipt或数值验收。
- run_id 来自SHA核定的同目录ATTEMPT；checkpoint本身**没有内部run_id**。只证明文件/SHA/identity/runtime关系，不给出更强的内部UUID保证。

现有模型读取器不锁schema，恢复文件保留其必需的status/method/terminal_epoch/config/normalization/state字段；元数据接口已用fixture验证，真实架构加载及实验验收仍需另做。严格fresh-terminal审计应保留拒绝新schema/unknown来源的边界，不自动登记为发布模型。

测试使用系统临时目录。`SYNTHETIC_NOT_TRAINING` CPU端到端fixture仅含36个小张量和150行假预算元数据，验证真实save/load、原子发布、字段/张量一致及输入SHA不变，**不是训练或科学复现证明**。

## English

Training commits its complete journal before exporting `model.pt`. A crash during export can leave an unusable/missing weights file even though the journal is complete; training resume correctly refuses to repeat a completed run. This separate tool exports the saved terminal state without advancing or changing training.

Run the commands above from the repository root, after the training process exits. Supply an explicit immutable own checkpoint and the same-directory `ATTEMPT.json`, with independently trusted SHA256 values, and a **new** destination in an existing directory. No LATEST pointer, private launcher or completion receipt is required. Default invocation is a no-read/no-environment-change plan.

Execution checks identity/runtime equality, complete formal budgets/history, basic optimizer/scheduler/RNG metadata and finite CPU tensors. It never constructs a model, restores training state, performs a forward pass or initializes CUDA. A process with CUDA already initialized is rejected. The verified temporary export is published by an atomic no-overwrite hard link; interrupted temporary files are retained. Local same-filesystem hard-link support is required; hardware-loss and network-filesystem durability are not promised.

The distinct recovery schema preserves consumer-required fields, but leaves unrecorded `fresh_training` and `same_run_resume_used` as null. Recorded source/data hashes are retained, not revalidated. The run ID comes from ATTEMPT, not from an internal checkpoint UUID. Recovery does not establish original invocation success, fresh-initialization provenance or numerical agreement with a publication.

Current consumer metadata compatibility and a small CPU serialization fixture are tested. Actual model loading and scientific acceptance remain separate. A strict fresh-artifact auditor must not silently promote this recovery product. The synthetic fixture is explicitly **SYNTHETIC_NOT_TRAINING**, not a completed scientific training run.
