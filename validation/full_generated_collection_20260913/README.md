# 完整再生数据集合 / Full generated-data collection

**独立输入完整性检查通过；实验数值验收尚未通过。** 2026-09-13，十个已完成的数据任务组装为新的独立集合，包含32个清单条目、13,155,083,047字节（不含清单本身）。它没有替换下载数据包，也没有进入Git仓库。

**Independent input integrity passed; experimental acceptance is not established.** All ten generation jobs are present, with no missing job slot. Native assembly took 88.000 s. The separate read-only audit then took 187.218 s; it neither assembled/copied the arrays again nor retrained a model.

| 已实际检查 / Executed check | 覆盖 / Coverage |
| --- | --- |
| 集合文件与来源 / Files and provenance | All 32 manifest-file hashes, 19 scientific-source hashes and 20 parent run/completion receipts |
| 原生完整数据检查 / Native full data checks | standard, FNO-training/test, coarse/short/cross, paired noise and three sampling jobs |
| 三项整数帧派生 / Three integer-frame derivations | Same newly generated dense parents; all selected float32 bits verified, no interpolation or new integration |
| GIFT输入 / GIFT inputs | clean, 1% and 10%; real `regenerated`, `full=True` |
| 基线输入 / Baseline inputs | FNO-2D, FNO-3D, U-NO and U-Net; each formal configuration and full input function, not training CLI execution |
| PINN输入 / PINN inputs | Sampling schema/design/parent checks only; runtime input gate not run |

集合清单SHA-256 / Collection manifest SHA-256:
`ab4da506dfa9790ca8c6c2df1d54c80b5cbb8b28a038ea2b433620f9b7da3e54`.

## 保留失败记录 / Preserved failed wrapper

首次工作目录执行器启动后发生并发文件更新，其最终身份保护检查拒绝该运行（rc1）。原生组装已经成功完成，输出保留；未恢复旧执行器、改写收据或把失败改成通过。独立核验使用另一份冻结脚本，只读检查现成集合，并证实科学源码、父收据与清单未变。两次状态分别记录：`original_wrapper_identity=FAIL_CHANGED`、`original_wrapper_run_accepted=false`，而本次独立核验为PASS。

The original work-only wrapper changed after launch and failed its final identity guard. Its failure remains a failure. The independently frozen verifier inspected the retained complete collection without rewriting the old journal, restoring the wrapper, copying arrays again or rerunning generation. This is separate input-integrity evidence, not acceptance of the failed wrapper invocation.

## 边界 / Limits

- No model construction, forward prediction, training, checkpoint restore, CUDA initialization or TensorFlow import was performed. Baseline input checks do not validate upstream architectures or U-Net's B7 training startup.
- PINN's runtime still requires the released sampling/data identities and separate initialization; the regenerated profile is unsupported there. No initialization is included or loaded, and its redistribution question remains unresolved.
- The new fields differ from released fields, as recorded in the [standard](../full_standard_20260913/README.md), [FNO-training](../full_fno_training_20260913/README.md) and [FNO-test](../full_fno_test_20260913/README.md) comparisons. This input audit neither repeats those comparisons nor removes their differences. No tolerance was changed.
- Extra950 parameters are declared initial conditions; their original random seed remains unknown. Complete job inventory is not complete six-experiment acceptance or training acceptance on new data.

本目录只有README、[result.json](result.json)和[manifest.json](manifest.json)。结果文件是独立audit的逐字节副本，保留原执行器失败及全部输入检查记录；其中UUID只标识实验任务，不是机器或人员。逻辑路径及哈希用于关联操作者保留的证据，不表示相应数据或代码随此目录上传。原始数组、日志、检查点、执行器和验证脚本均未纳入。

Only these three evidence files are distributed. The result is a byte-exact copy of the completed independent audit; relative labels identify undistributed evidence. No scientific arrays, logs, checkpoints, work-only launchers or verifier code are bundled here. See the [generation guide](../../docs/DATA_GENERATION.md) for the public independent commands.
