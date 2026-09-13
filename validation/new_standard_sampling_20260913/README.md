# 新 standard 的三条件采样 / Sampling from the new standard

2026-09-13，三个条件分别调用采样入口，均正常退出（rc=0）。独立 CPU 核验重算每条件全部10个数组，逐比特核对其自身新观测输入下的设计和 `u,v,omega` 目标，并检查完整设计 RNG 状态、schema、有限性、输入/输出哈希与完成身份：**完整性和独立公式核验通过**。

Three independently invoked condition jobs completed successfully. The CPU audit replayed all 10 arrays per condition and checked their bytes against an independent calculation from each job's own new observations, along with the full design RNG state, schema, finiteness, hashes and completion identity. This is an **integrity/formula pass**, not agreement with the released reference targets.

| 条件 / Condition | 子进程耗时 / Child seconds | 与发布缓存相同的设计数组 / Exact design arrays | 目标 max abs | 目标 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| `noise_000` | 2.1823325000004843 | 9/9 | 0.23547601699829102 | 0.002803406924103472 |
| `noise_001` | 1.4829630000167526 | 9/9 | 0.23547601699829102 | 0.0028031213252440795 |
| `noise_010` | 1.4200055999681354 | 9/9 | 0.23547887802124023 | 0.002790153181895904 |

三条件的9个设计数组均与各自发布缓存逐比特一致；三个 `targets` 数组均**不同**。上表差异按全体目标值以float64计算，relative L2分母为对应发布目标的L2范数，没有新增容差或将差异判作通过。耗时只含采样子进程，不含之后的独立核验。

All nine design arrays matched each corresponding released cache bitwise; **all three target arrays differed**. Differences use float64 over the complete target arrays, with the released-target norm as the relative-L2 denominator. No new tolerance was applied. Timings cover the sampling subprocesses, not the subsequent audits.

新 clean 祖先的 SHA-256 为 `8def9a45e0faf192ab6df488a95f7ae0817a05d6609b36c736a2b7d5f03a6260`，不是旧 clean。`noise_000` 直接使用该新 standard；1%与10%条件分别使用由它派生的新噪声文件。采样入口只读取各自输入，不用旧缓存目标生成新目标；发布缓存只在结束后的审计比较阶段读取。此处完整性通过不抹掉新 clean 与原发布场的数值差异。

The clean ancestor is the newly generated standard with the SHA above, **not the old clean input**. The other conditions use its newly derived noisy observations. Released targets were read only after generation for audit comparison, never supplied as generation inputs. Correct sampling does not erase the new clean field's numerical differences from the released field.

边界：本次采样不是重新积分、模型初始化或训练；没有宣布M1或六实验通过。**新噪声父文件的全量独立核验已另行通过**：完整性PASS、全部287,293,440个值与其自身新clean输入下的噪声公式逐比特一致，耗时12.890秒；见[独立父文件证据](../new_standard_noise_20260913/README.md)。这不是与旧噪声真值一致的声明，也不是由采样审计替代父文件核验。运行时为Python3.10.19、NumPy2.2.6、h5py3.16.0；CPU审计未导入Torch/TF。此次整理只读已有小收据，未重新读取H5/NPZ数组或重跑核验。

No integration, network initialization, model training, M1 or whole-project acceptance is claimed. **Separate full noisy-parent verification has now passed**: integrity PASS and exact formula bits for all 287,293,440 values, in 12.890 seconds; see the [independent parent evidence](../new_standard_noise_20260913/README.md). This is agreement with noise derived from the new clean input, not old noisy reference fields. The raw parent-verification result is bound in result.json by SHA-256 `0cea46af35b54dc03a2369b1d70e9a72886260bb0322a62453f457d644a9adc0`. Packaging read small receipts only and reran no numerical checks.

[result.json](result.json) 保留原始数值、输入/输出与源码哈希、绑定哈希及18份原始收据的SHA；工作机绝对路径和运行UUID未复制。逻辑收据名指向操作者归档，不是随此目录发布的文件。此目录仅含README、结果和清单，不含NPZ、H5、权重、原始日志或临时启动脚本。生产入口见[数据生成说明](../../docs/DATA_GENERATION.md)。

The portable result retains source/input/output/binding hashes and SHA-256 identifiers for 18 raw receipts held in the operator archive. Logical receipt names are archive aliases, not distributed files. No arrays, weights, machine paths, run UUIDs, raw logs or temporary launcher code are included. [manifest.json](manifest.json) inventories the two payload files; its own hash is reported separately to avoid circular self-hashing.
