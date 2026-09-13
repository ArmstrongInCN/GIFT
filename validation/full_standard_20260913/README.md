# Full standard clean generation / 完整 standard 清洁数据生成

**Integrity: PASS. Field comparison: DIFFERENT. Numerical acceptance: not assessed.**
On 2026-09-13, the independent CPU entry point completed all **270 trajectories,
37,270 saved frames and 152,657,920 float32 values** for the standard N64 dataset.
Generation constructed the declared initial conditions and integrated from
t=0; it did not read a released solution field or model checkpoint as input.
A separate read-only audit subsequently compared every stored field value
with the released standard dataset. All values were finite, but
**152,326,896 values had different float32 bit representations**.

**完整性通过，场数据存在差异，未据此判定数值验收通过。** 本次完成270条轨迹、
37,270帧、152,657,920个float32值的完整生成，并逐值核对参照。生成从明确初值
出发，不读取旧解场；参照文件仅在生成结束后的独立只读比较中使用。
完整人口、文件完整性及续算成功不等于“100%复现”。

## Independent phases and full coverage / 独立阶段与完整覆盖

The first process performed 2,101 batched advances and paused after the complete
training batch plus step 101 of the validation batch. It took **228.3664083 s**.
A separate process restored only the same attempt's hashed complex64 spectrum
and cursor, completed the remaining 3,899 advances, and took **1,311.0347891 s**.
The two measured phase durations sum to **1,539.4011974 s** (about 25.66 minutes),
excluding the gap between processes. The independent audit took **12.437 s**;
it did not reintegrate the equations or import PyTorch/TensorFlow.

两阶段使用同一任务UUID与绑定源码、参数、CPU环境；暂停状态为validation批次
step101，终态游标为全部3批已完成。两次进程的实际运行耗时相加，不把进程间
等待时间计入训练或积分耗时。检查点完整性已核对，但未由验证器重复推进全部频谱状态。

| Group / 分组 | Trajectories and IDs | Saved frames per trajectory |
| --- | --- | --- |
| Training | 50; IDs 0–49 | 501; t=0:0.02:10 |
| Validation | 20; IDs 50–69 | 501; t=0:0.02:10 |
| Test | 200; IDs 1000–1199 | 11; t=5:0.5:10 |

Each batch integrated 2,000 steps with dt=0.005, preserving the default batches
50/20/200: 6,000 batched advances and 540,000 trajectory-steps in total.
The source/runtime/owner/completion/checkpoint bindings, all IDs and times,
field shapes/dtypes/finiteness, solver metadata and independently reconstructed
initial parameters passed the recorded checks. The released standard H5 has
no initial-parameter datasets, so parameter equality is to the declared seed
protocol, not to nonexistent reference arrays.

## Measured differences / 实测差异

These are descriptive errors, **not a new acceptance threshold**. Each group's
aggregate relative L2 is computed over all its stored values as
`sqrt(sum((generated-reference)^2) / sum(reference^2))`, not an average or
maximum of per-frame relative errors.

| Group / 分组 | Aggregate relative L2 | Maximum absolute error |
| --- | ---: | ---: |
| Training | 0.0014927738753393398 | 0.5047175884246826 |
| Validation | 0.001538170946088378 | 0.32030296325683594 |
| Test | 0.002302492563107604 | 0.8025531768798828 |
| All stored values / 全部保存值 | 0.001559317406513346 | 0.8025531768798828 |

The global maximum absolute error, **0.8025531768798828**, occurs in test
trajectory **ID 1118 at t=10**. That particular frame's relative L2 is
**0.03232557445371127 (about 3.2326%)**. This is the relative error of the
maximum-absolute-error frame; it is **not reported as the global maximum
per-frame relative L2**.

全量最大绝对差出现在test ID1118、t=10；约3.2326%仅为该帧的relative L2，
不是“全局最大relative L2”。三个分组的聚合误差采用各组所有保存值计算，
不能与单帧误差或实验报告中的模型预测误差混用。

The earliest observable stored-field difference is already at **training ID 0,
t=0**, before integration: the frame's maximum absolute error is
`1.1920928955078125e-6` and relative L2 is `1.6653269235177098e-7`.
This is consistent with the earlier bounded t0 observation; it does not locate
the first differing internal arithmetic operation. The test reference starts
at t=5, so this audit cannot compare a test trajectory's t=0 field.

最早可观测差异已在training ID0的t0保存场出现，不能把后续增长单独归因于
积分步骤，也不能由当前证据确定最先不同的是哪一个底层浮点运算。
本次未调整精度、容差、参照值或观测点。

## Evidence and scope / 证据与范围

[result.json](result.json) is a **byte-for-byte copy of the original completed
audit result**, without rewriting its verdict, measured values or historical
file hashes. Its SHA256 is
`01AD8793E9E796FE62B3A6D44BB036E64239934AD11A28A65D04507ACE3661C1`.
[manifest.json](manifest.json) binds that result and this README.

The complete local per-frame CSV is 4,772,128 bytes, SHA256
`111DF963050ACC3A524A7385373DA47067989B08B933BA9E4A504D02E88961F6`.
It is **not included in Git**, nor are generated H5 data, parameter arrays,
spectral states or raw execution logs. Their hashes and metadata are recorded
in the result; those relative labels identify evidence, not download paths.

本目录仅分发说明、原始核验摘要和清单，不分发逐帧CSV、H5、初值数组、频谱状态
或原始运行日志。原数据包未被生成文件替换；原实验容差及模型数值失败仍保留。

This record covers the full **standard clean** dataset only. It does not prove
completion of other dataset generators, a full regenerated collection,
training on regenerated data, cross-device equality or the six experiments'
scientific acceptance. New provenance and parameter metadata also mean the
generated H5 is not a byte-identical replica of the historical container.
See [generation instructions](../../docs/DATA_GENERATION.md) and
[overall reproducibility status](../../docs/REPRODUCIBILITY_STATUS.md).
