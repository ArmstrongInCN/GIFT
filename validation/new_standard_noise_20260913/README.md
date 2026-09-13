# 新 standard 的完整成对噪声核验 / Paired noise from newly generated standard

2026-09-13 实际完成：完整性 **PASS**，噪声公式 **EXACT_BITS**。本目录只保存便携核验证据，不包含数据、代码、模型数组、检查点或原始日志。

本次输入是 [完整 standard 生成核验](../full_standard_20260913/README.md) 中新积分得到的 clean H5，**不是发布版 clean**。它与 [此前的成对噪声核验](../full_noise_20260913/README.md) 属于不同的 clean 数据来源，不能混用结论。

## 实际范围 / Executed scope

| 项目 / Item | 结果 / Result |
|---|---|
| 条件 / Conditions | 1% 和 10% 噪声，共用每块的标准正态随机数 |
| 每个条件 / Each condition | training 50 + validation 20；501 帧；64 × 64 |
| 完整覆盖 / Full coverage | 2,240 个独立随机数块；287,293,440 个 float32 输出值 |
| 逐位公式比较 / Formula comparison | 0 个 uint32 位模式差异；0 个非有限值 |
| 统计量 / Statistics | 两组完整 clean 数据按原顺序重算；count、mean、总体标准差、平方和全部精确一致 |
| 最终随机数状态 / Final RNG | 游标 2,240；MT19937 的 624 个状态字、位置及高斯缓存全部精确一致 |
| 实际耗时 / Elapsed time | 生成 18.8850 秒；独立核验 12.8900 秒；均返回码 0 |
| 本次续算 / Resume in this run | 无；无人工暂停，也不新增暂停续算验证声明 |

公式为 `float32(float64(clean) + eta * std(clean_group, ddof=0) * Z)`；每个至多 16 帧的块只抽取一次 `Z`，同时用于两个噪声比例。运行环境为 Python 3.10.19、NumPy 2.2.6、h5py 3.16.0；核验没有导入 Torch/TF 或执行训练。

## 证据与边界 / Evidence and limits

[result.json](result.json) 是已完成核验结果的**逐字节副本**，未重写数值、哈希或运行标识。其路径标签已是逻辑相对标签，不含个人绝对路径；保留的 UUID 是实验运行标识。文件包括源码、父数据及其完成证据、当前运行收据、最终检查点和两份输出的前后哈希，全部未变。

- 新 clean 父文件 SHA256：`8def9a45e0faf192ab6df488a95f7ae0817a05d6609b36c736a2b7d5f03a6260`。
- 本次核验原始结果 SHA256：`0cea46af35b54dc03a2369b1d70e9a72886260bb0322a62453f457d644a9adc0`。
- 本目录不含模型数组；H5、检查点、源码和日志仅以哈希绑定，未复制到本目录。[manifest.json](manifest.json) 校验本目录的两个载荷文件。

**完整性及公式通过不等于完整科学验收。** 父 standard 相对发布版 clean 的 `DIFFERENT` 结论仍保留。本次没有打开或比较旧噪声 H5，不要求不同 clean 输入产生相同旧噪声值，也不证明历史实验数字、模型训练、完整数据集合装配或跨设备逐位复现已通过。新噪声 H5 的部分旧版元数据缺省，由本次 `run.json` 绑定 clean 父文件；不声称整份 H5 与旧版相同。

English summary: This completed CPU run derives paired 1%/10% observations from the separately integrated and audited **new standard parent**, not the released clean data. All 287,293,440 output values match the independently replayed formula bit-for-bit; complete clean-group statistics and final MT19937 state also match. The raw result is copied unchanged. The parent's measured difference from released clean fields remains unresolved; no old-noise equivalence, downstream training acceptance, new tolerance, or universal reproducibility guarantee is claimed.
