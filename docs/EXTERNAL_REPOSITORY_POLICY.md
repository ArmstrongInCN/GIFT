# 外部算法边界与一致性验证

GIFT 的当前工作文件不得保存外部算法源码、复制的算法实现、可执行源码补丁或自动源码改写器。官方仓库、固定版本与必要的手工适配操作见 [外部算法适配说明](EXTERNAL_MANUAL_ADAPTATION.md)。`adapters/__init__.py` 仅提供通用路径解析、模块连接和发布元数据处理；外部方法的网络、求解器、源码加载与适配实现均须位于仓库之外。该要求同样适用于 checkpoint、NPZ、JSON 和 ZIP 内的源码文本，不能仅检查 `.py` 文件。

运行时通过 `GIFT_EXTERNAL_ADAPTER_ROOT` 指定使用者准备的外部接口目录，缺省为相邻 `gift-external-runtime/`；该目录及其实际解析路径不得位于 GIFT 内。重新取得 GIFT 的使用者需要在仓库外下载官方代码、按文字说明适配并提供接口。训练入口、评价协议及 GIFT 自身算法仍保留在 GIFT 中。

## 当前发布的边界

| 对象 | 当前处理与限制 |
| --- | --- |
| 当前源码工作树 | 只保留仓库引用、固定版本元数据、文字适配步骤和通用接口；具体外部实现仍由使用者在 GIFT 外准备。 |
| 新模型公开副本 | [源码元数据清理版 Release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020-source-free)提供 `GIFT-current-models-20261020-source-free.zip`，清理源码元数据，逐张量验证数值完全不变。[清单](CURRENT_MODEL_BUNDLE.json)记录直接下载地址及身份。 |
| 原始权重和实验记录 | 保持只读，不用新公开副本哈希覆盖原测量身份，不更改实验数值。 |
| 旧线上模型附件 | `GIFT-current-models-20261020.zip` 的 ANIE checkpoint 仍含四段 `before`/`after` 源码字符串，共441字符；新发布已取代旧包，旧附件作为来源记录保留，未物理删除。 |
| Git 历史及旧标签 | 旧 `main` 提交和 `reference-checkpoints-20260913` 标签仍可访问七个旧适配文件；本次未改写历史或移动旧标签。 |

新的普通提交只会记录新的树，不能从旧提交、旧标签或已经发布的附件中移除代码。因此当前文件检查通过只能说明相应工作文件或所指定附件通过，不能宣称整个 Git 历史或所有旧发布均无外部代码。如需从全部历史版本中移除旧代码，须另行制定并执行历史处理方案后复核。

ANIE 在固定提交 `2071ad709b260a6c305450affcf7168fd825f692` 未找到明确许可，当前也没有额外书面授权。来源引用和元数据清理解决的是披露与内容边界问题，不授予使用、修改或分发权利。固定来源见 [external_sources.json](../external_sources.json) 和 [anie_sources.json](../anie_sources.json)，原始研究引用见 [README](../README.md#references)。

## 公开模型副本

`scripts.package_current_models export` 默认检查并清理待公开 checkpoint 的源码元数据，仅向新的输出目录写入。没有需要清理字段的文件保持原字节复制；清理字段的文件另行序列化，并逐张量核对 dtype、shape 与值。`ORIGINS.json` 的模型条目分别记录来源文件与公开副本的字节数及 SHA256、`removed_source_fields` 和 `tensor_values_unchanged`；顶层 `tensor_values_changed` 为 false，并记录是否移除了源码元数据。张量等价在导出时实际核对；`verify` 检查公开副本的嵌套内容及清单绑定，不要求接收者拥有原训练机器上的来源文件。

原文件哈希、原测量哈希和公开副本哈希是三种可能不同的证据身份。元数据清理后的 ANIE 公开副本不能宣称与原文件逐字节相同；其模型张量保持逐位一致，也没有通过该操作取得额外许可。公开包不包含训练数据或完整 optimizer 恢复日志，使用方式见 [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md)。

## 2026-10-07 迁移的一致性证据

使用迁移前保存的正式权重、固定输入和独立 CPU 进程检查迁移后的运行结果：

| 验证对象 | 验证范围 | 结果 |
| --- | --- | --- |
| FNO2D、FNO3D、ANIE | 同一正式权重的完整 76 点预测 | 输出逐位一致，最大绝对差为 0 |
| U-NO、U-Net | 同一正式权重的两步预测 | 输出逐位一致，最大绝对差为 0 |
| 五种外部网络 | 参数数量、state tensor、来源身份与 checkpoint SHA256 | 一致 |
| 移出的 10 个模块 | 函数、类及算法常量的 AST 对照 | 一致；只调整项目根目录及来源文件位置 |
| ANIE 原 EarlyStopping | 1,011 个 loss 组成的序列，包括改善、相等及 patience=1000 | best、counter 与停止判定一致 |
| 正式权重加载 | 原模型规格、seed、epoch 和元数据校验 | 通过 |
| 465 个科学结果文件 | 数组、数值记录、权重、图片及引用哈希 | 431 个文件字节不变；34 个 JSON 仅移除源码片段元数据并更新相应文件哈希及字节数 |
| 14 个正式结果包 | 发布清单、文件哈希和结果包引用 | 通过 |

上述迁移检查覆盖的结果 JSON 已移除源码补丁原文。移除范围仅限补丁说明的 `before`、`after` 字段，及受这些文件字节变化影响的哈希、字节数引用；实验指标、训练设置、预测数组与图像没有重算或修改。该次检查未涵盖后来模型 ZIP 内的 checkpoint 字符串，也未清除 Git 历史。原始元数据、原始运行记录与 `external_boundary_20261007` 迁移验证证据保存在仓库外。

这些检查支持本次目录整理没有改变已验证的程序计算与既有实验结果。它们包括 CPU 固定输入测试和算法结构对照，没有重新执行完整 GPU 训练，也不声称证明所有设备和所有输入下的逐位确定性。

## 维护检查

从 GIFT 根目录执行 `python -B -m scripts.audit_repository`，检查当前拟发布工作文件的边界。目录中的忽略文件和缓存另作实际文件扫描，不能仅依赖 Git 文件选择。附件必须单独指定，避免只扫当前文本而遗漏 ZIP 内的 PT 或 NPZ：

```shell
python -B -m scripts.audit_repository --public-file /exports/GIFT-current-models-20261020-source-free.zip
python -B -m scripts.audit_repository --history-risk-ref reference-checkpoints-20260913
```

第二条命令只读报告指定历史引用中的风险，不清理历史。需要全面检查时应覆盖全部相关分支和标签；单个引用的结果不能代表未检查的历史。检查操作不会暂存、提交、上传或删除文件。旧适配文件从工作树移出后，其移除记录必须与通用接口、说明文档一起纳入后续的拟发布提交；仅提交新增说明不能改变旧的文件选择。
