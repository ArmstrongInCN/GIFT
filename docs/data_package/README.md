# GIFT 数据集 / GIFT dataset

本数据包汇集当前 W1、W2、W3、S1、S2、S3 使用的观测输入和正式结果的数值源数据。场数组轴统一说明为 `[trajectory,time,y,x]`。模型权重、训练恢复状态、外部算法源码和完整预测场不在本包内。

## 目录与用途

| 目录或文件 | 内容 |
| --- | --- |
| `prediction/four_vortex/` | 四涡初值的基础观测；供独立 N1000 辅助协议、GIFT-Lite 及 S1/S2 使用，也是 W1/W2 训练选择的父数据 |
| `prediction/w1_w2/` | 当前 W1/W2 的 N300、N50 训练切片和 N64/N96/N128 确认真值 |
| `identification/w3/` | 当前 W3 的 50 组无噪声参数、每组一条 ID0 轨迹及对应 PINN 采样 |
| `s3_gaussian/` | 当前 `gaussian_s3` 身份的 Gaussian 观测和原生成记录 |
| `source_data/W1/` 至 `source_data/S3/` | 六个正式实验的 CSV、JSON 和必要 NPZ；包括逐例结果、汇总和绘图源数据 |
| `metadata/` | 数组结构、样本划分、来源和实验覆盖表；W1/W2 原锁定记录位于 `metadata/provenance/w1_w2/` |
| `dataset.json` | `gift.dataset.v1` 路径角色表，供代码定位新版文件 |
| `manifest.json` | 当前包的文件大小和 SHA-256 清单 |

文件字段见 `DATA_DICTIONARY.md`；机器可读定义见 `metadata/schema.json`、`metadata/splits.json`、`metadata/provenance.json` 和 `metadata/experiment_coverage.json`。`source_data` 内保留的原结果或绘图清单记录历史来源，当前包成员以根 `manifest.json` 为准。

## 样本划分

当前 W1/W2 从基础训练池 IDs 0–999 中按固定随机顺序选择 300 条，validation 为 IDs 1000–1019 共 20 条；GIFT-Lite 使用这份有序名单的前 50 条，不能误解为基础池的 IDs 0–49。训练观测为 t=5–8、dt=0.04 的 76 个真实节点。

确认人口 IDs 100000–100179 和探索人口 IDs 1040–1219 各 180 条。W1 评价 t=5–8 的 75 个未来节点，W2 在 N64/N96/N128 评价 t=5–6 的 25 个未来节点；都排除输入初态。确认人口独立于训练和 validation，但成绩已经被查看，不作为新盲测。

独立 N1000 辅助协议使用训练 IDs 0–999、validation IDs 1000–1039、test IDs 1040–1219；其中完整时段 validation 为 1000–1019，`validation_aux` 为 1020–1039。其 GIFT-Lite 使用基础训练池 IDs 0–49。相同 ID 在不同网格上表示相同物理初值，validation 和 test 不进入训练。

W3 的 `parameter_000`–`parameter_049` 各含一条对应方程参数的无噪声轨迹、局部 ID0，无外部 validation；50 条轨迹不使用预测实验的全局 ID。旧 0%/1%/5% 噪声协议不属于当前 W3。

## 使用与版本

将 `GIFT_DATA_ROOT` 指向数据包根目录。本机新版位置为 `D:/GIFT/GIFT-dataset`，旧版完整备份为相邻的 `GIFT-dataset_legacy_20261020`。在 GIFT 项目根目录运行：

```shell
python -m scripts.verify_data --root /absolute/path/GIFT-dataset
```

新版通过角色表定位输入；没有 `dataset.json` 的旧包仍按旧布局读取。显式传入 HDF5 路径的命令应使用新版路径。重新生成数据与训练模型是两个独立步骤，参见项目的 `docs/DATA_GENERATION.md` 和 `docs/SETUP.md`。

数据许可为 CC BY 4.0，见 `LICENSE.txt`。既有 ScienceDB 数据记录为 *GIFT Navier–Stokes input data*，DOI `10.57760/sciencedb.013lo`。本次重整只生成本地版本，尚未上传或更新 ScienceDB 记录；引用时请核对实际使用的记录版本。