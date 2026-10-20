# 数据字典 / Data dictionary

## 共同字段

| 字段 | 类型、轴与含义 |
| --- | --- |
| `vorticity` | float32，`[trajectory,time,y,x]`，二维涡量场 |
| `time` | float64，`[time]`；绝对时刻或文件声明的相对时刻 |
| `absolute_time` | W1/W2 训练切片和确认真值的 float64 绝对时刻，`[time]` |
| `trajectory_index` | int64，`[trajectory]`，文件所属协议的轨迹 ID |

空间域为周期区域 `[0,2π)²`，不重复端点；空间轴依次为 y、x。未指定 SI 换算。NaN 和 Inf 不代表缺失值，观测数组应为有限数。数组组名、shape、dtype 见 `metadata/schema.json`；轨迹名单和保存时刻概况见 `metadata/splits.json`；完整时间轴与原始属性保留在各 HDF5 中。

## 四涡基础观测

下表路径均以 `prediction/four_vortex/` 为前缀。基础训练 IDs 0–999，validation IDs 1000–1039，test IDs 1040–1219；`validation_aux` 是 1020–1039，不是额外 test。独立辅助协议的 GIFT-Lite 使用 IDs 0–49。

| 文件 | 观测内容 |
| --- | --- |
| `train1000_n64_t0_t10_dt0p02.h5` | 1,000 条训练及 20 条完整 validation；N64，t=0–10，dt=0.02 |
| `train1000_n64_t0_t10_dt0p1.h5` | 同一训练及完整 validation；N64，t=0–10，dt=0.1 |
| `standard_n64.h5` | 50 条训练及 20 条完整 validation；180 条 test 和 20 条辅助 validation 在 t=5–10、dt=0.5 的观测 |
| `test_n64_t4p1_t6_dt0p1.h5` | 180 条 test 和 20 条辅助 validation；N64，t=4.1–6，dt=0.1 |
| `test_n96_n128_t4p1_t6_dt0p1.h5` | 配对 N96/N128 test 和辅助 validation；t=4.1–6，dt=0.1 |
| `test_n64_n96_n128_dt0p02.h5` | 配对 test 和辅助 validation；N64 为 t=4.1–8，N96/N128 为 t=4.1–6，dt=0.02 |

当前容器使用 canonical 轨迹编号。编号整理与目录重整不对观测插值、平滑、缩放、舍入或重新积分。`metadata/initial_conditions_extra950.json` 保存 IDs 50–999 的四涡初值参数和积分配置；每个初值为四行 `[circulation,sigma,cx,cy]`，供独立再生成使用。

## 当前 W1/W2

以下路径以 `prediction/w1_w2/` 为前缀。N300 名单为 `default_rng(2026100304).permutation(1000)[:300]`；N50 是该有序名单前 50 条。两份切片的 validation 均为 IDs 1000–1019。

| 文件 | 组、shape 与保存时刻 |
| --- | --- |
| `train300_validation20.h5` | `training/vorticity` 为 `[300,76,64,64]`，`validation/vorticity` 为 `[20,76,64,64]`；t=5–8，dt=0.04 |
| `train50_validation20.h5` | `training/vorticity` 为 `[50,76,64,64]`，`validation/vorticity` 为 `[20,76,64,64]`；t=5–8，dt=0.04 |
| `confirmation_n64.h5` | `test/vorticity` 为 `[180,151,64,64]`；IDs 100000–100179，t=5–8，dt=0.02 |
| `confirmation_n96_n128.h5` | `N96/test/vorticity` 和 `N128/test/vorticity` 分别为 `[180,51,96,96]` 和 `[180,51,128,128]`；同一确认 ID，t=5–6，dt=0.02 |

正式评价从 dense 真值每隔一个整数帧取样，W1 得到 76 个节点，W2 得到 26 个节点；不做时间插值。W1/W2 切片及确认容器的 `absolute_time` 为表中绝对时段，`time` 从零起算（训练切片及 N64 确认为 0–3，高网格确认为 0–1）。探索人口沿用基础观测的 test IDs 1040–1219。W1 七方法、W2 六方法的权重与完整预测场不在本包内。原锁定记录和数据 receipts 保存在 `metadata/provenance/w1_w2/`，其中历史绝对路径仍是原来源记录。

## 当前 W3

`identification/w3/parameters.csv` 保存固定的 50 组真值，`INITIAL_CONDITIONS.json` 保存初态，`DATA_COHORT_QA.json` 保存群体核查。`parameter_000`–`parameter_049` 各自保留：

- `standard_ns_n64_full_spectrum.h5`：一条局部 ID0 轨迹，float32 `[1,501,64,64]`，t=0–10，dt=0.02。
- `DATA_RECEIPT.json`、`DATA_LOCK.json`、`DATA_QA.json`、`OBSERVATION_PRESERVATION.json`：真值、数组身份及来源绑定。
- `auxiliary/pinn_sampling/noise_000_seed1234.npz` 和对应 `.receipt.json`：该 case 的固定 PINN 采样；`noise_000` 是采样文件保留的无噪声条件名。

方程为 `ω_t = -β(uω_x + vω_y) + νΔω − 4γcos(4y)`。ν 范围为 [0.0075,0.015]，β、γ 为 [0.75,1.25]；精确组合以参数 CSV 为准。各 case 一条训练轨迹、无外部 validation，不能将不同 case 的局部 ID0 合并为同一预测轨迹。

每个 case 的 PINN 采样包含 30,000 个实测 `(x,y,t)` 位置及 `(u,v,ω)` 目标，按 24,000/6,000 行分为训练/同轨迹 holdout；另有 60,000 个 LHS 位置，物理点总数为 84,000（包含 24,000 个训练位置）。holdout 行不是额外轨迹。GIFT 使用固定终态 checkpoint；模型及训练状态保留在独立运行目录中。旧 0%/1%/5% 噪声数据不属于当前包。

## S3 Gaussian 初值

`s3_gaussian/trajectories.h5` 和 `lite.h5` 使用当前 `gaussian_s3` 身份，保留原观测数组；网格 N64，t=0–10、dt=0.02，共 501 帧。`trajectories.h5` 含 1,000 条训练、20 条 validation、20 条辅助 validation 和 180 条 test；`lite.h5` 含对应训练池前 50 条及 20 条 validation。训练 IDs 为 1220–2219，validation IDs 为 2220–2259，test IDs 为 2260–2439。`COMPLETE.json`、`manifest.json`、`run.json` 保存原生成记录。完整字段和划分以本包元数据为准。

## 数值源数据与来源

`source_data/{W1,W2,W3,S1,S2,S3}/` 保存正式结果的 CSV、JSON 与必要 NPZ，供数值核查和重绘。W3 仅保留当前参数分布图的源数据，不将旧联合参数图作为当前图4。包内没有 SVG/PNG 成图、完整预测场或训练恢复状态。原结果/绘图 manifests 仍描述其原始产物；本次发布文件成员与哈希以根 `manifest.json` 为准。

基础四涡观测满足 `ω_t + u·∇ω = 0.01Δω − 4cos(4y)`。原生成采用 full-spectrum Fourier pseudospectral ETDRK4、内部 dt=0.005，N64/N96/N128 的 padding 分别为 99/147/195，不截断状态谱模态。数据目录重整不改变这些数值约定或历史科学结果。
