<div align="center">

# GIFT

**Generator Identification via Field Tomography**
**基于场层析的生成元识别：具备物理可解释性的流体动力学代理模型**

<sub>从流场轨迹中直接学习连续时间生成元，并把代理模型的预测机制还原为可检验的偏微分控制方程</sub>

</div>

<p align="center">
  <img src="assets/figures/gift_architecture.png" alt="GIFT architecture and prediction process" width="100%">
</p>

<p align="center"><sub><b>图 1｜GIFT 架构与预测过程。</b>(A) 稠密采样的连续流场轨迹。(B) 傅里叶变换与固定频带分解构造训练数据：<i>P</i><sub>21</sub> 分量进入主路径，<i>Q</i><sub>21</sub> 高频分量进入高频支路。(C) 场层析：利用各通道振幅响应差异，交替学习二次通道、线性通道与偏置场通道。(D) 高频支路。(E) 二次场相互作用单元。(F) 生成元经四阶 Runge–Kutta 积分推进至下一时刻。</sub></p>

## 核心思想

传统代理模型直接预测固定时间间隔后的流场；GIFT 换一个对象——它学习决定流场**瞬时演化**的**连续时间生成元**：

```
G(ω) = C + A(ω) + Q(ω, ω)
```

有限时间间隔后的状态只是生成元的时间积分结果，而生成元本身对应系统的控制规律。因此代理模型的预测与显式 PDE 参数识别可以共享同一个数学对象：预测机制不再是藏在网络权重里的黑盒映射，而是一个可以被检验物理正确性的连续动力学对象。

> Rather than predicting the field at a fixed time lag, GIFT learns the **continuous-time generator** that governs instantaneous evolution. A finite-lag state is merely its time integral, and the generator *is* the governing law — so prediction and explicit PDE-parameter identification share one checkable object instead of a black box.

## 方法要点

| 环节 | 内容 |
| --- | --- |
| 生成元分解 | 生成元拆为与状态无关但可随空间变化的偏置场 `C`、平移等变线性算子 `A(ω)` 与非线性算子 `Q(ω, ω)`；状态取二维不可压缩 Navier–Stokes 的涡量场。 |
| 场层析 | 线性算子的平移等变性与非线性算子的二次齐次性是不同特征响应；按此差异**交替训练**二次通道与线性、偏置场通道，把它们从数据中分离并重建，训练中无需显式构造齐次特征式。 |
| 二次场相互作用单元 | 非线性由结构化场算子而非激活函数提供：同一输入场经两个可学习谱滤波器后逐点相乘，再对各单元加权求和（图 1E）。 |
| 高频支路 | `Q`<sub>21</sub> 高频分量交由单独训练的网络支路预测，生成元训练结束后训练，两部分耦合积分（图 1D）。 |
| 递归局部修正 | 每个完整 RK4 步后执行的固定规则：按幅值尺度 `ℓ = 2.1σ` 裁剪去均值场的局部中值残差，并设异常网格点安全上限 `K`<sub>N</sub>（`N`=64、96、128 时分别为 40、90、160）。 |
| 物理可解释性检验 | 生成元冻结后，用已知 N–S 方程项拟合各通道输出，将读出系数与真实控制方程系数对比——对"是否学到正确物理"的可量化检验。 |

## 主要结果

以下均为本仓库当前协议下**完成并发布的实测结果**：1,000 条 `N`=64 训练轨迹（`t`=0.0–10.0，间隔 0.02），另保留使用其中前 50 条的 GIFT-Lite（Lite 表示训练数据减少，不表示网络更小）；测试集为未参与任何方法训练的 **180 条轨迹**（1040–1219），每个预测时刻单独评价。

### M1｜方程参数识别

真实系数 `(ν, β, γ) = (0.01, 1, 1)`，真实系数不参与训练。

| 噪声 | GIFT ν APE | GIFT β APE | GIFT γ APE |
| --- | ---: | ---: | ---: |
| 0% | **0.484%** | **0.466%** | **0.030%** |
| 1% | **4.361%** | **0.877%** | **0.040%** |
| 10% | 64.852% | 19.778% | **0.783%** |

0% 与 1% 噪声下，GIFT 对三个参数的绝对百分比误差（APE）均为五种配置中最低；10% 噪声下 `ν` 显著退化——当前场层析过程无法在高噪声下准确分离非线性项。

> ⚠️ 该能力属于**已知控制方程结构下的参数识别**，不是在未知候选结构中从头发现任意 PDE，不能替代 PDE-FIND 或 PINN-SR 所面向的"结构与参数均未知"任务。

### M2｜递归预测（`N`=64，`t`=5.0 → 8.0）

GIFT 与 GIFT-Lite 从 `t`=5.0 的真值状态出发以 Δ`t`=0.02 递归积分；两个 FNO 基线使用 `t`=4.1–5.0 的 46 帧历史状态。

| 时间 | GIFT | GIFT-Lite | FNO-2D | FNO-3D | U-NO | U-Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.5 | **0.021901** | 0.036061 | 0.221038 | 0.315175 | 0.030129 | 0.233574 |
| 6.0 | **0.021058** | 0.040763 | 0.271467 | 0.450682 | 0.041116 | 0.500918 |
| 6.5 | **0.021651** | 0.046963 | 0.335630 | 0.595549 | 0.060576 | 0.730157 |
| 7.0 | **0.024000** | 0.054451 | 0.413802 | 0.725142 | 0.086921 | 0.927827 |
| 7.5 | **0.028859** | 0.067797 | 0.515927 | 0.832534 | 0.122484 | 1.091212 |
| 8.0 | **0.036488** | 0.088190 | 0.624730 | 0.915958 | 0.165307 | 1.208035 |

六个报告时刻中 GIFT 的平均全场相对误差均最低；六种方法在完整预测区间都保持 180/180 条轨迹有限，统计未剔除失败样本。

<p align="center">
  <img src="results/formal/M2_recursive_prediction/figures/mean_relative_l2_vs_time.svg" alt="Mean relative L2 error vs time" width="72%">
</p>

<p align="center"><sub>六种方法在 180 条测试轨迹上的平均全场相对 <i>L</i><sup>2</sup> 误差随时间的变化；曲线为严格通过 7 个报告样本点的 PCHIP 插值，仅用于显示趋势。</sub></p>

<p align="center">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/panels/reference/traj1045_reference_scalar_T8p0.svg" alt="Reference vorticity at t=8.0" width="24%">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/panels/gift/traj1045_gift_scalar_T8p0.svg" alt="GIFT prediction at t=8.0" width="24%">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/panels/uno/traj1045_uno_scalar_T8p0.svg" alt="U-NO prediction at t=8.0" width="24%">
</p>

<p align="center"><sub>轨迹 1045 在 <i>t</i>=8.0 的涡量场：左为数值真值，中为 GIFT，右为表现最好的基线 U-NO（全场相对误差 0.036488 / 0.165307），三面板共用同一色标。</sub></p>

### M3｜跨分辨率预测（zero-shot）

仅用 `N`=64 数据训练，网络参数与 Fourier 模态数保持不变，直接在 `N`=96、128 网格上从 `t`=5.0 开始预测；目标分辨率不参与训练或模型选择。

| `t`=6.0 | `N`=64 | `N`=96 | `N`=128 |
| --- | ---: | ---: | ---: |
| **GIFT** | **0.021058** | **0.020947** | **0.020933** |
| GIFT-Lite | 0.040763 | 0.039754 | 0.039742 |
| FNO-2D | 0.271467 | 0.271199 | 0.271295 |
| FNO-3D | 0.450682 | 0.451181 | 0.452352 |

学习到的生成元是网格不变的：每个网格、每个 `t` > 5.0 的报告时刻，GIFT 与 GIFT-Lite 均低于两个 FNO 基线。跨分辨率实验评价的是模型在所测离散网格上的直接适用性，**不表示可以恢复无限高的频率**。

<p align="center">
  <img src="results/formal/M3_cross_resolution/figures/cross_resolution_mean_relative_l2.svg" alt="Cross-resolution mean relative L2 error" width="72%">
</p>

<p align="center"><sub>仅在 <i>N</i>=64 训练的四种方法在三个原生网格上的平均全场相对 <i>L</i><sup>2</sup> 误差（180 条配对轨迹）。</sub></p>

### 支线实验 S1–S3

| 实验 | 结论 |
| --- | --- |
| **S1 高频支路** | `N`=64、96、128 的 `t`=6.0 处分别降低 GIFT 全场误差 60.04%、57.54%、57.55%，降低 GIFT-Lite 33.63%、31.98%、31.98%。 |
| **S2 递归局部修正** | 在独立测试集上避免 GIFT-Lite 的轨迹 1062 出现非有限值；全数据 GIFT 未触发修正。 |
| **S3 随机种子稳定性** | 固定生成元、独立训练高频支路三个种子，GIFT 与 GIFT-Lite 主要指标的最大变异系数约 0.321% 与 0.313%。 |

> **适用范围。** 结论限于本项目的数据分布、训练协议与 `N`=64、96、128 网格，不构成任意分辨率上的误差保证或数值稳定性定理。相同 epoch 不代表相同更新次数或计算量。完整口径见 [EXPERIMENTS.md](EXPERIMENTS.md) 与 [TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md)。

## 论文口径说明

本仓库发布的结果与论文手稿的报告口径**不同，两者不可直接比较**：

| | 论文手稿 | 本仓库当前协议 |
| --- | --- | --- |
| 模型 | 50 条训练轨迹的配置 | 1,000 条训练轨迹（另保留 50 条的 GIFT-Lite） |
| 测试集 | 1,000–1199，共 200 条 | 1,040–1219，共 180 条 |
| `t`=8.0 终点误差 | 0.0875 | GIFT 0.036488；GIFT-Lite 0.088190 |
| 示例轨迹 | 1005 / 1130 | 1045 / 1150 |

手稿的主力结果对应本仓库现在称为 **GIFT-Lite** 的缩减数据配置与更早的数据划分；本 README 与 [EXPERIMENTS.md](EXPERIMENTS.md) 中所有数值均取本仓库当前已发布结果。

## 快速开始

数据与第三方实现位于项目之外，分别通过 `GIFT_DATA_ROOT` 与 `GIFT_EXTERNAL_ROOT` 指定；安装方式见 [SETUP.md](docs/SETUP.md)。每个模型与实验分别运行。

```shell
# 使用随项目提供的权重直接评价
python -m scripts.run_experiment M2 --output ../runs/M2 --skip-plots

# 独立训练一个模型（每个模型一条命令）
python -m scripts.run_training uno --data-profile canonical --run-training --output ../runs/uno

# 核对已发布结果的文件、完成记录与图像来源
python -m scripts.verify_published_results --experiment M3 --result-dir results/formal/M3_cross_resolution
```

代码组织：`src/gift/`（生成元模型、频带分解、数据划分）、`training/`（各模型独立训练入口）、`experiments/formal/`（M1–M3、S1–S3 启动与评价）、`results/formal/`（汇总 CSV/JSON 与 SVG 图像）、`artifacts/`（权重与检查点）、`tests/`（数据契约、检查点身份与结果完整性测试）。其余文档见 [docs/](docs/)。

## 外部代码与致谢

本项目在对比实验中使用了以下上游开源实现。上游源码**不随本仓库分发**，需按各自条款从上游获取；本仓库只记录固定提交与源码散列以便复现。对各项目的作者与贡献者表示感谢。

| 上游项目 | 用途 | 固定版本 | 许可状态 |
| --- | --- | --- | --- |
| [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | FNO-2D、FNO-3D 基线的网络结构与训练目标 | `01d2aeca` | 见上游（本项目内未声明） |
| [ashiq24/UNO](https://github.com/ashiq24/UNO) | U-NO 基线的 U 形神经算子与积分算子 | `19462d82` | BSD-2-Clause |
| [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | U-Net 基线的编码器 / 解码器结构 | `229da3e0` | 上游未声明许可，**不得再分发** |
| [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | PINN-SR 及 PINN-SR-KC 配置 | `9a20ebe6` | 上游未声明许可 |
| [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | PDE-FIND 及 PDE-FIND-KC 配置 | `86911349` | 上游未声明许可 |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)（tag `v1.15.0`） | NAdam 与 L-BFGS 外部优化器实现 | `590d6eef` | Apache-2.0 |

对上游实现只做必要、少量且可逐项说明的适配（数据张量布局、批量组织、跨分辨率输入、历史帧数等），不接管自动微分、不替换反向算子、不重新实现优化器内核。逐项说明见 [EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md)，固定提交与源码散列见 [`external_sources.json`](external_sources.json)。使用相关方法时请引用其原始研究。

> **Acknowledgements.** External implementations come from their upstream owners and are **not** distributed with this repository; only pinned commits and source hashes are recorded. We are grateful to the authors and contributors of the projects above.

## 许可

GIFT 软件与文档使用 [MIT 许可](LICENSE)。数据包与外部项目适用各自的条款。

<p align="center"><sub>项目语言：中文为正文语言，英文标注为对照说明。</sub></p>
