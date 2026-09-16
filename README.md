<div align="center">

# GIFT

**Generator Identification via Field Tomography**
**基于场层析的生成元识别：具备物理可解释性的流体动力学代理模型**

<sub>从流场轨迹中直接学习连续时间生成元，并把代理模型的预测机制还原为可检验的偏微分控制方程</sub>

<sub>Learning the continuous-time generator of a flow directly from trajectories, and turning the surrogate's prediction mechanism into an explicit, checkable PDE</sub>

</div>

<p align="center">
  <img src="assets/figures/gift_architecture.png" alt="GIFT architecture and prediction process" width="100%">
</p>

<p align="center"><sub><b>图 1｜GIFT 架构与预测过程。</b>(A) 稠密采样的连续流场轨迹。(B) 通过傅里叶变换与固定频带分解构造训练数据：<i>P</i><sub>21</sub> 频带内的分量进入 GIFT 主路径，<i>Q</i><sub>21</sub> 高频分量进入高频支路；中心差分构建的时间导数作为训练目标。(C) 场层析：利用不同通道的振幅响应差异，交替学习二次通道、线性通道与偏置场通道，其中含 8 个可学习的二次场相互作用单元。(D) 高频支路以常规神经网络构建。(E) 二次场相互作用单元以两个可学习谱滤波器经逐点相乘产生非线性响应，而非依赖激活函数。(F) 预测过程：生成元与高频支路给出时间导数，经四阶 Runge–Kutta 积分推进到下一时刻。<br><b>Figure 1 | GIFT architecture and prediction process.</b></sub></p>

---

## 一句话概括

传统代理模型直接预测固定时间间隔后的流场；GIFT 换个对象：它学习决定流场**瞬时演化**的**连续时间生成元**。有限时间间隔后的状态只是生成元的时间积分结果，而生成元本身对应系统的控制规律。因此代理模型的预测与显式 PDE 参数识别可以共享同一个数学对象——预测机制不再是藏在网络权重里的黑盒映射，而是一个可以被检验物理正确性的连续动力学对象。

> **In one sentence.** Rather than predicting the field at a fixed time lag, GIFT learns the **continuous-time generator** that governs instantaneous evolution. A finite-lag state is merely the time integral of that generator, and the generator *is* the governing law — so prediction and explicit PDE-parameter identification share one object, and the surrogate's mechanism becomes a checkable dynamical object instead of a black box.

## 为什么需要 GIFT

现代 CFD 已成为航空航天、汽车工程与气象预测中不可缺少的工具，但其计算成本在需要反复仿真的场景（设计优化、实时控制、数字孪生）中构成根本性瓶颈。数据驱动的代理模型提供了出路，然而当前代理模型普遍基于神经网络构建：

- **黑盒性。** 神经网络代理模型难以直接解释其预测所依据的物理规律，即使预测误差很小，也不能证明它满足正确的物理方程。
- **PINN 的前提。** PINN 把控制方程写进损失函数，但其基本前提是控制方程完全或部分已知。
- **候选字典依赖。** 结合稀疏回归的 PINN-SR 交替进行场表示学习与系数识别，PDE-FIND 从时空观测中恢复显式方程结构与参数——两者本质上都依赖候选方程项字典，候选项数量、噪声与方程项之间的相关性会显著影响识别精度。

GIFT 提出的问题是：**能否从有限的全场流体数据中，直接学习一个连续时间动力学模型，使它既能作为流场演化的动力学代理模型，又能在不进行广泛参数搜索的前提下映射回可检验的物理控制规律？**

> **Why.** Neural surrogates are black boxes: low prediction error does not prove the correct physics. Physics-informed approaches presuppose that the governing equation is partly known, and sparse-regression equation discovery depends on a candidate term library. GIFT asks whether a single continuous-time model can serve both as a dynamical surrogate and as a checkable governing law, without a broad parameter search.

## 方法

### 1. 生成元分解

对于瞬时演化仅由当前状态决定的系统，生成元把流场状态映射为它的时间变化率。GIFT 把可学习的生成元模型拆成三个通道：

```
G(ω) = C + A(ω) + Q(ω, ω)
```

`C` 是与状态无关但可随空间变化的偏置场，`A` 是自由的平移等变线性算子，`Q` 是非线性算子。对物理参数固定、具有周期边界和定常外力的二维不可压缩 Navier–Stokes 系统，状态取涡量场 `ω`。

### 2. 场层析

在二维不可压缩流动中，线性算子具有明确的**平移等变性**，非线性算子具有明确的**二次齐次性**。这些不同的振幅响应规律正是场层析的核心含义：从大量不同时空状态的数据中，按不同特征响应把生成元的各通道分离并分别重建。实际训练中并不显式构造各算子的齐次特征式，而是把二次通道与线性、偏置场通道**交替训练**，利用振幅响应差异从数据中把它们分开。

### 3. 二次场相互作用单元

GIFT 的非线性响应由结构化的场算子实现，而不是激活函数。**Quadratic field-interaction unit** 对同一输入场施加两个不同的可学习谱滤波器，把滤波后的场逐点相乘，再对各单元结果加权求和（图 1E）。它在模型连接结构中的位置类似神经网络中的神经元，但数学本质是一个输入输出均为场数据的场算子。

> **Method summary.** The generator decomposes into a field-bias path `C`, a translation-equivariant linear path `A(ω)`, and a quadratic path `Q(ω,ω)`. Field tomography separates these paths by exploiting their differing amplitude responses, training the quadratic path alternately with the linear and bias paths. Nonlinearity comes from a structured **quadratic field-interaction unit** — two learnable spectral filters applied to the same field, multiplied pointwise — not from an activation function.

### 4. 高频支路

用同一种结构化模型直接处理全部空间频带并非必要：高频分量幅值和能量占比较小，而场层析过程需要有较强幅值的信号。受 FNO 为谱算子保留逐点线性分支的思路启发，GIFT 为 `Q`<sub>21</sub> 高频分量设置**单独训练的高频支路**，以当前全场、高频分量和低频核心预测的时间导数作为输入，预测高频时间导数（图 1D）。该支路在生成元训练结束后单独训练，两部分随后耦合积分，既保留核心的结构化表示，又补充小尺度演化。

### 5. 递归局部修正

在每个完整 RK4 步之后执行、无需梯度训练的固定规则：对 `P`<sub>21</sub> 与 `Q`<sub>21</sub> 分别计算去均值场、局部中值残差与幅值尺度 `ℓ = 2.1σ`，当局部残差与中心幅值同时超过阈值时把残差裁剪回允许范围，并保持分量支持。每个频带在单条轨迹、单步内设有异常网格点安全上限 `K`<sub>N</sub> = max(1, ⌊40N²/4096⌋)（`N`=64、96、128 时分别为 40、90、160）；超过上限即停止该轨迹的后续积分并在运行记录中注明频带。

### 6. 物理可解释性

生成元模型训练完毕后冻结，用已知的 N–S 方程项拟合各通道的输出，再把得到的系数与真实控制方程系数对比。这是对"代理模型是否学到了正确物理"的**可量化检验**。

## 主要结果

所有结果均为本仓库当前协议下完成并发布的实测结果。预测模型使用同一批 1,000 条 `N`=64 训练轨迹（`t`=0.0 至 10.0，间隔 0.02）；GIFT 另保留使用其中前 50 条的 GIFT-Lite（Lite 表示训练数据减少，不表示网络更小）。测试集为**未参与任何方法训练的 180 条轨迹**（编号 1040–1219），每个预测时刻单独评价，不跨时刻求平均。

### M1｜方程参数识别

真实系数 `(ν, β, γ) = (0.01, 1, 1)`。GIFT 在冻结生成元后从常数、线性与二次作用中读出系数，真实系数不参与训练。

| 噪声 | GIFT ν APE | GIFT β APE | GIFT γ APE |
| --- | ---: | ---: | ---: |
| 0% | **0.484%** | **0.466%** | **0.030%** |
| 1% | **4.361%** | **0.877%** | **0.040%** |
| 10% | 64.852% | 19.778% | **0.783%** |

在 0% 和 1% 噪声下，GIFT 对三个参数的绝对百分比误差（APE）均为五种配置中最低。10% 噪声下 `ν`、`β`、`γ` 的最低 APE 分别来自 PDE-FIND-KC、PDE-FIND 和 GIFT——高噪声下 GIFT 的参数识别精度显著下降，当前场层析过程无法在高噪声下准确分离非线性项。

<p align="center">
  <img src="results/formal/M1_equation_identification/figures/parameter_identification_ape_vs_noise.svg" alt="Parameter identification APE vs noise" width="82%">
</p>

<p align="center"><sub>五种配置在 0%、1% 和 10% 噪声下对三个方程参数的绝对百分比误差（三个面板共用对数纵轴）。</sub></p>

> ⚠️ **适用范围。** GIFT 的这一能力属于**已知控制方程结构下的参数识别**，而不是在未知候选结构中从头发现任意 PDE，因此不能替代 PDE-FIND 或 PINN-SR 所面向的"结构与参数均未知"的方程识别任务。该表是单一方程、每个非零噪声水平一个配对噪声实现、单一随机种子和固定预算下的描述性结果。

### M2｜递归预测（`N`=64，`t`=5.0 → 8.0）

GIFT 与 GIFT-Lite 从 `t`=5.0 的一个真值状态出发，以 Δ`t`=0.02 递归积分；两个 FNO 基线使用 `t`=4.1 至 5.0 的 46 帧历史状态作为输入。

| 时间 | GIFT | GIFT-Lite | FNO-2D | FNO-3D | U-NO | U-Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.5 | **0.021901** | 0.036061 | 0.221038 | 0.315175 | 0.030129 | 0.233574 |
| 6.0 | **0.021058** | 0.040763 | 0.271467 | 0.450682 | 0.041116 | 0.500918 |
| 6.5 | **0.021651** | 0.046963 | 0.335630 | 0.595549 | 0.060576 | 0.730157 |
| 7.0 | **0.024000** | 0.054451 | 0.413802 | 0.725142 | 0.086921 | 0.927827 |
| 7.5 | **0.028859** | 0.067797 | 0.515927 | 0.832534 | 0.122484 | 1.091212 |
| 8.0 | **0.036488** | 0.088190 | 0.624730 | 0.915958 | 0.165307 | 1.208035 |

<p align="center">
  <img src="results/formal/M2_recursive_prediction/figures/mean_relative_l2_vs_time.svg" alt="Mean relative L2 error vs time" width="72%">
</p>

<p align="center"><sub>六种方法在 180 条测试轨迹上的平均全场相对 <i>L</i><sup>2</sup> 误差随时间的变化。曲线为严格通过 7 个报告样本点的 PCHIP 插值，仅用于显示趋势；<i>t</i>=5.0 是给定初始状态，不计作预测。</sub></p>

六个报告时刻中，GIFT 的平均全场相对误差均最低；六种方法在完整预测区间都保持 180/180 条轨迹有限，误差统计未剔除失败样本。下图为预先指定的独立测试轨迹 1045 在 `t`=8.0 的涡量场对比：

<p align="center">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/panels/reference/traj1045_reference_scalar_T8p0.svg" alt="Reference vorticity at t=8.0" width="24%">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/panels/gift/traj1045_gift_scalar_T8p0.svg" alt="GIFT prediction at t=8.0" width="24%">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/panels/uno/traj1045_uno_scalar_T8p0.svg" alt="U-NO prediction at t=8.0" width="24%">
</p>

<p align="center"><sub>轨迹 1045 在 <i>t</i>=8.0 的涡量场：左为数值真值，中为 GIFT，右为表现最好的基线 U-NO（全场相对误差 0.036488 / 0.165307）。三个面板共用同一色标；完整关键帧与有符号残差见 <a href="EXPERIMENTS.md">EXPERIMENTS.md</a> 图 M2-2。</sub></p>

### M3｜跨分辨率预测（zero-shot）

四种方法都只用 `N`=64 数据训练，网络参数与 Fourier 模态数保持不变，然后直接在 `N`=96 与 `N`=128 网格上从 `t`=5.0 开始预测；目标分辨率不参与训练或模型选择。

| `t`=6.0 | `N`=64 | `N`=96 | `N`=128 |
| --- | ---: | ---: | ---: |
| **GIFT** | **0.021058** | **0.020947** | **0.020933** |
| GIFT-Lite | 0.040763 | 0.039754 | 0.039742 |
| FNO-2D | 0.271467 | 0.271199 | 0.271295 |
| FNO-3D | 0.450682 | 0.451181 | 0.452352 |

<p align="center">
  <img src="results/formal/M3_cross_resolution/figures/cross_resolution_mean_relative_l2.svg" alt="Cross-resolution mean relative L2 error" width="72%">
</p>

<p align="center"><sub>仅在 <i>N</i>=64 训练的四种方法在三个原生网格上的平均全场相对 <i>L</i><sup>2</sup> 误差（180 条配对轨迹）。色带覆盖 GIFT 与 GIFT-Lite 三个训练种子的均值范围，不是置信区间；<i>t</i>=5.0 为零误差真值锚点，未纳入对数纵轴。</sub></p>

GIFT 学习到的生成元是网格不变的：在每个网格、每个 `t` > 5.0 的报告时刻，GIFT 与 GIFT-Lite 均低于两个 FNO 基线。需要说明的是，跨分辨率实验评价的是模型在所测试离散网格上的直接适用性，**不表示可以恢复无限高的频率**。

### 支线实验 S1–S3

| 实验 | 结论 |
| --- | --- |
| **S1 高频支路** | 在 `N`=64、96、128 的 `t`=6.0，高频支路分别降低 GIFT 全场误差 60.04%、57.54%、57.55%，降低 GIFT-Lite 33.63%、31.98%、31.98%；两者均改善所报告的递归 `Q`<sub>21</sub> 状态误差。 |
| **S2 递归局部修正** | 在独立测试集上避免 GIFT-Lite 的轨迹 1062 出现非有限值（关闭时三个种子均于 `t`=6.74 首次出现 NaN/Inf）。全数据 GIFT 未触发修正，本实验未显示修正对它的额外收益。 |
| **S3 随机种子稳定性** | 各自固定生成元、独立训练高频支路三个种子，GIFT 与 GIFT-Lite 主要指标的最大变异系数分别约 0.321% 与 0.313%。该范围只描述所测种子，不含生成元训练随机性。 |

## 论文口径说明

本仓库发布的结果与论文手稿的报告口径**不同，两者不可直接比较**，请按来源分别引用：

| | 论文手稿 | 本仓库当前协议 |
| --- | --- | --- |
| 模型 | 50 条训练轨迹的配置 | 1,000 条训练轨迹（另保留 50 条的 GIFT-Lite） |
| 测试集 | 1,000–1199，共 200 条 | 1,040–1219，共 180 条 |
| `t`=8.0 终点误差 | 0.0875 | GIFT 0.036488；GIFT-Lite 0.088190 |
| 示例轨迹 | 1005 / 1130 | 1045 / 1150 |

手稿中的主力结果对应本仓库现在称为 **GIFT-Lite** 的缩减数据配置，并采用更早的数据划分；本仓库另外提供在其基础上扩展的全数据 GIFT。本 README 与 [EXPERIMENTS.md](EXPERIMENTS.md) 中所有数值均取本仓库当前已发布结果。

> The manuscript reports the reduced-data configuration on an earlier split (200 test trajectories, trajectories 1005/1130); this repository publishes the current protocol (1,000 training trajectories plus the 50-trajectory GIFT-Lite, 180 test trajectories, trajectories 1045/1150). The two sets of numbers are **not directly comparable**. Every number in this README comes from the repository's own published results.

## 仓库内容

| 目录 | 内容 |
| --- | --- |
| `src/gift/` | 生成元模型、频带分解、数据划分与预测队列 |
| `training/` | 各模型独立训练入口（GIFT 生成元与预测器、FNO-2D/3D、U-NO、U-Net、PINN） |
| `experiments/formal/` | 六项实验的启动与评价入口（M1–M3、S1–S3） |
| `results/formal/` | 已完成实验的汇总 CSV/JSON 与 SVG 图像 |
| `artifacts/` | 完整模型权重与检查点目录 |
| `docs/` | [安装与独立运行](docs/SETUP.md)、[训练协议](docs/TRAINING_PROTOCOL.md)、[计算效率](docs/PERFORMANCE.md)、[实验图像](docs/FIGURES.md)、[结果说明](docs/RESULTS.md)、[外部适配说明](docs/EXTERNAL_ADAPTATIONS.md) |
| `tests/` | 数据契约、检查点身份、执行后端与结果完整性的测试 |

## 快速开始

数据与第三方实现位于项目之外，分别通过 `GIFT_DATA_ROOT` 与 `GIFT_EXTERNAL_ROOT` 指定。安装方式见 [SETUP.md](docs/SETUP.md)。

```shell
# 使用随项目提供的权重直接评价
python -m scripts.run_experiment M2 --output ../runs/M2 --skip-plots

# 独立训练一个模型（每个模型一条命令）
python -m scripts.run_training uno --data-profile canonical --run-training --output ../runs/uno

# 核对已发布结果的文件、完成记录与图像来源
python -m scripts.verify_published_results --experiment M3 --result-dir results/formal/M3_cross_resolution
```

每个模型与实验分别运行，不要求一次命令完成全部工作。GIFT 的 GPU 训练默认使用加速实现（CUDA graphs）并支持完整状态恢复；`--execution eager` 保留参考训练路径。断点续算只恢复同一次运行的模型、优化器、调度器与随机状态，不等于把不相关的已有模型当作初始化。完整命令与恢复边界见 [EXPERIMENT_EXECUTION.md](docs/EXPERIMENT_EXECUTION.md)、[CHECKPOINTS.md](docs/CHECKPOINTS.md) 与 [TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md)。

## 外部代码与致谢

本项目在对比实验中使用了以下上游开源实现。上游源码**不随本仓库分发**，需按各自条款从上游获取；本仓库只记录固定提交与源码散列，以便复现。对各项目的作者与贡献者表示感谢。

| 上游项目 | 用途 | 固定版本 | 许可状态 |
| --- | --- | --- | --- |
| [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | FNO-2D、FNO-3D 基线的网络结构与训练目标 | `01d2aeca` | 见上游（本项目内未声明） |
| [ashiq24/UNO](https://github.com/ashiq24/UNO) | U-NO 基线的 U 形神经算子与积分算子 | `19462d82` | BSD-2-Clause |
| [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | U-Net 基线的编码器 / 解码器结构 | `229da3e0` | 上游未声明许可，**不得再分发** |
| [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | PINN-SR 及 PINN-SR-KC 配置 | `9a20ebe6` | 上游未声明许可 |
| [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | PDE-FIND 及 PDE-FIND-KC 配置 | `86911349` | 上游未声明许可 |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)（tag `v1.15.0`） | NAdam 与 L-BFGS 外部优化器实现 | `590d6eef` | Apache-2.0 |

对上游实现只做必要、少量且可逐项说明的适配（数据张量布局、批量组织、跨分辨率输入、历史帧数等），不接管自动微分、不替换反向算子、不重新实现优化器内核。逐项适配说明见 [EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md)，固定提交与源码散列见 `external_sources.json`。使用相关方法时请引用其原始研究。

> **Acknowledgements.** External implementations are obtained from their upstream owners and are **not** distributed with this repository; only pinned commits and source hashes are recorded here. We are grateful to the authors and contributors of the projects listed above. Adaptations are limited to data layout, batching, cross-resolution input and history length; autograd, backward operators and optimizer kernels are untouched. See [`external_sources.json`](external_sources.json) for pinned commits and SHA-256 hashes.

## 结论与适用范围

GIFT 的价值不仅在于降低预测误差，更在于让代理模型内部的物理规律成为可直接检验的研究对象：一方面它像神经代理模型一样直接作用于全场状态，为时间推进提供连续动力学预测；另一方面它像 PINN-SR 和 PDE-FIND 一样，从全场数据中识别出具体的 PDE 参数。

一个值得关注的现象：用于训练 GIFT 的数据必须包含足够复杂的变化——丰富的振幅与空间结构变化有助于区分线性与二次响应，为场层析提供可辨识信息。当训练数据变化过于简单时，包括 FNO、U-NO 在内的基线预测性能大幅上升，而 GIFT 的性能反而下降。流动的复杂性不必然造成代理模型预测的困难，它可能恰恰是辨认动力学作用的关键信息来源。

**适用边界。** 结论限于本项目的数据分布、训练协议以及 `N`=64、96、128 网格，不构成任意分辨率上的误差保证或数值稳定性定理。相同 epoch 不代表相同参数更新次数或计算量：四个预测基线的训练预测长度、批量与梯度累积方式各不相同，更新次数与实测耗时见 [TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md) 与 [PERFORMANCE.md](docs/PERFORMANCE.md)。固定种子与完整状态恢复不保证跨设备逐位一致。

## 许可

GIFT 软件与文档使用 [MIT 许可](LICENSE)。数据包与外部项目适用各自的条款。GitHub 仓库内的精简结果包含汇总 CSV/JSON、SVG 与校验清单，不包含完整预测数组或运行日志；数据包单独提供。

<p align="center"><sub>Project language / 项目语言：中文为正文语言，英文标注为对照说明。</sub></p>
