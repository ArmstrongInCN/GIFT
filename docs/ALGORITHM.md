# GIFT algorithm / GIFT 算法

This is the algorithm reference reached from the architecture plate on the [project homepage](../README.md).
It describes what GIFT learns and what its components are. Measured numbers, data splits, training
budgets and run commands are kept in [EXPERIMENTS.md](../EXPERIMENTS.md) and [SETUP.md](SETUP.md);
this page contains no measurements of its own.

本页是从[项目主页](../README.md)结构图进入的算法说明，说明 GIFT 学的是什么、由哪些部件组成。
实测数值、数据划分、训练预算与运行命令不在此处，见 [EXPERIMENTS.md](../EXPERIMENTS.md) 与 [SETUP.md](SETUP.md)；
本页不含独立的测量结果。

## English

### What does GIFT learn?

Surrogates usually predict the field a fixed time lag ahead. GIFT changes the object of learning: it identifies the **continuous-time generator**, the dynamical operator that maps the current state field to its instantaneous time derivative,

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

where $\omega$ is the vorticity field of two-dimensional incompressible flow and $G^{\dagger}$ is the true system generator. The superscript distinguishes the true system generator from the approximate model obtained through learning; it is not an adjoint or a pseudoinverse. GIFT decomposes the generator model into a bias-field channel $C$, a linear channel $A(\omega)$ and a nonlinear channel $Q(\omega,\omega)$, each containing a learnable operator. The three channels differ in amplitude response — $A(\lambda\omega)=\lambda A(\omega)$ and $Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$, with $\lambda$ a scalar amplitude-scaling factor — and field tomography separates and learns them from flow-field trajectories by exploiting that difference.

A state one finite lag later is merely the time integral of the generator, and the generator *is* the governing law. One mathematical object therefore serves both purposes at once. GIFT thereby transforms the surrogate model's prediction mechanism from a black-box mapping hidden in neural network weights into an explicitly parameterized continuous-time generator, which can also be mapped back to testable governing physical laws.

### Components

| Component | In one line |
| --- | --- |
| Generator model | A parameterized model learned from scalar-field data to approximate the continuous-time generator: a bias-field channel $C$, a linear channel $A(\omega)$ and a nonlinear channel $Q(\omega,\omega)$, each containing a learnable operator. |
| Field tomography | Separating and learning different components of the generator from flow-field trajectories by exploiting differences in amplitude responses among the bias-field, linear and nonlinear channels, which are trained **alternately**, with no homogeneity identity written down explicitly. |
| Quadratic field-interaction unit | Nonlinearity comes from a structured field operator rather than an activation function: two learnable spectral filters act on the same input field, their outputs multiply pointwise, and the units are summed with weights (panel E). |
| Learnable spectral filter | A linear operator that applies learnable weights to individual spatial wavenumber components in the Fourier domain. |
| High-frequency branch | The complementary band $Q_{21}\omega$ is predicted by a separately trained branch, which is then coupled with the generator in a single fourth-order Runge–Kutta integration (panel D). |
| Recursive local correction | A fixed, gradient-free rule applied after each complete Runge–Kutta step during recursive rollouts: local median residuals of the de-meaned field are clipped at an amplitude scale, under a safety cap on anomalous grid points (40, 90 and 160 for $N$ = 64, 96 and 128). |
| Physical interpretability | With the generator frozen, known Navier–Stokes equation terms are fitted to the output of each channel and the recovered equation parameters are compared with the true ones. |

### Architecture and prediction process

![GIFT architecture and prediction process](../assets/figures/gift_architecture.png)

**GIFT architecture and prediction process.** (A) Densely sampled continuous flow-field trajectories. (B) Training data are constructed through Fourier transformation and frequency band decomposition: the band-limited state enters the main path, the complementary band enters the high-frequency branch. (C) Field tomography exploits differences in amplitude responses to separate the bias-field, linear and nonlinear channels, which are built from quadratic field-interaction units. (D) The high-frequency branch. (E) GIFT produces nonlinear responses through two learnable spectral filters. (F) The GIFT prediction procedure advances the generator to the next instant by fourth-order Runge–Kutta integration.

### Where the exact definitions are recorded

- Frequency decomposition, the projections $P_{21}$/$Q_{21}$ and the coupled Runge–Kutta integration: [EXPERIMENTS.md](../EXPERIMENTS.md) sections 2.1 and 3.1.
- Recursive local correction thresholds and the anomalous-point cap: [EXPERIMENTS.md](../EXPERIMENTS.md) section 3.2.
- Evaluation metrics and the seed-averaging convention: [EXPERIMENTS.md](../EXPERIMENTS.md) section 3.4.
- Training budgets, data splits and model selection: [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md).
- How to read, recompute and verify the published results: [RESULTS.md](RESULTS.md).

## 中文

### GIFT 学的是什么？

传统代理模型直接预测固定时间间隔后的流场；GIFT 换一个对象——它辨识**连续时间生成元**，即把当前状态场映射为其瞬时时间导数的动力学算子：

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

其中 $\omega$ 为二维不可压缩流动的涡量场，$G^{\dagger}$ 是真实系统生成元；上标用于区分真实系统生成元与通过学习得到的近似模型，不表示伴随或伪逆。GIFT 把生成元模型分解为偏置场通道 $C$、线性通道 $A(\omega)$ 与非线性通道 $Q(\omega,\omega)$，每个通道各含一个可学习算子。三个通道的振幅响应规律不同（$\lambda$ 为标量振幅缩放因子）——$A(\lambda\omega)=\lambda A(\omega)$、$Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$——场层析正是依据这一差异从流场轨迹中把它们分离并学习。

有限时间间隔后的状态只是生成元的时间积分结果，而生成元本身对应系统的控制规律，因此同一个数学对象可以同时承担两件事：GIFT 把代理模型的预测机制从隐藏在神经网络权重中的黑箱映射转变为可还原为可检验物理控制规律的连续时间模型。

### 组成部件

| 环节 | 一句话说明 |
| --- | --- |
| 生成元模型 | 从标量场数据中学习、用于近似连续时间生成元的参数化模型：偏置场通道 $C$、线性通道 $A(\omega)$ 与非线性通道 $Q(\omega,\omega)$，每个通道各含一个可学习算子。 |
| 场层析 | 利用偏置场、线性与非线性通道振幅响应的差异，从流场轨迹中分离并学习生成元的不同组成部分；各通道**交替训练**，训练中无需显式构造齐次特征式。 |
| 二次场相互作用单元 | 非线性由结构化场算子而非激活函数提供：同一输入场经两个可学习谱滤波器后逐点相乘，再对各单元加权求和（面板 E）。 |
| 可学习谱滤波器 | 在傅里叶域对各个空间波数分量施加可学习权重的线性算子。 |
| 高频支路 | 补频带 $Q_{21}\omega$ 交由单独训练的支路预测，并与生成元耦合在同一个四阶 Runge–Kutta 积分中推进（面板 D）。 |
| 递归局部修正 | 递归推演过程中每个完整 Runge–Kutta 步后执行的固定规则：按幅值尺度裁剪去均值场的局部中值残差，并设异常网格点安全上限（$N$ = 64、96、128 时分别为 40、90、160）。 |
| 物理可解释性 | 生成元冻结后，用已知 Navier–Stokes 方程项拟合各通道输出，将读出的方程参数与真实控制方程系数对比。 |

### 架构与预测过程

![GIFT 架构与预测过程](../assets/figures/gift_architecture.png)

**GIFT 架构与预测过程。**(A) 稠密采样的连续流场轨迹。(B) 通过傅里叶变换与频带分解构造训练数据：带限状态进入主路径，补频带状态进入高频支路。(C) 场层析依据各通道振幅响应差异，分离由二次场相互作用单元构成的偏置场通道、线性通道与非线性通道。(D) 高频支路。(E) GIFT 通过两个可学习谱滤波器产生非线性响应。(F) GIFT 的预测流程经四阶 Runge–Kutta 积分把生成元推进至下一时刻。

### 精确定义的记录位置

- 频率分解、投影 $P_{21}$/$Q_{21}$ 与耦合 Runge–Kutta 积分：[EXPERIMENTS.md](../EXPERIMENTS.md) 第 2.1、3.1 节。
- 递归局部修正的阈值与异常点上限：[EXPERIMENTS.md](../EXPERIMENTS.md) 第 3.2 节。
- 评价指标与种子统计口径：[EXPERIMENTS.md](../EXPERIMENTS.md) 第 3.4 节。
- 训练预算、数据划分与模型选择：[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)。
- 如何阅读、重算与校验已发布结果：[RESULTS.md](RESULTS.md)。
