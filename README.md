<div align="center">

# GIFT

**Generator Identification via Field Tomography**

<sub>Learning the continuous-time generator of a flow directly from trajectories, and reading the surrogate's prediction mechanism back out as an explicit, checkable PDE</sub>

[**English**](#english) &nbsp;·&nbsp; [**中文**](#中文)

</div>

---

## English

### 1. What GIFT learns

Surrogates usually predict the field a fixed time lag ahead. GIFT changes the object of learning: it identifies the **continuous-time generator** that maps a state to its instantaneous rate of change,

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

where $\omega$ is the vorticity field and $G^{\dagger}$ is the true generator to be identified. The dagger only distinguishes the true object from the learned approximation; it is not an adjoint or a pseudoinverse. $C$ is a bias field, $A$ a translation-equivariant linear operator and $Q$ a homogeneous-quadratic operator. The three channels differ in amplitude response, $A(\lambda\omega)=\lambda A(\omega)$ and $Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$, and that difference is what field tomography separates.

A state one finite lag later is merely the time integral of the generator, and the generator *is* the governing law. One mathematical object therefore serves both purposes at once. The surrogate's mechanism is no longer a black-box map hidden in network weights, but a continuous dynamical object whose physical correctness can be checked.

### 2. Method

| Component | In one line |
| --- | --- |
| Generator decomposition | Bias field $C$, translation-equivariant linear operator $A(\omega)$ and homogeneous-quadratic operator $Q(\omega,\omega)$; the state is the vorticity field of two-dimensional incompressible flow. |
| Field tomography | The quadratic path is trained **alternately** with the linear and bias paths, so the paths separate from data by their differing amplitude responses, with no homogeneity identity written down explicitly. |
| Quadratic field-interaction unit | Nonlinearity comes from a structured field operator rather than an activation function: two learnable spectral filters act on the same input field, their outputs multiply pointwise, and the units are summed with weights (panel E). |
| High-frequency branch | The complementary band $Q_{21}\omega$ is predicted by a separately trained branch, which is then coupled with the generator in a single integration (panel D). |
| Recursive local correction | A fixed, gradient-free rule applied after each complete RK4 step: local median residuals of the de-meaned field are clipped at an amplitude scale, under a safety cap on anomalous grid points (40, 90 and 160 for $N$ = 64, 96 and 128). |
| Physical interpretability check | With the generator frozen, known Navier–Stokes terms are fitted to each channel's output and the recovered coefficients are compared with the true ones. |

<p align="center"><img src="assets/figures/gift_architecture.png" alt="GIFT architecture and prediction process" width="100%"></p>

<p align="center"><b>GIFT architecture and prediction process.</b> (A) Densely sampled continuous flow trajectories. (B) Fourier transform and fixed-band decomposition build the training data: the band-limited state enters the main path, the complementary band enters the high-frequency branch. (C) Field tomography: differing amplitude responses separate the quadratic, linear and bias-field paths, which are learned alternately. (D) High-frequency branch. (E) Quadratic field-interaction unit. (F) The generator is advanced to the next instant by fourth-order Runge&ndash;Kutta integration.</p>

### 3. Results

Every number below is a completed and published measurement under the protocol recorded in [EXPERIMENTS.md](EXPERIMENTS.md) sections 2&ndash;3: 1,000 $N$ = 64 training trajectories ($t$ = 0.0&ndash;10.0, spacing 0.02), of which GIFT-Lite retains the first 50 (Lite denotes reduced training data, not a smaller network). The test set is **180 trajectories** (1040&ndash;1219) that took part in no method's training, and each prediction time is scored separately. This project numbers its experiments with the M series (M1 equation-parameter identification, M2 recursive prediction, M3 cross-resolution prediction); the supplementary materials use a separate W series, ordered as the main text reports the experiments: W1 recursive prediction (= M2), W2 cross-resolution prediction (= M3), W3 equation-parameter identification (= M1).

**M2, recursive prediction** ($N$ = 64, $t$ = 5.0 &rarr; 8.0). GIFT and GIFT-Lite start from the true state at $t$ = 5.0 and integrate recursively with $\Delta t$ = 0.02; the two FNO baselines take the 46 historical states from $t$ = 4.1&ndash;5.0 as input. Mean full-field relative $L^{2}$ error:

| Time | GIFT | GIFT-Lite | FNO-2D | FNO-3D | U-NO | U-Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.5 | **0.021901** | 0.036061 | 0.221038 | 0.315175 | 0.030129 | 0.233574 |
| 6.0 | **0.021058** | 0.040763 | 0.271467 | 0.450682 | 0.041116 | 0.500927 |
| 6.5 | **0.021651** | 0.046963 | 0.335630 | 0.595549 | 0.060576 | 0.730189 |
| 7.0 | **0.024000** | 0.054451 | 0.413802 | 0.725142 | 0.086921 | 0.927862 |
| 7.5 | **0.028859** | 0.067797 | 0.515927 | 0.832534 | 0.122484 | 1.091225 |
| 8.0 | **0.036488** | 0.088190 | 0.624730 | 0.915958 | 0.165307 | 1.208053 |

GIFT has the lowest mean error at all six reported times, and all six methods stay finite on 180/180 trajectories over the full interval; no failed sample is excluded from the statistics.

<p align="center"><img src="results/formal/M2_recursive_prediction/figures/mean_relative_l2_vs_time.svg" alt="Mean relative L2 error against time for the six methods" width="72%"></p>

<p align="center"><b>Figure M2-1 |</b> Mean full-field relative <i>L</i><sup>2</sup> error of the six methods over 180 test trajectories. GIFT and GIFT-Lite are averaged over three training seeds, whose seed SD is tabulated in the experiment record. Curves are PCHIP interpolants through exactly the seven reported sample points and are shown for trend only.</p>

<p align="center"><img src="results/formal/M2_recursive_prediction/figures/keyframes/traj1045_recursive_keyframes.svg" alt="Predicted vorticity fields and signed prediction errors of trajectory 1045 at four key times" width="100%"></p>

<p align="center"><b>Figure M2-2 |</b> Prespecified evaluation trajectory 1045 at <i>t</i> = 5.0, 6.0, 7.0, 8.0: the numerical reference and the six methods' predicted vorticity fields (left), each with the signed prediction error of the same row (right). A numerical reference does not define a prediction error, so its error cells are the flat zero tile. All vorticity fields share one colour scale and all prediction errors share a second, independent one. The trajectory belongs to the held-out test set.</p>

**M3, cross-resolution prediction** (zero-shot). Trained on $N$ = 64 only, with network parameters and Fourier modes unchanged, then evaluated directly on the $N$ = 96 and $N$ = 128 grids from $t$ = 5.0; the target resolution takes no part in training or model selection.

| $t$ = 6.0 | $N$ = 64 | $N$ = 96 | $N$ = 128 |
| --- | ---: | ---: | ---: |
| **GIFT** | **0.021058** | **0.020947** | **0.020933** |
| GIFT-Lite | 0.040763 | 0.039754 | 0.039742 |
| FNO-2D | 0.271467 | 0.271199 | 0.271295 |
| FNO-3D | 0.450682 | 0.451181 | 0.452352 |

The learned generator is grid-invariant: on every grid and at every reported time $t$ > 5.0, GIFT and GIFT-Lite stay below both FNO baselines. The experiment measures direct applicability on the discrete grids tested and **does not imply recovery of arbitrarily high frequencies**.

<p align="center"><img src="results/formal/M3_cross_resolution/figures/cross_resolution_mean_relative_l2.svg" alt="Cross-resolution mean relative L2 error on three native grids" width="72%"></p>

<p align="center"><b>Figure M3-1 |</b> Mean full-field relative <i>L</i><sup>2</sup> error of four methods trained on <i>N</i> = 64 only, evaluated on three native grids (180 paired trajectories). The solid dark-blue and dashed light-blue lines are the means over three training seeds for GIFT and GIFT-Lite; the band spans the range of those seed means and is not a confidence interval. <i>t</i> = 5.0 is the zero-error truth anchor and is not included in the logarithmic vertical axis.</p>

**M1, equation-parameter identification.** True coefficients $(\nu,\beta,\gamma)$ = (0.01, 1, 1), excluded from training. All five configurations identify those parameters from the same single local training trajectory: GIFT's generator is trained on it and the frozen generator is read out from it, exactly as PDE-FIND and PINN-SR consume that one trajectory. Absolute percentage error (APE):

| Noise | $\nu$ | $\beta$ | $\gamma$ |
| --- | ---: | ---: | ---: |
| 0% | **4.343%** | **1.619%** | **4.121%** |
| 1% | **11.890%** | **2.209%** | **4.958%** |
| 10% | 69.791% | 21.702% | **11.482%** |

At 0% and 1% noise GIFT attains the lowest APE on all three parameters among the five configurations; at 10% noise $\nu$ degrades sharply, so this protocol is not a high-noise method for the nonlinear coefficient. This capability is **parameter identification under a known governing-equation structure**, not discovery of an arbitrary PDE from an unknown candidate library, and it does not stand in for the "structure and parameters both unknown" setting that PDE-FIND or PINN-SR address.

<p align="center"><img src="results/formal/M1_equation_identification/figures/parameter_identification_ape_vs_noise.svg" alt="Absolute percentage error of the identified coefficients" width="72%"></p>

<p align="center"><b>Figure M1-1 |</b> Absolute percentage error of the identified coefficients for the five configurations at 0%, 1% and 10% noise. The three panels share one vertical axis, linear up to 20% and logarithmic above it, with labelled ticks.</p>

### 4. Side experiments

| Experiment | Result |
| --- | --- |
| **S1 High-frequency branch** | At $t$ = 6.0 it reduces the GIFT full-field error by 60.04%, 57.54% and 57.55% on $N$ = 64, 96 and 128, and the GIFT-Lite error by 33.63%, 31.98% and 31.98%. |
| **S2 Recursive local correction** | On the held-out test set it keeps trajectory 1062 of GIFT-Lite finite. Full-data GIFT never triggered the correction. |
| **S3 Random-seed stability** | With the generator fixed and the high-frequency branch trained under three seeds, the largest coefficient of variation on the main metrics is 0.321% for GIFT and 0.313% for GIFT-Lite. |
| **S4 Initial-condition distribution** | On a separate smooth Gaussian random-field population, full-data GIFT still has the lowest mean error at every reported time ($t$ = 8.0: 0.021035) and U-NO is the closest baseline (0.173805). Full-data GIFT and all four baselines stay finite on 180/180 test trajectories; GIFT-Lite loses 17 of 180 after $t$ &asymp; 6.9, so its later values are finite-subset statistics and are reported as such. |

### 5. Scope

The conclusions are limited to this project's data distribution, training protocol and the $N$ = 64, 96 and 128 grids. They are not an error guarantee or a numerical-stability theorem at arbitrary resolution, and equal epoch counts do not mean equal parameter-update counts or equal compute: GIFT starts from a single state at $t$ = 5.0 while the FNO baselines consume 46 historical frames, and GIFT's generator and high-frequency branch are trained under separate budgets. Full detail, including seed standard deviations, finiteness counts and per-job timings, is in [EXPERIMENTS.md](EXPERIMENTS.md) section 6 and [TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md).

### 6. Quick start

Data and third-party implementations live outside the project and are located through `GIFT_DATA_ROOT` and `GIFT_EXTERNAL_ROOT`; installation is described in [SETUP.md](docs/SETUP.md). Each model and each experiment runs separately.

```shell
# Evaluate directly with the weights shipped with the project
python -m scripts.run_experiment M2 --output ../runs/M2 --skip-plots

# Train a single model (one command per model)
python -m scripts.run_training uno --data-profile canonical --run-training --output ../runs/uno

# Check files, completion records and figure provenance of published results
python -m scripts.verify_published_results --experiment M3 --result-dir results/formal/M3_cross_resolution
```

Layout: `src/gift/` (generator model, band decomposition, data splits), `training/` (one training entry point per model), `experiments/formal/` (launchers and evaluation for M1&ndash;M3 and S1&ndash;S4), `results/formal/` (summary CSV/JSON and SVG figures), `artifacts/` (weights and checkpoints), `tests/` (data contract, checkpoint identity and result-integrity tests). Data and split definitions are in [EXPERIMENTS.md](EXPERIMENTS.md) section 8; further documentation is under [docs/](docs/).

### 7. Data, code and licence

GIFT software and documentation are released under the [MIT licence](LICENSE). Data packages and external projects are governed by their own terms.

The comparison experiments use the upstream open-source implementations below. Upstream sources are **not distributed with this repository** and must be obtained from their owners under their own terms; only pinned commits and source hashes are recorded here, in [`external_sources.json`](external_sources.json). Adaptations to upstream code are limited to what is necessary, small and individually explainable (data tensor layout, batching, cross-resolution input, history length); see [EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md) for the itemised list. We thank the authors and contributors of these projects, and ask that the original research be cited when using these methods.

| Upstream project | Used for | Pinned version | Licence |
| --- | --- | --- | --- |
| [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | FNO-2D and FNO-3D baselines | `01d2aeca` | See upstream (not declared in this project) |
| [ashiq24/UNO](https://github.com/ashiq24/UNO) | U-NO baseline | `19462d82` | BSD-2-Clause |
| [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | U-Net baseline | `229da3e0` | Not declared upstream; **do not redistribute** |
| [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | PINN-SR configurations | `9a20ebe6` | Not declared upstream |
| [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | PDE-FIND configurations | `86911349` | Not declared upstream |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) (tag `v1.15.0`) | NAdam and L-BFGS optimizer implementations | `590d6eef` | Apache-2.0 |

---

<a id="中文"></a>

## 中文

**基于场层析的生成元识别：具备物理可解释性的流体动力学代理模型**

<sub>从流场轨迹中直接学习连续时间生成元，并把代理模型的预测机制还原为可检验的偏微分控制方程</sub>

### 1. GIFT 学的是什么

传统代理模型直接预测固定时间间隔后的流场；GIFT 换一个对象——它辨识把当前状态映射为**瞬时变化率**的**连续时间生成元**：

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

其中 $\omega$ 为涡量场，$G^{\dagger}$ 是待辨识的真实生成元；上标 † 只用于区分真实对象与学习到的近似，不表示伴随或伪逆。$C$ 是偏置场，$A$ 是平移等变线性算子，$Q$ 是二次齐次算子。三个通道的振幅响应规律不同，$A(\lambda\omega)=\lambda A(\omega)$、$Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$，场层析正是依据这一差异把它们分离。

有限时间间隔后的状态只是生成元的时间积分结果，而生成元本身对应系统的控制规律。因此同一个数学对象可以同时承担两件事：预测机制不再是藏在网络权重里的黑盒映射，而是一个可以被检验物理正确性的连续动力学对象。

### 2. 方法

| 环节 | 一句话说明 |
| --- | --- |
| 生成元分解 | 生成元拆为偏置场 $C$、平移等变线性算子 $A(\omega)$ 与二次齐次算子 $Q(\omega,\omega)$；状态取二维不可压缩流动的涡量场。 |
| 场层析 | 二次通道与线性、偏置场通道**交替训练**，依据各通道振幅响应的差异把它们从数据中分离并重建，训练中无需显式构造齐次特征式。 |
| 二次场相互作用单元 | 非线性由结构化场算子而非激活函数提供：同一输入场经两个可学习谱滤波器后逐点相乘，再对各单元加权求和（面板 E）。 |
| 高频支路 | 补频带 $Q_{21}\omega$ 交由单独训练的网络支路预测，生成元训练结束后训练，两部分耦合积分（面板 D）。 |
| 递归局部修正 | 每个完整 RK4 步后执行的固定规则：按幅值尺度裁剪去均值场的局部中值残差，并设异常网格点安全上限（$N$ = 64、96、128 时分别为 40、90、160）。 |
| 物理可解释性检验 | 生成元冻结后，用已知 Navier–Stokes 方程项拟合各通道输出，将读出系数与真实控制方程系数对比。 |

<p align="center"><img src="assets/figures/gift_architecture.png" alt="GIFT 架构与预测过程" width="100%"></p>

<p align="center"><b>GIFT 架构与预测过程。</b>(A) 稠密采样的连续流场轨迹。(B) 傅里叶变换与固定频带分解构造训练数据：带限状态进入主路径，补频带状态进入高频支路。(C) 场层析：利用各通道振幅响应差异，交替学习二次通道、线性通道与偏置场通道。(D) 高频支路。(E) 二次场相互作用单元。(F) 生成元经四阶 Runge&ndash;Kutta 积分推进至下一时刻。</p>

### 3. 结果

以下均为 [EXPERIMENTS.md](EXPERIMENTS.md) 第 2、3 节所记录协议下**完成并发布**的实测结果：1,000 条 $N$ = 64 训练轨迹（$t$ = 0.0–10.0，间隔 0.02），其中前 50 条构成 GIFT-Lite（Lite 表示训练数据减少，不表示网络更小）；测试集为未参与任何方法训练的 **180 条轨迹**（1040–1219），每个预测时刻单独评价。本项目使用 M 系列编号（M1 方程参数识别、M2 递归预测、M3 跨分辨率预测）；论文附加材料使用另一套按正文叙述顺序排列的 W 系列：W1 递归预测（= M2）、W2 跨分辨率预测（= M3）、W3 方程参数识别（= M1）。

**M2 递归预测**（$N$ = 64，$t$ = 5.0 → 8.0）。GIFT 与 GIFT-Lite 从 $t$ = 5.0 的真值状态出发以 $\Delta t$ = 0.02 递归积分；两个 FNO 基线使用 $t$ = 4.1–5.0 的 46 帧历史状态。平均全场相对 $L^2$ 误差：

| 时间 | GIFT | GIFT-Lite | FNO-2D | FNO-3D | U-NO | U-Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.5 | **0.021901** | 0.036061 | 0.221038 | 0.315175 | 0.030129 | 0.233574 |
| 6.0 | **0.021058** | 0.040763 | 0.271467 | 0.450682 | 0.041116 | 0.500927 |
| 6.5 | **0.021651** | 0.046963 | 0.335630 | 0.595549 | 0.060576 | 0.730189 |
| 7.0 | **0.024000** | 0.054451 | 0.413802 | 0.725142 | 0.086921 | 0.927862 |
| 7.5 | **0.028859** | 0.067797 | 0.515927 | 0.832534 | 0.122484 | 1.091225 |
| 8.0 | **0.036488** | 0.088190 | 0.624730 | 0.915958 | 0.165307 | 1.208053 |

六个报告时刻中 GIFT 的平均误差均最低；六种方法在完整预测区间都保持 180/180 条轨迹有限，统计未剔除失败样本。

<p align="center"><img src="results/formal/M2_recursive_prediction/figures/mean_relative_l2_vs_time.svg" alt="六种方法的平均相对误差随时间变化" width="72%"></p>

<p align="center"><b>图 M2-1｜</b>六种方法在 180 条测试轨迹上的平均全场相对 <i>L</i><sup>2</sup> 误差。GIFT 和 GIFT-Lite 各对三个种子均值取平均，其种子 SD 见实验记录。曲线为严格通过 7 个报告样本点的 PCHIP 插值，仅用于显示趋势。</p>

<p align="center"><img src="results/formal/M2_recursive_prediction/figures/keyframes/traj1045_recursive_keyframes.svg" alt="测试轨迹 1045 在四个关键时刻的预测涡量场与有符号预测误差" width="100%"></p>

<p align="center"><b>图 M2-2｜</b>预先指定的评价轨迹 1045 在 <i>t</i> = 5.0、6.0、7.0、8.0 的涡量场与有符号预测误差。左侧按行给出数值真值与六种方法的预测涡量场，右侧在相同行给出对应的预测误差；数值真值不定义预测误差，其误差格为零误差块。全部涡量场使用同一色标，全部预测误差使用另一独立色标。该轨迹属于独立测试集。</p>

**M3 跨分辨率预测**（zero-shot）。仅用 $N$ = 64 数据训练，网络参数与 Fourier 模态数保持不变，直接在 $N$ = 96、128 网格上从 $t$ = 5.0 开始预测；目标分辨率不参与训练或模型选择。

| $t$ = 6.0 | $N$ = 64 | $N$ = 96 | $N$ = 128 |
| --- | ---: | ---: | ---: |
| **GIFT** | **0.021058** | **0.020947** | **0.020933** |
| GIFT-Lite | 0.040763 | 0.039754 | 0.039742 |
| FNO-2D | 0.271467 | 0.271199 | 0.271295 |
| FNO-3D | 0.450682 | 0.451181 | 0.452352 |

学习到的生成元是网格不变的：每个网格、每个 $t$ > 5.0 的报告时刻，GIFT 与 GIFT-Lite 均低于两个 FNO 基线。跨分辨率实验评价的是模型在所测离散网格上的直接适用性，**不表示可以恢复无限高的频率**。

<p align="center"><img src="results/formal/M3_cross_resolution/figures/cross_resolution_mean_relative_l2.svg" alt="跨分辨率平均全场相对 L2 误差" width="72%"></p>

<p align="center"><b>图 M3-1｜</b>仅在 <i>N</i> = 64 训练的四种方法在三个原生网格上的平均全场相对 <i>L</i><sup>2</sup> 误差，曲线统计 180 条配对基准轨迹。深蓝实线和浅蓝虚线分别为 GIFT 和 GIFT-Lite 三个训练种子的均值，对应色带覆盖三个种子均值的范围，不是置信区间。<i>t</i> = 5.0 是零误差真值锚点，未纳入对数纵轴。</p>

**M1 方程参数识别**。真实系数 $(\nu,\beta,\gamma)$ = (0.01, 1, 1)，不参与训练。五种配置都从同一条局部训练轨迹识别这些参数：GIFT 的生成元在该轨迹上训练，冻结后也在该轨迹上读出系数，与 PDE-FIND、PINN-SR 消费同一条轨迹的方式一致。绝对百分比误差（APE）：

| 噪声 | $\nu$ | $\beta$ | $\gamma$ |
| --- | ---: | ---: | ---: |
| 0% | **4.343%** | **1.619%** | **4.121%** |
| 1% | **11.890%** | **2.209%** | **4.958%** |
| 10% | 69.791% | 21.702% | **11.482%** |

0% 与 1% 噪声下，GIFT 对三个参数的 APE 均为五种配置中最低；10% 噪声下 $\nu$ 显著退化，说明本协议不是高噪声下识别非线性系数的方案。该能力属于**已知控制方程结构下的参数识别**，不是在未知候选结构中从头发现任意 PDE，不能替代 PDE-FIND 或 PINN-SR 所面向的「结构与参数均未知」任务。

<p align="center"><img src="results/formal/M1_equation_identification/figures/parameter_identification_ape_vs_noise.svg" alt="参数识别的绝对百分比误差" width="72%"></p>

<p align="center"><b>图 M1-1｜</b>五种配置在 0%、1% 和 10% 噪声条件下对三个方程参数的绝对百分比误差。三个面板共用一条纵轴：20% 及以下为线性、以上为对数，刻度已标注。</p>

### 4. 支线实验

| 实验 | 结论 |
| --- | --- |
| **S1 高频支路** | $N$ = 64、96、128 的 $t$ = 6.0 处分别降低 GIFT 全场误差 60.04%、57.54%、57.55%，降低 GIFT-Lite 33.63%、31.98%、31.98%。 |
| **S2 递归局部修正** | 在独立测试集上避免 GIFT-Lite 的轨迹 1062 出现非有限值；全数据 GIFT 未触发修正。 |
| **S3 随机种子稳定性** | 固定生成元、独立训练高频支路三个种子，GIFT 与 GIFT-Lite 主要指标的最大变异系数为 0.321% 与 0.313%。 |
| **S4 初值分布对照** | 在另一组平滑高斯随机场初值分布上，全数据 GIFT 的报告时刻平均误差仍为最低（$t$ = 8.0 为 0.021035），U-NO 为最接近的基线（0.173805）。全数据 GIFT 与四个基线均保持 180/180 条测试轨迹有限；GIFT-Lite 有 17/180 条在 $t$ ≈ 6.9 之后失稳，其后续数值为有限子集统计并已如实标注。 |

### 5. 适用范围

结论限于本项目的数据分布、训练协议与 $N$ = 64、96、128 网格，不构成任意分辨率上的误差保证或数值稳定性定理；相同 epoch 也不代表相同参数更新次数或计算量：GIFT 从 $t$ = 5.0 的单状态出发，而 FNO 基线使用 46 帧历史状态，且 GIFT 的生成元与高频支路分项训练。完整口径（含种子标准差、有限性计数与逐任务计时）见 [EXPERIMENTS.md](EXPERIMENTS.md) 第 6 节与 [TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md)。

### 6. 快速开始

数据与第三方实现位于项目之外，分别通过 `GIFT_DATA_ROOT` 与 `GIFT_EXTERNAL_ROOT` 指定；安装方式见 [SETUP.md](docs/SETUP.md)。每个模型与实验分别运行。

```shell
# 使用随项目提供的权重直接评价
python -m scripts.run_experiment M2 --output ../runs/M2 --skip-plots

# 独立训练一个模型（每个模型一条命令）
python -m scripts.run_training uno --data-profile canonical --run-training --output ../runs/uno

# 核对已发布结果的文件、完成记录与图像来源
python -m scripts.verify_published_results --experiment M3 --result-dir results/formal/M3_cross_resolution
```

代码组织：`src/gift/`（生成元模型、频带分解、数据划分）、`training/`（各模型独立训练入口）、`experiments/formal/`（M1–M3、S1–S4 启动与评价）、`results/formal/`（汇总 CSV/JSON 与 SVG 图像）、`artifacts/`（权重与检查点）、`tests/`（数据契约、检查点身份与结果完整性测试）。数据与划分定义见 [EXPERIMENTS.md](EXPERIMENTS.md) 第 8 节，其余文档见 [docs/](docs/)。

### 7. 数据、代码与许可

GIFT 软件与文档使用 [MIT 许可](LICENSE)。数据包与外部项目适用各自的条款。

对比实验使用以下上游开源实现。上游源码**不随本仓库分发**，需按各自条款从上游获取；本仓库只记录固定提交与源码散列（见 [`external_sources.json`](external_sources.json)）。对上游实现只做必要、少量且可逐项说明的适配（数据张量布局、批量组织、跨分辨率输入、历史帧数等），逐项说明见 [EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md)。对各项目的作者与贡献者表示感谢，使用相关方法时请引用其原始研究。

| 上游项目 | 用途 | 固定版本 | 许可状态 |
| --- | --- | --- | --- |
| [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | FNO-2D 与 FNO-3D 基线 | `01d2aeca` | 见上游（本项目内未声明） |
| [ashiq24/UNO](https://github.com/ashiq24/UNO) | U-NO 基线 | `19462d82` | BSD-2-Clause |
| [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | U-Net 基线 | `229da3e0` | 上游未声明许可，**不得再分发** |
| [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | PINN-SR 两种配置 | `9a20ebe6` | 上游未声明许可 |
| [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | PDE-FIND 两种配置 | `86911349` | 上游未声明许可 |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)（tag `v1.15.0`） | NAdam 与 L-BFGS 优化器实现 | `590d6eef` | Apache-2.0 |
