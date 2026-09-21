<div align="center">

# GIFT

**Generator Identification via Field Tomography**

<sub>Learning the continuous-time generator of a flow directly from trajectories, and turning the surrogate's prediction mechanism into an explicit, checkable PDE</sub>

[**English**](#english) &nbsp;·&nbsp; [**中文**](#中文)

</div>

---

## English

### In one sentence

Traditional surrogates predict the field a fixed time lag ahead; GIFT changes the object of learning. It learns the **continuous-time generator** that maps a state to its instantaneous rate of change:

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega)$$

where $\omega$ is the vorticity field, $G^{\dagger}$ is the true generator to be identified (the dagger only distinguishes the true object from the learned approximation; it does not denote an adjoint or a pseudoinverse), and $C$, $A$, $Q$ are the bias-field, linear and nonlinear channels. The channels differ in amplitude response — $A(\lambda\omega) = \lambda A(\omega)$ and $Q(\lambda\omega,\lambda\omega) = \lambda^{2}Q(\omega,\omega)$ — and that difference is what field tomography separates.

A state a finite lag later is merely the time integral of that generator, and the generator *is* the governing law. Prediction and explicit PDE-parameter identification therefore share one mathematical object: the surrogate's mechanism is no longer a black-box map hidden in network weights, but a continuous dynamical object whose physical correctness can be checked.

Section, figure, table and equation numbering throughout this README follow [EXPERIMENTS.md](EXPERIMENTS.md), the project's experiment record, so that every item can be looked up there directly; sections 1–2 of that document are folded into the overview above and into the protocol paragraph of section 4.

### 3. Method

| Component | Description |
| --- | --- |
| Generator decomposition | The generator splits into a state-independent but spatially varying bias field `C`, a translation-equivariant linear operator `A(ω)`, and a nonlinear operator `Q(ω, ω)`. The state is the vorticity field of the two-dimensional incompressible Navier–Stokes system. |
| Field tomography | Translation equivariance of the linear operator and quadratic homogeneity of the nonlinear operator are distinct signatures in amplitude response. Exploiting that difference, the quadratic path is trained **alternately** with the linear and bias paths so that the paths are separated and reconstructed from data, without writing down the homogeneity identities explicitly. |
| Quadratic field-interaction unit | Nonlinearity comes from a structured field operator rather than an activation function: two learnable spectral filters act on the same input field, their outputs are multiplied pointwise, and the per-unit results are summed with weights (panel E below). |
| High-frequency branch | The `Q`<sub>21</sub> high-frequency component is predicted by a separately trained branch, trained after the generator and then coupled with it in one integration (panel D below). |
| Recursive local correction | A fixed, gradient-free rule applied after each complete RK4 step: local median residuals of the de-meaned field are clipped at an amplitude scale of `ℓ` = 2.1`σ`, under an anomalous-grid-point safety cap `K`<sub>N</sub> (40, 90, 160 for `N` = 64, 96, 128). |
| Physical interpretability check | With the generator frozen, known N–S equation terms are fitted to each path's output and the recovered coefficients are compared with the true ones — a quantitative test of whether the surrogate learned the correct physics. |

<div align="center">
  <img src="assets/figures/gift_architecture.png" alt="GIFT architecture and prediction process" width="100%">
</div>

<div align="center"><sub><b>GIFT architecture and prediction process.</b> (A) Densely sampled continuous flow trajectories. (B) Fourier transform and fixed-band decomposition build the training data: the band-limited state ω<sub>K</sub> = <i>P</i><sub>21</sub>ω enters the main path, while the complement ω<sub>Q</sub> = <i>Q</i><sub>21</sub>ω enters the high-frequency branch. (C) Field tomography: differing amplitude responses separate the quadratic, linear and bias-field paths, which are learned alternately. (D) High-frequency branch. (E) Quadratic field-interaction unit. (F) The generator is advanced to the next instant by fourth-order Runge–Kutta integration.</sub></div>

### 4. Main experiments

Every number below is a completed and published measurement under the protocol recorded in [EXPERIMENTS.md](EXPERIMENTS.md) sections 2–3: 1,000 `N` = 64 training trajectories (`t` = 0.0–10.0, spacing 0.02), of which GIFT-Lite retains the first 50 (Lite denotes reduced training data, not a smaller network). The test set is **180 trajectories** (1040–1219) that took part in no method's training, and each prediction time is scored separately.

#### 4.1 M1 | Equation-parameter identification

True coefficients `(ν, β, γ)` = (0.01, 1, 1), excluded from training. All five configurations identify those parameters from the same single local training trajectory (trajectory 0): GIFT's generator is trained on it and the frozen generator is read out from it, exactly as PDE-FIND and PINN-SR consume that one trajectory.

| Noise | GIFT ν APE | GIFT β APE | GIFT γ APE |
| --- | ---: | ---: | ---: |
| 0% | **4.343%** | **1.619%** | **4.121%** |
| 1% | **11.890%** | **2.209%** | **4.958%** |
| 10% | 69.791% | 21.702% | **11.482%** |

At 0% and 1% noise GIFT attains the lowest absolute percentage error (APE) on all three parameters among the five configurations; at 10% noise `ν` degrades sharply, so this protocol is not a high-noise method for the nonlinear coefficient.

> ⚠️ This capability is **parameter identification under a known governing-equation structure**, not discovery of an arbitrary PDE from an unknown candidate library, and it does not stand in for the "structure and parameters both unknown" setting that PDE-FIND or PINN-SR address.

<div align="center">
  <img src="results/formal/M1_equation_identification/figures/parameter_identification_ape_vs_noise.svg" alt="Absolute percentage error of the identified coefficients" width="72%">
</div>

<div align="center"><sub><b>Figure M1-1 |</b> Absolute percentage error of the identified $\nu$, $\beta$ and $\gamma$ for the five configurations at 0%, 1% and 10% noise. The three panels share one vertical axis: linear up to 20% and logarithmic above it, with labelled ticks.</sub></div>

#### 4.2 M2 | Recursive prediction (`N` = 64, `t` = 5.0 → 8.0)

GIFT and GIFT-Lite start from the true state at `t` = 5.0 and integrate recursively with Δ`t` = 0.02; the two FNO baselines take the 46 historical states from `t` = 4.1–5.0 as input.

| Time | GIFT | GIFT-Lite | FNO-2D | FNO-3D | U-NO | U-Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.5 | **0.021901** | 0.036061 | 0.221038 | 0.315175 | 0.030129 | 0.233574 |
| 6.0 | **0.021058** | 0.040763 | 0.271467 | 0.450682 | 0.041116 | 0.500927 |
| 6.5 | **0.021651** | 0.046963 | 0.335630 | 0.595549 | 0.060576 | 0.730189 |
| 7.0 | **0.024000** | 0.054451 | 0.413802 | 0.725142 | 0.086921 | 0.927862 |
| 7.5 | **0.028859** | 0.067797 | 0.515927 | 0.832534 | 0.122484 | 1.091225 |
| 8.0 | **0.036488** | 0.088190 | 0.624730 | 0.915958 | 0.165307 | 1.208053 |

GIFT has the lowest mean full-field relative error at all six reported times, and all six methods stay finite on 180/180 trajectories over the full interval; no failed sample is excluded from the statistics.

<div align="center">
  <img src="results/formal/M2_recursive_prediction/figures/mean_relative_l2_vs_time.svg" alt="Mean relative L2 error vs time" width="72%">
</div>

<div align="center"><sub><b>Figure M2-1 |</b> Mean full-field relative <i>L</i><sup>2</sup> error of the six methods over 180 test trajectories; GIFT and GIFT-Lite are averaged over three training seeds, whose seed SD is tabulated in EXPERIMENTS.md. Curves are PCHIP interpolants through exactly the seven reported sample points, shown for trend only.</sub></div>

<div align="center">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/traj1045_recursive_keyframes.svg" alt="Predicted vorticity fields and signed prediction errors of trajectory 1045 at four key times" width="100%">
</div>

<div align="center"><sub><b>Figure M2-2 |</b> Prespecified evaluation trajectory 1045 at <i>t</i> = 5.0, 6.0, 7.0, 8.0: the numerical reference and the six methods' predicted vorticity fields (left), each with the signed prediction error of the same row (right); a numerical reference does not define a prediction error, so its error cells are the flat zero tile. Every method's field and prediction error are aligned strictly by row. All vorticity fields share one colour scale and all prediction errors share a second, independent one. GIFT and GIFT-Lite use the prespecified training seed 20260820, fixed before this evaluation and not reselected from these numbers; no field-level cross-seed averaging is applied. The trajectory belongs to the held-out test set.</sub></div>

#### 4.3 M3 | Cross-resolution prediction (zero-shot)

Trained on `N` = 64 only, with network parameters and Fourier modes unchanged, then evaluated directly on `N` = 96 and `N` = 128 grids from `t` = 5.0; the target resolution takes no part in training or model selection.

| `t` = 6.0 | `N` = 64 | `N` = 96 | `N` = 128 |
| --- | ---: | ---: | ---: |
| **GIFT** | **0.021058** | **0.020947** | **0.020933** |
| GIFT-Lite | 0.040763 | 0.039754 | 0.039742 |
| FNO-2D | 0.271467 | 0.271199 | 0.271295 |
| FNO-3D | 0.450682 | 0.451181 | 0.452352 |

The learned generator is grid-invariant: on every grid and at every reported time `t` > 5.0, GIFT and GIFT-Lite stay below both FNO baselines. The experiment measures direct applicability on the discrete grids tested and **does not imply recovery of arbitrarily high frequencies**.

<div align="center">
  <img src="results/formal/M3_cross_resolution/figures/cross_resolution_mean_relative_l2.svg" alt="Cross-resolution mean relative L2 error" width="72%">
</div>

<div align="center"><sub><b>Figure M3-1 |</b> Mean full-field relative <i>L</i><sup>2</sup> error of four methods trained on <i>N</i> = 64 only, on three native grids (180 paired trajectories). The solid dark-blue and dashed light-blue lines are the means over three training seeds for GIFT and GIFT-Lite; the band spans the range of those seed means, not a confidence interval. <i>t</i> = 5.0 is the zero-error truth anchor and is not included in the logarithmic vertical axis.</sub></div>

<div align="center">
  <img src="results/formal/M3_cross_resolution/figures/traj1150_n128_t6_prediction_fields.svg" alt="Reference field, four methods' predicted vorticity fields and signed prediction errors at N=128, t=6.0" width="100%">
</div>

<div align="center"><sub><b>Figure M3-2 |</b> Prespecified trajectory 1150 at <i>N</i> = 128, <i>t</i> = 6.0: vorticity prediction and prediction error. The trajectory and the GIFT training seed 20260820 were fixed before this figure was drawn. (a) The reference field and the predictions of GIFT, GIFT-Lite, FNO-2D and FNO-3D, sharing the symmetric `RdBu_r` limits [−13, 13]. (b) The signed prediction error, prediction minus reference, Δ<i>ω</i> = <i>ω̂</i> − <i>ω</i>; the four methods share `PuOr` zero-centred symmetric limits [−7, 7], and the number under each error panel is the full-field relative <i>L</i><sup>2</sup> error. All field panels cover exactly the same spatial extent and grid, with no colour clipping, cropping or smoothing.</sub></div>

### 5. Side experiments (S1–S4)

| Experiment | Result |
| --- | --- |
| **S1 High-frequency branch** | At `t` = 6.0 it reduces the GIFT full-field error by 60.04%, 57.54% and 57.55% on `N` = 64, 96 and 128, and the GIFT-Lite error by 33.63%, 31.98% and 31.98%. |
| **S2 Recursive local correction** | On the held-out test set it keeps trajectory 1062 of GIFT-Lite finite. Full-data GIFT never triggered the correction. |
| **S3 Random-seed stability** | With the generator fixed and the high-frequency branch trained under three seeds, the largest coefficient of variation on the main metrics is about 0.321% for GIFT and 0.313% for GIFT-Lite. |
| **S4 Initial-condition distribution** | Repeating the comparison on a separate smooth Gaussian random-field population (training 1220–2219, test 2260–2439), full-data GIFT still has the lowest mean error at every reported time (`t` = 8.0: 0.021035) and U-NO is the closest baseline (0.173805). Full-data GIFT and all four baselines stay finite on 180/180 test trajectories; GIFT-Lite loses 17 of 180 after `t` ≈ 6.9, so its later values are finite-subset statistics and are reported as such. |

<div align="center">
  <img src="results/formal/S4_initial_distribution/figures/curves/initial_distribution_comparison.svg" alt="Mean relative error against time for two initial-condition populations" width="100%">
</div>

<div align="center"><sub><b>Figure S4-1 |</b> Mean full-field relative error against time for the four-vortex (left) and Gaussian random-field (right) initial-condition populations. Both panels come from one plotting program and use the same vertical range and ticks, method colours, reported times and relative-<i>L</i><sup>2</sup> definition. The left panel is redrawn from the published M2 summary data; no M2 value or reported time changed, and M2 was not recomputed. The finite-subset interval for GIFT-Lite is marked inside the right panel.</sub></div>

<div align="center">
  <img src="results/formal/S4_initial_distribution/figures/keyframes/traj2265_recursive_keyframes.svg" alt="Predicted vorticity fields and signed prediction errors of trajectory 2265 at four key times" width="100%">
</div>

<div align="center"><sub><b>Figure S4-3 |</b> Prespecified test trajectory 2265 under Gaussian random-field initial conditions at <i>t</i> = 5.0, 6.0, 7.0, 8.0: vorticity fields and signed prediction error, on the same plate layout as Figure S4-2. Row and column order, alignment, colour-scale independence and the seed convention all follow M2.</sub></div>

Figure S4-2 is the published M2 example-trajectory plate, cited in the experiment document for side-by-side comparison; it is the same file as Figure M2-2 above and is not repeated here.

### 6. Conclusion and scope

On the protocol above, GIFT has the lowest mean full-field relative error at every reported positive prediction time, on all three native grids and under both initial-condition populations, and the identified generator is the object that carries that performance: it is read out as an explicit PDE for parameter identification (M1), applied unchanged to grids it was not trained on (M3), and reused on a different initial-condition population (S4).

> **Scope.** Conclusions are limited to this project's data distribution, training protocol and the `N` = 64, 96, 128 grids; they are not an error guarantee or a numerical-stability theorem at arbitrary resolution, and equal epochs do not mean equal parameter-update counts or equal compute. Full detail in [EXPERIMENTS.md](EXPERIMENTS.md) section 6 and [TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md).

### 7. Quick start and reproduction

Data and third-party implementations live outside the project and are located through `GIFT_DATA_ROOT` and `GIFT_EXTERNAL_ROOT`; installation is described in [SETUP.md](docs/SETUP.md). Each model and each experiment runs separately.

```shell
# Evaluate directly with the weights shipped with the project
python -m scripts.run_experiment M2 --output ../runs/M2 --skip-plots

# Train a single model (one command per model)
python -m scripts.run_training uno --data-profile canonical --run-training --output ../runs/uno

# Check files, completion records and figure provenance of published results
python -m scripts.verify_published_results --experiment M3 --result-dir results/formal/M3_cross_resolution
```

Layout: `src/gift/` (generator model, band decomposition, data splits), `training/` (one training entry point per model), `experiments/formal/` (launchers and evaluation for M1–M3 and S1–S4), `results/formal/` (summary CSV/JSON and SVG figures), `artifacts/` (weights and checkpoints), `tests/` (data contract, checkpoint identity and result-integrity tests). Further documentation is under [docs/](docs/).

### 8. Trajectory numbering and data splits

The prediction data form two independent populations. Within each population the training indices come first, then validation and test; a trajectory keeps its index across resolutions, and the Gaussian population continues the numbering after the four-vortex population rather than pairing with it frame by frame.

| Initial-condition population | Training | Validation | Test | GIFT-Lite training subset |
| --- | --- | --- | --- | --- |
| Four-vortex | 0–999 (1000) | 1000–1039 (40) | 1040–1219 (180) | 0–49 (50) |
| Smooth Gaussian random field | 1220–2219 (1000) | 2220–2259 (40) | 2260–2439 (180) | 1220–1269 (50) |

M1 uses its own local numbering (training 0–49, validation 50–69) and is not mixed with the prediction populations. Training, validation and test do not overlap; the GIFT-Lite subset is drawn from the corresponding training set, and validation or test data never update parameters. The per-experiment split table is in [EXPERIMENTS.md](EXPERIMENTS.md) section 8.

### 9. Cross-domain applicability (diagnostic experiments)

The method's core — learning a constant + linear + quadratic spectral right-hand side from observed field sequences and integrating it with RK4 — is not specific to the Navier–Stokes equations. Transfer was tested on three first-order scalar systems from other disciplines, each trained from scratch as a diagnostic `fixture` run with the unchanged full-data generator (500 trajectory epochs, 250 training trajectories, generator component only):

| Equation | Setting | Potential application areas | Recursive rollout, mean relative L² at lead 3.0 s (persistence baseline) |
| --- | --- | --- | ---: |
| `u_t = kappa * Lap(u)` | Linear diffusion | Heat conduction; mass diffusion (Fick) and contaminant transport; groundwater flow; heat-kernel smoothing | **0.0239** (0.4211) |
| `u_t = -c·grad(u)` | Linear advection | Pollutant and tracer transport; advection stages of weather and climate models; linear acoustic propagation in characteristic form | **0.0245** (1.4020) |
| `u_t = 0.02 * Lap(u) + r u (1 - u)` | Nonlinear reaction–diffusion | Population dynamics and epidemic invasion fronts (Fisher–KPP); chemical waves; combustion fronts; tumour growth | **0.0051** (0.0064) |

Rollouts start from the frame at `t` = 5.0 and predict 150 steps (`dt` = 0.02) through `t` = 8.0, averaged over 20 held-out trajectories. These diagnostics use their own entry points:

```shell
python -m scripts.generate_crossdomain_data --kind heat --output ../crossdomain/heat --execute
python -m scripts.run_crossdomain --kind heat --data ../crossdomain/heat/heat.h5 --output ../runs/crossdomain_heat --execute
```

> **Scope of the transfer claim.** These are diagnostic `fixture` measurements, not formal project experiments: one fixed parameter set per equation (a single autonomous system with varying initial conditions), a single scalar field per problem, and the generator component without the high-frequency branch. Mixing physical parameters across trajectories makes the state-to-derivative map ill-posed; second-order systems such as the full acoustic wave equation need an augmented state; non-periodic domains need another basis. See [EXPERIMENTS.md](EXPERIMENTS.md) section 9.

### External code and acknowledgements

The comparison experiments use the upstream open-source implementations below. Upstream sources are **not distributed with this repository** and must be obtained from their owners under their own terms; only pinned commits and source hashes are recorded here for reproducibility. We thank the authors and contributors of these projects.

| Upstream project | Used for | Pinned version | Licence |
| --- | --- | --- | --- |
| [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | Network structure and training objective of the FNO-2D and FNO-3D baselines | `01d2aeca` | See upstream (not declared in this project) |
| [ashiq24/UNO](https://github.com/ashiq24/UNO) | U-shaped neural operator and integral operator of the U-NO baseline | `19462d82` | BSD-2-Clause |
| [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | Encoder/decoder structure of the U-Net baseline | `229da3e0` | Not declared upstream; **do not redistribute** |
| [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | PINN-SR and PINN-SR-KC configurations | `9a20ebe6` | Not declared upstream |
| [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | PDE-FIND and PDE-FIND-KC configurations | `86911349` | Not declared upstream |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) (tag `v1.15.0`) | NAdam and L-BFGS external optimizer implementations | `590d6eef` | Apache-2.0 |

Adaptations to upstream code are limited to what is necessary, small, and individually explainable (data tensor layout, batching, cross-resolution input, history length). Autograd is not taken over, backward operators are not replaced, and optimizer kernels are not reimplemented. See [EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md) for the itemised list and [`external_sources.json`](external_sources.json) for pinned commits and SHA-256 hashes. Please cite the original research when using these methods.

### License

GIFT software and documentation are released under the [MIT licence](LICENSE). Data packages and external projects are governed by their own terms.

---

<a id="中文"></a>

## 中文

**基于场层析的生成元识别：具备物理可解释性的流体动力学代理模型**

<sub>从流场轨迹中直接学习连续时间生成元，并把代理模型的预测机制还原为可检验的偏微分控制方程</sub>

### 一句话概括

传统代理模型直接预测固定时间间隔后的流场；GIFT 换一个对象——它学习把当前状态映射为**瞬时变化率**的**连续时间生成元**：

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega)$$

其中 $\omega$ 为涡量场，$G^{\dagger}$ 是待辨识的真实生成元（上标 † 只用于区分真实对象与学习到的近似，不表示伴随或伪逆），$C$、$A$、$Q$ 分别是偏置场通道、线性通道与非线性通道。三个通道的振幅响应规律不同——$A(\lambda\omega) = \lambda A(\omega)$，$Q(\lambda\omega,\lambda\omega) = \lambda^{2}Q(\omega,\omega)$——场层析正是依据这一差异把它们分离。

有限时间间隔后的状态只是生成元的时间积分结果，而生成元本身对应系统的控制规律。因此代理模型的预测与显式 PDE 参数识别共享同一个数学对象：预测机制不再是藏在网络权重里的黑盒映射，而是一个可以被检验物理正确性的连续动力学对象。

本 README 的章节号、图号、表号与公式号一律以 [EXPERIMENTS.md](EXPERIMENTS.md)（本项目的实验记录）为准，便于逐条对应查阅；其中第 1、2 节的内容并入上文概述与第 4 节的协议说明。

### 3. 方法

| 环节 | 内容 |
| --- | --- |
| 生成元分解 | 生成元拆为与状态无关但可随空间变化的偏置场 `C`、平移等变线性算子 `A(ω)` 与非线性算子 `Q(ω, ω)`；状态取二维不可压缩 Navier–Stokes 的涡量场。 |
| 场层析 | 线性算子的平移等变性与非线性算子的二次齐次性是不同特征响应；按此差异**交替训练**二次通道与线性、偏置场通道，把它们从数据中分离并重建，训练中无需显式构造齐次特征式。 |
| 二次场相互作用单元 | 非线性由结构化场算子而非激活函数提供：同一输入场经两个可学习谱滤波器后逐点相乘，再对各单元加权求和（下方面板 E）。 |
| 高频支路 | `Q`<sub>21</sub> 高频分量交由单独训练的网络支路预测，生成元训练结束后训练，两部分耦合积分（下方面板 D）。 |
| 递归局部修正 | 每个完整 RK4 步后执行的固定规则：按幅值尺度 `ℓ` = 2.1`σ` 裁剪去均值场的局部中值残差，并设异常网格点安全上限 `K`<sub>N</sub>（`N` = 64、96、128 时分别为 40、90、160）。 |
| 物理可解释性检验 | 生成元冻结后，用已知 N–S 方程项拟合各通道输出，将读出系数与真实控制方程系数对比——对「是否学到正确物理」的可量化检验。 |

<div align="center">
  <img src="assets/figures/gift_architecture.png" alt="GIFT 架构与预测过程" width="100%">
</div>

<div align="center"><sub><b>GIFT 架构与预测过程。</b>(A) 稠密采样的连续流场轨迹。(B) 傅里叶变换与固定频带分解构造训练数据：带限状态 ω<sub>K</sub> = <i>P</i><sub>21</sub>ω 进入主路径，补状态 ω<sub>Q</sub> = <i>Q</i><sub>21</sub>ω 进入高频支路。(C) 场层析：利用各通道振幅响应差异，交替学习二次通道、线性通道与偏置场通道。(D) 高频支路。(E) 二次场相互作用单元。(F) 生成元经四阶 Runge–Kutta 积分推进至下一时刻。</sub></div>

### 4. 主线实验

以下均为 [EXPERIMENTS.md](EXPERIMENTS.md) 第 2、3 节所记录协议下**完成并发布**的实测结果：1,000 条 `N` = 64 训练轨迹（`t` = 0.0–10.0，间隔 0.02），另保留使用其中前 50 条的 GIFT-Lite（Lite 表示训练数据减少，不表示网络更小）；测试集为未参与任何方法训练的 **180 条轨迹**（1040–1219），每个预测时刻单独评价。

#### 4.1 M1｜方程参数识别

真实系数 `(ν, β, γ)` = (0.01, 1, 1)，真实系数不参与训练。五种配置都从同一条局部训练轨迹（轨迹 0）识别这些参数：GIFT 的生成元在该轨迹上训练，冻结后也在该轨迹上读出系数，与 PDE-FIND、PINN-SR 消费同一条轨迹的方式一致。

| 噪声 | GIFT ν APE | GIFT β APE | GIFT γ APE |
| --- | ---: | ---: | ---: |
| 0% | **4.343%** | **1.619%** | **4.121%** |
| 1% | **11.890%** | **2.209%** | **4.958%** |
| 10% | 69.791% | 21.702% | **11.482%** |

0% 与 1% 噪声下，GIFT 对三个参数的绝对百分比误差（APE）均为五种配置中最低；10% 噪声下 `ν` 显著退化，说明本协议不是高噪声下识别非线性系数的方案。

> ⚠️ 该能力属于**已知控制方程结构下的参数识别**，不是在未知候选结构中从头发现任意 PDE，不能替代 PDE-FIND 或 PINN-SR 所面向的「结构与参数均未知」任务。

<div align="center">
  <img src="results/formal/M1_equation_identification/figures/parameter_identification_ape_vs_noise.svg" alt="参数识别的绝对百分比误差" width="72%">
</div>

<div align="center"><sub><b>图 M1-1｜</b>五种配置在 0%、1% 和 10% 噪声条件下对 $\nu$、$\beta$ 和 $\gamma$ 的绝对百分比误差。三个面板共用一条纵轴：20% 及以下为线性、以上为对数，刻度已标注。</sub></div>

#### 4.2 M2｜递归预测（`N` = 64，`t` = 5.0 → 8.0）

GIFT 与 GIFT-Lite 从 `t` = 5.0 的真值状态出发以 Δ`t` = 0.02 递归积分；两个 FNO 基线使用 `t` = 4.1–5.0 的 46 帧历史状态。

| 时间 | GIFT | GIFT-Lite | FNO-2D | FNO-3D | U-NO | U-Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.5 | **0.021901** | 0.036061 | 0.221038 | 0.315175 | 0.030129 | 0.233574 |
| 6.0 | **0.021058** | 0.040763 | 0.271467 | 0.450682 | 0.041116 | 0.500927 |
| 6.5 | **0.021651** | 0.046963 | 0.335630 | 0.595549 | 0.060576 | 0.730189 |
| 7.0 | **0.024000** | 0.054451 | 0.413802 | 0.725142 | 0.086921 | 0.927862 |
| 7.5 | **0.028859** | 0.067797 | 0.515927 | 0.832534 | 0.122484 | 1.091225 |
| 8.0 | **0.036488** | 0.088190 | 0.624730 | 0.915958 | 0.165307 | 1.208053 |

六个报告时刻中 GIFT 的平均全场相对误差均最低；六种方法在完整预测区间都保持 180/180 条轨迹有限，统计未剔除失败样本。

<div align="center">
  <img src="results/formal/M2_recursive_prediction/figures/mean_relative_l2_vs_time.svg" alt="平均全场相对 L2 误差随时间的变化" width="72%">
</div>

<div align="center"><sub><b>图 M2-1｜</b>六种方法在 180 条测试轨迹上的平均全场相对误差随时间的变化；GIFT 和 GIFT-Lite 各对三个种子均值取平均，其种子 SD 见表。曲线为严格通过 7 个报告样本点的形状保持分段三次 Hermite（PCHIP）插值，仅用于显示趋势。</sub></div>

<div align="center">
  <img src="results/formal/M2_recursive_prediction/figures/keyframes/traj1045_recursive_keyframes.svg" alt="测试轨迹 1045 在四个关键时刻的预测涡量场与有符号预测误差" width="100%">
</div>

<div align="center"><sub><b>图 M2-2｜</b>预先指定的评价轨迹 1045 在 $t=5.0,6.0,7.0,8.0$ 的涡量场与有符号预测误差。左侧按行给出数值真值与六种方法的预测涡量场，右侧在相同行给出对应方法的预测值减去真值的预测误差；数值真值不定义预测误差，其误差格为零误差浅灰块。每种方法的预测场与预测误差严格按行对齐。全部涡量场使用同一色标，全部预测误差使用另一独立色标。GIFT 和 GIFT-Lite 图像均采用预先指定的随机种子 20260820，不根据本次数值重新挑选，也不对场变量进行跨种子平均。该轨迹属于独立测试集。</sub></div>

#### 4.3 M3｜跨分辨率预测（zero-shot）

仅用 `N` = 64 数据训练，网络参数与 Fourier 模态数保持不变，直接在 `N` = 96、128 网格上从 `t` = 5.0 开始预测；目标分辨率不参与训练或模型选择。

| `t` = 6.0 | `N` = 64 | `N` = 96 | `N` = 128 |
| --- | ---: | ---: | ---: |
| **GIFT** | **0.021058** | **0.020947** | **0.020933** |
| GIFT-Lite | 0.040763 | 0.039754 | 0.039742 |
| FNO-2D | 0.271467 | 0.271199 | 0.271295 |
| FNO-3D | 0.450682 | 0.451181 | 0.452352 |

学习到的生成元是网格不变的：每个网格、每个 `t` > 5.0 的报告时刻，GIFT 与 GIFT-Lite 均低于两个 FNO 基线。跨分辨率实验评价的是模型在所测离散网格上的直接适用性，**不表示可以恢复无限高的频率**。

<div align="center">
  <img src="results/formal/M3_cross_resolution/figures/cross_resolution_mean_relative_l2.svg" alt="跨分辨率平均全场相对 L2 误差" width="72%">
</div>

<div align="center"><sub><b>图 M3-1｜</b>仅在 $N=64$ 训练的 GIFT、GIFT-Lite、FNO-2D 和 FNO-3D 在三个原生网格上的平均全场相对 $L^2$ 误差。曲线统计 180 条配对基准轨迹；深蓝实线和浅蓝虚线分别为 GIFT 和 GIFT-Lite 三个训练种子的均值，对应色带覆盖三个种子均值的范围，不是置信区间；两个 FNO 基线各使用一组固定模型参数。$t=5.0$ 是零误差真值锚点，未纳入对数纵轴。</sub></div>

<div align="center">
  <img src="results/formal/M3_cross_resolution/figures/traj1150_n128_t6_prediction_fields.svg" alt="N=128、t=6.0 的真值场、四种方法预测场及有符号预测误差" width="100%">
</div>

<div align="center"><sub><b>图 M3-2｜</b>预先指定的轨迹 1150 在 $N=128$、$t=6.0$ 的涡量预测与预测误差。轨迹 1150 和两种 GIFT 的随机种子 20260820 在本次作图前固定，不根据本次数值重新挑选。a，真值场以及 GIFT、GIFT-Lite、FNO-2D 和 FNO-3D 的预测场，共用 `RdBu_r` 对称色限 $[-13,13]$。b，有符号预测误差，定义为预测值减去真值，即 $\Delta\omega=\hat\omega-\omega$；四种方法共用区别于标量场的 `PuOr` 零中心对称色限 $[-7,7]$，预测误差图下方数值为全场相对 $L^2$ 误差。全部场图的空间范围与网格尺寸完全一致，未发生色彩裁切，也未进行图像裁剪或平滑。</sub></div>

### 5. 支线实验（S1–S4）

| 实验 | 结论 |
| --- | --- |
| **S1 高频支路** | `N` = 64、96、128 的 `t` = 6.0 处分别降低 GIFT 全场误差 60.04%、57.54%、57.55%，降低 GIFT-Lite 33.63%、31.98%、31.98%。 |
| **S2 递归局部修正** | 在独立测试集上避免 GIFT-Lite 的轨迹 1062 出现非有限值；全数据 GIFT 未触发修正。 |
| **S3 随机种子稳定性** | 固定生成元、独立训练高频支路三个种子，GIFT 与 GIFT-Lite 主要指标的最大变异系数约 0.321% 与 0.313%。 |
| **S4 初值分布对照** | 在另一组平滑高斯随机场初值分布上重复同一比较（训练 1220–2219、测试 2260–2439），全数据 GIFT 的报告时刻平均误差仍为最低（`t` = 8.0 为 0.021035），U-NO 为最接近的基线（0.173805）。全数据 GIFT 与四个基线均保持 180/180 条测试轨迹有限；GIFT-Lite 有 17/180 条在 `t` ≈ 6.9 之后失稳，其后续数值为有限子集统计并已如实标注。 |

<div align="center">
  <img src="results/formal/S4_initial_distribution/figures/curves/initial_distribution_comparison.svg" alt="两种初值分布下的平均全场相对误差随时间变化" width="100%">
</div>

<div align="center"><sub><b>图 S4-1｜</b>左、右两幅分别是四涡旋初值和高斯随机场初值下的平均全场相对误差随时间变化。两幅由同一作图程序绘制、使用相同的纵轴范围与刻度、方法配色、报告时刻和相对 $L^2$ 定义。左幅按已发布的 M2 汇总数据重绘，M2 的数值与报告时刻均未改变，M2 也没有被重新计算。右图中 GIFT-Lite 的有限子集区间以图内注记标出。</sub></div>

<div align="center">
  <img src="results/formal/S4_initial_distribution/figures/keyframes/traj2265_recursive_keyframes.svg" alt="测试轨迹 2265 在四个关键时刻的预测涡量场与有符号预测误差" width="100%">
</div>

<div align="center"><sub><b>图 S4-3｜</b>高斯随机场初值下预先指定的测试轨迹 2265 在 $t=5.0,6.0,7.0,8.0$ 的涡量场与有符号预测误差，版面与图 S4-2 相同。行列顺序、对齐方式、色标独立性和种子约定均与 M2 一致。</sub></div>

图 S4-2 为 M2 已发布的示例轨迹 1045 场图，实验文档中直接引用以作并排对照；它与上文的图 M2-2 是同一文件，此处不重复嵌入。

### 6. 结论与适用范围

在上述协议下，GIFT 在所有正预测时刻、三个原生网格以及两组初值分布上的平均全场相对误差均为最低；而且承担这一性能的正是被识别出的生成元本身：它可被读出为显式 PDE 用于参数识别（M1），可原样应用到未参与训练的网格（M3），也可在另一组初值分布上复用（S4）。

> **适用范围。** 结论限于本项目的数据分布、训练协议与 `N` = 64、96、128 网格，不构成任意分辨率上的误差保证或数值稳定性定理；相同 epoch 不代表相同参数更新次数或计算量。完整口径见 [EXPERIMENTS.md](EXPERIMENTS.md) 第 6 节与 [TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md)。

### 7. 快速开始与复现

数据与第三方实现位于项目之外，分别通过 `GIFT_DATA_ROOT` 与 `GIFT_EXTERNAL_ROOT` 指定；安装方式见 [SETUP.md](docs/SETUP.md)。每个模型与实验分别运行。

```shell
# 使用随项目提供的权重直接评价
python -m scripts.run_experiment M2 --output ../runs/M2 --skip-plots

# 独立训练一个模型（每个模型一条命令）
python -m scripts.run_training uno --data-profile canonical --run-training --output ../runs/uno

# 核对已发布结果的文件、完成记录与图像来源
python -m scripts.verify_published_results --experiment M3 --result-dir results/formal/M3_cross_resolution
```

代码组织：`src/gift/`（生成元模型、频带分解、数据划分）、`training/`（各模型独立训练入口）、`experiments/formal/`（M1–M3、S1–S4 启动与评价）、`results/formal/`（汇总 CSV/JSON 与 SVG 图像）、`artifacts/`（权重与检查点）、`tests/`（数据契约、检查点身份与结果完整性测试）。其余文档见 [docs/](docs/)。

### 8. 轨迹编号与数据划分

预测数据按初值分布分为两个独立总体。每个总体内训练编号在前，随后为验证集与测试集；同一轨迹在不同分辨率上沿用同一编号，高斯随机场总体的编号接在四涡旋总体之后，两者不表示逐条配对的初值。

| 初值分布 | 训练集 | 验证集 | 测试集 | GIFT-Lite 训练子集 |
| --- | --- | --- | --- | --- |
| 四涡旋 | 0–999（1000 条） | 1000–1039（40 条） | 1040–1219（180 条） | 0–49（50 条） |
| 平滑高斯随机场 | 1220–2219（1000 条） | 2220–2259（40 条） | 2260–2439（180 条） | 1220–1269（50 条） |

M1 使用其独立的局部编号（训练 0–49、验证 50–69），不与预测总体编号混用。训练、验证与测试互不重叠；GIFT-Lite 子集取自对应训练集，验证或测试数据不参与参数更新。各实验的划分见 [EXPERIMENTS.md](EXPERIMENTS.md) 第 8 节。

### 9. 跨学科应用（诊断实验）

方法核心——从观测场序列学习“常数 + 线性 + 二次”的谱右端项并用 RK4 积分——并不专属于 Navier–Stokes 方程。我们在三个其它学科的一阶标量系统上做了迁移验证：各自以诊断性 `fixture` 运行从零训练，使用未经改动的全数据生成元（500 轨迹 epoch、250 条训练轨迹、仅生成元组件）。

| 方程 | 物理背景 | 潜在应用场景 | 递归预测领先 3.0 s 的平均相对 $L^2$（持续性基线） |
| --- | --- | --- | ---: |
| `u_t = kappa * Lap(u)` | 线性扩散 | 热传导；质量扩散（Fick 定律）与污染物迁移；地下水渗流；热核平滑 | **0.0239**（0.4211） |
| `u_t = -c·grad(u)` | 线性平流 | 污染物与示踪剂输运；天气与气候模式的平流阶段；特征形式下的线性声传播 | **0.0245**（1.4020） |
| `u_t = 0.02 * Lap(u) + r u (1 - u)` | 非线性反应扩散 | 种群动力学与传染病入侵波前（Fisher–KPP）；化学波；燃烧火焰面；肿瘤生长 | **0.0051**（0.0064） |

递归预测自 $t$ = 5.0 的观测帧起，向前积分 150 步（`dt` = 0.02）至 $t$ = 8.0，统计 20 条独立测试轨迹的平均值。这些诊断实验使用各自的入口：

```shell
python -m scripts.generate_crossdomain_data --kind heat --output ../crossdomain/heat --execute
python -m scripts.run_crossdomain --kind heat --data ../crossdomain/heat/heat.h5 --output ../runs/crossdomain_heat --execute
```

> **迁移结论的适用范围。** 上述为诊断性 `fixture` 测量，不属于项目正式实验：每个方程固定一组物理参数（单一自治系统、仅变化初值）、单一标量场观测，且只使用生成元组件、未启用高频支路。物理参数随轨迹变化会使“状态到导数”的映射不适定；完整声波方程等二阶系统需要状态增广；非周期区域需要更换基底。详见 [EXPERIMENTS.md](EXPERIMENTS.md) 第 9 节。

### 外部代码与致谢

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

### 许可

GIFT 软件与文档使用 [MIT 许可](LICENSE)。数据包与外部项目适用各自的条款。
