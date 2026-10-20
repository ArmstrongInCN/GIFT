# GIFT

**Generator identification by field tomography: Flow prediction with testable physical correctness**

A data-driven framework for learning continuous-time generator models from time series of flow fields. Identifying the continuous-time generator that governs instantaneous flow-field evolution yields a model that can also be mapped back to testable governing physical laws without extensive parameter search.

[English](#english) · [中文](#中文) · [Setup](docs/SETUP.md) · [Experiments](EXPERIMENTS.md)

Current-protocol reproduction: [fresh W1/W2 training and portable models](docs/CURRENT_REPRODUCTION.md) · [W3 preparation, training and aggregation](docs/W3_REPRODUCTION.md).

[![GIFT architecture and prediction process](assets/figures/gift_architecture.png)](docs/ALGORITHM.md)

[![W1: fixed trajectory100005, seven predictions and signed residuals](assets/figures/w1_flow_prediction_confirmation_traj100005_recursive_keyframes_20261007.png)](EXPERIMENTS.md)

*Both plates are links. The first opens the [algorithm description](docs/ALGORITHM.md); the second shows fixed confirmation ID100005 at t=5,6,7,8 with aligned vorticity fields and signed prediction-minus-truth errors and opens the [experiment record](EXPERIMENTS.md). The field/error colour scales are shared across all selected methods, without finite-value clipping.*

## English

### What does GIFT learn?

For a system whose instantaneous evolution is determined solely by its current state, the **continuous-time generator** is the dynamical operator that maps the current state field to its instantaneous time derivative,

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

where $\omega$ is the vorticity field of two-dimensional incompressible flow. GIFT decomposes the generator model into a bias-field channel, a linear channel and a nonlinear channel, each containing a learnable operator. The superscript $\dagger$ distinguishes the true system generator from the approximate model obtained through learning. GIFT learns through field tomography, which exploits differences in amplitude responses among the bias-field, linear and nonlinear channels to separate and learn different components of the generator: $A(\lambda\omega)=\lambda A(\omega)$ and $Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$, where $\lambda$ is a scalar amplitude-scaling factor.

Because a state one finite lag later is only the time integral of the generator, GIFT thereby transforms the surrogate model's prediction mechanism from a black-box mapping hidden in neural network weights into an explicitly parameterized continuous-time generator, which can also be mapped back to testable governing physical laws. The component-by-component description is in [docs/ALGORITHM.md](docs/ALGORITHM.md).

### How does it work?

- **Generator model** — a parameterized model learned from scalar-field data to approximate the continuous-time generator, organized into bias-field, linear and nonlinear channels.
- **Field tomography** — separating and learning different components of the generator from flow-field trajectories by exploiting differences in amplitude responses among the bias-field, linear and nonlinear channels.
- **Quadratic field-interaction unit** — a unit that applies two learnable spectral filters to the same input field and then multiplies the resulting fields pointwise to produce a quadratic nonlinear output.
- **Learnable spectral filter** — a linear operator that applies learnable weights to individual spatial wavenumber components in the Fourier domain.
- **Flow-field trajectory** — a time series of flow fields obtained for a given governing equation and initial field distribution.
- **High-frequency branch** — a separately trained branch that takes the current state and predicts the complementary band, then coupled with the generator in a single fourth-order Runge–Kutta integration.
- **Recursive local correction** — a fixed rule applied after each complete Runge–Kutta step during recursive rollouts, under a safety cap on anomalous grid points (40, 90 and 160 for $N$ = 64, 96 and 128).
- **Physical interpretability** — with the generator frozen, the Navier–Stokes equation terms are fitted to the output of each channel, reading out equation parameters with physical meaning.

### How well does it work?

Experiments follow the manuscript order: **W1 flow prediction, W2 cross-resolution prediction, W3 equation-parameter identification, S1 high-frequency branch ablation, S2 recursive local correction ablation, S3 initial-condition distribution comparison**. See [experiment nomenclature](docs/EXPERIMENT_NOMENCLATURE.md).

W1/W2 use 300 training trajectories, 76 observed time nodes at dt = 0.04, and one true initial field per trajectory. Each method has one training instance with fixed hyperparameters and declared seeds. W1 compares GIFT, GIFT-Lite, FNO-2D, FNO-3D, U-NO, U-Net and ANIE; W2 compares the same methods except ANIE. Their native losses, supervision horizons and compute budgets differ. Complete tables, failures and source data are in [EXPERIMENTS.md](EXPERIMENTS.md).

- **W1, 180 held-out confirmation trajectories:** GIFT future75 mean relative L2 is **2.4687%**, with t8 **3.8113%**; GIFT-Lite is **3.6608% / 6.5787%**. All seven methods retain 180/180 finite trajectories.
- **W2, N64 training with native N64/N96/N128 inference:** GIFT future25 errors are **2.1098% / 2.0633% / 2.0610%**. All six methods retain 180/180 finite trajectories per grid; no target-grid training is used. ANIE participates only in W1.
- **W3** evaluates five methods on 50 noise-free parameter combinations, with one corresponding ID0 trajectory per combination. All 250 method-case slots and 750 parameter readouts are complete; [per-case values, APE summaries and measured costs](EXPERIMENTS.md#43-w3-50组无噪声方程参数识别) are available. **S1–S3** retain their independent N1000 protocols and existing measurements.

W3 uses the paper Methods alternating subproblem (`paper_ado_v1`), native FP32 and the complete fixed observation/physics populations for 50 independent fits of each PINN method (100 terminal models in total). Each method retains its declared sampling and budget; measured costs and source runs are reported separately. See [W3 protocol](docs/W3_PROTOCOL.md).

U-NO supervises 20 steps through t = 5.80 and U-Net four steps through t = 5.16. GIFT-Lite (source key `GIFT-N50`) uses the first 50 trajectories of the ordered N300 sample with the same GIFT stage epochs, so both data and total updates decrease. The confirmation population is independent of training and validation, but results on both evaluation populations had already been examined; it is not a blind test. A separate 180-trajectory population is reported as exploratory. GIFT uses supervision leads of 0.08/0.48 seconds, internal RK4 dt = 0.02 and saved nodes every 0.04 seconds. The results quantify fixed-recipe, single-model performance; they do not isolate architecture effects, equalize compute or estimate variability across training seeds.

### How to get started?

For a new machine, follow [the current reproduction workflow](docs/CURRENT_REPRODUCTION.md). It prepares the N300/N50 recipes directly from the observation package, trains aligned ANIE and the other fixed methods, and evaluates separate complete populations. Download the [nine-file reference model bundle](https://github.com/ArmstrongInCN/GIFT/releases/download/current-protocol-20261020-source-free/GIFT-current-models-20261020-source-free.zip) from the [source-free release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020-source-free) and verify its [archive identity](docs/CURRENT_MODEL_BUNDLE.json). The new bundle removes ANIE source-text metadata while preserving every model tensor exactly; it supersedes the earlier model package. Historical experiment directories are not required for the fresh workflow.

Run from the repository root. The [training protocol](docs/W1_W2_PROTOCOL.md) gives the W1/W2 configurations and checkpoint bindings. Their weights and full predictions are stored in the declared external experiment directories; the repository contains result summaries and figure evidence. Set `GIFT_DATA_ROOT` for data. External baselines require user-prepared code outside this repository: `GIFT_EXTERNAL_ROOT` selects the pinned upstream checkouts, `GIFT_ANIE_SOURCE_ROOT` selects ANIE, and `GIFT_EXTERNAL_ADAPTER_ROOT` selects their external runtime interfaces. Follow the written [manual adaptation instructions](docs/EXTERNAL_MANUAL_ADAPTATION.md); the current working-tree code contains no external algorithm implementation or automatic source patcher.

```shell
# Recorded reference training environment: Python 3.10.19, PyTorch 2.10.0, CUDA 12.6
python -m venv ../gift-env            # activate it with your shell's command
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .

# Data and upstream sources; PowerShell examples are in docs/SETUP.md
export GIFT_DATA_ROOT=/absolute/path/GIFT-dataset
export GIFT_EXTERNAL_ROOT=/absolute/path/gift-external
export GIFT_ANIE_SOURCE_ROOT=/absolute/path/gift-external/anie
export GIFT_EXTERNAL_ADAPTER_ROOT=/absolute/path/gift-external-runtime
python -m scripts.verify_data --root "$GIFT_DATA_ROOT"
python -m scripts.prepare_external --root "$GIFT_EXTERNAL_ROOT"
# Download ANIE at its pinned commit and apply the written instructions outside GIFT.
# Prepare the external runtime interfaces before baseline training or inference.

# Evaluation arguments for the declared models and populations
python -m experiments.additional.w1_current_evaluate --help
python -m experiments.additional.w2_single_evaluate --help

# Integrity of the result summaries and figures
python -m scripts.verify_published_results --experiment W2 --result-dir results/formal/W2_cross_resolution_20261007
```

GIFT and the prediction baselines support CPU and CUDA. Hardware, numerical libraries and upstream GPU operators can affect floating-point results. Environment and per-model instructions are in [docs/SETUP.md](docs/SETUP.md).

### Where does the data come from?

The separate dataset has an existing ScienceDB record; downloading observations is optional because they can be regenerated. The package reorganized on 2026-10-20 has not been uploaded as a new ScienceDB version. Its [README and layout](docs/data_package/README.md) distinguish current W1/W2 N300/N50 inputs, independent N1000 auxiliary data, 50 current W3 cases, S3 Gaussian data and numerical source data for all six formal experiments. It includes no trained weights or full prediction archive. Set `GIFT_DATA_ROOT` to the extracted dataset directory outside GIFT.

- **Dataset record** — *GIFT Navier–Stokes input data*, ScienceDB, DOI `10.57760/sciencedb.013lo` (CSTR `31253.11.sciencedb.013lo`), licensed CC BY 4.0.
- **Generating observations locally instead** — `scripts/generate_data` integrates clean fields, `generate_sampling` prepares protocol-specific inputs, `derive_dense_frames` selects exact coarse/cross-resolution frames, and `generate_gaussian_data` generates S3. `assemble_generated_data` and `prepare_prediction_package` build their declared generated collections; they do not recreate the complete six-experiment publication package. Optional `generate_noise` supports the separate historical noisy protocol. Commands and continuation boundaries are in [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md); regeneration starts from recorded initial conditions and does not promise bitwise identity across devices.

### Documentation

- [docs/ALGORITHM.md](docs/ALGORITHM.md) — the generator model, field tomography, the quadratic field-interaction unit, the high-frequency branch and the recursive local correction.
- [EXPERIMENTS.md](EXPERIMENTS.md) — the numerical experiment record: W1/W2 single-model tables, independent W3 and S1–S3 protocols, finiteness counts and source entry points.
- [docs/SETUP.md](docs/SETUP.md) — environments, data package, external sources, one model per command.
- [docs/TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md) — training budgets, data splits, model selection.
- [docs/RESULTS.md](docs/RESULTS.md) — how to read the published results, and why reading a table, recomputing inference and training from scratch are different operations.
- [docs/FIGURES.md](docs/FIGURES.md) — curve and field specifications, selected trajectories, colour scales and numerical evidence.
- [docs/EXPERIMENT_EXECUTION.md](docs/EXPERIMENT_EXECUTION.md) · [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) · [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) · [docs/VALIDATION.md](docs/VALIDATION.md) · [docs/PERFORMANCE.md](docs/PERFORMANCE.md) · [docs/S3_PROTOCOL.md](docs/S3_PROTOCOL.md) · [docs/EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md)

### Repository layout

- `src/gift/` — generator model, band decomposition, data splits, the identified-generator readout.
- `adapters/` — a generic bridge to runtime interfaces prepared outside GIFT; no external networks, solvers or executable source patches.
- `training/` — one training entry point per model.
- `experiments/additional/` — W1/W2 single-run evaluation.
- `experiments/formal/` — parameter identification and independent auxiliary protocols.
- `results/formal/` — published summaries, CSV/JSON records and SVG figures with their manifests.
- `artifacts/` — weights and checkpoints with their recorded identities.
- `scripts/` — data preparation, verification, packaging and figure rendering.
- `tests/` — data readers, numerical methods, checkpoint recovery and result-integrity tests.
- `docs/` — setup, protocol, execution and validation documentation.
- `assets/figures/` — the two plates shown above, with the vector original of the architecture plate.

The root also holds the numerical experiment record [`EXPERIMENTS.md`](EXPERIMENTS.md), the pinned upstream identities [`external_sources.json`](external_sources.json) and the working conventions [`AGENTS.md`](AGENTS.md).

### Data, code and licence

GIFT software and documentation are released under the [MIT licence](LICENSE). The data package uses CC BY 4.0; external projects are governed by their own terms.

The comparison experiments use the upstream implementations below. The current working tree retains source references, pinned metadata, written adaptation instructions and generic interfaces; external algorithm implementations and executable source patches must remain outside GIFT. Users obtain the official code under its own terms and perform the stated adaptations outside the repository; see [docs/EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md) and the detailed [manual procedure](docs/EXTERNAL_MANUAL_ADAPTATION.md). The external runtime used on the experiment machine is a separate local dependency. Earlier Git history and the `reference-checkpoints-20260913` tag retain old executable adapters; a normal new commit does not remove them. The superseded model attachment retains four ANIE source-text metadata strings (441 characters); it remains available for provenance, and the new release provides the cleaned publication copy. This release does not rewrite Git history; [the boundary policy](docs/EXTERNAL_REPOSITORY_POLICY.md) distinguishes these scopes. We thank the authors and contributors of these projects, and ask that the original research be cited when using these methods.

| Upstream project | Used for | Pinned version | Licence |
| --- | --- | --- | --- |
| [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | FNO-2D and FNO-3D baselines | `01d2aeca` | MIT |
| [ashiq24/UNO](https://github.com/ashiq24/UNO) | U-NO baseline | `19462d82` | BSD-2-Clause |
| [emazap7/ANIE](https://github.com/emazap7/ANIE) | ANIE baseline | `2071ad70` | No explicit licence found at the pinned commit; no additional written authorization obtained |
| [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | U-Net baseline | `229da3e0` | No licence declared upstream; **do not redistribute** |
| [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | PINN-SR configurations | `9a20ebe6` | No licence declared upstream |
| [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | PDE-FIND configurations | `86911349` | No licence declared upstream |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) (tag `v1.15.0`) | NAdam and L-BFGS optimizer implementations | `590d6eef` | Apache-2.0 |

The ANIE baseline uses [emazap7/ANIE](https://github.com/emazap7/ANIE) at commit `2071ad709b260a6c305450affcf7168fd825f692`; its selected source metadata are recorded in [anie_sources.json](anie_sources.json), and the necessary coordinate/dropout operations are described in [the manual adaptation guide](docs/EXTERNAL_MANUAL_ADAPTATION.md). The JSON contains no executable patch text.

Each licence entry above refers to the upstream repository **at the pinned commit**. Four of the seven have no explicit licence identified, so GIFT grants no rights over their code. ANIE usage, modification and weight-distribution authorization remain unresolved; adding attribution or removing source metadata does not provide that authorization. ANIE attributes its Galerkin attention to [Galerkin Transformer](https://github.com/scaomath/galerkin-transformer), which has an MIT licence; that licence does not automatically cover ANIE's other code. TensorFlow is Apache-2.0; any redistribution of covered source must preserve the applicable notices.

### Citation

The comparison methods have primary references, and two upstream repositories publish an explicit citation request. Please cite the GIFT manuscript and the upstream work you use.

- **GIFT** — *Generator identification by field tomography: Flow prediction with testable physical correctness* (manuscript; the Chinese version is titled 基于场层析的生成元识别：可检验物理正确性的流场预测).
- **GIFT data** — *GIFT Navier–Stokes input data*, ScienceDB (CC BY 4.0), DOI `10.57760/sciencedb.013lo`. Cite this record when reusing the observation data.
- **FNO-2D / FNO-3D** — Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, "Fourier neural operator for parametric partial differential equations", ICLR 2021 (arXiv:2010.08895). Cited by the upstream repository.
- **U-NO** — M. A. Rahman, Z. E. Ross, K. Azizzadenesheli, "U-NO: U-shaped neural operators", Trans. Mach. Learn. Res. 2023.
- **U-Net baseline** — R. Wang, K. Kashinath, M. Mustafa, A. Albert, R. Yu, "Towards physics-informed deep learning for turbulent flow prediction", KDD 2020. Cited by the upstream repository.
- **ANIE** — E. Zappala et al., "Learning integral operators via neural integral equations", Nat. Mach. Intell. 6, 1046–1062 (2024), [doi:10.1038/s42256-024-00886-8](https://doi.org/10.1038/s42256-024-00886-8).
- **PINN-SR** — Z. Chen, Y. Liu, H. Sun, "Physics-informed learning of governing equations from scarce data", Nat. Commun. 12, 6136 (2021).
- **PDE-FIND** — S. H. Rudy, S. L. Brunton, J. L. Proctor, J. N. Kutz, "Data-driven discovery of partial differential equations", Sci. Adv. 3, e1602614 (2017).
- **TensorFlow optimizers** — Apache-2.0 software; the upstream project requests no citation.

<a id="中文"></a>

## 中文

**基于场层析的生成元识别：可检验物理正确性的流场预测**

从流场时间序列中学习生成元模型的数据驱动框架。辨识支配瞬时流场演化的连续时间生成元，所得到的模型还能够被还原为可检验的物理控制规律，而无需大规模参数搜索。

### GIFT 学的是什么？

对于瞬时演化只由当前状态决定的系统，**连续时间生成元**是把当前状态场映射为其瞬时时间导数的动力学算子：

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

其中 $\omega$ 为二维不可压缩流动的涡量场。GIFT 把生成元模型分解为偏置场通道、线性通道与非线性通道，每个通道各含一个可学习算子；上标 $\dagger$ 用于区分真实系统生成元与通过学习得到的近似模型。GIFT 的学习过程通过场层析实现，即利用偏置场、线性和非线性通道的振幅响应差异，分离并学习生成元的不同分量：$A(\lambda\omega)=\lambda A(\omega)$、$Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$，其中 $\lambda$ 是标量振幅缩放因子。

有限时间间隔后的状态只是生成元的时间积分结果，因此 GIFT 把代理模型的预测机制从隐藏在神经网络权重中的黑箱映射转变为可还原为可检验物理控制规律的连续时间模型。逐部件说明见 [docs/ALGORITHM.md](docs/ALGORITHM.md)。

### 它是怎么工作的？

- **生成元模型**——从标量场数据中学习、用于近似连续时间生成元的参数化模型，组织为偏置场通道、线性通道与非线性通道。
- **场层析**——利用偏置场、线性与非线性通道振幅响应的差异，从流场轨迹中分离并学习生成元的不同组成部分。
- **二次场相互作用单元**——对同一输入场施加两个可学习谱滤波器，再把所得场逐点相乘，产生二次非线性输出。
- **可学习谱滤波器**——在傅里叶域对各个空间波数分量施加可学习权重的线性算子。
- **流场轨迹**——给定控制方程和初始分布后得到的流场时间序列数据。
- **高频支路**——单独训练、读取当前状态并预测补频带的支路，与生成元耦合在同一个四阶 Runge–Kutta 积分中推进。
- **递归局部修正**——递归推演过程中每个完整 Runge–Kutta 步后执行的固定规则，并设异常网格点安全上限（$N$ = 64、96、128 时分别为 40、90、160）。
- **物理可解释性**——生成元冻结后，用 Navier–Stokes 方程项拟合各通道输出，读出具有物理意义的方程参数。

### 效果如何？

实验按手稿的 **W1流场演化预测、W2跨分辨率预测、W3方程参数识别、S1高频支路消融、S2递归局部修正消融、S3初值分布对照** 排列，见[实验命名](docs/EXPERIMENT_NOMENCLATURE.md)。

W1/W2使用300条训练轨迹、76个真实观测时间点（dt = 0.04）和每条轨迹的单一真实初态。各方法采用固定超参数与声明的种子，只训练一个模型实例。W1比较GIFT、GIFT-Lite、FNO-2D、FNO-3D、U-NO、U-Net和ANIE；W2除ANIE外比较六种方法。各自的原生损失、监督范围和计算预算不同，完整表格、失败计数和数据来源见 [EXPERIMENTS.md](EXPERIMENTS.md)。

- **W1确认集180条：** GIFT未来75点的平均relative L2为 **2.4687%**，t8为 **3.8113%**；GIFT-Lite为 **3.6608%／6.5787%**。七种方法各180/180条预测有限。
- **W2仅N64训练、原生跨网格：** GIFT在N64／N96／N128的未来25点误差为 **2.1098%／2.0633%／2.0610%**，六种方法每个网格各180/180条预测有限，没有目标网格训练。ANIE仅参加W1。
- **W3**在50组无噪声方程参数组合上评估五种方法；每组各使用对应的一条ID0轨迹，共250个方法/参数槽位、750个参数读出。全部数值、APE汇总与实测成本见[实验记录](EXPERIMENTS.md#43-w3-50组无噪声方程参数识别)及[W3结果包](results/formal/W3_single_trajectory_20261010/README.md)。**S1–S3**保留各自独立N1000协议和既有结果。

W3两种PINN采用论文Methods交替子问题`paper_ado_v1`、原生float32及全部观测点和物理约束点。各方法保留各自采样与预算，真实成本和运行来源分别报告，见[W3协议](docs/W3_PROTOCOL.md)。

U-NO监督20步至t5.80，U-Net监督4步至t5.16。GIFT-Lite（来源键`GIFT-N50`）取有序N300名单前50条，沿用GIFT阶段epoch，因此样本量与总更新数同时减少。确认集与训练、验证集独立，但其中部分方法的结果此前已被查看，故不视为盲测；另一180条轨迹仅作探索性评价。GIFT采用0.08/0.48秒监督时距、内部RK4 dt = 0.02，每0.04秒保存一次。结果反映固定配方的单模型表现，不能单独归因于架构，也不提供等计算预算或跨训练种子变异的证据。

### 如何开始？

新机器请按[当前协议复现流程](docs/CURRENT_REPRODUCTION.md)从观测包直接准备N300/N50配方、训练时空对齐ANIE及其余固定方法，再评估各组完整轨迹集。从[源码元数据清理版 Release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020-source-free)下载[九文件参考模型包](https://github.com/ArmstrongInCN/GIFT/releases/download/current-protocol-20261020-source-free/GIFT-current-models-20261020-source-free.zip)，并按[归档身份](docs/CURRENT_MODEL_BUNDLE.json)校验。新包移除了 ANIE 源码文本元数据，全部模型张量逐位不变，取代此前的模型包。从零训练不依赖作者的历史实验目录。当前W3的准备、单轨迹再生、训练和250槽汇总见[专用流程](docs/W3_REPRODUCTION.md)。

命令均在项目根目录运行。[训练协议](docs/W1_W2_PROTOCOL.md) 给出W1/W2配置和权重绑定；其权重与完整预测保存在声明的外部实验目录，仓库提供结果汇总及图像证据。数据包由 `GIFT_DATA_ROOT` 指定。外部基线须由使用者在 GIFT 仓库外准备：`GIFT_EXTERNAL_ROOT` 指定固定官方 checkout，`GIFT_ANIE_SOURCE_ROOT` 指定 ANIE，`GIFT_EXTERNAL_ADAPTER_ROOT` 指定外部运行接口。具体步骤见[手工适配说明](docs/EXTERNAL_MANUAL_ADAPTATION.md)；当前工作树代码不包含外部算法实现或自动源码修改器。

```shell
# 原记录训练参考环境：Python 3.10.19、PyTorch 2.10.0、CUDA 12.6
python -m venv ../gift-env            # 再用所在shell的命令激活
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .

# 数据、仓库外官方源码与运行接口；PowerShell写法见docs/SETUP.md
export GIFT_DATA_ROOT=/absolute/path/GIFT-dataset
export GIFT_EXTERNAL_ROOT=/absolute/path/gift-external
export GIFT_ANIE_SOURCE_ROOT=/absolute/path/gift-external/anie
export GIFT_EXTERNAL_ADAPTER_ROOT=/absolute/path/gift-external-runtime
python -m scripts.verify_data --root "$GIFT_DATA_ROOT"
python -m scripts.prepare_external --root "$GIFT_EXTERNAL_ROOT"
# 单独下载固定版本ANIE，按文字步骤在仓库外适配并准备运行接口。

# 指定模型和评估样本集的参数接口
python -m experiments.additional.w1_current_evaluate --help
python -m experiments.additional.w2_single_evaluate --help

# 结果汇总与图像的完整性
python -m scripts.verify_published_results --experiment W2 --result-dir results/formal/W2_cross_resolution_20261007
```

GIFT与预测基线支持CPU和CUDA，硬件、数学库及上游GPU算子可能影响浮点结果。环境和各模型说明见 [docs/SETUP.md](docs/SETUP.md)。

### 数据从哪里来？

数据已有独立 ScienceDB 记录；观测也可从初值重新生成。2026年10月20日重整的数据包尚未作为新版本上传 ScienceDB。[包说明与目录](docs/data_package/README.md) 分别列出当前 W1/W2 的 N300/N50 输入、独立 N1000 辅助数据、当前 W3 的 50 个 case、S3 Gaussian 数据和六个正式实验的数值源数据。模型权重和完整预测场不在本包内。请将 `GIFT_DATA_ROOT` 指向 GIFT 仓库外的数据包解压目录。

- **数据集记录**——*GIFT Navier–Stokes input data*，ScienceDB，DOI `10.57760/sciencedb.013lo`（CSTR `31253.11.sciencedb.013lo`），许可 CC BY 4.0。
- **也可以本地生成观测**——`scripts/generate_data` 积分干净场，`generate_sampling` 准备协议采样，`derive_dense_frames` 精取粗帧与跨分辨率帧，`generate_gaussian_data` 生成 S3。`assemble_generated_data` 和 `prepare_prediction_package` 汇集各自声明的再生数据，不自动重建六个实验的完整发布包。可选 `generate_noise` 用于独立历史噪声协议。命令与断点边界见 [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md)；再生从已记录初值出发，不承诺跨设备逐比特一致。

### 文档

- [docs/ALGORITHM.md](docs/ALGORITHM.md)——生成元模型、场层析、二次场相互作用单元、高频支路与递归局部修正。
- [EXPERIMENTS.md](EXPERIMENTS.md)——数值实验记录：W1/W2单模型表、独立W3及S1–S3协议、有限性计数与运行入口。
- [docs/SETUP.md](docs/SETUP.md)——环境、数据包、外部源码与「每个模型一条命令」。
- [docs/TRAINING_PROTOCOL.md](docs/TRAINING_PROTOCOL.md)——训练预算、数据划分与模型选择。
- [docs/RESULTS.md](docs/RESULTS.md)——如何阅读已发布结果，以及读表、重新推理与从零训练为何是三种不同操作。
- [docs/FIGURES.md](docs/FIGURES.md)——曲线与场板规范、固定轨迹、色阶及数值源。
- [docs/EXPERIMENT_EXECUTION.md](docs/EXPERIMENT_EXECUTION.md) · [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) · [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) · [docs/VALIDATION.md](docs/VALIDATION.md) · [docs/PERFORMANCE.md](docs/PERFORMANCE.md) · [docs/S3_PROTOCOL.md](docs/S3_PROTOCOL.md) · [docs/EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md)

### 代码组织

- `src/gift/`——生成元模型、频带分解、数据划分与生成元读出。
- `adapters/`——仅连接仓库外用户准备的运行接口，不包含外部网络、求解器或可执行源码补丁。
- `training/`——各模型独立训练入口。
- `experiments/additional/`——W1/W2单次评价。
- `experiments/formal/`——参数识别与独立辅助协议。
- `results/formal/`——已发布汇总、CSV/JSON 记录、SVG 图像及其清单。
- `artifacts/`——权重与检查点及其身份记录。
- `scripts/`——数据准备、校验、打包与图像渲染。
- `tests/`——数据契约、检查点身份与结果完整性测试。
- `docs/`——安装、协议、执行与验证文档。
- `assets/figures/`——上文两幅图，以及结构图的矢量原件。

根目录另置数值实验记录 [`EXPERIMENTS.md`](EXPERIMENTS.md)、固定上游身份 [`external_sources.json`](external_sources.json) 与工作约定 [`AGENTS.md`](AGENTS.md)。

### 数据、代码与许可

GIFT 软件与文档使用 [MIT 许可](LICENSE)；数据包使用 CC BY 4.0；外部项目适用各自条款。

对比实验使用以下上游实现。当前工作树仅保留来源引用、固定版本元数据、文字适配说明和通用接口；外部算法实现及可执行源码补丁必须保存在 GIFT 之外。使用者按上游条款自行取得官方代码并在仓库外完成适配，步骤见 [docs/EXTERNAL_ADAPTATIONS.md](docs/EXTERNAL_ADAPTATIONS.md) 和[详细操作说明](docs/EXTERNAL_MANUAL_ADAPTATION.md)。实验机器上已有的外部运行接口是独立本地依赖。此前 Git 历史和 `reference-checkpoints-20260913` 标签仍保留旧的可执行适配器，普通新提交不会移除这些内容；已被取代的旧模型附件仍含四段 ANIE 源码文本元数据（441字符），作为来源记录保留，新发布提供清理后的公开副本。本次发布未改写 Git 历史，范围见[外部算法边界说明](docs/EXTERNAL_REPOSITORY_POLICY.md)。对各项目的作者与贡献者表示感谢，使用相关方法时请引用其原始研究。

| 上游项目 | 用途 | 固定版本 | 许可状态 |
| --- | --- | --- | --- |
| [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | FNO-2D 与 FNO-3D 基线 | `01d2aeca` | MIT |
| [ashiq24/UNO](https://github.com/ashiq24/UNO) | U-NO 基线 | `19462d82` | BSD-2-Clause |
| [emazap7/ANIE](https://github.com/emazap7/ANIE) | ANIE 基线 | `2071ad70` | 固定提交未找到明确许可，尚未取得额外书面授权 |
| [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | U-Net 基线 | `229da3e0` | 上游未声明许可，**不得再分发** |
| [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | PINN-SR 两种配置 | `9a20ebe6` | 上游未声明许可 |
| [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | PDE-FIND 两种配置 | `86911349` | 上游未声明许可 |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)（tag `v1.15.0`） | NAdam 与 L-BFGS 优化器实现 | `590d6eef` | Apache-2.0 |

表中许可状态均对应上述**固定提交**。七个上游项目中有四个未找到明确许可，GIFT 不授予其代码的任何权利。ANIE 的使用、修改和权重公开分发授权仍未闭环；补齐引用或清理源码元数据不能替代授权。ANIE 声明其 Galerkin attention 来自采用 MIT 许可的 [Galerkin Transformer](https://github.com/scaomath/galerkin-transformer)，该许可不会自动覆盖 ANIE 的其余代码。TensorFlow 使用 Apache-2.0；如再分发受其覆盖的源码，须保留适用声明。

ANIE基线采用 [emazap7/ANIE](https://github.com/emazap7/ANIE) 的固定提交 `2071ad709b260a6c305450affcf7168fd825f692`，选定源码元数据见 [anie_sources.json](anie_sources.json)，坐标与 dropout 必要操作见[手工适配说明](docs/EXTERNAL_MANUAL_ADAPTATION.md)。该 JSON 不含可执行补丁文本。

### 引用

对比方法各有原始文献，其中两个上游仓库明确提出了引用请求。使用本项目时请引用 GIFT 手稿以及所使用的外部工作。

- **GIFT**——*Generator identification by field tomography: Flow prediction with testable physical correctness*（手稿；中文版题为「基于场层析的生成元识别：可检验物理正确性的流场预测」）。
- **GIFT 数据**——*GIFT Navier–Stokes input data*，ScienceDB（CC BY 4.0），DOI `10.57760/sciencedb.013lo`。使用观测数据时请引用该记录。
- **FNO-2D / FNO-3D**——Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, "Fourier neural operator for parametric partial differential equations", ICLR 2021 (arXiv:2010.08895)。上游仓库明确要求引用。
- **U-NO**——M. A. Rahman, Z. E. Ross, K. Azizzadenesheli, "U-NO: U-shaped neural operators", Trans. Mach. Learn. Res. 2023。
- **U-Net 基线**——R. Wang, K. Kashinath, M. Mustafa, A. Albert, R. Yu, "Towards physics-informed deep learning for turbulent flow prediction", KDD 2020。上游仓库明确要求引用。
- **ANIE**——E. Zappala et al., "Learning integral operators via neural integral equations", Nat. Mach. Intell. 6, 1046–1062 (2024)，[doi:10.1038/s42256-024-00886-8](https://doi.org/10.1038/s42256-024-00886-8)。
- **PINN-SR**——Z. Chen, Y. Liu, H. Sun, "Physics-informed learning of governing equations from scarce data", Nat. Commun. 12, 6136 (2021)。
- **PDE-FIND**——S. H. Rudy, S. L. Brunton, J. L. Proctor, J. N. Kutz, "Data-driven discovery of partial differential equations", Sci. Adv. 3, e1602614 (2017)。
- **TensorFlow 优化器**——Apache-2.0 软件，上游未提出引用要求。
