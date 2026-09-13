# 第三方算法来源与不随包发布的边界

核查日期：2026-09-10。本文件依据旧临时项目的 `source_manifest.json`、`SOURCE_AUDIT.md`、实际模型/训练/锁定运行器及上游主来源的只读核查。它是发布来源清单，不是法律保证，也不是从零复现已经通过的证书。

## 发布规则

本候选按项目所有者要求仅提供上游 URL、固定版本、文件身份及必要的原创最小适配；不打包他人的源代码或实质性派生大段代码。该规则也适用于具有 MIT/BSD/Apache 许可的代码：许可存在不改变本次不打包的选择。Zenodo 是外置数值数据/权重与真实审计结果的载体，不是第三方源码的转存处。

“位于 upstream 之外”“叫 wrapper/adapter”“有来源 manifest”或“注释写 clean-room”均不单独证明文件没有第三方实现。相反，纯原创数据协议、独立训练循环或调用外部库的接口，不会仅因依赖第三方库就自动被判定为第三方派生代码；必须检查实际内容。

## 已核对的固定上游

| 方法 | 上游与固定提交 | 本次找到的许可证据 | 候选处理 |
| --- | --- | --- | --- |
| FNO2d/FNO3d | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator/tree/01d2aeca407f85ce1c1e78b313edb749973fc051)，`01d2aeca407f85ce1c1e78b313edb749973fc051` | 固定提交的 [MIT LICENSE](https://github.com/neuraloperator/neuraloperator/blob/01d2aeca407f85ce1c1e78b313edb749973fc051/LICENSE) | 用户外部 clone；不内置旧脚本或提取模型。 |
| U-NO | [ashiq24/UNO](https://github.com/ashiq24/UNO/tree/19462d82729ef64ef7b9e97056ddcaaf3044ad47)，`19462d82729ef64ef7b9e97056ddcaaf3044ad47` | 固定提交的 [BSD-2-Clause LICENSE](https://github.com/ashiq24/UNO/blob/19462d82729ef64ef7b9e97056ddcaaf3044ad47/LICENSE) | 用户外部 clone，包含该版本实际使用的网络、积分算子、自定义 Adam 和工具函数。 |
| U-Net | [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets/tree/229da3e01acefa6f4dd6c77244da5015c9b965af)，`229da3e01acefa6f4dd6c77244da5015c9b965af` | 固定提交的完整文件树与所选源文件未发现明确许可声明 | 记录再分发授权未查明；只提供来源与外部加载方法。 |
| PINN-SR | [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery/tree/9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496)，`9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496` | 同上，未发现明确许可声明 | 不内置完整 PINN 网络、STRidge、训练适配体。详见 [M1](adapters/M1.md)。 |
| PDE-FIND | [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND/tree/86911349b9fac82996e35465a13d06205730cc05)，`86911349b9fac82996e35465a13d06205730cc05` | 同上，未发现明确许可声明 | 不内置上游回归及复制/展开实现。详见 [M1](adapters/M1.md)。 |

“未发现声明”的具体检查：GitHub 固定提交的递归文件树均完整返回、非 truncated；EQDiscovery 53 项、PDE-FIND 31 项、Turbulent-Flow-Nets 23 项，未发现 LICENSE/COPYING/NOTICE/COPYRIGHT 命名的许可文件，选定算法源文件也未发现明确许可声明。依据为上游 [EQDiscovery tree API](https://api.github.com/repos/isds-neu/EQDiscovery/git/trees/9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496?recursive=1)、[PDE-FIND tree API](https://api.github.com/repos/snagcliffs/PDE-FIND/git/trees/86911349b9fac82996e35465a13d06205730cc05?recursive=1)、[U-Net tree API](https://api.github.com/repos/Rui1521/Turbulent-Flow-Nets/git/trees/229da3e01acefa6f4dd6c77244da5015c9b965af?recursive=1)。这不是穷尽全部可能授权的法律意见，更不是断言读者不能自行使用；如希望再分发，应向权利方取得明确说明。

PINN 还有 [TensorFlow v1.15.0 NAdam](https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/contrib/opt/python/training/nadam_optimizer.py)（该文件 Apache-2.0）和 [SciPy v1.7.3 L-BFGS-B 包装器](https://github.com/scipy/scipy/blob/v1.7.3/scipy/optimize/lbfgsb.py)来源。它们不能因被复制进 PINN 包装文件而从清单消失。旧 manifest 没有把前者作为独立外部源字节固定项；当前 `external_sources.json` 已补齐 `tensorflow115` 提交 `590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b`，以及 NAdam、external_optimizer 两文件 SHA/Git blob 身份。源码身份补齐不等于完整 PINN 集成验收。UNO 的 `Adam.py` 有 PyTorch 风格功能性优化器代码线索，本次未完成其更早历史的逐行归属确认，不应声称所有文件最初均由 UNO 作者独立创作；本候选不再分发该文件。

## 不能只排除 upstream 目录

以下路径相对旧临时项目。表中“整文件不迁入”是本候选对混合来源的保守发布处理，不等于对文件每一行的法律性质作出结论。

| 范围 | 已发现的内容 | 最小处理 |
| --- | --- | --- |
| `benchmarks/fno/models.py` | 完整 `SpectralConv2d_fast`、`FNO2d`、`SpectralConv3d`、`FNO3d`，保留上游频谱切片、参数布局和前向计算。 | 不发布提取类；由外部固定源码提供。 |
| `benchmarks/fno/utilities.py` | `UnitGaussianNormalizer`、`LpLoss` 等上游语义实现。 | 外部导入；不要把复制 loss/normalizer 当自有工具。 |
| `benchmarks/fno/training.py` | 混合本项目协议与源兼容的训练目标/输入准备。 | 不整文件迁入；原创数据协议和独立调度可拆分，来源相关算法函数外部提供。 |
| `benchmarks/unet/model.py` | 实际 builder 导入上游，但文件仍保留未使用的完整本地编码器/解码器/UNet 类与 forward。 | 连同未使用的实现排除，只写新的路径校验与外部类加载。 |
| `benchmarks/uno/model.py` | 主要为来源校验和外部导入，未发现完整本地 UNO 模型复制；旧实现仍把 vendored 目录插入全局搜索路径。 | 可重新写成原创外部加载接口；不能保留 vendored 路径回退。 |
| `baselines/_uno_training.py` | 混合原创数据/状态流程、上游逐样本 loss 语义、针对原插值的确定性反向适配。 | 不整文件作为纯自有代码迁入；loss 从外部提供，原创确定性扩展单独评审与验证。 |
| `baselines/train_unet.py` 等训练入口 | 读取部分未发现完整复制的网络；包含独立训练/审计代码。 | 不能仅因名字含 training 就认定派生，也不能未经全文件来源审查就加入原创白名单。只迁入已确认原创的必要接口。 |
| `benchmarks/pinn_sr/locked_runtime/` | 完整网络/自动微分/稀疏回归、复制的 NAdam 更新、阶段优化与 SciPy 续算包装，另夹有原创 I/O。 | 不迁入完整目录或大文件；见 [M1 具体范围与阻碍](adapters/M1.md)。 |
| `benchmarks/pde_find/locked_runtime/run_pdefind_batch.py` | 数据适配混合上游候选库构造的向量化实现。 | 外部调用 `build_Theta` / `TrainSTRidge`，重新界定原创数据接口。 |
| `reproduction/historical_m1_sources_r4/`、`origin_sources/`、源快照/备份/代码归档 | 保存相同上游及派生实现。 | 同样排除；不能仅过滤 `.py` 后保留含源正文的 JSON/ZIP。 |

旧 FNO source_manifest 主要固定上游文件，不等于把 `models.py` 等变成无来源问题的代码。PINN/PDE 的 manifest 固定了锁定运行器，能说明实际执行身份，也不等于具有再分发授权。另已核实 PINN/PDE 的旧源快照使用 CRLF，官方 Git blob 使用 LF；外部 clone 的 canonical hash 与旧归档 hash 必须分列，详见 M1 文档及外部源清单，不能误把可解释的换行差异判成算法版本差异。

## 外置来源与科学设置必须同时固定

建议由用户在发布目录之外自行取得 clone，并提供显式路径。加载器只接受已列固定提交和源字节；不自动下载，不自动改源码，不搜索正式目录作为回退。存在顶层训练语句的脚本只从验证后的外部文件中选择所需定义；不要整体 import 触发数据加载、环境修改、Session 或训练。

不同方法使用独立子进程和局部依赖搜索空间，避免 `utilities3`、`Adam` 等通用模块名冲突，以及一个依赖修改环境后影响另一个模型。每次运行记录实际外部源、适配规则、数据、数学配置和环境。生成的外部适配缓存即使不进入仓库，也不能被误称为原创源码。

最容易被“仅换 loader”破坏的设置如下：

- FNO2d 的历史长度适配是输入 48 通道；FNO3d 为 49 通道。必须在构造前改变输入维度，不能先按原维度实例化再换第一层，否则后续权重的随机初始化序列发生变化。保留原无 padding、2d BN 使用/3d BN 声明但前向不使用、复杂权重初始化、坐标顺序与 loss/normalizer。不能换成最新版 neuraloperator 的不同网络。
- U-NO 必须继续取得上游自定义 Adam、原积分算子、周期坐标、宽度/缩放，以及训练闭环目标。旧临时实现为确定性 bicubic 反向专门增加过原创适配；一个成功 import 的新 loader 并不自动具备相同的确定性完整训练功能。
- U-Net 必须保留 46 输入通道、上游层布局、4 步闭环训练目标、该模型实际 TF32 设置，而非统一替换成其他模型的网络/全局精度策略。
- PINN/PDE 的数据模态、候选库顺序、dtype、优化器与完整预算不能用更现代库“近似替代”。PINN 新外置执行路径仍有实质技术工作，见 [M1](adapters/M1.md)。

详细 FNO、UNO、UNet 加载说明由对应 `docs/adapters/` 文档及外部源清单给出；其中静态校验、模型构造小测试与完整训练证据应分别标记，不能混称“已复现”。

## 数值包与发布前尚需确认的事项

模型权重/数值输出与源代码不是同一类对象，但不能由“不是 .py”推断可无条件再分发。对源数据、生成方式、权重及初始化分别记录来源与权利状态；不要用统一 LICENSE 把未确认对象宣称为项目独占。特别要检查模型序列化是否内嵌可执行模型定义、TensorFlow 图、源快照或代码文本，而不只看文件扩展名。

PINN 未训练初始化 NPZ 已有可定位的生成函数和固定 hash，不是训练后检查点；生成来源/发布权利仍保留待确认标记。详见 [初始化说明](adapters/M1.md#43-未训练初始化文件与自身续算的区别)。

发布前的最小剩余工作：

1. 对候选采用原创文件白名单，不用整项目复制后仅排除 upstream 的方式。
2. 按现有 `external_sources.json` 核对外部 clone 的固定提交、选定文件 hash、许可来源/缺口；PINN 的 TF 嵌套优化器来源身份已补齐，仍须保持实际运行绑定。
3. 完成新的薄接口与必要原创适配；先做小型结构/参数初始化/输入与 loss 或梯度等价检查，再决定是否需要重跑受影响任务。不能把小测试当完整预算结果。
4. 保留既有真实失败与未验证状态。旧公开总入口的工程失败、部分数值未通过、新加载器尚未等价验证是不同问题，分别处理，不能通过换参考/降容差或伪造成功掩盖。

这些来源整理、外置加载和结果读取工程本身不要求重新训练所有模型；但在新接口没有相应真实执行证据前，不承诺其从零训练和自身续算已达到 100% 复现。
