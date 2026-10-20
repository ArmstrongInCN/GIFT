# 外部算法来源与必要适配

**GIFT 仓库不包含外部算法实现、自动源码改写器或可执行补丁。** 本文与[手工适配操作说明](EXTERNAL_MANUAL_ADAPTATION.md)只记录官方仓库、固定版本、需要实施的操作和外部接口。使用者须在 GIFT 仓库文件夹之外下载、适配和保存外部算法，不能因目录已被 Git 忽略就把代码放在仓库内。

GIFT 的 `adapters/` 只连接使用者准备的仓库外运行接口。`GIFT_EXTERNAL_ADAPTER_ROOT` 指向包含外部 `adapters/` 模块的根目录，缺省为相邻 `gift-external-runtime/`；`GIFT_EXTERNAL_ROOT` 指向官方 checkout 父目录；`GIFT_ANIE_SOURCE_ROOT` 指向 ANIE checkout。本机现有接口保存于 `D:/2_GIFT/gift-external-runtime`，原官方 checkout 保持只读。这些本地依赖不属于 GIFT Git 仓库，也不随 GIFT 发布。新使用者需自行按文字步骤提供接口；纯下载工具不执行算法适配。

## 官方来源

| 方法 | 官方仓库 | 固定 commit |
| --- | --- | --- |
| FNO-2D / FNO-3D | [NeuralOperator](https://github.com/neuraloperator/neuraloperator) | `01d2aeca407f85ce1c1e78b313edb749973fc051` |
| U-NO | [UNO](https://github.com/ashiq24/UNO) | `19462d82729ef64ef7b9e97056ddcaaf3044ad47` |
| U-Net | [Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | `229da3e01acefa6f4dd6c77244da5015c9b965af` |
| ANIE | [ANIE](https://github.com/emazap7/ANIE) | `2071ad709b260a6c305450affcf7168fd825f692` |
| PDE-FIND | [PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | `86911349b9fac82996e35465a13d06205730cc05` |
| PINN-SR | [EQDiscovery](https://github.com/isds-neu/EQDiscovery) | `9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496` |
| PINN 原生优化器 | [TensorFlow](https://github.com/tensorflow/tensorflow)，tag `v1.15.0` | `590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b` |

仓库内 `external_sources.json` 与 `anie_sources.json` 仅保存来源、版本、选定文件和文字操作标识。固定 commit 与散列用于来源记录，不能代替上游许可。许可状态和原始研究引用见 [README](../README.md)；安装、独立命令及目录设置见 [SETUP.md](SETUP.md)。

## 当前单初态 W1/W2

当前[固定协议](W1_W2_PROTOCOL.md)使用 `single_native76_n300`，各方法一个固定种子模型，不进行选参或多seed。来源适配与实验设置须分别理解：使用公开架构不等于执行作者原数据和训练脚本，也不意味着各方法监督时域或预算相等。

| 方法 | 仓库外必要操作 | 保留的计算与当前任务设置 |
| --- | --- | --- |
| FNO-2D / FNO-3D | 隔离模块顶层示例；在原唯一fc0位置把输入数改为3/4，包含一场及两/三坐标；提供原工具类。 | 保留谱层、forward、初始化、原LpLoss和normalizer。FNO-2D闭环课程属于训练协议；FNO-3D输出75未来点。W2保持75 joint queries，冻结normalizer空间Fourier插值后截前25，不拟合目标网格统计。 |
| U-NO | 提供原模型/积分算子/Adam/LpLoss；导入时不重设训练种子；用原构造参数设置in_width=5、width=32、pad=0、factor=.75。 | 一场加原四坐标；保留谱、插值梯度与BatchNorm。当前20步完整闭环监督及非重入activation checkpoint、每epoch调度和终态选择由协议声明。 |
| U-Net | 从原Baselines/U-net.py提供U_net，原构造参数设输入/输出通道1/1、kernel_size=3、dropout_rate=0。 | 保留卷积/BN/skip和初始化。四步训练预算与优化器为项目配方；N64/N96/N128使用同一全卷积模型，像素核物理尺度随N改变。 |
| ANIE | 提供原NS网络和积分求解器；将空间和时间坐标与状态统一为space-major/time-minor；功能性attention dropout跟随train/eval状态，训练概率仍0.5。 | 当前aligned_ns_single_v1使用原c=0/y_init分支；保持原全局decoder、积分归一化、迭代规则和autograd。训练/评价均76查询，micro6/effective32为显存组织；仅参加W1。 |

ANIE 的坐标与评价 dropout 修正会改变未经修正的函数，因此当前结果称为带明确修正的官方 notebook 架构实验，不称为作者实现逐字原样复现。此处的仓库边界整理只移动既有本地依赖，不再引入这些数值变化，也不将旧权重切换到另一profile。

各方法的参数源、监督覆盖、loss、真实预算和数值环境见[单次固定配方实现](W1_W2_PROTOCOL.md)、[补充基线实现](W1_W2_PROTOCOL.md)、[W2实现](W1_W2_PROTOCOL.md)与[结果解释范围](W1_W2_PROTOCOL.md)。ANIE的必要修正及其当前配置见[ANIE实现说明](ANIE_ADAPTATION.md)。实际模型及权重身份见 [CHECKPOINTS.md](CHECKPOINTS.md)，成本见 [PERFORMANCE.md](PERFORMANCE.md)。

## 独立46帧与方程识别协议

历史46帧预测使用独立输入投影：FNO-2D/FNO-3D为48/49，U-NO为50，U-Net输入通道46。不能把单初态权重和历史46帧权重互换。ANIE不同profile同样须按权重记录的坐标和自由项分支加载；不能静默升级旧模型的运算路径。

PDE-FIND 和 PINN-SR 服务独立 W3，不随 W1/W2改动。其仓库外操作为：

| 方法 | 操作范围 |
| --- | --- |
| PDE-FIND | 提供原库构造/回归函数；所有NumPy版本均将唯一空索引比较改为显式长度判断（1.26已会报错）。按原意执行非空索引的最终最小二乘；旧单元素数组误判的细节见手工说明。保持seed0样本和顺序；局部RNG及成员查找仅隔离/加速抽样。周期速度、SVD及导数处理沿用声明协议。 |
| PINN-SR 网络 | 提供原PhysicsInformedNN；TF1引用变量使原SciPy候选feed实际进入目标；已知库4/开放库90。保留Xavier、网络、损失及完整人口。默认原生FP32；可选同式坐标导数/残差计算与XLA须通过完整配对精度验收。 |
| PINN-SR 稀疏回归 | 使用原TrainSTRidge/STRidge；保留seed0随机训练行原顺序和float32继承系数，rcond=−1及唯一空索引长度判断。立即明确赋值，纯读取无副作用；ADO固定回归系数，只更新网络，post固定最后STRidge支持并联合细化。 |
| TensorFlow优化器 | 使用固定官方NAdam和ScipyOptimizerInterface，不复写内核。保留同一NAdam slots/beta；记录真实SciPy终止与目标变化。耦合保存回归支持与模型/optimizer/FSM；L-BFGS内部中断从阶段边界重做，不宣称内部求解器状态续算。 |

当前W3的两种PINN方法各完成50次独立训练，共100份正式终态模型，采用论文Methods式(6b)/(8)的`paper_ado_v1`，保持固定回归系数ADO与最后回归支持后训练。TF1引用变量、纯读取与原回归行顺序的操作见[手工适配说明](EXTERNAL_MANUAL_ADAPTATION.md#pinn-sr-与-tensorflow-优化器)，完整数据和实际结果见[W3协议](W3_PROTOCOL.md)。

具体目标文件、构造顺序、张量轴、坐标规则、库列序、接口名称与检查步骤均见[手工适配操作说明](EXTERNAL_MANUAL_ADAPTATION.md)。小规模加载、前向或续算检查只能验证对应计算条件，不能代替完整科学实验。

## 致谢

感谢 NeuralOperator、U-NO、Turbulent-Flow-Nets、ANIE、EQDiscovery、PDE-FIND、TensorFlow 与 SciPy 的作者和贡献者。请引用实际使用的研究并遵守上游条款；GIFT 许可不授予外部代码的额外权利。
